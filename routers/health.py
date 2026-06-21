"""Health check and status endpoints."""

import time
from datetime import datetime, timedelta
from typing import Annotated

import httpx
import redis
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from cache.accessors import get_cache_stats, get_open_meteo_stats
from config import ANALYTICS_RATE_LIMIT, HTTP_TIMEOUT_SHORT, OPEN_METEO_ARCHIVE_URL, RATE_LIMIT_ENABLED, WEATHER_PROVIDER
from routers.dependencies import get_redis_client
from utils.daily_temperature_store import get_daily_temperature_store
from version import __version__

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple health check endpoint for Render load balancers."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def _check_redis(redis_client: redis.Redis) -> dict:
    try:
        redis_client.ping()
        test_key = "health_check:test"
        redis_client.setex(test_key, 60, "test_value")
        test_value = redis_client.get(test_key)
        redis_client.delete(test_key)
        if test_value == "test_value":
            return {"status": "healthy", "message": "Connection and read/write successful"}
        return {"status": "degraded", "message": "Connection successful but read/write test failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def _check_firebase() -> dict:
    try:
        from firebase_admin import auth
        auth.verify_id_token("invalid_token_test", check_revoked=False)
        return {"status": "unknown", "message": "Firebase accepted invalid token (unexpected)"}
    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "token" in error_str or "expired" in error_str:
            return {"status": "healthy", "message": "Service responding correctly"}
        return {"status": "degraded", "error": f"Unexpected error: {str(e)}"}


async def _check_open_meteo() -> dict:
    if WEATHER_PROVIDER != "open_meteo":
        return {"status": "skipped", "provider": WEATHER_PROVIDER, "message": "Open-Meteo monitoring is informational when another provider is active"}
    probe_status, probe_status_code, probe_error = "unknown", None, None
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SHORT) as client:
            resp = await client.get(
                f"{OPEN_METEO_ARCHIVE_URL}?latitude=51.5&longitude=-0.1"
                f"&start_date={yesterday}&end_date={yesterday}&daily=temperature_2m_mean&timezone=UTC",
                timeout=HTTP_TIMEOUT_SHORT,
            )
            probe_status_code = resp.status_code
            probe_status = "healthy" if resp.status_code < 400 else "degraded"
    except Exception as e:
        probe_status, probe_error = "unhealthy", str(e)

    open_meteo_stats = get_open_meteo_stats()
    result = (
        open_meteo_stats.get_health(probe_status=probe_status)
        if open_meteo_stats
        else {"status": probe_status, "monitoring_enabled": False, "message": "Open-Meteo stats unavailable"}
    )
    result.update({"provider": WEATHER_PROVIDER, "probe_status": probe_status, "probe_status_code": probe_status_code})
    if probe_error:
        result["probe_error"] = probe_error
    return result


def _check_worker(redis_client: redis.Redis) -> dict:
    try:
        heartbeat = redis_client.get("worker:heartbeat")
        if not heartbeat:
            return {"status": "unknown", "message": "No worker heartbeat found (worker may not be running)"}
        try:
            age_seconds = time.time() - float(heartbeat)
            if age_seconds < 300:
                return {"status": "healthy", "last_heartbeat_seconds_ago": round(age_seconds, 2)}
            return {"status": "degraded", "message": f"Worker heartbeat is {round(age_seconds / 60, 1)} minutes old (may be idle)"}
        except (ValueError, TypeError):
            return {"status": "healthy", "message": "Worker heartbeat present"}
    except Exception as e:
        return {"status": "unknown", "error": f"Could not check worker heartbeat: {str(e)}"}


def _check_cache_stats() -> dict:
    try:
        cache_stats_instance = get_cache_stats()
        if cache_stats_instance and hasattr(cache_stats_instance, "get_cache_health"):
            return cache_stats_instance.get_cache_health()
        return {"status": "unknown", "message": "Cache stats not available"}
    except Exception as e:
        return {"status": "unknown", "message": f"Cache stats unavailable: {str(e)}"}


async def _check_postgres() -> dict:
    try:
        store = await get_daily_temperature_store()
        return await store.ping()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def _check_analytics_rate_limit(redis_client: redis.Redis) -> dict:
    try:
        rejected = redis_client.get("analytics_429:total")
        rejected_count = int(rejected) if rejected else 0
        return {
            "status": "healthy" if rejected_count == 0 else "degraded",
            "enabled": RATE_LIMIT_ENABLED and ANALYTICS_RATE_LIMIT > 0,
            "limit_per_hour": ANALYTICS_RATE_LIMIT,
            "rejected_last_hour": rejected_count,
        }
    except Exception as e:
        return {"status": "unknown", "error": str(e)}


def _apply_check(health_status: dict, overall_healthy: bool, key: str, result: dict) -> bool:
    health_status["checks"][key] = result
    status = result.get("status")
    if status == "unhealthy":
        overall_healthy = False
        health_status["status"] = "unhealthy"
    elif status == "degraded" and overall_healthy:
        health_status["status"] = "degraded"
    return overall_healthy


@router.get("/health/detailed")
async def detailed_health_check(redis_client: Annotated[redis.Redis, Depends(get_redis_client)]):
    """Comprehensive health check endpoint for debugging and monitoring (LOW-007: Enhanced dependencies check)."""
    health_status = {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": __version__, "checks": {}}
    overall_healthy = True

    redis_result = _check_redis(redis_client)
    overall_healthy = _apply_check(health_status, overall_healthy, "redis", redis_result)

    firebase_result = _check_firebase()
    overall_healthy = _apply_check(health_status, overall_healthy, "firebase", firebase_result)

    open_meteo_result = await _check_open_meteo()
    overall_healthy = _apply_check(health_status, overall_healthy, "open_meteo_api", open_meteo_result)

    worker_result = _check_worker(redis_client)
    health_status["checks"]["worker"] = worker_result  # worker status doesn't change overall_healthy

    cache_result = _check_cache_stats()
    if cache_result.get("status") != "unhealthy":
        health_status["checks"]["cache"] = cache_result
    else:
        overall_healthy = False
        health_status["checks"]["cache"] = cache_result

    pg_result = await _check_postgres()
    if pg_result["status"] == "disabled":
        overall_healthy = _apply_check(health_status, overall_healthy, "postgres", pg_result)
    else:
        overall_healthy = _apply_check(health_status, overall_healthy, "postgres", pg_result)

    analytics_result = _check_analytics_rate_limit(redis_client)
    health_status["checks"]["analytics_rate_limit"] = analytics_result
    if analytics_result.get("rejected_last_hour", 0) > 0 and health_status["status"] == "healthy":
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code, headers={"Cache-Control": "no-cache"})
