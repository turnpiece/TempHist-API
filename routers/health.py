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


@router.get("/health/detailed")
async def detailed_health_check(redis_client: Annotated[redis.Redis, Depends(get_redis_client)]):
    """Comprehensive health check endpoint for debugging and monitoring (LOW-007: Enhanced dependencies check)."""
    health_status = {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": __version__, "checks": {}}

    overall_healthy = True

    # Check Redis connection (LOW-007: Enhanced)
    try:
        redis_client.ping()
        # Test read/write
        test_key = "health_check:test"
        redis_client.setex(test_key, 60, "test_value")
        test_value = redis_client.get(test_key)
        redis_client.delete(test_key)

        if test_value == "test_value":
            health_status["checks"]["redis"] = {"status": "healthy", "message": "Connection and read/write successful"}
        else:
            health_status["checks"]["redis"] = {
                "status": "degraded",
                "message": "Connection successful but read/write test failed",
            }
            overall_healthy = False
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False
        health_status["status"] = "unhealthy"

    # Check Firebase auth service (LOW-007)
    try:
        from firebase_admin import auth

        # Attempt to verify a clearly invalid token - if it rejects properly, service is up
        auth.verify_id_token("invalid_token_test", check_revoked=False)
        health_status["checks"]["firebase"] = {
            "status": "unknown",
            "message": "Firebase accepted invalid token (unexpected)",
        }
    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "token" in error_str or "expired" in error_str:
            # Expected rejection of invalid token means service is healthy
            health_status["checks"]["firebase"] = {"status": "healthy", "message": "Service responding correctly"}
        else:
            health_status["checks"]["firebase"] = {"status": "degraded", "error": f"Unexpected error: {str(e)}"}
            if overall_healthy:
                health_status["status"] = "degraded"

    # Check Open-Meteo API availability and recent production-traffic failures.
    if WEATHER_PROVIDER != "open_meteo":
        health_status["checks"]["open_meteo_api"] = {
            "status": "skipped",
            "provider": WEATHER_PROVIDER,
            "message": "Open-Meteo monitoring is informational when another provider is active",
        }
    else:
        probe_status = "unknown"
        probe_status_code = None
        probe_error = None
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SHORT) as client:
                resp = await client.get(
                    f"{OPEN_METEO_ARCHIVE_URL}?latitude=51.5&longitude=-0.1"
                    f"&start_date={yesterday}&end_date={yesterday}"
                    f"&daily=temperature_2m_mean&timezone=UTC",
                    timeout=HTTP_TIMEOUT_SHORT,
                )
                probe_status_code = resp.status_code
                probe_status = "healthy" if resp.status_code < 400 else "degraded"
        except Exception as e:
            probe_status = "unhealthy"
            probe_error = str(e)

        open_meteo_stats = get_open_meteo_stats()
        if open_meteo_stats:
            open_meteo_health = open_meteo_stats.get_health(probe_status=probe_status)
        else:
            open_meteo_health = {
                "status": probe_status,
                "monitoring_enabled": False,
                "message": "Open-Meteo stats unavailable",
            }

        open_meteo_health.update(
            {
                "provider": WEATHER_PROVIDER,
                "probe_status": probe_status,
                "probe_status_code": probe_status_code,
            }
        )
        if probe_error:
            open_meteo_health["probe_error"] = probe_error
        health_status["checks"]["open_meteo_api"] = open_meteo_health

        if open_meteo_health["status"] == "unhealthy":
            overall_healthy = False
            health_status["status"] = "unhealthy"
        elif open_meteo_health["status"] == "degraded" and overall_healthy:
            health_status["status"] = "degraded"

    # Check worker service health (LOW-007)
    try:
        heartbeat_key = "worker:heartbeat"
        heartbeat = redis_client.get(heartbeat_key)
        if heartbeat:
            # Try to parse timestamp if it's there
            try:
                heartbeat_time = float(heartbeat)
                age_seconds = time.time() - heartbeat_time
                if age_seconds < 300:  # Less than 5 minutes old
                    health_status["checks"]["worker"] = {
                        "status": "healthy",
                        "last_heartbeat_seconds_ago": round(age_seconds, 2),
                    }
                else:
                    health_status["checks"]["worker"] = {
                        "status": "degraded",
                        "message": f"Worker heartbeat is {round(age_seconds / 60, 1)} minutes old (may be idle)",
                    }
            except (ValueError, TypeError):
                # Heartbeat exists but not in expected format
                health_status["checks"]["worker"] = {"status": "healthy", "message": "Worker heartbeat present"}
        else:
            health_status["checks"]["worker"] = {
                "status": "unknown",
                "message": "No worker heartbeat found (worker may not be running)",
            }
    except Exception as e:
        health_status["checks"]["worker"] = {
            "status": "unknown",
            "error": f"Could not check worker heartbeat: {str(e)}",
        }

    # Check cache statistics if available
    try:
        cache_stats_instance = get_cache_stats()
        if cache_stats_instance and hasattr(cache_stats_instance, "get_cache_health"):
            cache_health = cache_stats_instance.get_cache_health()
            health_status["checks"]["cache"] = cache_health
            # Only consider cache "unhealthy" status as a failure, not "degraded"
            if cache_health.get("status") == "unhealthy":
                overall_healthy = False
    except Exception as e:
        health_status["checks"]["cache"] = {"status": "unknown", "message": f"Cache stats unavailable: {str(e)}"}

    # Check Postgres connectivity
    try:
        store = await get_daily_temperature_store()
        pg_result = await store.ping()
        health_status["checks"]["postgres"] = pg_result
        if pg_result["status"] == "unhealthy":
            overall_healthy = False
            health_status["status"] = "unhealthy"
        elif pg_result["status"] == "disabled" and overall_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["postgres"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False
        health_status["status"] = "unhealthy"

    # Check analytics rate limit pressure (reports 429s fired in the last hour)
    try:
        rejected = redis_client.get("analytics_429:total")
        rejected_count = int(rejected) if rejected else 0
        analytics_rl_status = "healthy" if rejected_count == 0 else "degraded"
        health_status["checks"]["analytics_rate_limit"] = {
            "status": analytics_rl_status,
            "enabled": RATE_LIMIT_ENABLED and ANALYTICS_RATE_LIMIT > 0,
            "limit_per_hour": ANALYTICS_RATE_LIMIT,
            "rejected_last_hour": rejected_count,
        }
        if rejected_count > 0 and health_status["status"] == "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["analytics_rate_limit"] = {"status": "unknown", "error": str(e)}

    # Return appropriate HTTP status code
    status_code = 200 if health_status["status"] == "healthy" else 503

    return JSONResponse(content=health_status, status_code=status_code, headers={"Cache-Control": "no-cache"})
