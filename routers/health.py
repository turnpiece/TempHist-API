"""Health check and status endpoints."""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
import time
import httpx
import redis
from config import (
    DEBUG, API_KEY, OPENWEATHER_API_KEY, REDIS_URL
)
from cache_utils import get_cache_stats
from routers.dependencies import get_redis_client

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple health check endpoint for Render load balancers."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@router.get("/health/detailed")
async def detailed_health_check(redis_client: redis.Redis = Depends(get_redis_client)):
    """Comprehensive health check endpoint for debugging and monitoring (LOW-007: Enhanced dependencies check)."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
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
            health_status["checks"]["redis"] = {
                "status": "healthy",
                "message": "Connection and read/write successful"
            }
        else:
            health_status["checks"]["redis"] = {
                "status": "degraded",
                "message": "Connection successful but read/write test failed"
            }
            overall_healthy = False
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_healthy = False
        health_status["status"] = "unhealthy"
    
    # Check Firebase auth service (LOW-007)
    try:
        from firebase_admin import auth
        # Attempt to verify a clearly invalid token - if it rejects properly, service is up
        auth.verify_id_token("invalid_token_test", check_revoked=False)
        health_status["checks"]["firebase"] = {
            "status": "unknown",
            "message": "Firebase accepted invalid token (unexpected)"
        }
    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "token" in error_str or "expired" in error_str:
            # Expected rejection of invalid token means service is healthy
            health_status["checks"]["firebase"] = {
                "status": "healthy",
                "message": "Service responding correctly"
            }
        else:
            health_status["checks"]["firebase"] = {
                "status": "degraded",
                "error": f"Unexpected error: {str(e)}"
            }
            if overall_healthy:
                health_status["status"] = "degraded"
    
    # Check external API (Visual Crossing) availability (LOW-007)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Just check if the base URL is reachable (don't use API key in health check)
            resp = await client.get(
                "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/",
                timeout=5.0
            )
            if resp.status_code < 500:
                health_status["checks"]["visual_crossing_api"] = {
                    "status": "healthy",
                    "status_code": resp.status_code,
                    "message": "API reachable" if API_KEY else "API reachable but key not configured"
                }
                if not API_KEY:
                    health_status["status"] = "degraded" if overall_healthy else health_status["status"]
            else:
                health_status["checks"]["visual_crossing_api"] = {
                    "status": "degraded",
                    "status_code": resp.status_code,
                    "message": f"API returned status {resp.status_code}"
                }
                if overall_healthy:
                    health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["visual_crossing_api"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        if overall_healthy:
            health_status["status"] = "degraded"
    
    # Check API keys configuration
    if not OPENWEATHER_API_KEY:
        health_status["checks"]["openweather_api"] = {
            "status": "degraded",
            "message": "API key not configured"
        }
        if overall_healthy:
            health_status["status"] = "degraded"
    else:
        health_status["checks"]["openweather_api"] = {
            "status": "healthy",
            "message": "API key configured"
        }
    
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
                        "last_heartbeat_seconds_ago": round(age_seconds, 2)
                    }
                else:
                    health_status["checks"]["worker"] = {
                        "status": "degraded",
                        "message": f"Worker heartbeat is {round(age_seconds/60, 1)} minutes old (may be idle)"
                    }
            except (ValueError, TypeError):
                # Heartbeat exists but not in expected format
                health_status["checks"]["worker"] = {
                    "status": "healthy",
                    "message": "Worker heartbeat present"
                }
        else:
            health_status["checks"]["worker"] = {
                "status": "unknown",
                "message": "No worker heartbeat found (worker may not be running)"
            }
    except Exception as e:
        health_status["checks"]["worker"] = {
            "status": "unknown",
            "error": f"Could not check worker heartbeat: {str(e)}"
        }
    
    # Check cache statistics if available
    try:
        cache_stats_instance = get_cache_stats()
        if cache_stats_instance and hasattr(cache_stats_instance, 'get_cache_health'):
            cache_health = cache_stats_instance.get_cache_health()
            health_status["checks"]["cache"] = cache_health
            # Only consider cache "unhealthy" status as a failure, not "degraded"
            if cache_health.get("status") == "unhealthy":
                overall_healthy = False
    except Exception as e:
        health_status["checks"]["cache"] = {
            "status": "unknown",
            "message": f"Cache stats unavailable: {str(e)}"
        }
    
    # Return appropriate HTTP status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(
        content=health_status,
        status_code=status_code,
        headers={"Cache-Control": "no-cache"}
    )
