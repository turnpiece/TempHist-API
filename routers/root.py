"""Root endpoint and API information."""
import redis
from datetime import timedelta
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from version import __version__
from config import CACHE_CONTROL_HEADER
from cache_utils import set_cache_value, get_cache_value, get_cache_stats
from routers.dependencies import get_redis_client

router = APIRouter()


@router.api_route("/", methods=["GET", "OPTIONS"])
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "Temperature History API",
        "version": __version__,
        "description": "Temperature history API with v1 unified endpoints and legacy support",
        "v1_endpoints": {
            "records": [
                "/v1/records/{period}/{location}/{identifier}",
                "/v1/records/{period}/{location}/{identifier}/average",
                "/v1/records/{period}/{location}/{identifier}/trend",
                "/v1/records/{period}/{location}/{identifier}/summary",
                "/v1/records/{period}/{location}/{identifier}/updated"
            ],
            "periods": ["daily", "weekly", "monthly", "yearly"],
            "locations": [
                "/v1/locations/preapproved",
                "/v1/locations/preapproved/status"
            ],
            "examples": [
                "/v1/records/daily/london/01-15",
                "/v1/records/weekly/london/01-15",
                "/v1/records/monthly/london/01-15",
                "/v1/records/yearly/london/01-15",
                "/v1/records/daily/london/01-15/updated"
            ]
        },
        "removed_endpoints": {
            "status": "removed",
            "endpoints": [
                "/data/{location}/{month_day}",
                "/average/{location}/{month_day}",
                "/trend/{location}/{month_day}",
                "/summary/{location}/{month_day}"
            ],
            "note": "These endpoints have been removed. Please use v1 endpoints instead.",
            "migration": "Use /v1/records/daily/{location}/{month_day} and subresources"
        },
        "other_endpoints": [
            "/weather/{location}/{date}",
            "/forecast/{location}",
            "/rate-limit-status",
            "/rate-limit-stats",
            "/usage-stats",
            "/usage-stats/{location}",
            "/cache-warm",
            "/cache-warm/status",
            "/cache-warm/locations",
            "/cache-warm/startup",
            "/cache-warm/schedule",
            "/cache-stats",
            "/cache-stats/health",
            "/cache-stats/endpoints",
            "/cache-stats/locations",
            "/cache-stats/hourly",
            "/cache-stats/reset",
            "/cache/info",
            "/cache/invalidate/key/{cache_key}",
            "/cache/invalidate/pattern",
            "/cache/invalidate/endpoint/{endpoint}",
            "/cache/invalidate/location/{location}",
            "/cache/invalidate/date/{date}",
            "/cache/invalidate/forecast",
            "/cache/invalidate/today",
            "/cache/invalidate/expired",
            "/cache/clear"
        ],
        "analytics_endpoints": [
            "/analytics",
            "/analytics/summary", 
            "/analytics/recent",
            "/analytics/session/{session_id}"
        ]
    }


@router.api_route("/test-cors", methods=["GET", "OPTIONS"])
async def test_cors():
    """Test endpoint for CORS"""
    return {"message": "CORS is working"}


@router.api_route("/test-cors-rolling", methods=["GET", "OPTIONS"])
async def test_cors_rolling():
    """Test endpoint for CORS"""
    return {"message": "CORS is working", "path": "/test-cors-rolling"}


@router.get("/test-redis")
async def test_redis(redis_client: redis.Redis = Depends(get_redis_client)):
    """Test Redis connection."""
    try:
        # Try to set a test value
        set_cache_value("test_key", timedelta(minutes=5), "test_value", redis_client)
        # Try to get the test value
        test_value = get_cache_value("test_key", redis_client, "test", "test", get_cache_stats())
        if test_value:
            # Handle both bytes and string responses
            test_str = test_value.decode('utf-8') if isinstance(test_value, bytes) else test_value
            if test_str == "test_value":
                return JSONResponse(
                    content={"status": "success", "message": "Redis connection is working"},
                    headers={"Cache-Control": CACHE_CONTROL_HEADER}
                )
        return JSONResponse(
            content={"status": "error", "message": "Redis connection test failed"},
            headers={"Cache-Control": CACHE_CONTROL_HEADER}
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Redis connection error: {str(e)}"
            },
            headers={"Cache-Control": CACHE_CONTROL_HEADER}
        )
