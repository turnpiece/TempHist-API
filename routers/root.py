"""Root endpoint and API information."""
from fastapi import APIRouter
from version import __version__

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
            "rolling_bundle": [
                "/v1/records/rolling-bundle/{location}/{anchor}"
            ],
            "periods": ["daily", "weekly", "monthly", "yearly"],
            "examples": [
                "/v1/records/daily/london/01-15",
                "/v1/records/weekly/london/01-15",
                "/v1/records/monthly/london/01-15",
                "/v1/records/yearly/london/01-15",
                "/v1/records/daily/london/01-15/updated",
                "/v1/records/rolling-bundle/london/2024-01-15"
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
    """Test endpoint for CORS with rolling-bundle path"""
    return {"message": "CORS is working for rolling-bundle", "path": "/test-cors-rolling"}
