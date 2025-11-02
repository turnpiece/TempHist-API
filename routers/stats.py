"""Statistics and rate limiting status endpoints."""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from config import (
    RATE_LIMIT_ENABLED, API_ACCESS_TOKEN, MAX_LOCATIONS_PER_HOUR,
    MAX_REQUESTS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS, USAGE_TRACKING_ENABLED
)
from utils.ip_utils import get_client_ip, is_ip_whitelisted, is_ip_blacklisted
from cache_utils import get_usage_tracker
from routers.dependencies import (
    get_service_token_rate_limiter, get_location_monitor, get_request_monitor
)

router = APIRouter()


def verify_firebase_token(request: Request):
    """Verify Firebase authentication token."""
    from fastapi import HTTPException
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    id_token = auth_header.split(" ")[1]
    try:
        from firebase_admin import auth
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid token")


@router.get("/rate-limit-status")
async def get_rate_limit_status(
    request: Request,
    location_monitor=Depends(get_location_monitor),
    request_monitor=Depends(get_request_monitor),
    service_token_rate_limiter=Depends(get_service_token_rate_limiter)
):
    """Get rate limiting status for the current client IP, including service token rate limits if applicable."""
    if not RATE_LIMIT_ENABLED:
        return {"status": "disabled", "message": "Rate limiting is not enabled"}
    
    client_ip = get_client_ip(request)
    
    # Check if this is a service job using API_ACCESS_TOKEN
    auth_header = request.headers.get("Authorization")
    is_service_job = False
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
            is_service_job = True
    
    # Check IP status
    is_whitelisted = is_ip_whitelisted(client_ip)
    is_blacklisted = is_ip_blacklisted(client_ip)
    
    # Get stats based on whether this is a service job or regular user
    if is_service_job and service_token_rate_limiter:
        # Service token rate limits (high limits, but still protected)
        service_stats = service_token_rate_limiter.get_stats(client_ip)
        return {
            "client_ip": client_ip,
            "ip_status": {
                "whitelisted": is_whitelisted,
                "blacklisted": is_blacklisted,
                "service_job": True,
                "rate_limited": True  # Service tokens are rate limited (high limits)
            },
            "service_token_rate_limits": service_stats,
            "rate_limits": {
                "requests_per_hour": service_token_rate_limiter.requests_per_hour,
                "locations_per_hour": service_token_rate_limiter.locations_per_hour,
                "window_hours": service_token_rate_limiter.window_seconds / 3600
            }
        }
    else:
        # Regular user rate limits (standard limits)
        location_stats = location_monitor.get_stats(client_ip) if location_monitor and not is_whitelisted else {}
        request_stats = request_monitor.get_stats(client_ip) if request_monitor and not is_whitelisted else {}
        
        return {
            "client_ip": client_ip,
            "ip_status": {
                "whitelisted": is_whitelisted,
                "blacklisted": is_blacklisted,
                "service_job": False,
                "rate_limited": not is_whitelisted and not is_blacklisted
            },
            "location_monitor": location_stats,
            "request_monitor": request_stats,
            "rate_limits": {
                "max_locations_per_hour": MAX_LOCATIONS_PER_HOUR,
                "max_requests_per_hour": MAX_REQUESTS_PER_HOUR,
                "window_hours": RATE_LIMIT_WINDOW_HOURS
            }
        }


@router.get("/rate-limit-stats")
async def get_rate_limit_stats(
    request: Request,
    user=Depends(verify_firebase_token),
    location_monitor=Depends(get_location_monitor),
    request_monitor=Depends(get_request_monitor)
):
    """Get overall rate limiting statistics (admin endpoint - requires authentication)."""
    if not RATE_LIMIT_ENABLED:
        return {"status": "disabled", "message": "Rate limiting is not enabled"}
    
    # Get stats for all monitored IPs
    all_stats = {}
    
    if location_monitor:
        for ip in location_monitor.ip_locations.keys():
            all_stats[ip] = {
                "location_stats": location_monitor.get_stats(ip),
                "request_stats": request_monitor.get_stats(ip) if request_monitor else {},
                "suspicious": location_monitor.is_suspicious(ip),
                "whitelisted": is_ip_whitelisted(ip),
                "blacklisted": is_ip_blacklisted(ip)
            }
    
    return {
        "total_monitored_ips": len(all_stats),
        "suspicious_ips": list(location_monitor.suspicious_ips) if location_monitor else [],
        "whitelisted_ips": None,  # Will be injected from config
        "blacklisted_ips": None,  # Will be injected from config
        "ip_details": all_stats
    }


@router.get("/usage-stats")
async def get_usage_stats(
    request: Request,
    user=Depends(verify_firebase_token)
):
    """Get usage tracking statistics."""
    if not USAGE_TRACKING_ENABLED or not get_usage_tracker():
        return {"status": "disabled", "message": "Usage tracking is not enabled"}
    
    return {
        "enabled": USAGE_TRACKING_ENABLED,
        "retention_days": None,  # Will be injected from config
        "popular_locations_24h": get_usage_tracker().get_popular_locations(limit=10, hours=24),
        "popular_locations_7d": get_usage_tracker().get_popular_locations(limit=10, hours=168),
        "all_location_stats": get_usage_tracker().get_all_location_stats()
    }


@router.get("/usage-stats/{location}")
async def get_location_usage_stats(location: str):
    """Get usage statistics for a specific location."""
    if not USAGE_TRACKING_ENABLED or not get_usage_tracker():
        return {"status": "disabled", "message": "Usage tracking is not enabled"}
    
    stats = get_usage_tracker().get_location_stats(location)
    if not stats:
        return {"status": "not_found", "message": f"No usage data found for location: {location}"}
    
    return stats
