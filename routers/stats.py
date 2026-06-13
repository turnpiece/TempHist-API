"""Statistics and rate limiting status endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cache.accessors import get_usage_tracker
from config import (
    API_ACCESS_TOKEN,
    MAX_LOCATIONS_PER_HOUR,
    MAX_REQUESTS_PER_HOUR,
    POPULARITY_WINDOW_DAYS,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_WINDOW_HOURS,
    USAGE_TRACKING_ENABLED,
)
from routers.dependencies import get_location_monitor, get_request_monitor, get_service_token_rate_limiter
from utils.admin_auth import verify_admin_key
from utils.ip_utils import get_client_ip, is_ip_blacklisted, is_ip_whitelisted

router = APIRouter()


@router.get("/rate-limit-status")
async def get_rate_limit_status(
    request: Request,
    location_monitor=Depends(get_location_monitor),
    request_monitor=Depends(get_request_monitor),
    service_token_rate_limiter=Depends(get_service_token_rate_limiter),
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
                "rate_limited": True,  # Service tokens are rate limited (high limits)
            },
            "service_token_rate_limits": service_stats,
            "rate_limits": {
                "requests_per_hour": service_token_rate_limiter.requests_per_hour,
                "locations_per_hour": service_token_rate_limiter.locations_per_hour,
                "window_hours": service_token_rate_limiter.window_seconds / 3600,
            },
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
                "rate_limited": not is_whitelisted and not is_blacklisted,
            },
            "location_monitor": location_stats,
            "request_monitor": request_stats,
            "rate_limits": {
                "max_locations_per_hour": MAX_LOCATIONS_PER_HOUR,
                "max_requests_per_hour": MAX_REQUESTS_PER_HOUR,
                "window_hours": RATE_LIMIT_WINDOW_HOURS,
            },
        }


@router.get("/rate-limit-stats")
async def get_rate_limit_stats(
    request: Request,
    _admin: Annotated[bool, Depends(verify_admin_key)],
    location_monitor=Depends(get_location_monitor),
    request_monitor=Depends(get_request_monitor),
):
    """Get overall rate limiting statistics (admin endpoint)."""
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
                "blacklisted": is_ip_blacklisted(ip),
            }

    return {
        "total_monitored_ips": len(all_stats),
        "suspicious_ips": list(location_monitor.suspicious_ips) if location_monitor else [],
        "whitelisted_ips": None,  # Will be injected from config
        "blacklisted_ips": None,  # Will be injected from config
        "ip_details": all_stats,
    }


@router.get("/usage-stats")
async def get_usage_stats(
    request: Request,
    _admin: Annotated[bool, Depends(verify_admin_key)],
):
    """Get usage tracking statistics from the selection signal (admin endpoint).

    Backed by the canonicalized selection sorted sets — the same source that
    feeds /v1/locations/popular and cache warming.
    """
    tracker = get_usage_tracker()
    if not USAGE_TRACKING_ENABLED or not tracker:
        return {"status": "disabled", "message": "Usage tracking is not enabled"}

    ranked = tracker.get_popular_from_selections(limit=500, days=POPULARITY_WINDOW_DAYS)
    return {
        "enabled": USAGE_TRACKING_ENABLED,
        "window_days": POPULARITY_WINDOW_DAYS,
        "total_selections": tracker.get_total_selections(days=POPULARITY_WINDOW_DAYS),
        "popular_locations": [{"location_id": loc_id, "selections": count} for loc_id, count in ranked],
    }


@router.get("/usage-stats/{location}")
async def get_location_usage_stats(
    location: str,
    _admin: Annotated[bool, Depends(verify_admin_key)],
):
    """Get selection statistics for a specific location ID (admin endpoint)."""
    tracker = get_usage_tracker()
    if not USAGE_TRACKING_ENABLED or not tracker:
        return {"status": "disabled", "message": "Usage tracking is not enabled"}

    ranked = tracker.get_popular_from_selections(limit=500, days=POPULARITY_WINDOW_DAYS)
    for rank, (loc_id, count) in enumerate(ranked, start=1):
        if loc_id == location:
            return {
                "location_id": loc_id,
                "selections": count,
                "rank": rank,
                "window_days": POPULARITY_WINDOW_DAYS,
            }

    return {"status": "not_found", "message": f"No selection data found for location: {location}"}
