"""Cache management endpoints."""
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Path, Query, Depends
from cache_utils import (
    get_cache_stats, get_cache_warmer, get_cache_invalidator,
    get_job_manager, CACHE_WARMING_ENABLED, CACHE_STATS_ENABLED,
    CACHE_INVALIDATION_ENABLED, CACHE_WARMING_INTERVAL_HOURS
)
from utils.firebase import verify_firebase_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/cache-warm")
async def trigger_cache_warming():
    """Trigger manual cache warming for all popular locations (legacy endpoint - now uses job system)."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    # Create cache warming job
    job_id = job_manager.create_job("cache_warming", {
        "type": "all",
        "locations": [],
        "triggered_by": "legacy_api",
        "triggered_at": datetime.now(timezone.utc).isoformat()
    })
    
    return {
        "status": "job_created",
        "message": "Cache warming job created successfully (legacy endpoint)",
        "job_id": job_id,
        "job_status_url": f"/cache-warm/job/{job_id}",
        "locations_to_warm": get_cache_warmer().get_locations_to_warm()
    }


@router.get("/cache-warm/status")
async def get_cache_warming_status():
    """Get cache warming status and statistics."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    return get_cache_warmer().get_warming_stats()


@router.get("/cache-warm/locations")
async def get_locations_to_warm():
    """Get list of locations that would be warmed."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    return {
        "locations": get_cache_warmer().get_locations_to_warm(),
        "dates": get_cache_warmer().get_dates_to_warm(),
        "month_days": get_cache_warmer().get_month_days_to_warm()
    }


@router.post("/cache-warm/startup")
async def trigger_startup_warming():
    """Trigger cache warming on startup (useful for deployment) - now uses job system."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    # Create cache warming job
    job_id = job_manager.create_job("cache_warming", {
        "type": "all",
        "locations": [],
        "triggered_by": "startup_api",
        "triggered_at": datetime.now(timezone.utc).isoformat()
    })
    
    return {
        "status": "job_created",
        "message": "Startup cache warming job created successfully",
        "job_id": job_id,
        "job_status_url": f"/cache-warm/job/{job_id}",
        "locations_to_warm": get_cache_warmer().get_locations_to_warm()
    }


@router.get("/cache-warm/schedule")
async def get_warming_schedule():
    """Get information about the warming schedule."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    cache_warmer = get_cache_warmer()
    return {
        "enabled": CACHE_WARMING_ENABLED,
        "interval_hours": CACHE_WARMING_INTERVAL_HOURS,
        "next_warming_in_hours": CACHE_WARMING_INTERVAL_HOURS,  # Simplified - would need more complex logic for exact timing
        "last_warming": cache_warmer.last_warming_time.isoformat() if cache_warmer and cache_warmer.last_warming_time else None,
        "warming_in_progress": cache_warmer.warming_in_progress if cache_warmer else False
    }


@router.post("/cache-warm/job")
async def trigger_cache_warming_job(
    warming_type: str = "all",
    locations: Optional[List[str]] = None
):
    """Trigger cache warming as a background job."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    # Validate warming type
    if warming_type not in ["all", "popular", "specific"]:
        return {"status": "error", "message": "Invalid warming type. Must be 'all', 'popular', or 'specific'"}
    
    # Validate locations for specific warming
    if warming_type == "specific" and not locations:
        return {"status": "error", "message": "Locations must be provided for specific warming"}
    
    # Create cache warming job
    job_id = job_manager.create_job("cache_warming", {
        "type": warming_type,
        "locations": locations or [],
        "triggered_by": "api",
        "triggered_at": datetime.now(timezone.utc).isoformat()
    })
    
    return {
        "status": "job_created",
        "message": "Cache warming job created successfully",
        "job_id": job_id,
        "warming_type": warming_type,
        "locations": locations or [],
        "job_status_url": f"/jobs/{job_id}"
    }


@router.get("/cache-warm/job/{job_id}")
async def get_cache_warming_job_status(job_id: str):
    """Get status of a cache warming job."""
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    job_status = job_manager.get_job_status(job_id)
    if not job_status:
        return {"status": "not_found", "message": "Job not found"}
    
    return job_status


@router.get("/cache-stats")
async def get_cache_statistics(
    user=Depends(verify_firebase_token)
):
    """Get comprehensive cache statistics and performance metrics."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return cache_stats_instance.get_comprehensive_stats()


@router.get("/cache-stats/health")
async def get_cache_health(
    user=Depends(verify_firebase_token)
):
    """Get cache health assessment and alerts."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return cache_stats_instance.get_cache_health()


@router.get("/cache-stats/endpoints")
async def get_cache_endpoint_stats(
    user=Depends(verify_firebase_token)
):
    """Get cache statistics broken down by endpoint."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return {
        "by_endpoint": cache_stats_instance.get_endpoint_stats(),
        "overall": {
            "total_requests": cache_stats_instance.stats["total_requests"],
            "hit_rate": cache_stats_instance.get_hit_rate(),
            "error_rate": cache_stats_instance.get_error_rate()
        }
    }


@router.get("/cache-stats/locations")
async def get_cache_location_stats(
    user=Depends(verify_firebase_token)
):
    """Get cache statistics broken down by location."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return {
        "by_location": cache_stats_instance.get_location_stats(),
        "overall": {
            "total_requests": cache_stats_instance.stats["total_requests"],
            "hit_rate": cache_stats_instance.get_hit_rate(),
            "error_rate": cache_stats_instance.get_error_rate()
        }
    }


@router.get("/cache-stats/hourly")
async def get_cache_hourly_stats(
    hours: int = Query(24, ge=1, le=168),
    user=Depends(verify_firebase_token)
):
    """Get hourly cache statistics for the last N hours."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return {
        "hourly_data": cache_stats_instance.get_hourly_stats(hours),
        "requested_hours": hours
    }


@router.post("/cache-stats/reset")
async def reset_cache_statistics(
    user=Depends(verify_firebase_token)
):
    """Reset all cache statistics (admin endpoint)."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    cache_stats_instance.reset_stats()
    return {
        "status": "success",
        "message": "Cache statistics have been reset",
        "timestamp": datetime.now().isoformat()
    }


@router.delete("/cache/invalidate/key/{cache_key:path}")
async def invalidate_cache_key(
    cache_key: str,
    dry_run: bool = False
):
    """Invalidate a specific cache key."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_key(cache_key, dry_run)
    return result


@router.delete("/cache/invalidate/pattern")
async def invalidate_by_pattern(
    pattern: str,
    dry_run: bool = False
):
    """Invalidate cache keys matching a pattern."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_pattern(pattern, dry_run)
    return result


@router.delete("/cache/invalidate/endpoint/{endpoint}")
async def invalidate_by_endpoint(
    endpoint: str,
    location: Optional[str] = None,
    dry_run: bool = False
):
    """Invalidate cache keys for a specific endpoint and optionally location."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_endpoint(endpoint, location, dry_run)
    return result


@router.delete("/cache/invalidate/location/{location}")
async def invalidate_by_location(
    location: str,
    dry_run: bool = False
):
    """Invalidate all cache keys for a specific location."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_location(location, dry_run)
    return result


@router.delete("/cache/invalidate/date/{date}")
async def invalidate_by_date(
    date: str,
    dry_run: bool = False
):
    """Invalidate cache keys for a specific date."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_date(date, dry_run)
    return result


@router.delete("/cache/invalidate/forecast")
async def invalidate_forecast_data(
    dry_run: bool = False
):
    """Invalidate all forecast data."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_forecast_data(dry_run)
    return result


@router.delete("/cache/invalidate/today")
async def invalidate_today_data(
    dry_run: bool = False
):
    """Invalidate all data for today."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_today_data(dry_run)
    return result


@router.delete("/cache/invalidate/expired")
async def invalidate_expired_keys(
    dry_run: bool = False
):
    """Invalidate keys that have expired."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_expired_keys(dry_run)
    return result


@router.get("/cache/info")
async def get_cache_info():
    """Get information about current cache state."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    return get_cache_invalidator().get_cache_info()


@router.delete("/cache/clear")
async def clear_all_cache(
    dry_run: bool = False
):
    """Clear all cache data (use with caution!)."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().clear_all_cache(dry_run)
    return result
