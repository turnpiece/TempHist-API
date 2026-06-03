"""
Singleton accessor functions and cache initialisation.

All module-level cache instances live here.  Call initialize_cache() once
at app startup (lifespan) before calling any get_*() function.
"""

import asyncio
import logging
from datetime import date as dt_date
from typing import Optional

import redis
from fastapi import Request, Response

from cache.core import (
    CACHE_CONTROL_DAILY_TODAY,
    CACHE_CONTROL_PUBLIC,
    CACHE_TTL_SHORT,
    CacheHeaders,
    CacheInvalidator,
    EnhancedCache,
)
from cache.warming import (
    CACHE_STATS_ENABLED,
    CACHE_WARMING_DAYS_BACK,
    CACHE_WARMING_ENABLED,
    CACHE_WARMING_INTERVAL_HOURS,
    CacheStats,
    CacheWarmer,
)
from config import (
    DEBUG,
    USAGE_RETENTION_DAYS,
    USAGE_TRACKING_ENABLED,
)
from jobs.manager import JobManager
from tracking.usage import LocationUsageTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (set by initialize_cache)
# ---------------------------------------------------------------------------

enhanced_cache: Optional[EnhancedCache] = None
job_manager: Optional[JobManager] = None
usage_tracker: Optional[LocationUsageTracker] = None
cache_warmer: Optional[CacheWarmer] = None
cache_stats: Optional[CacheStats] = None
cache_invalidator: Optional[CacheInvalidator] = None


def initialize_cache(redis_client: redis.Redis):
    """Initialise all global cache/job singleton instances."""
    global enhanced_cache, job_manager, usage_tracker, cache_warmer, cache_stats, cache_invalidator

    enhanced_cache = EnhancedCache(redis_client)
    job_manager = JobManager(redis_client)

    if USAGE_TRACKING_ENABLED:
        usage_tracker = LocationUsageTracker(redis_client, USAGE_RETENTION_DAYS)
        if DEBUG:
            logger.info(f"📊 USAGE TRACKING INITIALIZED: {USAGE_RETENTION_DAYS} days retention")
        else:
            logger.info(f"Usage tracking enabled: {USAGE_RETENTION_DAYS} days retention")
    else:
        usage_tracker = None
        logger.info("Usage tracking disabled" if not DEBUG else "⚠️  USAGE TRACKING DISABLED")

    if CACHE_WARMING_ENABLED:
        cache_warmer = CacheWarmer(redis_client, usage_tracker)
        if DEBUG:
            logger.info(
                f"🔥 CACHE WARMING INITIALIZED: {CACHE_WARMING_INTERVAL_HOURS}h interval, "
                f"{CACHE_WARMING_DAYS_BACK} days back"
            )
        else:
            logger.info(f"Cache warming enabled: {CACHE_WARMING_INTERVAL_HOURS}h interval")
    else:
        cache_warmer = None
        logger.info("Cache warming disabled" if not DEBUG else "⚠️  CACHE WARMING DISABLED")

    if CACHE_STATS_ENABLED:
        cache_stats = CacheStats(redis_client)
        cache_stats._load_stats_from_redis()
        logger.info("Cache statistics enabled" if not DEBUG else "📊 CACHE STATS INITIALIZED")
    else:
        cache_stats = None
        logger.info("Cache statistics disabled" if not DEBUG else "⚠️  CACHE STATS DISABLED")

    from cache.core import CACHE_INVALIDATION_BATCH_SIZE, CACHE_INVALIDATION_ENABLED

    if CACHE_INVALIDATION_ENABLED:
        cache_invalidator = CacheInvalidator(redis_client)
        if DEBUG:
            logger.info(f"🗑️  CACHE INVALIDATION INITIALIZED: Batch size {CACHE_INVALIDATION_BATCH_SIZE}")
        else:
            logger.info("Cache invalidation enabled")
    else:
        cache_invalidator = None
        logger.info("Cache invalidation disabled" if not DEBUG else "⚠️  CACHE INVALIDATION DISABLED")

    logger.info("Enhanced cache and job manager initialized")


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------


def get_cache() -> EnhancedCache:
    if enhanced_cache is None:
        raise RuntimeError("Cache not initialized. Call initialize_cache() first.")
    return enhanced_cache


def get_job_manager() -> JobManager:
    if job_manager is None:
        raise RuntimeError("Job manager not initialized. Call initialize_cache() first.")
    return job_manager


def get_usage_tracker() -> Optional[LocationUsageTracker]:
    return usage_tracker


def get_cache_warmer() -> Optional[CacheWarmer]:
    return cache_warmer


def get_cache_stats() -> Optional[CacheStats]:
    return cache_stats


def get_cache_invalidator() -> Optional[CacheInvalidator]:
    return cache_invalidator


# ---------------------------------------------------------------------------
# High-level endpoint helper
# ---------------------------------------------------------------------------


async def cached_endpoint_response(
    request: Request,
    response: Response,
    cache_key: str,
    compute_func,
    ttl: int = None,
    cache_control: str = CACHE_CONTROL_PUBLIC,
    period: Optional[str] = None,
    date: Optional[dt_date] = None,
    location: Optional[str] = None,
    *args,
    **kwargs,
):
    """Wrap any endpoint with cache get-or-compute logic."""
    from cache.core import CACHE_TTL_DEFAULT
    from cache.keys import _is_today_in_location_timezone

    if ttl is None:
        ttl = CACHE_TTL_DEFAULT

    cache = get_cache()

    if period == "daily" and date is not None:
        if _is_today_in_location_timezone(date, location, cache.redis):
            ttl = CACHE_TTL_SHORT
            cache_control = CACHE_CONTROL_DAILY_TODAY

    try:
        data, etag, last_modified = await cache.get_or_compute(cache_key, compute_func, ttl, *args, **kwargs)
        if CacheHeaders.check_conditional_headers(request, etag, last_modified):
            response.status_code = 304
            return None
        CacheHeaders.set_cache_headers(response, etag, last_modified, cache_control)
        return data
    except Exception as e:
        logger.error(f"Cache error for {cache_key}: {e}")
        data = (
            await compute_func(*args, **kwargs)
            if asyncio.iscoroutinefunction(compute_func)
            else compute_func(*args, **kwargs)
        )
        return data
