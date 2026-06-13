"""
Singleton accessor functions and cache initialisation.

All module-level cache instances live here.  Call initialize_cache() once
at app startup (lifespan) before calling any get_*() function.
"""

import logging
from typing import Optional

import redis

from cache.core import (
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
from utils.open_meteo_stats import OpenMeteoStats

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
open_meteo_stats: Optional[OpenMeteoStats] = None


def initialize_cache(redis_client: redis.Redis):
    """Initialise all global cache/job singleton instances."""
    global enhanced_cache, job_manager, usage_tracker, cache_warmer, cache_stats, cache_invalidator, open_meteo_stats

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

    open_meteo_stats = OpenMeteoStats(redis_client)
    logger.info("Open-Meteo monitoring initialized" if not DEBUG else "📈 OPEN-METEO MONITORING INITIALIZED")

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


def get_open_meteo_stats() -> Optional[OpenMeteoStats]:
    return open_meteo_stats


def get_cache_invalidator() -> Optional[CacheInvalidator]:
    return cache_invalidator


