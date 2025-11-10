"""Shared dependencies for routers."""
import redis
from typing import Optional
from utils.location_validation import InvalidLocationCache
from analytics_storage import AnalyticsStorage

# Global instances - will be set by main.py during app initialization
_redis_client: Optional[redis.Redis] = None
_invalid_location_cache: Optional[InvalidLocationCache] = None
_service_token_rate_limiter = None
_location_monitor = None
_request_monitor = None
_analytics_storage: Optional[AnalyticsStorage] = None


def get_redis_client() -> redis.Redis:
    """Dependency to get Redis client."""
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized. Ensure app is properly started.")
    return _redis_client


def get_invalid_location_cache() -> InvalidLocationCache:
    """Dependency to get invalid location cache."""
    if _invalid_location_cache is None:
        raise RuntimeError("Invalid location cache not initialized. Ensure app is properly started.")
    return _invalid_location_cache


def get_service_token_rate_limiter():
    """Dependency to get service token rate limiter."""
    return _service_token_rate_limiter


def get_location_monitor():
    """Dependency to get location diversity monitor."""
    return _location_monitor


def get_request_monitor():
    """Dependency to get request rate monitor."""
    return _request_monitor


def get_analytics_storage() -> AnalyticsStorage:
    """Dependency to get analytics storage."""
    if _analytics_storage is None:
        raise RuntimeError("Analytics storage not initialized. Ensure app is properly started.")
    return _analytics_storage


def initialize_dependencies(
    redis_client: redis.Redis,
    invalid_location_cache: InvalidLocationCache,
    service_token_rate_limiter,
    location_monitor,
    request_monitor,
    analytics_storage: AnalyticsStorage
):
    """Initialize shared dependencies. Called from main.py during app startup."""
    global _redis_client, _invalid_location_cache, _service_token_rate_limiter
    global _location_monitor, _request_monitor, _analytics_storage
    
    _redis_client = redis_client
    _invalid_location_cache = invalid_location_cache
    _service_token_rate_limiter = service_token_rate_limiter
    _location_monitor = location_monitor
    _request_monitor = request_monitor
    _analytics_storage = analytics_storage
