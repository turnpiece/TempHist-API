"""Shared dependencies for routers."""
import redis
from typing import Optional
from utils.location_validation import InvalidLocationCache

# Global instances - will be set by main.py during app initialization
_redis_client: Optional[redis.Redis] = None
_invalid_location_cache: Optional[InvalidLocationCache] = None


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


def initialize_dependencies(redis_client: redis.Redis):
    """Initialize shared dependencies. Called from main.py during app startup."""
    global _redis_client, _invalid_location_cache
    _redis_client = redis_client
    _invalid_location_cache = InvalidLocationCache(redis_client)
