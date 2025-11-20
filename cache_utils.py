"""
Enhanced caching utilities for Cloudflare-friendly API responses.

This module provides:
- Strong cache headers (Cache-Control, ETag, Last-Modified)
- Canonical cache key generation with parameter normalization
- Single-flight protection to prevent cache stampedes
- Cache hit/miss metrics and health monitoring
- Deterministic JSON serialization for consistent ETags
"""

import hashlib
import json
import time
import logging
import asyncio
import os
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta, timezone, date as dt_date

import redis
import aiohttp
from fastapi import Request, Response

from config import USAGE_TRACKING_ENABLED, USAGE_RETENTION_DAYS, DEBUG, API_ACCESS_TOKEN, BASE_URL

# Try to import zoneinfo (Python 3.9+). Fall back to UTC if unavailable.
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # Will fall back to UTC if not available

logger = logging.getLogger(__name__)

# Cache for preapproved locations timezone data (loaded on first use)
_preapproved_locations_cache: Optional[List[Dict[str, Any]]] = None

# Cache configuration with validation
def validate_ttl(ttl: int, name: str, default: int) -> int:
    """Validate cache TTL value with min/max bounds."""
    MIN_TTL = 60  # 1 minute minimum
    MAX_TTL = 31536000  # 365 days maximum (allows for very old historical data caching)
    
    if not isinstance(ttl, int):
        logger.warning(f"{name} TTL is not integer, using default {default}s")
        return default
    if ttl < MIN_TTL:
        logger.warning(f"{name} TTL {ttl}s too low (min {MIN_TTL}s), using {MIN_TTL}s")
        return MIN_TTL
    if ttl > MAX_TTL:
        logger.warning(f"{name} TTL {ttl}s too high (max {MAX_TTL}s), using {MAX_TTL}s")
        return MAX_TTL
    return ttl

# Validate TTL values (MED-012)
CACHE_TTL_DEFAULT = validate_ttl(int(os.getenv("CACHE_TTL_DEFAULT", "86400")), "CACHE_TTL_DEFAULT", 86400)  # 24 hours
CACHE_TTL_SHORT = validate_ttl(int(os.getenv("CACHE_TTL_SHORT", "3600")), "CACHE_TTL_SHORT", 3600)  # 1 hour  
CACHE_TTL_LONG = validate_ttl(int(os.getenv("CACHE_TTL_LONG", "604800")), "CACHE_TTL_LONG", 604800)  # 7 days
CACHE_TTL_JOB = validate_ttl(int(os.getenv("CACHE_TTL_JOB", "7200")), "CACHE_TTL_JOB", 7200)  # 2 hours for job results

# Year-based caching TTL constants
TTL_STABLE = validate_ttl(int(os.getenv("TTL_STABLE", "7776000")), "TTL_STABLE", 7776000)  # 90 days for past years
TTL_HISTORICAL = validate_ttl(int(os.getenv("TTL_HISTORICAL", "31536000")), "TTL_HISTORICAL", 31536000)  # 365 days for very old data (7+ years)
TTL_CURRENT_DAILY = validate_ttl(int(os.getenv("TTL_CURRENT_DAILY", "7200")), "TTL_CURRENT_DAILY", 7200)  # 2 hours for current year daily
TTL_CURRENT_WEEKLY = validate_ttl(int(os.getenv("TTL_CURRENT_WEEKLY", "14400")), "TTL_CURRENT_WEEKLY", 14400)  # 4 hours for current year weekly
TTL_CURRENT_MONTHLY = validate_ttl(int(os.getenv("TTL_CURRENT_MONTHLY", "43200")), "TTL_CURRENT_MONTHLY", 43200)  # 12 hours for current year monthly
TTL_CURRENT_YEARLY = validate_ttl(int(os.getenv("TTL_CURRENT_YEARLY", "86400")), "TTL_CURRENT_YEARLY", 86400)  # 24 hours for current year yearly
TTL_BUNDLE = validate_ttl(int(os.getenv("TTL_BUNDLE", "900")), "TTL_BUNDLE", 900)  # 15 minutes for assembled bundle

def get_ttl_for_year(year: int) -> int:
    """
    Get appropriate cache TTL based on how old the data is.

    Args:
        year: The year of the data

    Returns:
        TTL in seconds

    Strategy:
        - Historical data (7+ years old): 365 days (won't change)
        - Stable data (past years < 7 years): 90 days (rarely changes)
        - Current year: Determined by period-specific logic elsewhere
    """
    from datetime import datetime
    current_year = datetime.now().year
    age = current_year - year

    if age >= 7:
        # Very old historical data - cache for 1 year
        return TTL_HISTORICAL
    elif age > 0:
        # Recent past years - cache for 90 days
        return TTL_STABLE
    else:
        # Current year - use period-specific logic (handled by caller)
        return TTL_STABLE

# Coordinate precision for cache key normalization
COORD_PRECISION = 4  # 4 decimal places (~11m precision)

# Cache header configuration
CACHE_CONTROL_PUBLIC = "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800"
CACHE_CONTROL_PRIVATE = "private, max-age=300, stale-while-revalidate=3600"
# Shorter cache for today's daily data (uses forecast which may change)
CACHE_CONTROL_DAILY_TODAY = "public, max-age=1800, s-maxage=3600, stale-while-revalidate=7200"

# Cache Warming Configuration
CACHE_WARMING_ENABLED = os.getenv("CACHE_WARMING_ENABLED", "true").lower() == "true"
CACHE_WARMING_INTERVAL_HOURS = int(os.getenv("CACHE_WARMING_INTERVAL_HOURS", "4"))
CACHE_WARMING_DAYS_BACK = int(os.getenv("CACHE_WARMING_DAYS_BACK", "7"))
CACHE_WARMING_CONCURRENT_REQUESTS = int(os.getenv("CACHE_WARMING_CONCURRENT_REQUESTS", "3"))
CACHE_WARMING_MAX_LOCATIONS = int(os.getenv("CACHE_WARMING_MAX_LOCATIONS", "15"))

# Cache Statistics Configuration
CACHE_STATS_ENABLED = os.getenv("CACHE_STATS_ENABLED", "true").lower() == "true"
CACHE_STATS_RETENTION_HOURS = int(os.getenv("CACHE_STATS_RETENTION_HOURS", "24"))
CACHE_HEALTH_THRESHOLD = float(os.getenv("CACHE_HEALTH_THRESHOLD", "0.7"))  # 70% hit rate threshold

# Cache Invalidation Configuration
CACHE_INVALIDATION_ENABLED = os.getenv("CACHE_INVALIDATION_ENABLED", "true").lower() == "true"
CACHE_INVALIDATION_DRY_RUN = os.getenv("CACHE_INVALIDATION_DRY_RUN", "false").lower() == "true"
CACHE_INVALIDATION_BATCH_SIZE = int(os.getenv("CACHE_INVALIDATION_BATCH_SIZE", "100"))

# Usage Tracking Configuration (imported from config)
# USAGE_TRACKING_ENABLED and USAGE_RETENTION_DAYS are imported from config

# Environment variables for cache warming (imported from config)
# DEBUG, BASE_URL, and API_ACCESS_TOKEN are imported from config

class LocationUsageTracker:
    """Track location usage patterns for analytics and cache warming."""
    
    def __init__(self, redis_client: redis.Redis, retention_days: int = 7):
        """Initialize location usage tracker.

        Args:
            redis_client: Redis client for storing usage data
            retention_days: Number of days to retain usage statistics (default: 7)
        """
        self.redis_client = redis_client
        self.retention_days = retention_days
        self.retention_seconds = retention_days * 24 * 3600
        self.usage_prefix = "usage_"
        self.timestamp_prefix = "usage_ts_"
        
    def track_location_request(self, location: str, endpoint: str = None):
        """Track a location request for analytics and popularity detection."""
        if not USAGE_TRACKING_ENABLED:
            return
            
        location = location.lower()
        current_time = time.time()
        
        # Track total requests for this location
        usage_key = f"{self.usage_prefix}{location}"
        self.redis_client.incr(usage_key)
        self.redis_client.expire(usage_key, self.retention_seconds)
        
        # Track timestamp for time-based analysis
        timestamp_key = f"{self.timestamp_prefix}{location}"
        self.redis_client.lpush(timestamp_key, current_time)
        self.redis_client.expire(timestamp_key, self.retention_seconds)
        
        # Track endpoint-specific usage
        if endpoint:
            endpoint_key = f"{self.usage_prefix}{location}_{endpoint}"
            self.redis_client.incr(endpoint_key)
            self.redis_client.expire(endpoint_key, self.retention_seconds)
        
        # Add location to tracked locations set (for easy retrieval without KEYS command)
        tracked_locations_key = "tracked_locations"
        self.redis_client.sadd(tracked_locations_key, location)
        self.redis_client.expire(tracked_locations_key, self.retention_seconds)
        
        if DEBUG:
            logger.debug(f"📊 USAGE TRACKED: {location} | Endpoint: {endpoint or 'unknown'}")
    
    def get_popular_locations(self, limit: int = 10, hours: int = 24) -> List[tuple]:
        """Get most popular locations in the last N hours."""
        if not USAGE_TRACKING_ENABLED:
            return []
            
        cutoff_time = time.time() - (hours * 3600)
        location_counts = []
        
        # Get all tracked locations from the set (no KEYS command needed)
        tracked_locations_key = "tracked_locations"
        tracked_locations = self.redis_client.smembers(tracked_locations_key)
        
        for location_bytes in tracked_locations:
            location = location_bytes.decode() if isinstance(location_bytes, bytes) else location_bytes
            
            # Check if location has recent activity
            timestamp_key = f"{self.timestamp_prefix}{location}"
            timestamps = self.redis_client.lrange(timestamp_key, 0, -1)
            
            recent_count = 0
            for ts in timestamps:
                if float(ts) > cutoff_time:
                    recent_count += 1
            
            if recent_count > 0:
                location_counts.append((location, recent_count))
        
        # Sort by count and return top locations
        return sorted(location_counts, key=lambda x: x[1], reverse=True)[:limit]
    
    def get_location_stats(self, location: str) -> Dict:
        """Get detailed stats for a specific location."""
        if not USAGE_TRACKING_ENABLED:
            return {}
            
        location = location.lower()
        usage_key = f"{self.usage_prefix}{location}"
        timestamp_key = f"{self.timestamp_prefix}{location}"
        
        total_requests = int(self.redis_client.get(usage_key) or 0)
        timestamps = [float(ts) for ts in self.redis_client.lrange(timestamp_key, 0, -1)]
        
        # Calculate recent activity
        now = time.time()
        last_hour = sum(1 for ts in timestamps if ts > now - 3600)
        last_24h = sum(1 for ts in timestamps if ts > now - 86400)
        
        return {
            "location": location,
            "total_requests": total_requests,
            "last_hour": last_hour,
            "last_24h": last_24h,
            "first_request": min(timestamps) if timestamps else None,
            "last_request": max(timestamps) if timestamps else None
        }
    
    def get_all_location_stats(self) -> Dict:
        """Get stats for all tracked locations."""
        if not USAGE_TRACKING_ENABLED:
            return {}
            
        all_stats = {}
        tracked_locations_key = "tracked_locations"
        tracked_locations = self.redis_client.smembers(tracked_locations_key)
        
        for location_bytes in tracked_locations:
            location = location_bytes.decode() if isinstance(location_bytes, bytes) else location_bytes
            all_stats[location] = self.get_location_stats(location)
        
        return all_stats

class CacheKeyBuilder:
    """Build canonical cache keys with normalized parameters."""
    
    @staticmethod
    def normalize_params(params: Dict[str, Any]) -> Dict[str, str]:
        """Normalize query parameters for consistent cache keys."""
        normalized = {}
        
        for key, value in sorted(params.items()):
            # Skip None values and empty strings
            if value is None or value == "":
                continue
                
            # Convert to string and strip whitespace
            str_value = str(value).strip()
            if not str_value:
                continue
            
            # Normalize location names
            if key in ['location']:
                str_value = str_value.lower().replace(" ", "_").replace(",", "_")
                
            # Normalize coordinate precision
            if key in ['lat', 'lon', 'latitude', 'longitude']:
                try:
                    coord = float(str_value)
                    str_value = f"{coord:.{COORD_PRECISION}f}"
                except (ValueError, TypeError):
                    pass
            
            # Skip default values to reduce cache key variations
            if key == 'unit_group' and str_value == 'celsius':
                continue
            if key == 'month_mode' and str_value == 'rolling1m':
                continue
            if key == 'days_back' and str_value in ['7', '0']:
                continue
                
            normalized[key] = str_value
            
        return normalized
    
    @staticmethod
    def build_cache_key(
        endpoint: str, 
        path_params: Dict[str, str] = None,
        query_params: Dict[str, Any] = None,
        prefix: str = "temphist"
    ) -> str:
        """Build a canonical cache key from endpoint and parameters."""
        path_params = path_params or {}
        query_params = query_params or {}
        
        # Normalize path parameters
        path_parts = []
        for key in sorted(path_params.keys()):
            value = str(path_params[key]).strip().lower()
            # Apply same normalization as query params for location
            if key == 'location':
                value = value.replace(" ", "_").replace(",", "_")
            path_parts.append(f"{key}={value}")
        
        # Normalize query parameters
        normalized_query = CacheKeyBuilder.normalize_params(query_params)
        query_parts = []
        for key, value in sorted(normalized_query.items()):
            query_parts.append(f"{key}={value}")
        
        # Build the key
        key_parts = [prefix, endpoint]
        if path_parts:
            key_parts.extend(path_parts)
        if query_parts:
            key_parts.extend(query_parts)
            
        return ":".join(key_parts)

class ETagGenerator:
    """Generate and validate ETags for responses."""
    
    @staticmethod
    def generate_etag(data: Any) -> str:
        """Generate ETag from response data using SHA256 (128-bit security)."""
        # Ensure deterministic JSON serialization
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        # Use SHA256 with 32 characters (128-bit security) instead of broken MD5
        return f'"{hashlib.sha256(json_str.encode()).hexdigest()[:32]}"'
    
    @staticmethod
    def parse_etag(etag: str) -> Optional[str]:
        """Parse ETag from If-None-Match header."""
        if not etag:
            return None
        # Remove quotes if present
        return etag.strip('"\'')
    
    @staticmethod
    def matches_etag(response_etag: str, request_etag: str) -> bool:
        """Check if request ETag matches response ETag."""
        if not response_etag or not request_etag:
            return False
        return response_etag.strip('"\'') == request_etag.strip('"\'')

class CacheHeaders:
    """Manage cache headers for responses."""
    
    @staticmethod
    def set_cache_headers(
        response: Response,
        etag: str,
        last_modified: datetime,
        cache_control: str = CACHE_CONTROL_PUBLIC,
        vary: str = "Accept-Encoding"
    ):
        """Set comprehensive cache headers."""
        response.headers["Cache-Control"] = cache_control
        response.headers["ETag"] = etag
        response.headers["Last-Modified"] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
        if vary:
            response.headers["Vary"] = vary
    
    @staticmethod
    def check_conditional_headers(request: Request, etag: str, last_modified: datetime) -> bool:
        """Check if request has conditional headers that match cached response."""
        # Check If-None-Match (ETag)
        if_none_match = request.headers.get("If-None-Match")
        if if_none_match and ETagGenerator.matches_etag(etag, if_none_match):
            return True
            
        # Check If-Modified-Since
        if_modified_since = request.headers.get("If-Modified-Since")
        if if_modified_since:
            try:
                # Parse the If-Modified-Since header
                request_time = datetime.strptime(if_modified_since, "%a, %d %b %Y %H:%M:%S GMT")
                # Add 1 second tolerance for clock skew
                if last_modified <= request_time + timedelta(seconds=1):
                    return True
            except ValueError:
                # Invalid date format, ignore
                pass
                
        return False

class SingleFlightLock:
    """Prevent cache stampedes with Redis-based locks."""
    
    def __init__(self, redis_client: redis.Redis, lock_ttl: int = 30):
        """Initialize single-flight lock manager.

        Args:
            redis_client: Redis client for distributed locking
            lock_ttl: Lock time-to-live in seconds (default: 30)
        """
        self.redis = redis_client
        self.lock_ttl = lock_ttl
        self.lock_prefix = "lock:"
    
    async def acquire(self, key: str) -> bool:
        """Acquire a lock for the given key."""
        lock_key = f"{self.lock_prefix}{key}"
        try:
            # Use SET with NX and EX for atomic lock acquisition
            result = self.redis.set(lock_key, "1", nx=True, ex=self.lock_ttl)
            return result is not None
        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis error acquiring lock for {key}: {e}")
            return False

    def release(self, key: str):
        """Release a lock for the given key."""
        lock_key = f"{self.lock_prefix}{key}"
        try:
            self.redis.delete(lock_key)
        except (redis.RedisError, redis.ConnectionError) as e:
            logger.warning(f"Redis error releasing lock for {key}: {e}")

class EnhancedCache:
    """Enhanced Redis cache with single-flight protection and metrics."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize enhanced cache with single-flight protection.

        Args:
            redis_client: Redis client for caching operations
        """
        self.redis = redis_client
        self.lock_manager = SingleFlightLock(redis_client)
        self.cache_prefix = "cache:"
        self.metrics_prefix = "metrics:"

        # Metrics tracking
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.compute_times = []
    
    def _get_cache_key(self, key: str) -> str:
        """Get full cache key with prefix."""
        return f"{self.cache_prefix}{key}"
    
    def _get_etag_key(self, key: str) -> str:
        """Get ETag key for cache entry."""
        return f"{self.cache_prefix}{key}:etag"
    
    def _get_metrics_key(self, key: str) -> str:
        """Get metrics key for cache entry."""
        return f"{self.metrics_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Tuple[Any, str, datetime]]:
        """Get cached value with ETag and Last-Modified timestamp."""
        cache_key = self._get_cache_key(key)
        etag_key = self._get_etag_key(key)
        
        try:
            # Get cached data and metadata
            cached_data = self.redis.get(cache_key)
            cached_etag = self.redis.get(etag_key)
            cached_timestamp = self.redis.hget(cache_key, "timestamp")
            
            if cached_data and cached_etag and cached_timestamp:
                # Parse the data
                data = json.loads(cached_data)
                etag = cached_etag.decode() if isinstance(cached_etag, bytes) else cached_etag
                last_modified = datetime.fromisoformat(cached_timestamp.decode() if isinstance(cached_timestamp, bytes) else cached_timestamp)
                
                self.hits += 1
                return data, etag, last_modified
            else:
                self.misses += 1
                return None

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as redis_error:
            self.errors += 1
            logger.error(f"Redis error getting cache key {key}: {redis_error}")
            return None
        except (json.JSONDecodeError, ValueError) as decode_error:
            self.errors += 1
            logger.error(f"Decode error for cache key {key}: {decode_error}")
            return None
    
    async def get_updated_timestamp(self, key: str) -> Optional[datetime]:
        """Get the last updated timestamp for a cache key without fetching the full data."""
        cache_key = self._get_cache_key(key)
        
        try:
            cached_timestamp = self.redis.hget(cache_key, "timestamp")
            if cached_timestamp:
                return datetime.fromisoformat(cached_timestamp.decode() if isinstance(cached_timestamp, bytes) else cached_timestamp)
            return None

        except (redis.RedisError, redis.ConnectionError) as redis_error:
            logger.error(f"Redis error getting timestamp for {key}: {redis_error}")
            return None
        except (ValueError, TypeError) as parse_error:
            logger.error(f"Date parsing error for timestamp {key}: {parse_error}")
            return None
    
    async def set(
        self, 
        key: str, 
        data: Any, 
        ttl: int = CACHE_TTL_DEFAULT,
        etag: Optional[str] = None,
        last_modified: Optional[datetime] = None
    ) -> str:
        """Set cached value with ETag and metadata."""
        cache_key = self._get_cache_key(key)
        etag_key = self._get_etag_key(key)
        
        if etag is None:
            etag = ETagGenerator.generate_etag(data)
        if last_modified is None:
            last_modified = datetime.now(timezone.utc)
        
        try:
            # Store data and ETag
            json_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
            self.redis.setex(cache_key, ttl, json_data)
            self.redis.setex(etag_key, ttl, etag)
            
            # Store metadata
            self.redis.hset(cache_key, mapping={
                "timestamp": last_modified.isoformat(),
                "ttl": ttl,
                "size": len(json_data)
            })
            self.redis.expire(cache_key, ttl)

            return etag

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as redis_error:
            self.errors += 1
            logger.error(f"Redis error setting cache key {key}: {redis_error}")
            raise
        except (json.JSONEncodeError, TypeError) as encode_error:
            self.errors += 1
            logger.error(f"JSON encoding error for cache key {key}: {encode_error}")
            raise

    async def mset(
        self,
        items: List[Tuple[str, Any, int, Optional[str]]],
    ) -> List[str]:
        """
        Set multiple cache values using Redis pipeline for better performance.

        Args:
            items: List of tuples (key, data, ttl, optional_etag)

        Returns:
            List of ETags for each item
        """
        if not items:
            return []

        try:
            pipeline = self.redis.pipeline()
            etags = []
            last_modified = datetime.now(timezone.utc)

            for key, data, ttl, etag in items:
                cache_key = self._get_cache_key(key)
                etag_key = self._get_etag_key(key)

                if etag is None:
                    etag = ETagGenerator.generate_etag(data)
                etags.append(etag)

                # Add operations to pipeline
                json_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
                pipeline.setex(cache_key, ttl, json_data)
                pipeline.setex(etag_key, ttl, etag)
                pipeline.hset(cache_key, mapping={
                    "timestamp": last_modified.isoformat(),
                    "ttl": ttl,
                    "size": len(json_data)
                })
                pipeline.expire(cache_key, ttl)

            # Execute all operations in a single round trip
            pipeline.execute()
            return etags

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as redis_error:
            self.errors += len(items)
            logger.error(f"Redis error in batch set: {redis_error}")
            raise
        except (json.JSONEncodeError, TypeError) as encode_error:
            self.errors += len(items)
            logger.error(f"JSON encoding error in batch set: {encode_error}")
            raise

    async def mget(self, keys: List[str]) -> Dict[str, Optional[Tuple[Any, str, datetime]]]:
        """
        Get multiple cache values using Redis pipeline for better performance.

        Args:
            keys: List of cache keys to fetch

        Returns:
            Dictionary mapping keys to (data, etag, last_modified) tuples or None if not found
        """
        if not keys:
            return {}

        try:
            pipeline = self.redis.pipeline()

            # Build pipeline for all gets
            for key in keys:
                cache_key = self._get_cache_key(key)
                etag_key = self._get_etag_key(key)

                pipeline.get(cache_key)
                pipeline.get(etag_key)
                pipeline.hget(cache_key, "timestamp")

            # Execute all operations in a single round trip
            results = pipeline.execute()

            # Parse results (every 3 results = data, etag, timestamp for one key)
            parsed_results = {}
            for i, key in enumerate(keys):
                idx = i * 3
                cached_data = results[idx]
                cached_etag = results[idx + 1]
                cached_timestamp = results[idx + 2]

                if cached_data and cached_etag and cached_timestamp:
                    try:
                        data = json.loads(cached_data)
                        etag = cached_etag.decode() if isinstance(cached_etag, bytes) else cached_etag
                        last_modified = datetime.fromisoformat(
                            cached_timestamp.decode() if isinstance(cached_timestamp, bytes) else cached_timestamp
                        )
                        parsed_results[key] = (data, etag, last_modified)
                        self.hits += 1
                    except (json.JSONDecodeError, ValueError) as parse_error:
                        logger.warning(f"Parse error for key {key}: {parse_error}")
                        parsed_results[key] = None
                        self.errors += 1
                else:
                    parsed_results[key] = None
                    self.misses += 1

            return parsed_results

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as redis_error:
            self.errors += len(keys)
            logger.error(f"Redis error in batch get: {redis_error}")
            return {key: None for key in keys}

    async def get_or_compute(
        self,
        key: str,
        compute_func,
        ttl: int = CACHE_TTL_DEFAULT,
        *args,
        **kwargs
    ) -> Tuple[Any, str, datetime]:
        """Get from cache or compute with single-flight protection."""
        # Try to get from cache first
        cached_result = await self.get(key)
        if cached_result is not None:
            return cached_result
        
        # Try to acquire lock to prevent stampedes
        lock_acquired = await self.lock_manager.acquire(key)
        
        try:
            if lock_acquired:
                # We got the lock, compute the value
                start_time = time.time()
                data = await compute_func(*args, **kwargs) if asyncio.iscoroutinefunction(compute_func) else compute_func(*args, **kwargs)
                compute_time = time.time() - start_time
                
                # Store in cache
                etag = await self.set(key, data, ttl)
                last_modified = datetime.now(timezone.utc)
                
                # Track compute time
                self.compute_times.append(compute_time)
                
                return data, etag, last_modified
            else:
                # Another process is computing, wait and retry
                await asyncio.sleep(0.1)
                return await self.get_or_compute(key, compute_func, ttl, *args, **kwargs)
                
        finally:
            if lock_acquired:
                self.lock_manager.release(key)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        avg_compute_time = sum(self.compute_times) / len(self.compute_times) if self.compute_times else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "avg_compute_time": round(avg_compute_time, 3),
            "compute_times_count": len(self.compute_times)
        }
    
    def reset_metrics(self):
        """Reset cache metrics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.compute_times = []

class CacheWarmer:
    """Proactive cache warming for popular locations and recent dates."""
    
    def __init__(self, redis_client: redis.Redis, usage_tracker: 'LocationUsageTracker' = None):
        """Initialize cache warmer for proactive cache population.

        Args:
            redis_client: Redis client for checking cache status
            usage_tracker: Optional usage tracker for identifying popular locations
        """
        self.redis_client = redis_client
        self.usage_tracker = usage_tracker
        self.warming_in_progress = False
        self.last_warming_time = None
        self.warming_stats = {
            "total_warmed": 0,
            "successful_warmed": 0,
            "failed_warmed": 0,
            "last_warming_duration": 0
        }
    
    def get_locations_to_warm(self) -> List[str]:
        """Get list of locations to warm, prioritizing preapproved locations."""
        locations: List[str] = []
        seen_normalized = set()

        def add_location(candidate: str) -> None:
            if not candidate:
                return
            normalized = normalize_location_for_cache(candidate)
            if normalized in seen_normalized:
                return
            seen_normalized.add(normalized)
            locations.append(candidate)
        
        # Priority 1: Preapproved locations in web app format
        preapproved_locations = self.get_preapproved_locations()
        if preapproved_locations:
            for loc in preapproved_locations:
                add_location(loc)
            if DEBUG:
                logger.info(f"🔥 CACHE WARMING: Added {len(preapproved_locations)} preapproved locations")
        
        # Priority 2: Recently popular locations from usage tracking
        if self.usage_tracker and USAGE_TRACKING_ENABLED:
            recent_popular = self.usage_tracker.get_popular_locations(limit=10, hours=24)
            for location, count in recent_popular:
                add_location(location)
            if DEBUG and recent_popular:
                logger.info(f"🔥 CACHE WARMING: Added {len(recent_popular)} usage-based locations")
        
        # Limit to max locations
        final_locations = locations[:CACHE_WARMING_MAX_LOCATIONS]
        
        if DEBUG:
            logger.info(f"🔥 CACHE WARMING: Total locations to warm: {len(final_locations)}")
            logger.info(f"🔥 SAMPLE LOCATIONS: {final_locations[:5]}...")
        
        return final_locations
    
    def get_preapproved_locations(self) -> List[str]:
        """Get preapproved locations in the exact format the web app requests them."""
        try:
            import os
            import json
            
            # Path to preapproved locations data file
            # Find project root by looking for pyproject.toml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != os.path.dirname(project_root):  # Stop at filesystem root
                if os.path.exists(os.path.join(project_root, "pyproject.toml")):
                    break
                project_root = os.path.dirname(project_root)
            
            data_file = os.path.join(project_root, "data", "preapproved_locations.json")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract locations in the format: "Name, Admin1, Country Name"
            locations = []
            seen = set()
            for item in data:
                if 'name' in item and 'admin1' in item and 'country_name' in item:
                    full_name = f"{item['name']}, {item['admin1']}, {item['country_name']}"
                    normalized = normalize_location_for_cache(full_name)
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    locations.append(full_name)
            
            if DEBUG:
                logger.info(f"📋 PREAPPROVED LOCATIONS: Loaded {len(locations)} locations in web app format")
                logger.info(f"📋 SAMPLE LOCATIONS: {locations[:3]}...")
            
            return locations

        except FileNotFoundError as e:
            if DEBUG:
                logger.warning(f"⚠️  Preapproved locations file not found: {e}")
            return []
        except json.JSONDecodeError as e:
            if DEBUG:
                logger.warning(f"⚠️  Invalid JSON in preapproved locations: {e}")
            return []
        except (IOError, PermissionError) as e:
            if DEBUG:
                logger.warning(f"⚠️  Cannot read preapproved locations file: {e}")
            return []
    
    def get_dates_to_warm(self) -> List[str]:
        """Get list of dates to warm."""
        dates = []
        today = datetime.now().date()
        
        # Add recent dates
        for days_back in range(CACHE_WARMING_DAYS_BACK):
            date = today - timedelta(days=days_back)
            dates.append(date.strftime("%Y-%m-%d"))
        
        # Add current month days for current year
        current_year = today.year
        current_month = today.month
        for day in range(1, 32):
            try:
                date = dt_date(current_year, current_month, day)
                if date <= today:  # Only include past and current days
                    dates.append(date.strftime("%Y-%m-%d"))
            except ValueError:
                continue  # Skip invalid dates (e.g., Feb 30)
        
        return dates
    
    def get_month_days_to_warm(self) -> List[str]:
        """Get list of month-day combinations to warm."""
        month_days = []
        today = datetime.now()
        
        # Add current month-day
        month_days.append(f"{today.month:02d}-{today.day:02d}")
        
        # Add recent month-days
        for days_back in range(1, min(CACHE_WARMING_DAYS_BACK, 30)):
            date = today - timedelta(days=days_back)
            month_days.append(f"{date.month:02d}-{date.day:02d}")
        
        return month_days
    
    async def warm_location_data(self, location: str) -> Dict:
        """Warm all data for a specific location."""
        if DEBUG:
            logger.info(f"🔥 WARMING LOCATION: {location}")
        
        results = {
            "location": location,
            "warmed_endpoints": [],
            "errors": []
        }
        
        try:
            # Warm forecast data
            try:
                forecast_url = f"{BASE_URL}/forecast/{location}"
                auth_token = API_ACCESS_TOKEN
                if not auth_token:
                    results["errors"].append("forecast: No authentication token available")
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(forecast_url, headers={"Authorization": f"Bearer {auth_token}"}) as resp:
                            if resp.status == 200:
                                results["warmed_endpoints"].append("forecast")
                            else:
                                results["errors"].append(f"forecast: {resp.status}")
            except (aiohttp.ClientError, aiohttp.ClientConnectorError) as http_error:
                results["errors"].append(f"forecast: HTTP error - {str(http_error)}")
            except asyncio.TimeoutError:
                results["errors"].append(f"forecast: Request timeout")
            
            # Warm legacy endpoints (forecast, weather) - daily data now handled by v1 endpoints below
            month_days = self.get_month_days_to_warm()
            
            # Warm v1 endpoints (daily, weekly, monthly) - skipping yearly for phase 1
            for month_day in month_days:
                for period in ["daily", "weekly", "monthly"]:
                    # Main record endpoint
                    try:
                        v1_url = f"{BASE_URL}/v1/records/{period}/{location}/{month_day}"
                        auth_token = API_ACCESS_TOKEN
                        if not auth_token:
                            results["errors"].append(f"v1/records/{period}/{month_day}: No authentication token available")
                        else:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(v1_url, headers={"Authorization": f"Bearer {auth_token}"}) as resp:
                                    if resp.status == 200:
                                        results["warmed_endpoints"].append(f"v1/records/{period}/{month_day}")
                                    else:
                                        results["errors"].append(f"v1/records/{period}/{month_day}: {resp.status}")
                    except (aiohttp.ClientError, aiohttp.ClientConnectorError) as http_error:
                        results["errors"].append(f"v1/records/{period}/{month_day}: HTTP error - {str(http_error)}")
                    except asyncio.TimeoutError:
                        results["errors"].append(f"v1/records/{period}/{month_day}: Request timeout")
                    
                    # Subresource endpoints (average, trend, summary)
                    for subresource in ["average", "trend", "summary"]:
                        try:
                            sub_url = f"{BASE_URL}/v1/records/{period}/{location}/{month_day}/{subresource}"
                            auth_token = API_ACCESS_TOKEN
                            if not auth_token:
                                results["errors"].append(f"v1/records/{period}/{month_day}/{subresource}: No authentication token available")
                            else:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(sub_url, headers={"Authorization": f"Bearer {auth_token}"}) as resp:
                                        if resp.status == 200:
                                            results["warmed_endpoints"].append(f"v1/records/{period}/{month_day}/{subresource}")
                                        else:
                                            results["errors"].append(f"v1/records/{period}/{month_day}/{subresource}: {resp.status}")
                        except (aiohttp.ClientError, aiohttp.ClientConnectorError) as http_error:
                            results["errors"].append(f"v1/records/{period}/{month_day}/{subresource}: HTTP error - {str(http_error)}")
                        except asyncio.TimeoutError:
                            results["errors"].append(f"v1/records/{period}/{month_day}/{subresource}: Request timeout")
            
            # Warm individual weather data for recent dates
            dates = self.get_dates_to_warm()
            for date in dates[:5]:  # Limit to last 5 dates to avoid too many requests
                try:
                    weather_url = f"{BASE_URL}/weather/{location}/{date}"
                    auth_token = API_ACCESS_TOKEN
                    if not auth_token:
                        results["errors"].append(f"weather/{date}: No authentication token available")
                    else:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(weather_url, headers={"Authorization": f"Bearer {auth_token}"}) as resp:
                                if resp.status == 200:
                                    results["warmed_endpoints"].append(f"weather/{date}")
                                else:
                                    results["errors"].append(f"weather/{date}: {resp.status}")
                except (aiohttp.ClientError, aiohttp.ClientConnectorError) as http_error:
                    results["errors"].append(f"weather/{date}: HTTP error - {str(http_error)}")
                except asyncio.TimeoutError:
                    results["errors"].append(f"weather/{date}: Request timeout")

        except (aiohttp.ClientError, aiohttp.ClientConnectorError) as http_error:
            results["errors"].append(f"location_warming: HTTP error - {str(http_error)}")
        except Exception as e:
            results["errors"].append(f"location_warming: Unexpected error - {str(e)}")
            logger.error(f"Unexpected error warming location {location}: {e}", exc_info=True)
        
        return results
    
    async def warm_all_locations(self) -> Dict:
        """Warm cache for all popular locations."""
        if self.warming_in_progress:
            return {"status": "already_in_progress", "message": "Cache warming already in progress"}
        
        if not CACHE_WARMING_ENABLED:
            return {"status": "disabled", "message": "Cache warming is disabled"}
        
        # Check if Redis is available before attempting to warm cache
        try:
            self.redis_client.ping()
        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"⚠️ Cache warming skipped - Redis not available: {e}")
            return {"status": "skipped", "message": "Redis not available", "error": str(e)}
        
        self.warming_in_progress = True
        start_time = time.time()
        
        try:
            locations = self.get_locations_to_warm()
            if DEBUG:
                logger.info(f"🔥 STARTING CACHE WARMING: {len(locations)} locations")
                logger.info(f"🏙️  LOCATIONS: {', '.join(locations)}")
            
            # Warm preapproved locations endpoint once
            try:
                preapproved_url = f"{BASE_URL}/v1/locations/preapproved"
                auth_token = API_ACCESS_TOKEN
                if auth_token:
                    # Set a reasonable timeout for the request
                    timeout = aiohttp.ClientTimeout(total=10)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(preapproved_url, headers={"Authorization": f"Bearer {auth_token}"}) as resp:
                            if resp.status == 200:
                                if DEBUG:
                                    logger.info("✅ PREAPPROVED ENDPOINT: Warmed successfully")
                            else:
                                logger.warning(f"⚠️  PREAPPROVED ENDPOINT: HTTP {resp.status}")
                else:
                    logger.warning("⚠️  PREAPPROVED ENDPOINT: No authentication token available")
            except aiohttp.ClientConnectorError as e:
                logger.warning(f"⚠️  PREAPPROVED ENDPOINT: Cannot connect to {BASE_URL} - {str(e)}")
                logger.info("💡  TIP: Set BASE_URL environment variable to your API server URL on Railway")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️  PREAPPROVED ENDPOINT: Request timeout to {BASE_URL}")
            except Exception as e:
                logger.warning(f"⚠️  PREAPPROVED ENDPOINT: {str(e)}")
            
            # Warm locations with concurrency limit
            semaphore = asyncio.Semaphore(CACHE_WARMING_CONCURRENT_REQUESTS)
            
            async def warm_with_semaphore(location):
                async with semaphore:
                    return await self.warm_location_data(location)
            
            # Execute warming tasks
            tasks = [warm_with_semaphore(location) for location in locations]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_locations = 0
            total_endpoints = 0
            total_errors = 0
            
            for result in results:
                if isinstance(result, dict):
                    if result.get("warmed_endpoints"):
                        successful_locations += 1
                        total_endpoints += len(result["warmed_endpoints"])
                    total_errors += len(result.get("errors", []))
                else:
                    total_errors += 1
            
            # Update stats
            duration = time.time() - start_time
            self.warming_stats.update({
                "total_warmed": len(locations),
                "successful_warmed": successful_locations,
                "failed_warmed": len(locations) - successful_locations,
                "last_warming_duration": duration
            })
            self.last_warming_time = datetime.now()
            
            if DEBUG:
                logger.info(f"✅ CACHE WARMING COMPLETED: {successful_locations}/{len(locations)} locations | {total_endpoints} endpoints | {duration:.1f}s")
            
            return {
                "status": "completed",
                "locations_processed": len(locations),
                "successful_locations": successful_locations,
                "total_endpoints_warmed": total_endpoints,
                "total_errors": total_errors,
                "duration_seconds": duration,
                "results": results
            }
        
        except Exception as e:
            if DEBUG:
                logger.error(f"❌ CACHE WARMING FAILED: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "duration_seconds": time.time() - start_time
            }
        
        finally:
            self.warming_in_progress = False
    
    def get_warming_stats(self) -> Dict:
        """Get cache warming statistics."""
        return {
            "enabled": CACHE_WARMING_ENABLED,
            "in_progress": self.warming_in_progress,
            "last_warming_time": self.last_warming_time.isoformat() if self.last_warming_time else None,
            "stats": self.warming_stats,
            "configuration": {
                "interval_hours": CACHE_WARMING_INTERVAL_HOURS,
                "days_back": CACHE_WARMING_DAYS_BACK,
                "concurrent_requests": CACHE_WARMING_CONCURRENT_REQUESTS,
                "max_locations": CACHE_WARMING_MAX_LOCATIONS
            }
        }

class CacheStats:
    """Track and analyze cache performance statistics."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize cache statistics tracker.

        Args:
            redis_client: Redis client for storing statistics
        """
        self.redis_client = redis_client
        self.stats_prefix = "cache_stats_"
        self.retention_seconds = CACHE_STATS_RETENTION_HOURS * 3600
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_errors": 0,
            "endpoint_stats": {},
            "location_stats": {},
            "hourly_stats": {},
            "last_reset": time.time()
        }
    
    def track_cache_request(self, cache_key: str, hit: bool, endpoint: str = None, location: str = None, error: bool = False):
        """Track a cache request (hit, miss, or error)."""
        if not CACHE_STATS_ENABLED:
            return
        
        current_time = time.time()
        hour_key = int(current_time // 3600)  # Hour bucket
        
        # Update overall stats
        self.stats["total_requests"] += 1
        if error:
            self.stats["cache_errors"] += 1
        elif hit:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1
        
        # Update endpoint stats
        if endpoint:
            if endpoint not in self.stats["endpoint_stats"]:
                self.stats["endpoint_stats"][endpoint] = {"hits": 0, "misses": 0, "errors": 0, "total": 0}
            
            self.stats["endpoint_stats"][endpoint]["total"] += 1
            if error:
                self.stats["endpoint_stats"][endpoint]["errors"] += 1
            elif hit:
                self.stats["endpoint_stats"][endpoint]["hits"] += 1
            else:
                self.stats["endpoint_stats"][endpoint]["misses"] += 1
        
        # Update location stats
        if location:
            if location not in self.stats["location_stats"]:
                self.stats["location_stats"][location] = {"hits": 0, "misses": 0, "errors": 0, "total": 0}
            
            self.stats["location_stats"][location]["total"] += 1
            if error:
                self.stats["location_stats"][location]["errors"] += 1
            elif hit:
                self.stats["location_stats"][location]["hits"] += 1
            else:
                self.stats["location_stats"][location]["misses"] += 1
        
        # Update hourly stats
        if hour_key not in self.stats["hourly_stats"]:
            self.stats["hourly_stats"][hour_key] = {"hits": 0, "misses": 0, "errors": 0, "total": 0}
        
        self.stats["hourly_stats"][hour_key]["total"] += 1
        if error:
            self.stats["hourly_stats"][hour_key]["errors"] += 1
        elif hit:
            self.stats["hourly_stats"][hour_key]["hits"] += 1
        else:
            self.stats["hourly_stats"][hour_key]["misses"] += 1
        
        # Store in Redis for persistence
        self._store_stats_in_redis()
    
    def _store_stats_in_redis(self):
        """Store current stats in Redis for persistence."""
        if not CACHE_STATS_ENABLED:
            return
        
        try:
            stats_key = f"{self.stats_prefix}current"
            self.redis_client.setex(stats_key, self.retention_seconds, json.dumps(self.stats))
        except Exception as e:
            if DEBUG:
                logger.error(f"Failed to store cache stats in Redis: {e}")
    
    def _load_stats_from_redis(self):
        """Load stats from Redis on startup."""
        if not CACHE_STATS_ENABLED:
            return
        
        try:
            stats_key = f"{self.stats_prefix}current"
            cached_stats = self.redis_client.get(stats_key)
            if cached_stats:
                loaded_stats = json.loads(cached_stats)
                # Merge with current stats (in case of partial data)
                for key, value in loaded_stats.items():
                    if key in self.stats and isinstance(value, dict):
                        if isinstance(self.stats[key], dict):
                            self.stats[key].update(value)
                        else:
                            self.stats[key] = value
                    else:
                        self.stats[key] = value
        except Exception as e:
            if DEBUG:
                logger.error(f"Failed to load cache stats from Redis: {e}")
    
    def get_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total == 0:
            return 0.0
        return self.stats["cache_hits"] / total
    
    def get_error_rate(self) -> float:
        """Calculate cache error rate."""
        total = self.stats["total_requests"]
        if total == 0:
            return 0.0
        return self.stats["cache_errors"] / total
    
    def get_endpoint_stats(self) -> Dict:
        """Get cache statistics by endpoint."""
        endpoint_stats = {}
        for endpoint, stats in self.stats["endpoint_stats"].items():
            total = stats["hits"] + stats["misses"]
            hit_rate = stats["hits"] / total if total > 0 else 0.0
            error_rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            
            endpoint_stats[endpoint] = {
                "total_requests": stats["total"],
                "cache_hits": stats["hits"],
                "cache_misses": stats["misses"],
                "cache_errors": stats["errors"],
                "hit_rate": hit_rate,
                "error_rate": error_rate
            }
        
        return endpoint_stats
    
    def get_location_stats(self) -> Dict:
        """Get cache statistics by location."""
        location_stats = {}
        for location, stats in self.stats["location_stats"].items():
            total = stats["hits"] + stats["misses"]
            hit_rate = stats["hits"] / total if total > 0 else 0.0
            error_rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
            
            location_stats[location] = {
                "total_requests": stats["total"],
                "cache_hits": stats["hits"],
                "cache_misses": stats["misses"],
                "cache_errors": stats["errors"],
                "hit_rate": hit_rate,
                "error_rate": error_rate
            }
        
        return location_stats
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict]:
        """Get hourly cache statistics for the last N hours."""
        current_hour = int(time.time() // 3600)
        hourly_data = []
        
        for i in range(hours):
            hour_key = current_hour - i
            if hour_key in self.stats["hourly_stats"]:
                stats = self.stats["hourly_stats"][hour_key]
                total = stats["hits"] + stats["misses"]
                hit_rate = stats["hits"] / total if total > 0 else 0.0
                
                hourly_data.append({
                    "hour": hour_key,
                    "timestamp": hour_key * 3600,
                    "total_requests": stats["total"],
                    "cache_hits": stats["hits"],
                    "cache_misses": stats["misses"],
                    "cache_errors": stats["errors"],
                    "hit_rate": hit_rate
                })
            else:
                hourly_data.append({
                    "hour": hour_key,
                    "timestamp": hour_key * 3600,
                    "total_requests": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_errors": 0,
                    "hit_rate": 0.0
                })
        
        return list(reversed(hourly_data))  # Return in chronological order
    
    def get_cache_health(self) -> Dict:
        """Get cache health assessment."""
        hit_rate = self.get_hit_rate()
        error_rate = self.get_error_rate()
        
        health_status = "healthy"
        if hit_rate < CACHE_HEALTH_THRESHOLD:
            health_status = "degraded"
        if error_rate > 0.1:  # 10% error rate threshold
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "hit_rate": hit_rate,
            "error_rate": error_rate,
            "threshold": CACHE_HEALTH_THRESHOLD,
            "total_requests": self.stats["total_requests"],
            "uptime_hours": (time.time() - self.stats["last_reset"]) / 3600
        }
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        return {
            "overall": {
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_errors": self.stats["cache_errors"],
                "hit_rate": self.get_hit_rate(),
                "error_rate": self.get_error_rate()
            },
            "by_endpoint": self.get_endpoint_stats(),
            "by_location": self.get_location_stats(),
            "hourly": self.get_hourly_stats(24),
            "health": self.get_cache_health(),
            "configuration": {
                "enabled": CACHE_STATS_ENABLED,
                "retention_hours": CACHE_STATS_RETENTION_HOURS,
                "health_threshold": CACHE_HEALTH_THRESHOLD
            }
        }
    
    def reset_stats(self):
        """Reset all cache statistics."""
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_errors": 0,
            "endpoint_stats": {},
            "location_stats": {},
            "hourly_stats": {},
            "last_reset": time.time()
        }
        self._store_stats_in_redis()
        if DEBUG:
            logger.info("📊 CACHE STATS RESET: All statistics cleared")

class CacheInvalidator:
    """Manage cache invalidation with various strategies and patterns."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize cache invalidator for managing cache invalidation.

        Args:
            redis_client: Redis client for cache operations
        """
        self.redis_client = redis_client
        self.invalidation_prefix = "invalidation_"
        self.batch_size = CACHE_INVALIDATION_BATCH_SIZE
        
    def invalidate_by_key(self, cache_key: str, dry_run: bool = False) -> Dict:
        """Invalidate a specific cache key."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        try:
            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                # Check if key exists without deleting
                exists = self.redis_client.exists(cache_key)
                return {
                    "status": "dry_run",
                    "cache_key": cache_key,
                    "exists": bool(exists),
                    "action": "would_delete" if exists else "no_action"
                }
            else:
                # Actually delete the key
                deleted = self.redis_client.delete(cache_key)
                return {
                    "status": "success",
                    "cache_key": cache_key,
                    "deleted": bool(deleted),
                    "action": "deleted" if deleted else "not_found"
                }
        except Exception as e:
            return {
                "status": "error",
                "cache_key": cache_key,
                "error": str(e)
            }
    
    def invalidate_by_pattern(self, pattern: str, dry_run: bool = False, max_keys: int = 10000) -> Dict:
        """Invalidate cache keys matching a pattern (MED-010: Add DoS protection)."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        # MED-010: Limit pattern matching to prevent DoS
        if max_keys > 100000:
            logger.warning(f"Cache invalidation max_keys ({max_keys}) is very high, capping at 100000")
            max_keys = 100000
        
        try:
            # Use SCAN instead of KEYS to avoid blocking Redis (O(N) blocking operation)
            matching_keys = []
            cursor = 0
            scan_max_keys = max_keys  # Use parameter limit for safety
            
            try:
                while cursor != 0 or len(matching_keys) == 0:  # Start iteration
                    cursor, keys = self.redis_client.scan(
                        cursor,
                        match=pattern,
                        count=100  # Process in batches
                    )
                    matching_keys.extend(keys)
                    
                    # Safety limit to prevent infinite loops or excessive memory usage (MED-010)
                    if len(matching_keys) > scan_max_keys:
                        logger.warning(f"Pattern matches >{scan_max_keys} keys, stopping scan at {len(matching_keys)} keys")
                        return {
                            "status": "error",
                            "pattern": pattern,
                            "error": f"Pattern matches too many keys (>{scan_max_keys}). Use a more specific pattern or increase max_keys parameter."
                        }
                    
                    # Prevent infinite loop (shouldn't happen, but safety check)
                    if cursor == 0:
                        break
            except Exception as e:
                # Handle Redis permissions error or other issues
                if "permissions" in str(e).lower() or "scan" in str(e).lower():
                    return {
                        "status": "error",
                        "pattern": pattern,
                        "error": "Redis SCAN command not permitted on this instance",
                        "message": "Cache invalidation by pattern is not available on managed Redis services"
                    }
                else:
                    raise e
            
            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                return {
                    "status": "dry_run",
                    "pattern": pattern,
                    "matching_keys": [key.decode() if isinstance(key, bytes) else key for key in matching_keys],
                    "count": len(matching_keys),
                    "action": "would_delete"
                }
            else:
                # Delete keys in batches
                deleted_count = 0
                for i in range(0, len(matching_keys), self.batch_size):
                    batch = matching_keys[i:i + self.batch_size]
                    if batch:
                        deleted_count += self.redis_client.delete(*batch)
                
                return {
                    "status": "success",
                    "pattern": pattern,
                    "matching_keys": [key.decode() if isinstance(key, bytes) else key for key in matching_keys],
                    "deleted_count": deleted_count,
                    "total_found": len(matching_keys)
                }
        except Exception as e:
            return {
                "status": "error",
                "pattern": pattern,
                "error": str(e)
            }
    
    def invalidate_by_endpoint(self, endpoint: str, location: str = None, dry_run: bool = False) -> Dict:
        """Invalidate cache keys for a specific endpoint and optionally location."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        # Build pattern based on endpoint and location
        if location:
            pattern = f"{endpoint}_{location.lower()}_*"
        else:
            pattern = f"{endpoint}_*"
        
        return self.invalidate_by_pattern(pattern, dry_run)
    
    def invalidate_by_location(self, location: str, dry_run: bool = False) -> Dict:
        """Invalidate all cache keys for a specific location."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        location = location.lower()
        pattern = f"*_{location}_*"
        return self.invalidate_by_pattern(pattern, dry_run)
    
    def invalidate_by_date(self, date: str, dry_run: bool = False) -> Dict:
        """Invalidate cache keys for a specific date."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        # Handle different date formats
        date_patterns = [
            f"*_{date}_*",  # YYYY-MM-DD format
            f"*_{date.replace('-', '_')}_*",  # YYYY_MM_DD format
            f"*_{date.replace('-', '_')}",  # End of key
        ]
        
        results = []
        total_deleted = 0
        total_found = 0
        
        for pattern in date_patterns:
            result = self.invalidate_by_pattern(pattern, dry_run)
            results.append(result)
            if result.get("status") == "success":
                total_deleted += result.get("deleted_count", 0)
                total_found += result.get("total_found", 0)
            elif result.get("status") == "dry_run":
                total_found += result.get("count", 0)
        
        return {
            "status": "success" if not dry_run and not CACHE_INVALIDATION_DRY_RUN else "dry_run",
            "date": date,
            "patterns_checked": len(date_patterns),
            "total_deleted": total_deleted,
            "total_found": total_found,
            "pattern_results": results
        }
    
    def invalidate_forecast_data(self, dry_run: bool = False) -> Dict:
        """Invalidate all forecast data (since it changes frequently)."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        return self.invalidate_by_pattern("forecast_*", dry_run)
    
    def invalidate_today_data(self, dry_run: bool = False) -> Dict:
        """Invalidate all data for today (since it includes forecasts)."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        today_pattern = today.strftime("%m_%d")
        
        # Invalidate by both date formats
        results = []
        for date_format in [today_str, today_pattern]:
            result = self.invalidate_by_date(date_format, dry_run)
            results.append(result)
        
        return {
            "status": "success" if not dry_run and not CACHE_INVALIDATION_DRY_RUN else "dry_run",
            "date": today_str,
            "results": results
        }
    
    def invalidate_expired_keys(self, dry_run: bool = False) -> Dict:
        """Invalidate keys that have expired (TTL = 0 or negative)."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        try:
            # Use SCAN instead of KEYS to avoid blocking Redis
            all_keys = []
            cursor = 0
            max_keys = 100000  # Safety limit
            
            try:
                while cursor != 0 or len(all_keys) == 0:  # Start iteration
                    cursor, keys = self.redis_client.scan(
                        cursor,
                        match="*",
                        count=100  # Process in batches
                    )
                    all_keys.extend(keys)
                    
                    # Safety limit
                    if len(all_keys) > max_keys:
                        logger.warning(f"Found >{max_keys} keys, stopping scan at {len(all_keys)} keys")
                        break
                    
                    if cursor == 0:
                        break
            except Exception as e:
                # Handle Redis permissions error
                if "permissions" in str(e).lower() or "scan" in str(e).lower():
                    return {
                        "status": "error",
                        "error": "Redis SCAN command not permitted on this instance",
                        "message": "Expired key invalidation is not available on managed Redis services"
                    }
                else:
                    raise e
            
            expired_keys = []
            
            for key in all_keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # Key exists but has no expiration
                    continue
                elif ttl <= 0:  # Key has expired
                    expired_keys.append(key)
            
            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                return {
                    "status": "dry_run",
                    "expired_keys": [key.decode() if isinstance(key, bytes) else key for key in expired_keys],
                    "count": len(expired_keys),
                    "action": "would_delete"
                }
            else:
                # Delete expired keys
                deleted_count = 0
                if expired_keys:
                    deleted_count = self.redis_client.delete(*expired_keys)
                
                return {
                    "status": "success",
                    "expired_keys": [key.decode() if isinstance(key, bytes) else key for key in expired_keys],
                    "deleted_count": deleted_count,
                    "total_found": len(expired_keys)
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_cache_info(self) -> Dict:
        """Get information about current cache state."""
        try:
            # Get basic Redis info
            info = self.redis_client.info()
            
            # Get key count by pattern
            patterns = {
                "weather": "weather_*",
                "data": "data_*",
                "trend": "trend_*",
                "average": "average_*",
                "summary": "summary_*",
                "forecast": "forecast_*",
                "series": "series_*",
                "usage": "usage_*",
                "cache_stats": "cache_stats_*",
                "v1_records": "records:*",
                "v1_records_average": "records:*:average",
                "v1_records_trend": "records:*:trend",
                "v1_records_summary": "records:*:summary"
            }
            
            pattern_counts = {}
            for name, pattern in patterns.items():
                try:
                    # Use SCAN instead of KEYS to avoid blocking
                    matching_keys = []
                    cursor = 0
                    max_keys = 10000  # Limit for stats gathering
                    
                    while cursor != 0 or len(matching_keys) == 0:
                        cursor, keys = self.redis_client.scan(
                            cursor,
                            match=pattern,
                            count=100
                        )
                        matching_keys.extend(keys)
                        if len(matching_keys) > max_keys:
                            break
                        if cursor == 0:
                            break
                    
                    pattern_counts[name] = len(matching_keys)
                except Exception as e:
                    # Handle Redis permissions error
                    if "permissions" in str(e).lower() or "scan" in str(e).lower():
                        pattern_counts[name] = "N/A (permissions required)"
                    else:
                        pattern_counts[name] = f"N/A (error: {str(e)[:50]})"
            
            return {
                "redis_info": {
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "key_counts": pattern_counts,
                "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_all_cache(self, dry_run: bool = False) -> Dict:
        """Clear all cache data (use with caution!)."""
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        
        try:
            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                # Use SCAN to count keys without deleting (may fail on managed Redis)
                try:
                    all_keys = []
                    cursor = 0
                    max_keys = 100000  # Safety limit
                    
                    while cursor != 0 or len(all_keys) == 0:
                        cursor, keys = self.redis_client.scan(
                            cursor,
                            match="*",
                            count=100
                        )
                        all_keys.extend(keys)
                        if len(all_keys) > max_keys:
                            break
                        if cursor == 0:
                            break
                    
                    return {
                        "status": "dry_run",
                        "total_keys": len(all_keys),
                        "action": "would_delete_all"
                    }
                except Exception as e:
                    if "permissions" in str(e).lower() or "scan" in str(e).lower():
                        return {
                            "status": "error",
                            "error": "Redis SCAN command not permitted on this instance",
                            "message": "Cannot count keys on managed Redis service for dry run"
                        }
                    else:
                        raise e
            else:
                # Flush all data
                self.redis_client.flushdb()
                return {
                    "status": "success",
                    "action": "cleared_all_cache",
                    "message": "All cache data has been cleared"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Job management for async processing
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class JobManager:
    """Manage async job processing with Redis storage."""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize async job manager for background processing.

        Args:
            redis_client: Redis client for job storage and results
        """
        self.redis = redis_client
        self.job_prefix = "job:"
        self.result_prefix = "result:"
        self.job_ttl = CACHE_TTL_JOB
    
    def create_job(self, job_type: str, params: Dict[str, Any]) -> str:
        """
        Create a new job and return job ID.
        Implements deduplication - if an identical job is already pending or processing, returns existing job_id.
        """
        try:
            # Generate deterministic job ID based on params (without timestamp for deduplication)
            params_hash = hashlib.sha256(str(params).encode()).hexdigest()[:16]
            dedup_key = f"job:dedup:{job_type}:{params_hash}"

            # Check if identical job already exists
            existing_job_id = self.redis.get(dedup_key)
            if existing_job_id:
                existing_job_id = existing_job_id.decode() if isinstance(existing_job_id, bytes) else existing_job_id
                # Verify the job still exists and is pending/processing
                job_key = f"{self.job_prefix}{existing_job_id}"
                job_data = self.redis.get(job_key)
                if job_data:
                    job = json.loads(job_data)
                    status = job.get("status")
                    if status in [JobStatus.PENDING, JobStatus.PROCESSING]:
                        logger.info(f"Deduplicated job {existing_job_id}: identical job already {status}")
                        return existing_job_id

            # Create new job with timestamp for uniqueness
            job_id = f"{job_type}_{int(time.time() * 1000)}_{params_hash[:8]}"
            job_key = f"{self.job_prefix}{job_id}"

            job_data = {
                "id": job_id,
                "type": job_type,
                "status": JobStatus.PENDING,
                "params": params,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            logger.info(f"Creating job with ID: {job_id}")
            logger.info(f"Job params: {params}")

            # Store job data
            self.redis.setex(job_key, self.job_ttl, json.dumps(job_data))
            logger.info(f"Job data stored in Redis with key: {job_key}")

            # Store deduplication key (expires after 5 minutes)
            self.redis.setex(dedup_key, 300, job_id)

            # Add job to queue for worker processing
            job_queue_key = "job_queue"
            self.redis.lpush(job_queue_key, job_id)
            logger.info(f"Job {job_id} added to queue")

            return job_id
        except Exception as e:
            logger.error(f"Error in create_job: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Job type: {job_type}, Params: {params}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result if ready."""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.redis.get(job_key)
        
        if not job_data:
            return None
        
        job = json.loads(job_data)
        
        # If job is ready, include result
        if job["status"] == JobStatus.READY:
            result_key = f"{self.result_prefix}{job_id}"
            result_data = self.redis.get(result_key)
            if result_data:
                job["result"] = json.loads(result_data)
        
        return job
    
    def update_job_status(self, job_id: str, status: str, result: Any = None, error: str = None, error_details: Dict = None):
        """Update job status and optionally store result."""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.redis.get(job_key)

        if not job_data:
            return

        job = json.loads(job_data)
        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

        if error:
            job["error"] = error

        if error_details:
            job["error_details"] = error_details

        self.redis.setex(job_key, self.job_ttl, json.dumps(job))

        # Store result if provided
        if result is not None:
            result_key = f"{self.result_prefix}{job_id}"
            self.redis.setex(result_key, self.job_ttl, json.dumps(result))

        # Clean up deduplication key when job completes (READY or ERROR status)
        if status in [JobStatus.READY, JobStatus.ERROR]:
            job_type = job.get("type")
            params = job.get("params")
            if job_type and params:
                params_hash = hashlib.sha256(str(params).encode()).hexdigest()[:16]
                dedup_key = f"job:dedup:{job_type}:{params_hash}"
                self.redis.delete(dedup_key)
    
    def cleanup_expired_jobs(self) -> int:
        """Clean up expired jobs (Redis TTL should handle this automatically)."""
        # This is mainly for manual cleanup if needed
        # Redis TTL should handle automatic cleanup
        return 0

# Cache utility functions
def get_cache_value(cache_key, redis_client: redis.Redis, endpoint: str = None, location: str = None, cache_stats=None):
    """Get a value from the cache with statistics tracking."""
    if DEBUG:
        logger.debug(f"🔍 CACHE GET: {cache_key}")
    
    try:
        result = redis_client.get(cache_key)
        hit = result is not None
        
        # Track cache statistics
        if CACHE_STATS_ENABLED and cache_stats:
            cache_stats.track_cache_request(cache_key, hit, endpoint, location)
        
        if DEBUG and result:
            logger.debug(f"✅ CACHE HIT: {cache_key}")
        elif DEBUG:
            logger.debug(f"❌ CACHE MISS: {cache_key}")
        
        return result
    
    except Exception as e:
        # Track cache errors
        if CACHE_STATS_ENABLED and cache_stats:
            cache_stats.track_cache_request(cache_key, False, endpoint, location, error=True)
        
        if DEBUG:
            logger.error(f"❌ CACHE ERROR: {cache_key} - {str(e)}")
        return None

def set_cache_value(cache_key, lifetime, value, redis_client: redis.Redis):
    """Set a value in the cache with specified lifetime."""
    if DEBUG:
        logger.debug(f"💾 CACHE SET: {cache_key} | TTL: {lifetime}")
    redis_client.setex(cache_key, lifetime, value)

def normalize_location_for_cache(location: str) -> str:
    """Normalize location string for consistent cache keys.
    
    Args:
        location: Location name (e.g., "London, England, United Kingdom")
        
    Returns:
        str: Normalized location (e.g., "london_england_united_kingdom")
    """
    return location.lower().replace(" ", "_").replace(",", "_")

def get_weather_cache_key(location: str, date_str: str) -> str:
    """Generate cache key for weather data."""
    normalized_location = normalize_location_for_cache(location)
    return f"{normalized_location}_{date_str}"

def _get_location_timezone_from_cache(location: str, redis_client: Optional[redis.Redis] = None) -> Optional[str]:
    """Get timezone for a location from Redis cache (stored from Visual Crossing API responses).
    
    Args:
        location: Location name
        redis_client: Optional Redis client (will try to get global cache if not provided)
        
    Returns:
        IANA timezone string (e.g., "Europe/London") or None if not found
    """
    try:
        if redis_client is None:
            # Try to get Redis client from global cache
            try:
                cache = get_cache()
                redis_client = cache.redis
            except Exception:
                return None
        
        if not redis_client:
            return None
        
        # Normalize location for cache key
        normalized_location = normalize_location_for_cache(location)
        timezone_key = f"location_timezone:{normalized_location}"
        
        cached_timezone = redis_client.get(timezone_key)
        if cached_timezone:
            timezone_str = cached_timezone.decode('utf-8') if isinstance(cached_timezone, bytes) else cached_timezone
            return timezone_str
        
        return None
    except Exception as e:
        if DEBUG:
            logger.debug(f"Could not get timezone from cache for location '{location}': {e}")
        return None

def store_location_timezone(location: str, timezone_str: str, redis_client: Optional[redis.Redis] = None, ttl: int = 604800) -> None:
    """Store timezone for a location in Redis cache.
    
    Args:
        location: Location name
        timezone_str: IANA timezone string from Visual Crossing API
        redis_client: Optional Redis client (will try to get global cache if not provided)
        ttl: Time to live in seconds (default: 7 days)
    """
    try:
        if redis_client is None:
            # Try to get Redis client from global cache
            try:
                cache = get_cache()
                redis_client = cache.redis
            except Exception:
                return
        
        if not redis_client:
            return
        
        # Normalize location for cache key
        normalized_location = normalize_location_for_cache(location)
        timezone_key = f"location_timezone:{normalized_location}"
        
        # Store with long TTL (timezone doesn't change often)
        redis_client.setex(timezone_key, ttl, timezone_str)
        
        if DEBUG:
            logger.debug(f"Stored timezone for location '{location}': {timezone_str}")
    except Exception as e:
        if DEBUG:
            logger.debug(f"Could not store timezone for location '{location}': {e}")

def _get_location_timezone_from_preapproved(location: str) -> Optional[str]:
    """Get timezone for a location from preapproved locations data (fallback).
    
    Args:
        location: Location name (can be in various formats)
        
    Returns:
        IANA timezone string (e.g., "Europe/London") or None if not found
    """
    global _preapproved_locations_cache
    
    try:
        # Load locations data once and cache it
        if _preapproved_locations_cache is None:
            # Find project root by looking for pyproject.toml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != os.path.dirname(project_root):
                if os.path.exists(os.path.join(project_root, "pyproject.toml")):
                    break
                project_root = os.path.dirname(project_root)
            
            data_file = os.path.join(project_root, "data", "preapproved_locations.json")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                _preapproved_locations_cache = json.load(f)
        
        # Normalize location for comparison
        location_lower = location.lower().strip()
        
        # Try to match by various formats
        for item in _preapproved_locations_cache:
            # Match by full name format: "Name, Admin1, Country Name"
            if 'name' in item and 'admin1' in item and 'country_name' in item:
                full_name = f"{item['name']}, {item['admin1']}, {item['country_name']}".lower()
                if full_name == location_lower:
                    return item.get('timezone')
            
            # Match by name only
            if item.get('name', '').lower() == location_lower:
                return item.get('timezone')
            
            # Match by slug
            if item.get('slug', '').lower() == location_lower:
                return item.get('timezone')
        
        return None
    except Exception as e:
        if DEBUG:
            logger.debug(f"Could not get timezone from preapproved locations for '{location}': {e}")
        return None

def _get_location_timezone(location: str, redis_client: Optional[redis.Redis] = None) -> Optional[str]:
    """Get timezone for a location, trying multiple sources in order:
    1. Redis cache (from Visual Crossing API responses)
    2. Preapproved locations JSON file
    3. None (fallback to UTC)
    
    Args:
        location: Location name
        redis_client: Optional Redis client
        
    Returns:
        IANA timezone string (e.g., "Europe/London") or None if not found
    """
    # First try Redis cache (from Visual Crossing API)
    timezone = _get_location_timezone_from_cache(location, redis_client)
    if timezone:
        return timezone
    
    # Fallback to preapproved locations
    return _get_location_timezone_from_preapproved(location)

def _is_today_in_location_timezone(date: dt_date, location: Optional[str] = None, redis_client: Optional[redis.Redis] = None) -> bool:
    """Check if a date is today in the location's timezone, or UTC if location not found.
    
    Args:
        date: Date to check
        location: Optional location name to look up timezone
        redis_client: Optional Redis client for timezone cache lookup
        
    Returns:
        True if date is today in location's timezone (or UTC if location not found)
    """
    if location and ZoneInfo:
        timezone_str = _get_location_timezone(location, redis_client)
        if timezone_str:
            try:
                tz = ZoneInfo(timezone_str)
                today_in_tz = datetime.now(tz).date()
                return date == today_in_tz
            except Exception as e:
                if DEBUG:
                    logger.debug(f"Error using timezone {timezone_str} for location '{location}': {e}")
                # Fall through to UTC fallback
    
    # Fallback to UTC if location not found or zoneinfo not available
    today = datetime.now(timezone.utc).date()
    return date == today

def generate_cache_key(prefix: str, location: str, date_part: str = "") -> str:
    """Generate standardized cache keys.
    
    Args:
        prefix: Cache key prefix (e.g., 'weather', 'data', 'trend')
        location: Location name
        date_part: Date part (can be YYYY-MM-DD, MM-DD, or MM_DD format). Empty string for no date.
        
    Returns:
        str: Standardized cache key in format: {prefix}_{location}_{date_part}
    """
    # Normalize location using centralized function
    location = normalize_location_for_cache(location)
    
    if date_part:
        # Normalize date part by replacing hyphens with underscores
        date_part = date_part.replace('-', '_')
        return f"{prefix}_{location}_{date_part}"
    else:
        return f"{prefix}_{location}"

# Year-based caching helper functions
def rec_key(scope: str, slug: str, identifier: str, year: int) -> str:
    """Generate cache key for a per-year record.
    
    Args:
        scope: Period scope ('daily', 'weekly', 'monthly', 'yearly')
        slug: Location slug (normalized)
        identifier: Date identifier (e.g., '11-06', '2025-W45')
        year: Year (e.g., 1979, 2024)
        
    Returns:
        str: Cache key in format records:v1:{scope}:{slug}:{identifier}:{year}
    """
    return f"records:v1:{scope}:{slug}:{identifier}:{year}"

def bundle_key(scope: str, slug: str, identifier: str) -> str:
    """Generate cache key for assembled bundle.
    
    Args:
        scope: Period scope ('daily', 'weekly', 'monthly', 'yearly')
        slug: Location slug (normalized)
        identifier: Date identifier (e.g., '11-06', '2025-W45')
        
    Returns:
        str: Cache key in format records:v1:{scope}:{slug}:{identifier}:bundle
    """
    return f"records:v1:{scope}:{slug}:{identifier}:bundle"

def rec_etag_key(scope: str, slug: str, identifier: str, year: int) -> str:
    """Generate cache key for per-year ETag.
    
    Args:
        scope: Period scope ('daily', 'weekly', 'monthly', 'yearly')
        slug: Location slug (normalized)
        identifier: Date identifier (e.g., '11-06', '2025-W45')
        year: Year (e.g., 1979, 2024)
        
    Returns:
        str: ETag cache key in format records:v1:{scope}:{slug}:{identifier}:{year}:etag
    """
    return f"records:v1:{scope}:{slug}:{identifier}:{year}:etag"

async def get_records(
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    years: List[int]
) -> Tuple[Dict[int, Dict], List[int], bool]:
    """Get per-year records from cache using MGET.
    
    Args:
        redis_client: Redis client instance
        scope: Period scope ('daily', 'weekly', 'monthly', 'yearly')
        slug: Location slug (normalized)
        identifier: Date identifier (e.g., '11-06', '2025-W45')
        years: List of years to fetch
        
    Returns:
        Tuple of (year_data dict, missing_past list, missing_current bool):
        - year_data: Dict mapping year -> record data (only for found years)
        - missing_past: List of past years that are missing
        - missing_current: True if current year is missing
    """
    if not years:
        return {}, [], False
    
    current_year = datetime.now(timezone.utc).year
    
    # Build all keys for MGET
    keys = [rec_key(scope, slug, identifier, year) for year in years]
    
    # MGET all year keys
    try:
        values = redis_client.mget(keys)
    except Exception as e:
        logger.error(f"Error in MGET for records: {e}")
        return {}, years, current_year in years
    
    year_data = {}
    missing_past = []
    missing_current = False
    
    for year, value in zip(years, values):
        if value:
            try:
                data = json.loads(value.decode() if isinstance(value, bytes) else value)
                year_data[year] = data
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Error decoding cached data for year {year}: {e}")
                if year < current_year:
                    missing_past.append(year)
                elif year == current_year:
                    missing_current = True
        else:
            if year < current_year:
                missing_past.append(year)
            elif year == current_year:
                missing_current = True
    
    return year_data, missing_past, missing_current

async def assemble_and_cache(
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    year_data: Dict[int, Dict],
    year_etags: Optional[Dict[int, str]] = None
) -> Tuple[Dict, str]:
    """Assemble final payload from per-year records and cache bundle with ETag.
    
    Args:
        redis_client: Redis client instance
        scope: Period scope ('daily', 'weekly', 'monthly', 'yearly')
        slug: Location slug (normalized)
        identifier: Date identifier (e.g., '11-06', '2025-W45')
        year_data: Dict mapping year -> record data
        year_etags: Optional dict mapping year -> ETag (if not provided, will fetch)
        
    Returns:
        Tuple of (payload dict, bundle_etag string)
    """
    # Sort records by year
    sorted_years = sorted(year_data.keys())
    ordered_records = [year_data[y] for y in sorted_years]
    
    # Build final payload
    payload = {
        "version": 1,
        "count": len(ordered_records),
        "records": ordered_records
    }
    
    # Get year ETags if not provided
    if year_etags is None:
        year_etags = await get_year_etags(redis_client, scope, slug, identifier, sorted_years)
    
    # Compute bundle ETag from per-year ETags
    bundle_etag_computed = compute_bundle_etag(year_etags)
    
    # Cache bundle with short TTL
    bundle_key_str = bundle_key(scope, slug, identifier)
    bundle_etag_key = f"{bundle_key_str}:etag"
    try:
        json_data = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        redis_client.setex(bundle_key_str, TTL_BUNDLE, json_data)
        redis_client.setex(bundle_etag_key, TTL_BUNDLE, bundle_etag_computed)
        if DEBUG:
            logger.debug(f"Cached bundle: {bundle_key_str} with ETag")
    except Exception as e:
        logger.warning(f"Error caching bundle {bundle_key_str}: {e}")
    
    return payload, bundle_etag_computed

def compute_bundle_etag(year_etags: Dict[int, str]) -> str:
    """Compute bundle ETag from ordered list of per-year ETags.
    
    Args:
        year_etags: Dict mapping year -> ETag string
        
    Returns:
        str: SHA256 hash of ordered per-year ETags (32 chars)
    """
    # Sort by year and extract ETag values (remove quotes if present)
    sorted_etags = []
    for year in sorted(year_etags.keys()):
        etag = year_etags[year]
        # Remove quotes if present
        etag_clean = etag.strip('"\'')
        sorted_etags.append(f"{year}:{etag_clean}")
    
    # Hash the concatenated ETags
    etag_string = "|".join(sorted_etags)
    return f'"{hashlib.sha256(etag_string.encode()).hexdigest()[:32]}"'

async def get_year_etags(
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    years: List[int]
) -> Dict[int, str]:
    """Get ETags for all years using MGET.
    
    Args:
        redis_client: Redis client instance
        scope: Period scope ('daily', 'weekly', 'monthly', 'yearly')
        slug: Location slug (normalized)
        identifier: Date identifier (e.g., '11-06', '2025-W45')
        years: List of years to fetch ETags for
        
    Returns:
        Dict mapping year -> ETag string (empty string if not found)
    """
    if not years:
        return {}
    
    # Build all ETag keys for MGET
    keys = [rec_etag_key(scope, slug, identifier, year) for year in years]
    
    # MGET all ETag keys
    try:
        values = redis_client.mget(keys)
    except Exception as e:
        logger.error(f"Error in MGET for ETags: {e}")
        return {year: "" for year in years}
    
    year_etags = {}
    for year, value in zip(years, values):
        if value:
            etag = value.decode() if isinstance(value, bytes) else value
            year_etags[year] = etag
        else:
            year_etags[year] = ""
    
    return year_etags

async def cached_endpoint_response(
    request: Request,
    response: Response,
    cache_key: str,
    compute_func,
    ttl: int = CACHE_TTL_DEFAULT,
    cache_control: str = CACHE_CONTROL_PUBLIC,
    period: Optional[str] = None,
    date: Optional[dt_date] = None,
    location: Optional[str] = None,
    *args,
    **kwargs
):
    """Helper function to add caching to any endpoint.
    
    Args:
        request: FastAPI request object
        response: FastAPI response object
        cache_key: Cache key for the data
        compute_func: Function to compute the data if not cached
        ttl: Cache TTL in seconds (default: CACHE_TTL_DEFAULT)
        cache_control: Cache-Control header value (default: CACHE_CONTROL_PUBLIC)
        period: Optional period type (e.g., "daily", "weekly", "monthly", "yearly")
        date: Optional date object to check if it's today
        location: Optional location name to determine timezone for "today" check
        *args, **kwargs: Additional arguments passed to compute_func
    """
    cache = get_cache()  # This returns EnhancedCache instance
    
    # If this is a daily endpoint with today's date, use shorter cache TTL
    # since today's data uses forecast which may change
    # Check if date is "today" in the location's timezone (or UTC if location not found)
    if period == "daily" and date is not None:
        if _is_today_in_location_timezone(date, location, cache.redis):
            # Use shorter TTL for today's daily data
            ttl = CACHE_TTL_SHORT  # 1 hour instead of default
            cache_control = CACHE_CONTROL_DAILY_TODAY  # Shorter max-age
    
    try:
        # Try to get from cache or compute
        data, etag, last_modified = await cache.get_or_compute(
            cache_key, compute_func, ttl, *args, **kwargs
        )
        
        # Check if client has cached version
        if CacheHeaders.check_conditional_headers(request, etag, last_modified):
            response.status_code = 304
            return None
        
        # Set cache headers
        CacheHeaders.set_cache_headers(response, etag, last_modified, cache_control)
        
        return data
        
    except Exception as e:
        logger.error(f"Cache error for {cache_key}: {e}")
        # Fallback to direct computation
        data = await compute_func(*args, **kwargs) if asyncio.iscoroutinefunction(compute_func) else compute_func(*args, **kwargs)
        return data

async def get_cache_updated_timestamp(cache_key: str, redis_client: redis.Redis) -> Optional[datetime]:
    """Get the last updated timestamp for a cache key."""
    try:
        if not redis_client:
            return None
        
        # Check if the cache key exists
        if not redis_client.exists(cache_key):
            return None
        
        # First, try to get the timestamp from the hash (new enhanced cache format)
        # Only try hget if the key is a hash type
        try:
            key_type = redis_client.type(cache_key)
            if key_type == 'hash':
                timestamp_data = redis_client.hget(cache_key, "timestamp")
                if timestamp_data:
                    return datetime.fromisoformat(timestamp_data.decode() if isinstance(timestamp_data, bytes) else timestamp_data)
        except Exception:
            # If hget fails, continue to string-based approach
            pass
        
        # Fallback: check if it's a simple string key (current cache format)
        # Try to parse the JSON data to extract the updated timestamp
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                data = json.loads(cached_data)
                if "updated" in data and data["updated"]:
                    return datetime.fromisoformat(data["updated"])
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Final fallback: estimate from TTL
        ttl = redis_client.ttl(cache_key)
        if ttl > 0:
            # Estimate creation time as now - (original_ttl - current_ttl)
            # This is approximate but gives a reasonable estimate
            return datetime.now(timezone.utc) - timedelta(seconds=ttl)
        
        return None
        
    except Exception as e:
        logger.warning(f"Error getting cache timestamp for {cache_key}: {e}")
        return None

async def scheduled_cache_warming(cache_warmer: CacheWarmer):
    """Background task that schedules cache warming jobs on a schedule."""
    if not CACHE_WARMING_ENABLED or not cache_warmer:
        return
    
    # Import here to avoid circular imports
    from cache_utils import get_job_manager
    
    while True:
        try:
            # Wait for the specified interval
            await asyncio.sleep(CACHE_WARMING_INTERVAL_HOURS * 3600)
            
            # Check if warming is already in progress
            if not cache_warmer.warming_in_progress:
                if DEBUG:
                    logger.info("🕐 SCHEDULED CACHE WARMING: Creating cache warming job")
                
                # Create a cache warming job instead of running directly
                job_manager = get_job_manager()
                if job_manager:
                    job_id = job_manager.create_job("cache_warming", {
                        "type": "all",
                        "locations": [],
                        "scheduled": True,
                        "scheduled_at": datetime.now(timezone.utc).isoformat()
                    })
                    logger.info(f"✅ Cache warming job created: {job_id}")
                else:
                    logger.warning("⚠️  Job manager not available, falling back to direct warming")
                    # Fallback to direct warming if job manager not available
                    asyncio.create_task(cache_warmer.warm_all_locations())
            else:
                if DEBUG:
                    logger.info("⏭️  SCHEDULED CACHE WARMING: Skipping - warming already in progress")
        
        except Exception as e:
            if DEBUG:
                logger.error(f"❌ SCHEDULED CACHE WARMING ERROR: {str(e)}")
            # Continue the loop even if there's an error
            await asyncio.sleep(300)  # Wait 5 minutes before retrying


# Global cache instances (will be initialized in main.py)
enhanced_cache: Optional[EnhancedCache] = None
job_manager: Optional[JobManager] = None
usage_tracker: Optional[LocationUsageTracker] = None
cache_warmer: Optional[CacheWarmer] = None
cache_stats: Optional[CacheStats] = None
cache_invalidator: Optional[CacheInvalidator] = None

def initialize_cache(redis_client: redis.Redis):
    """Initialize global cache instances."""
    global enhanced_cache, job_manager, usage_tracker, cache_warmer, cache_stats, cache_invalidator
    
    enhanced_cache = EnhancedCache(redis_client)
    job_manager = JobManager(redis_client)
    
    # Initialize usage tracker
    if USAGE_TRACKING_ENABLED:
        usage_tracker = LocationUsageTracker(redis_client, USAGE_RETENTION_DAYS)
        if DEBUG:
            logger.info(f"📊 USAGE TRACKING INITIALIZED: {USAGE_RETENTION_DAYS} days retention")
        else:
            logger.info(f"Usage tracking enabled: {USAGE_RETENTION_DAYS} days retention")
    else:
        usage_tracker = None
        if DEBUG:
            logger.info("⚠️  USAGE TRACKING DISABLED")
        else:
            logger.info("Usage tracking disabled")
    
    # Initialize cache warmer
    if CACHE_WARMING_ENABLED:
        cache_warmer = CacheWarmer(redis_client, usage_tracker)
        if DEBUG:
            logger.info(f"🔥 CACHE WARMING INITIALIZED: {CACHE_WARMING_INTERVAL_HOURS}h interval, {CACHE_WARMING_DAYS_BACK} days back")
        else:
            logger.info(f"Cache warming enabled: {CACHE_WARMING_INTERVAL_HOURS}h interval")
    else:
        cache_warmer = None
        if DEBUG:
            logger.info("⚠️  CACHE WARMING DISABLED")
        else:
            logger.info("Cache warming disabled")
    
    # Initialize cache stats
    if CACHE_STATS_ENABLED:
        cache_stats = CacheStats(redis_client)
        cache_stats._load_stats_from_redis()  # Load existing stats from Redis
        if DEBUG:
            logger.info("📊 CACHE STATS INITIALIZED")
        else:
            logger.info("Cache statistics enabled")
    else:
        cache_stats = None
        if DEBUG:
            logger.info("⚠️  CACHE STATS DISABLED")
        else:
            logger.info("Cache statistics disabled")
    
    # Initialize cache invalidator
    if CACHE_INVALIDATION_ENABLED:
        cache_invalidator = CacheInvalidator(redis_client)
        if DEBUG:
            logger.info(f"🗑️  CACHE INVALIDATION INITIALIZED: Batch size {CACHE_INVALIDATION_BATCH_SIZE}, Dry run: {CACHE_INVALIDATION_DRY_RUN}")
        else:
            logger.info(f"Cache invalidation enabled: batch size {CACHE_INVALIDATION_BATCH_SIZE}")
    else:
        cache_invalidator = None
        if DEBUG:
            logger.info("⚠️  CACHE INVALIDATION DISABLED")
        else:
            logger.info("Cache invalidation disabled")
    
    logger.info("Enhanced cache and job manager initialized")

def get_cache() -> EnhancedCache:
    """Get the global cache instance."""
    if enhanced_cache is None:
        raise RuntimeError("Cache not initialized. Call initialize_cache() first.")
    return enhanced_cache

def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    if job_manager is None:
        raise RuntimeError("Job manager not initialized. Call initialize_cache() first.")
    return job_manager

def get_usage_tracker() -> Optional[LocationUsageTracker]:
    """Get the global usage tracker instance."""
    return usage_tracker

def get_cache_warmer() -> Optional[CacheWarmer]:
    """Get the global cache warmer instance."""
    return cache_warmer

def get_cache_stats() -> Optional[CacheStats]:
    """Get the global cache stats instance."""
    return cache_stats

def get_cache_invalidator() -> Optional[CacheInvalidator]:
    """Get the global cache invalidator instance."""
    return cache_invalidator
