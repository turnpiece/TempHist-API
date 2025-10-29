"""
Improved caching utilities for TempHist API with canonicalized location keys and temporal tolerance.

This module provides:
- Canonicalized location keys for consistent caching
- Temporal tolerance for different aggregation periods
- Redis sorted set-based date matching
- Metadata tracking for approximate data usage
"""

import json
import logging
import re
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, Tuple, List
import redis

logger = logging.getLogger(__name__)

# Temporal tolerance configuration
TEMPORAL_TOLERANCE = {
    "yearly": 7,   # ±7 days for yearly data
    "monthly": 2,  # ±2 days for monthly data
    "daily": 0     # Exact match only for daily data
}

# Common location suffixes to simplify
COMMON_SUFFIXES = [
    ", united_kingdom",
    ", united states",
    ", usa",
    ", us",
    ", uk",
    ", england",
    ", scotland",
    ", wales",
    ", northern ireland",
    ", ireland",
    ", canada",
    ", australia",
    ", new zealand",
    ", france",
    ", germany",
    ", spain",
    ", italy",
    ", netherlands",
    ", belgium",
    ", switzerland",
    ", austria",
    ", sweden",
    ", norway",
    ", denmark",
    ", finland",
    ", poland",
    ", czech republic",
    ", hungary",
    ", romania",
    ", bulgaria",
    ", croatia",
    ", slovenia",
    ", slovakia",
    ", estonia",
    ", latvia",
    ", lithuania",
    ", portugal",
    ", greece",
    ", turkey",
    ", russia",
    ", china",
    ", japan",
    ", south korea",
    ", india",
    ", brazil",
    ", argentina",
    ", chile",
    ", mexico",
    ", south africa",
    ", egypt",
    ", morocco",
    ", nigeria",
    ", kenya",
    ", israel",
    ", saudi arabia",
    ", uae",
    ", qatar",
    ", kuwait",
    ", bahrain",
    ", oman",
    ", jordan",
    ", lebanon",
    ", syria",
    ", iraq",
    ", iran",
    ", afghanistan",
    ", pakistan",
    ", bangladesh",
    ", sri lanka",
    ", thailand",
    ", vietnam",
    ", philippines",
    ", indonesia",
    ", malaysia",
    ", singapore",
    ", taiwan",
    ", hong kong",
    ", macau"
]


def canonicalize_location(location: str) -> str:
    """
    Canonicalize location string for consistent caching.
    
    Args:
        location: Raw location string (e.g., "London, England, United Kingdom")
        
    Returns:
        str: Canonicalized location (e.g., "london_england")
        
    Examples:
        "London, England, United Kingdom" → "london_england"
        "New York, NY, USA" → "new_york_ny"
        "Paris, France" → "paris"
    """
    if not location:
        return ""
    
    # Convert to lowercase and trim
    canonical = location.lower().strip()
    
    # Replace commas and spaces with underscores
    canonical = re.sub(r'[,\s]+', '_', canonical)
    
    # Remove multiple consecutive underscores
    canonical = re.sub(r'_+', '_', canonical)
    
    # Remove leading/trailing underscores
    canonical = canonical.strip('_')
    
    # Simplify common suffixes
    for suffix in COMMON_SUFFIXES:
        suffix_normalized = suffix.lower().replace(',', '_').replace(' ', '_')
        suffix_normalized = re.sub(r'_+', '_', suffix_normalized).strip('_')
        
        if canonical.endswith(f"_{suffix_normalized}"):
            canonical = canonical[:-len(f"_{suffix_normalized}")]
            break
    
    return canonical


def get_temporal_tolerance(agg: str) -> int:
    """
    Get temporal tolerance in days for a given aggregation period.
    
    Args:
        agg: Aggregation period ("daily", "monthly", "yearly")
        
    Returns:
        int: Tolerance in days
    """
    return TEMPORAL_TOLERANCE.get(agg.lower(), 0)


def date_to_timestamp(date_str: str) -> float:
    """
    Convert date string to timestamp for Redis sorted set operations.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        float: Unix timestamp
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.timestamp()
    except ValueError:
        # Try MM-DD format
        try:
            current_year = datetime.now().year
            dt = datetime.strptime(f"{current_year}-{date_str}", "%Y-%m-%d")
            return dt.timestamp()
        except ValueError:
            logger.error(f"Invalid date format: {date_str}")
            return 0.0


def timestamp_to_date(timestamp: float) -> str:
    """
    Convert timestamp back to date string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        str: Date in YYYY-MM-DD format
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d")


def calculate_temporal_delta(requested_date: str, cached_date: str) -> int:
    """
    Calculate the temporal delta in days between requested and cached dates.
    
    Args:
        requested_date: Requested date in YYYY-MM-DD format
        cached_date: Cached date in YYYY-MM-DD format
        
    Returns:
        int: Absolute difference in days
    """
    try:
        req_dt = datetime.strptime(requested_date, "%Y-%m-%d")
        cache_dt = datetime.strptime(cached_date, "%Y-%m-%d")
        return abs((req_dt - cache_dt).days)
    except ValueError:
        return 0


class ImprovedCache:
    """
    Improved cache with canonicalized location keys and temporal tolerance.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_prefix = "temphist:improved:"
    
    def _get_sorted_set_key(self, canonical_location: str, agg: str) -> str:
        """Get Redis sorted set key for a canonical location and aggregation."""
        return f"{self.cache_prefix}{canonical_location}:{agg}"
    
    def _get_data_key(self, canonical_location: str, agg: str, end_date: str) -> str:
        """Get Redis data key for cached payload."""
        return f"{self.cache_prefix}{canonical_location}:{agg}:{end_date}:data"
    
    async def cache_get(self, r: redis.Redis, agg: str, location: str, end_date: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Get cached data with temporal tolerance.
        
        Args:
            r: Redis client
            agg: Aggregation period ("daily", "monthly", "yearly")
            location: Location string
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple of (payload, meta) if found, None otherwise
        """
        canonical_location = canonicalize_location(location)
        tolerance = get_temporal_tolerance(agg)
        
        # Get sorted set key for this location and aggregation
        sorted_set_key = self._get_sorted_set_key(canonical_location, agg)
        
        try:
            # Convert end_date to timestamp
            target_timestamp = date_to_timestamp(end_date)
            
            if tolerance > 0:
                # Use temporal tolerance - find nearest cached date within tolerance
                min_timestamp = target_timestamp - (tolerance * 24 * 3600)
                max_timestamp = target_timestamp + (tolerance * 24 * 3600)
                
                # Get cached dates within tolerance range
                cached_dates = r.zrangebyscore(
                    sorted_set_key, 
                    min_timestamp, 
                    max_timestamp,
                    withscores=True
                )
                
                if cached_dates:
                    # Find the closest date
                    closest_date = None
                    min_delta = float('inf')
                    
                    for date_timestamp, score in cached_dates:
                        delta = abs(score - target_timestamp)
                        if delta < min_delta:
                            min_delta = delta
                            closest_date = timestamp_to_date(score)
                    
                    if closest_date:
                        # Get the cached data
                        data_key = self._get_data_key(canonical_location, agg, closest_date)
                        cached_data = r.get(data_key)
                        
                        if cached_data:
                            payload = json.loads(cached_data)
                            
                            # Calculate temporal delta
                            temporal_delta = calculate_temporal_delta(end_date, closest_date)
                            
                            # Create metadata
                            meta = {
                                "requested": {
                                    "location": location,
                                    "end_date": end_date
                                },
                                "served_from": {
                                    "canonical_location": canonical_location,
                                    "end_date": closest_date,
                                    "temporal_delta_days": temporal_delta
                                },
                                "approximate": {
                                    "temporal": temporal_delta > 0
                                }
                            }
                            
                            logger.debug(f"Cache hit with temporal tolerance: {canonical_location}:{agg}:{end_date} -> {closest_date} (Δ{temporal_delta}d)")
                            return payload, meta
            else:
                # Exact match only (daily data)
                cached_data = r.zscore(sorted_set_key, target_timestamp)
                if cached_data is not None:
                    data_key = self._get_data_key(canonical_location, agg, end_date)
                    cached_payload = r.get(data_key)
                    
                    if cached_payload:
                        payload = json.loads(cached_payload)
                        
                        # Create metadata for exact match
                        meta = {
                            "requested": {
                                "location": location,
                                "end_date": end_date
                            },
                            "served_from": {
                                "canonical_location": canonical_location,
                                "end_date": end_date,
                                "temporal_delta_days": 0
                            },
                            "approximate": {
                                "temporal": False
                            }
                        }
                        
                        logger.debug(f"Cache hit exact match: {canonical_location}:{agg}:{end_date}")
                        return payload, meta
            
            logger.debug(f"Cache miss: {canonical_location}:{agg}:{end_date}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for {canonical_location}:{agg}:{end_date}: {e}")
            return None
    
    async def cache_set(self, r: redis.Redis, agg: str, location: str, end_date: str, payload: Dict[str, Any]) -> bool:
        """
        Set cached data with canonicalized location key.
        
        Args:
            r: Redis client
            agg: Aggregation period ("daily", "monthly", "yearly")
            location: Location string
            end_date: End date in YYYY-MM-DD format
            payload: Data to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        canonical_location = canonicalize_location(location)
        
        try:
            # Convert end_date to timestamp for sorted set
            timestamp = date_to_timestamp(end_date)
            
            # Get keys
            sorted_set_key = self._get_sorted_set_key(canonical_location, agg)
            data_key = self._get_data_key(canonical_location, agg, end_date)
            
            # Store data
            r.setex(data_key, 86400 * 7, json.dumps(payload))  # 7 days TTL
            
            # Add to sorted set for temporal tolerance lookup
            r.zadd(sorted_set_key, {timestamp: timestamp})
            
            # Set TTL for sorted set
            r.expire(sorted_set_key, 86400 * 7)  # 7 days TTL
            
            logger.debug(f"Cache set: {canonical_location}:{agg}:{end_date}")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for {canonical_location}:{agg}:{end_date}: {e}")
            return False
    
    async def cache_invalidate(self, r: redis.Redis, agg: str, location: str, end_date: str = None) -> bool:
        """
        Invalidate cached data for a specific location and aggregation.
        
        Args:
            r: Redis client
            agg: Aggregation period
            location: Location string
            end_date: Specific date to invalidate (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        canonical_location = canonicalize_location(location)
        
        try:
            sorted_set_key = self._get_sorted_set_key(canonical_location, agg)
            
            if end_date:
                # Invalidate specific date
                timestamp = date_to_timestamp(end_date)
                data_key = self._get_data_key(canonical_location, agg, end_date)
                
                r.delete(data_key)
                r.zrem(sorted_set_key, timestamp)
                
                logger.debug(f"Cache invalidated: {canonical_location}:{agg}:{end_date}")
            else:
                # Invalidate all dates for this location/aggregation
                # Get all timestamps
                timestamps = r.zrange(sorted_set_key, 0, -1)
                
                # Delete all data keys
                for timestamp in timestamps:
                    date_str = timestamp_to_date(float(timestamp))
                    data_key = self._get_data_key(canonical_location, agg, date_str)
                    r.delete(data_key)
                
                # Delete sorted set
                r.delete(sorted_set_key)
                
                logger.debug(f"Cache invalidated all: {canonical_location}:{agg}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation error for {canonical_location}:{agg}: {e}")
            return False
    
    async def cache_stats(self, r: redis.Redis) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            r: Redis client
            
        Returns:
            Dict with cache statistics
        """
        try:
            # Get all cache keys
            pattern = f"{self.cache_prefix}*"
            keys = r.keys(pattern)
            
            stats = {
                "total_keys": len(keys),
                "sorted_sets": 0,
                "data_keys": 0,
                "locations": set(),
                "aggregations": set()
            }
            
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                
                if ":data" in key_str:
                    stats["data_keys"] += 1
                else:
                    stats["sorted_sets"] += 1
                
                # Extract location and aggregation from key
                parts = key_str.replace(self.cache_prefix, "").split(":")
                if len(parts) >= 2:
                    stats["locations"].add(parts[0])
                    stats["aggregations"].add(parts[1])
            
            stats["locations"] = list(stats["locations"])
            stats["aggregations"] = list(stats["aggregations"])
            
            return stats
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}


# Global cache instance
_improved_cache: Optional[ImprovedCache] = None


def initialize_improved_cache(redis_client: redis.Redis):
    """Initialize the global improved cache instance."""
    global _improved_cache
    _improved_cache = ImprovedCache(redis_client)
    logger.info("Improved cache initialized with canonicalized location keys and temporal tolerance")


def get_improved_cache() -> ImprovedCache:
    """Get the global improved cache instance."""
    if _improved_cache is None:
        raise RuntimeError("Improved cache not initialized. Call initialize_improved_cache() first.")
    return _improved_cache


# Convenience functions for backward compatibility
async def cache_get(r: redis.Redis, agg: str, location: str, end_date: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Convenience function for cache_get."""
    try:
        cache = get_improved_cache()
        return await cache.cache_get(r, agg, location, end_date)
    except RuntimeError as e:
        # Cache not initialized yet - return None (cache miss)
        logger.debug(f"Improved cache not initialized, skipping cache_get: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in cache_get: {e}")
        return None


async def cache_set(r: redis.Redis, agg: str, location: str, end_date: str, payload: Dict[str, Any]) -> bool:
    """Convenience function for cache_set."""
    try:
        cache = get_improved_cache()
        return await cache.cache_set(r, agg, location, end_date, payload)
    except RuntimeError as e:
        # Cache not initialized yet - return False (cache not stored)
        logger.debug(f"Improved cache not initialized, skipping cache_set: {e}")
        return False
    except Exception as e:
        logger.error(f"Error in cache_set: {e}")
        return False
