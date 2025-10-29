"""
Improved caching utilities for TempHist API with canonicalized location keys and temporal tolerance.

This module provides:
- Async Redis operations (non-blocking)
- Canonicalized location keys using VC's resolvedAddress when available
- Temporal tolerance for different aggregation periods
- Redis sorted set-based date matching
- Metadata tracking for approximate data usage
- Gzip compression for efficient storage
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date, datetime, timezone, timedelta
from typing import Literal, Optional, Dict, Any

from redis.asyncio import Redis  # async client

logger = logging.getLogger(__name__)

# Temporal tolerance configuration
Agg = Literal["daily", "monthly", "yearly"]

TEMPORAL_TOLERANCE: dict[Agg, int] = {
    "daily": 0,    # Exact match only
    "monthly": 2,  # ±2 days for monthly data
    "yearly": 7    # ±7 days for yearly data
}

KEY_NS = "thv1"  # Namespace for cache keys (allows migration without mass deletes)


def _epoch(d: date) -> int:
    """Convert date to Unix timestamp (seconds since epoch)."""
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _ensure_date_str(d: date | str) -> str:
    """Ensure date is in ISO format string."""
    return d if isinstance(d, str) else d.isoformat()


def canonicalize_location(original: str, resolved_address: Optional[str] = None) -> str:
    """
    Canonicalize location string for consistent caching.
    
    Prefer VC's resolvedAddress (stable) if provided; otherwise do safe lexical normalization.
    No hard-coded suffix lists - let VC's data be the source of truth.
    
    Args:
        original: Original location string from user
        resolved_address: Visual Crossing's resolvedAddress (preferred if available)
        
    Returns:
        Canonicalized location string
        
    Examples:
        canonicalize_location("London, UK", "London, England, United Kingdom") 
        -> "london_england_united_kingdom"
        
        canonicalize_location("New York", None)
        -> "new_york"
    """
    # Prefer resolvedAddress from VC API (most stable)
    s = (resolved_address or original or "").strip().lower()
    
    if not s:
        return ""
    
    # Safe lexical normalization: collapse whitespace, replace commas with spaces,
    # then split and join with underscores, dropping duplicate underscores
    parts = " ".join(part.strip() for part in s.replace(",", " ").split())
    canonical = "_".join(part for part in parts.split() if part)
    
    return canonical


def _val_key(agg: Agg, canonical_loc: str, end_iso: str) -> str:
    """Generate Redis key for cached value."""
    return f"{KEY_NS}:{agg}:{canonical_loc}:{end_iso}"


def _tindex_key(agg: Agg, canonical_loc: str) -> str:
    """Generate Redis key for temporal index (sorted set)."""
    return f"{KEY_NS}:idx:{agg}:{canonical_loc}"


async def cache_set(
    r: Redis,
    *,
    agg: Agg,
    original_location: str,
    end_date: date,
    payload: dict,
    resolved_address: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> bool:
    """
    Store data in cache with canonicalized location key and temporal index.
    
    Args:
        r: Async Redis client
        agg: Aggregation period ("daily", "monthly", "yearly")
        original_location: Original location string
        end_date: End date (use full YYYY-MM-DD for monthly/yearly)
        payload: Data payload to cache
        resolved_address: VC's resolvedAddress (preferred for canonicalization)
        ttl_seconds: Time to live in seconds (default: 7 days)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        canonical = canonicalize_location(original_location, resolved_address)
        end_iso = _ensure_date_str(end_date)
        vkey = _val_key(agg, canonical, end_iso)
        tkey = _tindex_key(agg, canonical)
        
        # Wrap with baseline metadata (exact match)
        wrapped = {
            "data": payload,
            "meta": {
                "requested": {
                    "location": original_location,
                    "end_date": end_iso
                },
                "served_from": {
                    "canonical_location": canonical,
                    "end_date": end_iso,
                    "temporal_delta_days": 0,
                },
                "approximate": {"temporal": False},
            },
        }
        
        # Compress with gzip
        raw = json.dumps(wrapped, separators=(",", ":")).encode("utf-8")
        gz = gzip.compress(raw)
        
        # Use pipeline for atomic operations
        ttl = ttl_seconds if ttl_seconds and ttl_seconds > 0 else 86400 * 7  # Default 7 days
        pipe = r.pipeline()
        pipe.set(vkey, gz, ex=ttl)
        pipe.zadd(tkey, {end_iso: _epoch(date.fromisoformat(end_iso))})
        pipe.expire(tkey, ttl)
        await pipe.execute()
        
        logger.debug(f"Cache set: {canonical}:{agg}:{end_iso}")
        return True
        
    except Exception as e:
        logger.error(f"Cache set error for {agg}:{original_location}:{end_date}: {e}")
        return False


async def cache_get(
    r: Redis,
    *,
    agg: Agg,
    original_location: str,
    end_date: date,
    resolved_address: Optional[str] = None,
) -> Optional[dict]:
    """
    Retrieve cached data with temporal tolerance support.
    
    Args:
        r: Async Redis client
        agg: Aggregation period ("daily", "monthly", "yearly")
        original_location: Original location string
        end_date: End date to look up
        resolved_address: VC's resolvedAddress (for canonicalization)
        
    Returns:
        Wrapped data with metadata if found, None otherwise
    """
    try:
        canonical = canonicalize_location(original_location, resolved_address)
        end_iso = _ensure_date_str(end_date)
        vkey = _val_key(agg, canonical, end_iso)
        
        # Try exact match first
        gz = await r.get(vkey)
        if gz:
            obj = json.loads(gzip.decompress(gz))
            logger.debug(f"Cache hit exact match: {canonical}:{agg}:{end_iso}")
            return obj
        
        # Try temporal tolerance for non-daily aggregations
        tol_days = TEMPORAL_TOLERANCE[agg]
        if tol_days <= 0:
            logger.debug(f"Cache miss: {canonical}:{agg}:{end_iso}")
            return None
        
        # Look for nearest date within tolerance
        tkey = _tindex_key(agg, canonical)
        center = _epoch(end_date)
        window = tol_days * 86400
        
        candidates = await r.zrangebyscore(
            tkey,
            center - window,
            center + window,
            withscores=False
        )
        
        if not candidates:
            logger.debug(f"Cache miss (no candidates in tolerance): {canonical}:{agg}:{end_iso}")
            return None
        
        # Pick nearest by absolute delta
        def _delta(iso_bytes: bytes) -> int:
            """Calculate absolute days difference."""
            d = date.fromisoformat(iso_bytes.decode())
            return abs((end_date - d).days)
        
        nearest_iso = min(candidates, key=_delta).decode()
        gz2 = await r.get(_val_key(agg, canonical, nearest_iso))
        
        if not gz2:
            logger.debug(f"Cache miss (candidate found but data missing): {canonical}:{agg}:{end_iso}")
            return None
        
        # Decompress and update metadata for approximate match
        obj = json.loads(gzip.decompress(gz2))
        delta_days = abs((end_date - date.fromisoformat(nearest_iso)).days)
        
        obj["meta"]["approximate"]["temporal"] = True
        obj["meta"]["served_from"]["canonical_location"] = canonical
        obj["meta"]["served_from"]["end_date"] = nearest_iso
        obj["meta"]["served_from"]["temporal_delta_days"] = delta_days
        obj["meta"]["requested"] = {
            "location": original_location,
            "end_date": end_iso
        }
        
        logger.debug(f"Cache hit with temporal tolerance: {canonical}:{agg}:{end_iso} -> {nearest_iso} (Δ{delta_days}d)")
        return obj
        
    except Exception as e:
        logger.error(f"Cache get error for {agg}:{original_location}:{end_date}: {e}")
        return None


async def cache_invalidate(
    r: Redis,
    *,
    agg: Agg,
    original_location: str,
    end_date: Optional[date] = None,
    resolved_address: Optional[str] = None,
) -> bool:
    """
    Invalidate cached data for a specific location and aggregation.
    
    Args:
        r: Async Redis client
        agg: Aggregation period
        original_location: Original location string
        end_date: Specific date to invalidate (None = all dates)
        resolved_address: VC's resolvedAddress (for canonicalization)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        canonical = canonicalize_location(original_location, resolved_address)
        tkey = _tindex_key(agg, canonical)
        
        if end_date:
            # Invalidate specific date
            end_iso = _ensure_date_str(end_date)
            vkey = _val_key(agg, canonical, end_iso)
            timestamp = _epoch(end_date)
            
            pipe = r.pipeline()
            pipe.delete(vkey)
            pipe.zrem(tkey, end_iso)
            await pipe.execute()
            
            logger.debug(f"Cache invalidated: {canonical}:{agg}:{end_iso}")
        else:
            # Invalidate all dates for this location/aggregation
            timestamps = await r.zrange(tkey, 0, -1)
            
            pipe = r.pipeline()
            # Delete all data keys
            for timestamp_iso in timestamps:
                vkey = _val_key(agg, canonical, timestamp_iso.decode())
                pipe.delete(vkey)
            # Delete sorted set
            pipe.delete(tkey)
            await pipe.execute()
            
            logger.debug(f"Cache invalidated all: {canonical}:{agg}")
        
        return True
        
    except Exception as e:
        logger.error(f"Cache invalidation error for {agg}:{original_location}:{end_date}: {e}")
        return False


async def cache_stats(r: Redis) -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Args:
        r: Async Redis client
        
    Returns:
        Dict with cache statistics
    """
    try:
        # Get all cache keys with the namespace
        pattern = f"{KEY_NS}:*"
        keys = []
        async for key in r.scan_iter(match=pattern):
            keys.append(key)
        
        stats = {
            "total_keys": len(keys),
            "sorted_sets": 0,
            "data_keys": 0,
            "locations": set(),
            "aggregations": set()
        }
        
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            
            if ":idx:" in key_str:
                stats["sorted_sets"] += 1
            else:
                stats["data_keys"] += 1
            
            # Extract location and aggregation from key
            parts = key_str.replace(f"{KEY_NS}:", "").split(":")
            if len(parts) >= 2:
                # Skip idx: prefix for sorted sets
                idx_offset = 1 if ":idx:" in key_str else 0
                if len(parts) > idx_offset:
                    stats["aggregations"].add(parts[idx_offset])
                    if len(parts) > idx_offset + 1:
                        stats["locations"].add(parts[idx_offset + 1])
        
        stats["locations"] = list(stats["locations"])
        stats["aggregations"] = list(stats["aggregations"])
        
        return stats
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {"error": str(e)}


# Legacy compatibility functions (will be deprecated)
async def initialize_improved_cache(redis_client: Redis):
    """Initialize wrapper - no-op for async Redis."""
    logger.info("Improved cache ready (using async Redis client)")


# Note: These convenience functions are kept for backward compatibility
# but should migrate to using cache_get/cache_set directly with proper date objects
async def _legacy_cache_get(r: Redis, agg: str, location: str, end_date: str) -> Optional[dict]:
    """Legacy wrapper for string dates."""
    try:
        end_d = date.fromisoformat(end_date) if "-" in end_date else date.fromisoformat(f"{datetime.now().year}-{end_date}")
        return await cache_get(r, agg=agg, original_location=location, end_date=end_d)  # type: ignore
    except ValueError:
        logger.error(f"Invalid date format in legacy cache_get: {end_date}")
        return None


async def _legacy_cache_set(r: Redis, agg: str, location: str, end_date: str, payload: dict) -> bool:
    """Legacy wrapper for string dates."""
    try:
        end_d = date.fromisoformat(end_date) if "-" in end_date else date.fromisoformat(f"{datetime.now().year}-{end_date}")
        return await cache_set(r, agg=agg, original_location=location, end_date=end_d, payload=payload)  # type: ignore
    except ValueError:
        logger.error(f"Invalid date format in legacy cache_set: {end_date}")
        return False
