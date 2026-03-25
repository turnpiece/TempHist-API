"""
Improved caching utilities for TempHist API with canonicalized location keys and temporal tolerance.

This module provides:
- Canonicalized location keys using VC's resolvedAddress when available
- Temporal tolerance for different aggregation periods
- Redis sorted set-based date matching
- Metadata tracking for approximate data usage
- Gzip compression for efficient storage

Uses sync redis.Redis to match the rest of the codebase.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import date, datetime, timezone
from typing import Literal, Optional, Dict, Any

import redis

logger = logging.getLogger(__name__)

# Temporal tolerance configuration
Agg = Literal["daily", "weekly", "monthly", "yearly"]

TEMPORAL_TOLERANCE: dict[str, int] = {
    "daily": 0,    # Exact match only
    "weekly": 0,   # Exact match only
    "monthly": 2,  # ±2 days for monthly data
    "yearly": 7,   # ±7 days for yearly data
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

    Examples:
        canonicalize_location("London, UK", "London, England, United Kingdom")
        -> "london_england_united_kingdom"

        canonicalize_location("New York", None)
        -> "new_york"
    """
    s = (resolved_address or original or "").strip().lower()

    if not s:
        return ""

    parts = " ".join(part.strip() for part in s.replace(",", " ").split())
    canonical = "_".join(part for part in parts.split() if part)

    return canonical


def _val_key(agg: str, canonical_loc: str, end_iso: str) -> str:
    """Generate Redis key for cached value."""
    return f"{KEY_NS}:{agg}:{canonical_loc}:{end_iso}"


def _tindex_key(agg: str, canonical_loc: str) -> str:
    """Generate Redis key for temporal index (sorted set)."""
    return f"{KEY_NS}:idx:{agg}:{canonical_loc}"


def cache_set(
    r: redis.Redis,
    *,
    agg: str,
    original_location: str,
    end_date: date,
    payload: dict,
    resolved_address: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> bool:
    """
    Store data in cache with canonicalized location key and temporal index.

    Args:
        r: Redis client
        agg: Aggregation period ("daily", "weekly", "monthly", "yearly")
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

        wrapped = {
            "data": payload,
            "meta": {
                "requested": {
                    "location": original_location,
                    "end_date": end_iso,
                },
                "served_from": {
                    "canonical_location": canonical,
                    "end_date": end_iso,
                    "temporal_delta_days": 0,
                },
                "approximate": {"temporal": False},
            },
        }

        raw = json.dumps(wrapped, separators=(",", ":")).encode("utf-8")
        gz = gzip.compress(raw)

        ttl = ttl_seconds if ttl_seconds and ttl_seconds > 0 else 86400 * 7
        pipe = r.pipeline()
        pipe.set(vkey, gz, ex=ttl)
        pipe.zadd(tkey, {end_iso: _epoch(date.fromisoformat(end_iso))})
        pipe.expire(tkey, ttl)
        pipe.execute()

        logger.debug(f"Temporal cache set: {canonical}:{agg}:{end_iso}")
        return True

    except Exception as e:
        logger.error(f"Temporal cache set error for {agg}:{original_location}:{end_date}: {e}")
        return False


def cache_get(
    r: redis.Redis,
    *,
    agg: str,
    original_location: str,
    end_date: date,
    resolved_address: Optional[str] = None,
) -> Optional[dict]:
    """
    Retrieve cached data with temporal tolerance support.

    Args:
        r: Redis client
        agg: Aggregation period ("daily", "weekly", "monthly", "yearly")
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
        gz = r.get(vkey)
        if gz:
            obj = json.loads(gzip.decompress(gz))
            # Update the requested location to match what the caller asked for
            obj["meta"]["requested"]["location"] = original_location
            logger.debug(f"Temporal cache hit (exact): {canonical}:{agg}:{end_iso}")
            return obj

        # Try temporal tolerance for non-daily aggregations
        tol_days = TEMPORAL_TOLERANCE.get(agg, 0)
        if tol_days <= 0:
            return None

        # Look for nearest date within tolerance via sorted set
        tkey = _tindex_key(agg, canonical)
        center = _epoch(end_date)
        window = tol_days * 86400

        candidates = r.zrangebyscore(tkey, center - window, center + window)

        if not candidates:
            return None

        # Pick nearest by absolute delta
        def _delta(iso_bytes: bytes) -> int:
            d = date.fromisoformat(iso_bytes.decode() if isinstance(iso_bytes, bytes) else iso_bytes)
            return abs((end_date - d).days)

        nearest_iso = min(candidates, key=_delta)
        if isinstance(nearest_iso, bytes):
            nearest_iso = nearest_iso.decode()

        gz2 = r.get(_val_key(agg, canonical, nearest_iso))

        if not gz2:
            return None

        obj = json.loads(gzip.decompress(gz2))
        delta_days = abs((end_date - date.fromisoformat(nearest_iso)).days)

        obj["meta"]["approximate"]["temporal"] = True
        obj["meta"]["served_from"]["canonical_location"] = canonical
        obj["meta"]["served_from"]["end_date"] = nearest_iso
        obj["meta"]["served_from"]["temporal_delta_days"] = delta_days
        obj["meta"]["requested"] = {
            "location": original_location,
            "end_date": end_iso,
        }

        logger.debug(
            f"Temporal cache hit (approx): {canonical}:{agg}:{end_iso} -> {nearest_iso} (Δ{delta_days}d)"
        )
        return obj

    except Exception as e:
        logger.error(f"Temporal cache get error for {agg}:{original_location}:{end_date}: {e}")
        return None


def cache_invalidate(
    r: redis.Redis,
    *,
    agg: str,
    original_location: str,
    end_date: Optional[date] = None,
    resolved_address: Optional[str] = None,
) -> bool:
    """
    Invalidate cached data for a specific location and aggregation.

    Args:
        r: Redis client
        agg: Aggregation period
        original_location: Original location string
        end_date: Specific date to invalidate (None = all dates)
        resolved_address: VC's resolvedAddress (for canonicalization)
    """
    try:
        canonical = canonicalize_location(original_location, resolved_address)
        tkey = _tindex_key(agg, canonical)

        if end_date:
            end_iso = _ensure_date_str(end_date)
            vkey = _val_key(agg, canonical, end_iso)

            pipe = r.pipeline()
            pipe.delete(vkey)
            pipe.zrem(tkey, end_iso)
            pipe.execute()
        else:
            timestamps = r.zrange(tkey, 0, -1)

            pipe = r.pipeline()
            for ts_iso in timestamps:
                vkey = _val_key(agg, canonical, ts_iso.decode() if isinstance(ts_iso, bytes) else ts_iso)
                pipe.delete(vkey)
            pipe.delete(tkey)
            pipe.execute()

        return True

    except Exception as e:
        logger.error(f"Temporal cache invalidation error for {agg}:{original_location}:{end_date}: {e}")
        return False
