"""
Improved caching utilities for TempHist API with canonicalized location keys and temporal tolerance.

This module provides:
- Canonicalized location keys using a resolved canonical name when available
- Temporal tolerance for different aggregation periods
- Redis sorted set-based date matching
- Metadata tracking for approximate data usage
- Gzip compression for efficient storage

Uses sync redis.Redis to match the rest of the codebase.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
from datetime import date, datetime, timezone
from typing import Literal, Optional

import redis

logger = logging.getLogger(__name__)

# Temporal tolerance configuration
Agg = Literal["daily", "weekly", "monthly", "yearly"]

TEMPORAL_TOLERANCE: dict[str, int] = {
    "daily": 0,  # Exact match only
    "weekly": 0,  # Exact match only
    "monthly": 2,  # ±2 days for monthly data
    "yearly": 7,  # ±7 days for yearly data
}

KEY_NS = "thv1"  # Namespace for cache keys (allows migration without mass deletes)


def _epoch(d: date) -> int:
    """Convert date to Unix timestamp (seconds since epoch)."""
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def _ensure_date_str(d: date | str) -> str:
    """Ensure date is in ISO format string."""
    return d if isinstance(d, str) else d.isoformat()


def _preferred_canonical_name(
    canonical_name: Optional[str] = None,
    resolved_address: Optional[str] = None,
) -> Optional[str]:
    """Return the preferred canonical location label for key generation."""
    return canonical_name or resolved_address


def canonicalize_location(
    original: str,
    resolved_address: Optional[str] = None,
    *,
    canonical_name: Optional[str] = None,
) -> str:
    """
    Canonicalize location string for consistent temporal cache keys.

    Prefer a resolved canonical name (from Postgres, preapproved data, or legacy
    provider metadata) when provided; otherwise do safe lexical normalization.

    Examples:
        canonicalize_location("London", canonical_name="London, England, United Kingdom")
        -> "london_england_united_kingdom"

        canonicalize_location("New York")
        -> "new_york"
    """
    preferred = _preferred_canonical_name(canonical_name, resolved_address)
    s = (preferred or original or "").strip().lower()

    if not s:
        return ""

    parts = " ".join(part.strip() for part in s.replace(",", " ").split())
    canonical = "_".join(part for part in parts.split() if part)

    return canonical


def _temporal_lookup_canonicals(
    original: str,
    resolved_address: Optional[str] = None,
    canonical_name: Optional[str] = None,
) -> list[str]:
    """Ordered temporal cache location keys: canonical first, then legacy lexical."""
    primary = canonicalize_location(original, resolved_address, canonical_name=canonical_name)
    legacy = canonicalize_location(original)
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in (primary, legacy):
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _migrate_temporal_entry(
    r: redis.Redis,
    *,
    agg: str,
    from_canonical: str,
    to_canonical: str,
    end_iso: str,
) -> None:
    """Copy a temporal cache entry from a legacy canonical key to the preferred key."""
    if from_canonical == to_canonical:
        return
    old_vkey = _val_key(agg, from_canonical, end_iso)
    new_vkey = _val_key(agg, to_canonical, end_iso)
    if r.exists(new_vkey):
        r.delete(old_vkey)
        old_tkey = _tindex_key(agg, from_canonical)
        r.zrem(old_tkey, end_iso)
        return
    if not r.exists(old_vkey):
        return
    try:
        ttl = r.ttl(old_vkey)
        value = r.get(old_vkey)
        if value is None:
            return
        if ttl and ttl > 0:
            r.setex(new_vkey, ttl, value)
        else:
            r.set(new_vkey, value)
        r.delete(old_vkey)
        old_tkey = _tindex_key(agg, from_canonical)
        new_tkey = _tindex_key(agg, to_canonical)
        score = r.zscore(old_tkey, end_iso)
        if score is not None:
            r.zadd(new_tkey, {end_iso: score})
            r.zrem(old_tkey, end_iso)
            index_ttl = r.ttl(old_tkey)
            if index_ttl and index_ttl > 0:
                r.expire(new_tkey, index_ttl)
    except Exception as exc:
        logger.debug("Temporal cache migrate %s -> %s failed: %s", from_canonical, to_canonical, exc)


def _decode_temporal_entry(stored, original_location: str) -> Optional[dict]:
    if not stored:
        return None
    raw_gz = base64.b64decode(stored) if isinstance(stored, str) else stored
    obj = json.loads(gzip.decompress(raw_gz))
    obj["meta"]["requested"]["location"] = original_location
    return obj


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
    canonical_name: Optional[str] = None,
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
        resolved_address: Deprecated alias for canonical_name
        canonical_name: Canonical location label for key generation
        ttl_seconds: Time to live in seconds (default: 7 days)

    Returns:
        True if successful, False otherwise
    """
    try:
        canonical = canonicalize_location(
            original_location,
            resolved_address,
            canonical_name=canonical_name,
        )
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
        # Store as base64 so it's safe with decode_responses=True Redis clients
        stored = base64.b64encode(gz).decode("ascii")

        ttl = ttl_seconds if ttl_seconds and ttl_seconds > 0 else 86400 * 7
        pipe = r.pipeline()
        pipe.set(vkey, stored, ex=ttl)
        pipe.zadd(tkey, {end_iso: _epoch(date.fromisoformat(end_iso))})
        pipe.expire(tkey, ttl)
        pipe.execute()

        logger.debug(f"Temporal cache set: {canonical}:{agg}:{end_iso}")
        return True

    except Exception as e:
        logger.error(f"Temporal cache set error for {agg}:{original_location}:{end_date}: {e}")
        return False


def _safe_redis_get(r: redis.Redis, key: str):
    """Get a Redis value, deleting the key and returning None on UnicodeDecodeError."""
    try:
        return r.get(key)
    except UnicodeDecodeError:
        logger.info(f"Temporal cache legacy key deleted (non-UTF-8): {key}")
        try:
            r.delete(key)
        except Exception as _del_err:
            logger.debug("Failed to delete legacy cache key %s: %s", key, _del_err)
        return None


def _try_exact_match(
    r: redis.Redis,
    lookup_canonicals,
    primary_canonical: str,
    agg: str,
    end_iso: str,
    original_location: str,
) -> Optional[dict]:
    for canonical in lookup_canonicals:
        stored = _safe_redis_get(r, _val_key(agg, canonical, end_iso))
        if not stored:
            continue
        obj = _decode_temporal_entry(stored, original_location)
        if obj and canonical != primary_canonical:
            _migrate_temporal_entry(r, agg=agg, from_canonical=canonical, to_canonical=primary_canonical, end_iso=end_iso)
        if obj:
            logger.debug(f"Temporal cache hit (exact): {canonical}:{agg}:{end_iso}")
        return obj
    return None


def _try_approximate_match(
    r: redis.Redis,
    lookup_canonicals,
    primary_canonical: str,
    agg: str,
    end_date: date,
    end_iso: str,
    original_location: str,
) -> Optional[dict]:
    tol_days = TEMPORAL_TOLERANCE.get(agg, 0)
    if tol_days <= 0:
        return None
    center = _epoch(end_date)
    window = tol_days * 86400

    for canonical in lookup_canonicals:
        candidates = r.zrangebyscore(_tindex_key(agg, canonical), center - window, center + window)
        if not candidates:
            continue
        nearest_iso = min(candidates, key=lambda b: abs((end_date - date.fromisoformat(b.decode() if isinstance(b, bytes) else b)).days))
        if isinstance(nearest_iso, bytes):
            nearest_iso = nearest_iso.decode()
        stored = _safe_redis_get(r, _val_key(agg, canonical, nearest_iso))
        if not stored:
            continue
        obj = _decode_temporal_entry(stored, original_location)
        if not obj:
            continue
        delta_days = abs((end_date - date.fromisoformat(nearest_iso)).days)
        if canonical != primary_canonical:
            _migrate_temporal_entry(r, agg=agg, from_canonical=canonical, to_canonical=primary_canonical, end_iso=nearest_iso)
        obj["meta"]["approximate"]["temporal"] = True
        obj["meta"]["served_from"]["canonical_location"] = primary_canonical
        obj["meta"]["served_from"]["end_date"] = nearest_iso
        obj["meta"]["served_from"]["temporal_delta_days"] = delta_days
        obj["meta"]["requested"] = {"location": original_location, "end_date": end_iso}
        logger.debug(f"Temporal cache hit (approx): {canonical}:{agg}:{end_iso} -> {nearest_iso} (Δ{delta_days}d)")
        return obj
    return None


def cache_get(
    r: redis.Redis,
    *,
    agg: str,
    original_location: str,
    end_date: date,
    resolved_address: Optional[str] = None,
    canonical_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Retrieve cached data with temporal tolerance support.

    Args:
        r: Redis client
        agg: Aggregation period ("daily", "weekly", "monthly", "yearly")
        original_location: Original location string
        end_date: End date to look up
        resolved_address: Deprecated alias for canonical_name
        canonical_name: Canonical location label for key generation

    Returns:
        Wrapped data with metadata if found, None otherwise
    """
    try:
        end_iso = _ensure_date_str(end_date)
        lookup_canonicals = _temporal_lookup_canonicals(original_location, resolved_address, canonical_name=canonical_name)
        if not lookup_canonicals:
            return None
        primary_canonical = lookup_canonicals[0]
        return (
            _try_exact_match(r, lookup_canonicals, primary_canonical, agg, end_iso, original_location)
            or _try_approximate_match(r, lookup_canonicals, primary_canonical, agg, end_date, end_iso, original_location)
        )
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
    canonical_name: Optional[str] = None,
) -> bool:
    """
    Invalidate cached data for a specific location and aggregation.

    Args:
        r: Redis client
        agg: Aggregation period
        original_location: Original location string
        end_date: Specific date to invalidate (None = all dates)
        resolved_address: Deprecated alias for canonical_name
        canonical_name: Canonical location label for key generation
    """
    try:
        lookup_canonicals = _temporal_lookup_canonicals(
            original_location,
            resolved_address,
            canonical_name=canonical_name,
        )
        if not lookup_canonicals:
            return False

        if end_date:
            end_iso = _ensure_date_str(end_date)
            for canonical in lookup_canonicals:
                vkey = _val_key(agg, canonical, end_iso)
                tkey = _tindex_key(agg, canonical)
                pipe = r.pipeline()
                pipe.delete(vkey)
                pipe.zrem(tkey, end_iso)
                pipe.execute()
        else:
            for canonical in lookup_canonicals:
                tkey = _tindex_key(agg, canonical)
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
