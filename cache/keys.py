"""
Cache key builders, key-generator functions, and timezone helpers.
"""

import hashlib
import json
import logging
import os
from datetime import date as dt_date
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import redis

from cache.core import COORD_PRECISION
from config import DEBUG

logger = logging.getLogger(__name__)

# Module-level cache for preapproved locations data
_preapproved_locations_cache: Optional[List[Dict[str, Any]]] = None

# Try to import zoneinfo (Python 3.9+); fall back to UTC if unavailable.
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ---------------------------------------------------------------------------
# CacheKeyBuilder
# ---------------------------------------------------------------------------


_DEFAULT_SKIPS = {
    "unit_group": {"celsius"},
    "month_mode": {"rolling1m"},
    "days_back": {"7", "0"},
}
_COORD_KEYS = {"lat", "lon", "latitude", "longitude"}


def _normalize_coord_value(str_value: str) -> str:
    try:
        coord = float(str_value)
        return f"{coord:.{COORD_PRECISION}f}"
    except (ValueError, TypeError):
        return str_value


def _is_default_skip(key: str, str_value: str) -> bool:
    skips = _DEFAULT_SKIPS.get(key)
    return skips is not None and str_value in skips


class CacheKeyBuilder:
    """Build canonical cache keys with normalized parameters."""

    @staticmethod
    def normalize_params(params: Dict[str, Any]) -> Dict[str, str]:
        normalized = {}
        for key, value in sorted(params.items()):
            if value is None or value == "":
                continue
            str_value = str(value).strip()
            if not str_value:
                continue
            if key in ["location"]:
                str_value = str_value.lower().replace(" ", "_").replace(",", "_")
            if key in _COORD_KEYS:
                str_value = _normalize_coord_value(str_value)
            if _is_default_skip(key, str_value):
                continue
            normalized[key] = str_value
        return normalized

    @staticmethod
    def build_cache_key(
        endpoint: str,
        path_params: Dict[str, str] = None,
        query_params: Dict[str, Any] = None,
        prefix: str = "temphist",
    ) -> str:
        path_params = path_params or {}
        query_params = query_params or {}

        path_parts = []
        for key in sorted(path_params.keys()):
            value = str(path_params[key]).strip().lower()
            if key == "location":
                value = value.replace(" ", "_").replace(",", "_")
            path_parts.append(f"{key}={value}")

        normalized_query = CacheKeyBuilder.normalize_params(query_params)
        query_parts = [f"{k}={v}" for k, v in sorted(normalized_query.items())]

        key_parts = [prefix, endpoint]
        if path_parts:
            key_parts.extend(path_parts)
        if query_parts:
            key_parts.extend(query_parts)

        return ":".join(key_parts)


# ---------------------------------------------------------------------------
# Simple key functions
# ---------------------------------------------------------------------------


def normalize_location_for_cache(location: str) -> str:
    return location.lower().replace(" ", "_").replace(",", "_")


def get_weather_cache_key(location: str, date_str: str) -> str:
    return f"{normalize_location_for_cache(location)}_{date_str}"


def generate_cache_key(prefix: str, location: str, date_part: str = "") -> str:
    location = normalize_location_for_cache(location)
    if date_part:
        date_part = date_part.replace("-", "_")
        return f"{prefix}_{location}_{date_part}"
    return f"{prefix}_{location}"


def rec_key(scope: str, slug: str, identifier: str, year: int) -> str:
    return f"records:v1:{scope}:{slug}:{identifier}:{year}"


def bundle_key(scope: str, slug: str, identifier: str) -> str:
    return f"records:v1:{scope}:{slug}:{identifier}:bundle"


def rec_etag_key(scope: str, slug: str, identifier: str, year: int) -> str:
    return f"records:v1:{scope}:{slug}:{identifier}:{year}:etag"


def compute_bundle_etag(year_etags: Dict[int, str]) -> str:
    sorted_etags = [f"{year}:{year_etags[year].strip(chr(34) + chr(39))}" for year in sorted(year_etags.keys())]
    etag_string = "|".join(sorted_etags)
    return f'"{hashlib.sha256(etag_string.encode()).hexdigest()[:32]}"'


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------


def _resolve_redis_client(redis_client: Optional[redis.Redis]) -> Optional[redis.Redis]:
    """Return the supplied client, or lazy-load one from the cache accessor."""
    if redis_client is not None:
        return redis_client
    try:
        from cache.accessors import get_cache  # lazy to avoid circular import

        return get_cache().redis
    except Exception:
        return None


def _get_location_timezone_from_cache(
    location: str,
    redis_client: Optional[redis.Redis] = None,
) -> Optional[str]:
    """Get timezone for a location from Redis cache."""
    try:
        redis_client = _resolve_redis_client(redis_client)
        if not redis_client:
            return None
        normalized = normalize_location_for_cache(location)
        cached = redis_client.get(f"location_timezone:{normalized}")
        if cached:
            return cached.decode("utf-8") if isinstance(cached, bytes) else cached
        return None
    except Exception as e:
        if DEBUG:
            logger.debug(f"Could not get timezone from cache for '{location}': {e}")
        return None


def store_location_timezone(
    location: str,
    timezone_str: str,
    redis_client: Optional[redis.Redis] = None,
    ttl: int = 604800,
) -> None:
    """Store timezone for a location in Redis cache.

    Called from the active weather provider (Open-Meteo or Visual Crossing)
    after each fetch that resolves a timezone, so subsequent
    `_get_location_timezone` reads for non-preapproved locations hit the
    cache instead of falling through to `None`.
    """
    try:
        redis_client = _resolve_redis_client(redis_client)
        if not redis_client:
            return
        normalized = normalize_location_for_cache(location)
        redis_client.setex(f"location_timezone:{normalized}", ttl, timezone_str)
        if DEBUG:
            logger.debug(f"Stored timezone for location '{location}': {timezone_str}")
    except Exception as e:
        if DEBUG:
            logger.debug(f"Could not store timezone for location '{location}': {e}")


def _get_location_timezone_from_preapproved(location: str) -> Optional[str]:
    global _preapproved_locations_cache
    try:
        if _preapproved_locations_cache is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != os.path.dirname(project_root):
                if os.path.exists(os.path.join(project_root, "pyproject.toml")):
                    break
                project_root = os.path.dirname(project_root)
            with open(os.path.join(project_root, "data", "preapproved_locations.json"), "r", encoding="utf-8") as f:
                _preapproved_locations_cache = json.load(f)

        location_lower = location.lower().strip()
        for item in _preapproved_locations_cache:
            if "name" in item and "admin1" in item and "country_name" in item:
                if f"{item['name']}, {item['admin1']}, {item['country_name']}".lower() == location_lower:
                    return item.get("timezone")
            if item.get("name", "").lower() == location_lower:
                return item.get("timezone")
            if item.get("slug", "").lower() == location_lower:
                return item.get("timezone")
        return None
    except Exception as e:
        if DEBUG:
            logger.debug(f"Could not get timezone from preapproved locations for '{location}': {e}")
        return None


def _get_location_timezone(location: str, redis_client: Optional[redis.Redis] = None) -> Optional[str]:
    tz = _get_location_timezone_from_cache(location, redis_client)
    return tz if tz else _get_location_timezone_from_preapproved(location)


def get_local_today(
    location: Optional[str] = None,
    redis_client: Optional[redis.Redis] = None,
) -> dt_date:
    """Return the current date in the location's local timezone, falling back to UTC."""
    if location and ZoneInfo:
        timezone_str = _get_location_timezone(location, redis_client)
        if timezone_str:
            try:
                return datetime.now(ZoneInfo(timezone_str)).date()
            except Exception as e:
                if DEBUG:
                    logger.debug(f"Error using timezone {timezone_str} for '{location}': {e}")
    return datetime.now(timezone.utc).date()


# ---------------------------------------------------------------------------
# Legacy slug fallback + lazy migration (Phase 4)
# ---------------------------------------------------------------------------


def _copy_key_with_ttl(redis_client: redis.Redis, old_key: str, new_key: str) -> bool:
    """Copy old_key to new_key preserving TTL, then delete old_key. Returns success."""
    try:
        ttl = redis_client.ttl(old_key)
        value = redis_client.get(old_key)
        if value is None:
            return False
        if ttl and ttl > 0:
            redis_client.setex(new_key, ttl, value)
        else:
            redis_client.set(new_key, value)
        redis_client.delete(old_key)
        return True
    except Exception as exc:
        if DEBUG:
            logger.debug("Could not migrate cache key %s -> %s: %s", old_key, new_key, exc)
        return False


def migrate_redis_key(redis_client: redis.Redis, old_key: str, new_key: str) -> bool:
    """Rename a Redis key to its canonical form, preserving TTL when possible."""
    if old_key == new_key or not redis_client.exists(old_key):
        return False
    if redis_client.exists(new_key):
        redis_client.delete(old_key)
        return False
    try:
        redis_client.rename(old_key, new_key)
        return True
    except redis.RedisError:
        return _copy_key_with_ttl(redis_client, old_key, new_key)


def migrate_bundle_slug(
    redis_client: redis.Redis,
    scope: str,
    from_slug: str,
    to_slug: str,
    identifier: str,
) -> None:
    """Migrate bundle and ETag keys from a legacy slug to the canonical slug."""
    if from_slug == to_slug:
        return
    old_bundle = bundle_key(scope, from_slug, identifier)
    new_bundle = bundle_key(scope, to_slug, identifier)
    migrate_redis_key(redis_client, old_bundle, new_bundle)
    migrate_redis_key(redis_client, f"{old_bundle}:etag", f"{new_bundle}:etag")


def migrate_year_record_slugs(
    redis_client: redis.Redis,
    scope: str,
    from_slug: str,
    to_slug: str,
    identifier: str,
    years: Sequence[int],
) -> None:
    """Migrate per-year record and ETag keys from a legacy slug to the canonical slug."""
    if from_slug == to_slug:
        return
    for year in years:
        migrate_redis_key(
            redis_client, rec_key(scope, from_slug, identifier, year), rec_key(scope, to_slug, identifier, year)
        )
        migrate_redis_key(
            redis_client,
            rec_etag_key(scope, from_slug, identifier, year),
            rec_etag_key(scope, to_slug, identifier, year),
        )


def get_bundle_with_slug_fallback(
    redis_client: redis.Redis,
    scope: str,
    lookup_slugs: Sequence[str],
    identifier: str,
) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """Try bundle keys in order; lazily migrate a legacy hit to the canonical slug."""
    if not lookup_slugs:
        return None, None, None

    canonical_slug = lookup_slugs[0]
    for slug in lookup_slugs:
        bundle_key_str = bundle_key(scope, slug, identifier)
        bundle_data = redis_client.get(bundle_key_str)
        bundle_etag = redis_client.get(f"{bundle_key_str}:etag")
        if bundle_data and bundle_etag:
            if slug != canonical_slug:
                migrate_bundle_slug(redis_client, scope, slug, canonical_slug, identifier)
            return bundle_data, bundle_etag, slug
    return None, None, None


# ---------------------------------------------------------------------------
# Batch record helpers (MGET-based)
# ---------------------------------------------------------------------------


async def get_records(  # NOSONAR S7503 — async by design (awaited by callers)
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    years: List[int],
    lookup_slugs: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, Dict], List[int], bool]:
    if not years:
        return {}, [], False

    slugs = tuple(lookup_slugs) if lookup_slugs else (slug,)
    if not slugs:
        slugs = (slug,)
    canonical_slug = slugs[0]
    current_year = datetime.now(timezone.utc).year
    year_data: Dict[int, Dict] = {}
    migrated_from: Optional[str] = None

    for try_slug in slugs:
        remaining_years = [year for year in years if year not in year_data]
        if not remaining_years:
            break

        keys = [rec_key(scope, try_slug, identifier, year) for year in remaining_years]
        try:
            values = redis_client.mget(keys)
        except Exception:
            logger.exception("Error in MGET for records")
            return year_data, [y for y in years if y not in year_data and y < current_year], current_year in years

        for year, value in zip(remaining_years, values):
            if not value:
                continue
            try:
                year_data[year] = json.loads(value.decode() if isinstance(value, bytes) else value)
                if try_slug != canonical_slug and migrated_from is None:
                    migrated_from = try_slug
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Error decoding cached data for year {year}: {e}")

    if migrated_from:
        migrate_year_record_slugs(
            redis_client, scope, migrated_from, canonical_slug, identifier, list(year_data.keys())
        )

    missing_past: List[int] = []
    missing_current = False
    for year in years:
        if year in year_data:
            continue
        if year < current_year:
            missing_past.append(year)
        elif year == current_year:
            missing_current = True

    return year_data, missing_past, missing_current


async def get_year_etags(  # NOSONAR S7503 — async by design (awaited by callers)
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    years: List[int],
    lookup_slugs: Optional[Sequence[str]] = None,
) -> Dict[int, str]:
    if not years:
        return {}

    slugs = tuple(lookup_slugs) if lookup_slugs else (slug,)
    if not slugs:
        slugs = (slug,)

    etags: Dict[int, str] = {}
    for try_slug in slugs:
        remaining_years = [year for year in years if year not in etags]
        if not remaining_years:
            break
        keys = [rec_etag_key(scope, try_slug, identifier, year) for year in remaining_years]
        try:
            values = redis_client.mget(keys)
        except Exception:
            logger.exception("Error in MGET for ETags")
            return {year: etags.get(year, "") for year in years}
        for year, value in zip(remaining_years, values):
            if value:
                etags[year] = value.decode() if isinstance(value, bytes) else value

    return {year: etags.get(year, "") for year in years}


async def assemble_and_cache(
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    year_data: Dict[int, Dict],
    year_etags: Optional[Dict[int, str]] = None,
) -> Tuple[Dict, str]:
    from cache.core import TTL_BUNDLE

    sorted_years = sorted(year_data.keys())
    payload = {
        "version": 1,
        "count": len(sorted_years),
        "records": [year_data[y] for y in sorted_years],
    }

    if year_etags is None:
        year_etags = await get_year_etags(redis_client, scope, slug, identifier, sorted_years)

    bundle_etag_computed = compute_bundle_etag(year_etags)
    bkey = bundle_key(scope, slug, identifier)
    bundle_etag_key_str = f"{bkey}:etag"

    try:
        json_data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        redis_client.setex(bkey, TTL_BUNDLE, json_data)
        redis_client.setex(bundle_etag_key_str, TTL_BUNDLE, bundle_etag_computed)
        if DEBUG:
            logger.debug(f"Cached bundle: {bkey} with ETag")
    except Exception as e:
        logger.warning(f"Error caching bundle {bkey}: {e}")

    return payload, bundle_etag_computed
