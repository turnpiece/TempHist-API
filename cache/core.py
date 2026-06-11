"""
Core caching primitives: TTL constants, ETag helpers, EnhancedCache, CacheInvalidator.

This is the lowest-level cache module.  Higher-level helpers (warming, stats,
singleton accessors) live in sibling modules.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis
from fastapi import Request, Response

from config import DEBUG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL validation
# ---------------------------------------------------------------------------


def validate_ttl(ttl: int, name: str, default: int) -> int:
    """Validate cache TTL value with min/max bounds."""
    MIN_TTL = 60
    MAX_TTL = 31536000  # 365 days

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


# ---------------------------------------------------------------------------
# TTL constants
# ---------------------------------------------------------------------------

CACHE_TTL_DEFAULT = validate_ttl(int(os.getenv("CACHE_TTL_DEFAULT", "86400")), "CACHE_TTL_DEFAULT", 86400)
CACHE_TTL_SHORT = validate_ttl(int(os.getenv("CACHE_TTL_SHORT", "3600")), "CACHE_TTL_SHORT", 3600)
CACHE_TTL_LONG = validate_ttl(int(os.getenv("CACHE_TTL_LONG", "604800")), "CACHE_TTL_LONG", 604800)

TTL_STABLE = validate_ttl(int(os.getenv("TTL_STABLE", "7776000")), "TTL_STABLE", 7776000)
TTL_HISTORICAL = validate_ttl(int(os.getenv("TTL_HISTORICAL", "31536000")), "TTL_HISTORICAL", 31536000)
TTL_CURRENT_DAILY = validate_ttl(int(os.getenv("TTL_CURRENT_DAILY", "7200")), "TTL_CURRENT_DAILY", 7200)
TTL_CURRENT_WEEKLY = validate_ttl(int(os.getenv("TTL_CURRENT_WEEKLY", "14400")), "TTL_CURRENT_WEEKLY", 14400)
TTL_CURRENT_MONTHLY = validate_ttl(int(os.getenv("TTL_CURRENT_MONTHLY", "43200")), "TTL_CURRENT_MONTHLY", 43200)
TTL_CURRENT_YEARLY = validate_ttl(int(os.getenv("TTL_CURRENT_YEARLY", "86400")), "TTL_CURRENT_YEARLY", 86400)
TTL_BUNDLE = validate_ttl(int(os.getenv("TTL_BUNDLE", "900")), "TTL_BUNDLE", 900)

# ---------------------------------------------------------------------------
# Cache header constants
# ---------------------------------------------------------------------------

COORD_PRECISION = 4  # 4 decimal places (~11m precision)

CACHE_CONTROL_PUBLIC = "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800"
CACHE_CONTROL_PRIVATE = "private, max-age=300, stale-while-revalidate=3600"
CACHE_CONTROL_DAILY_TODAY = "public, max-age=1800, s-maxage=3600, stale-while-revalidate=7200"

# ---------------------------------------------------------------------------
# Cache invalidation config
# ---------------------------------------------------------------------------

CACHE_INVALIDATION_ENABLED = os.getenv("CACHE_INVALIDATION_ENABLED", "true").lower() == "true"
CACHE_INVALIDATION_DRY_RUN = os.getenv("CACHE_INVALIDATION_DRY_RUN", "false").lower() == "true"
CACHE_INVALIDATION_BATCH_SIZE = int(os.getenv("CACHE_INVALIDATION_BATCH_SIZE", "100"))

# ---------------------------------------------------------------------------
# Year-based TTL helper
# ---------------------------------------------------------------------------


def get_ttl_for_year(year: int) -> int:
    """Return appropriate cache TTL based on how old the data is."""
    current_year = datetime.now().year
    age = current_year - year

    if age >= 7:
        return TTL_HISTORICAL
    elif age > 0:
        return TTL_STABLE
    else:
        return TTL_STABLE


# ---------------------------------------------------------------------------
# ETag helpers
# ---------------------------------------------------------------------------


class ETagGenerator:
    """Generate and validate ETags for responses."""

    @staticmethod
    def generate_etag(data: Any) -> str:
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return f'"{hashlib.sha256(json_str.encode()).hexdigest()[:32]}"'

    @staticmethod
    def parse_etag(etag: str) -> Optional[str]:
        if not etag:
            return None
        return etag.strip("\"'")

    @staticmethod
    def matches_etag(response_etag: str, request_etag: str) -> bool:
        if not response_etag or not request_etag:
            return False
        return response_etag.strip("\"'") == request_etag.strip("\"'")


class CacheHeaders:
    """Manage cache headers for responses."""

    @staticmethod
    def set_cache_headers(
        response: Response,
        etag: str,
        last_modified: datetime,
        cache_control: str = CACHE_CONTROL_PUBLIC,
        vary: str = "Accept-Encoding",
    ):
        response.headers["Cache-Control"] = cache_control
        response.headers["ETag"] = etag
        response.headers["Last-Modified"] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
        if vary:
            response.headers["Vary"] = vary

    @staticmethod
    def check_conditional_headers(request: Request, etag: str, last_modified: datetime) -> bool:
        if_none_match = request.headers.get("If-None-Match")
        if if_none_match and ETagGenerator.matches_etag(etag, if_none_match):
            return True

        if_modified_since = request.headers.get("If-Modified-Since")
        if if_modified_since:
            try:
                request_time = datetime.strptime(if_modified_since, "%a, %d %b %Y %H:%M:%S GMT")
                if last_modified <= request_time + timedelta(seconds=1):
                    return True
            except ValueError:
                pass

        return False


# ---------------------------------------------------------------------------
# EnhancedCache
# ---------------------------------------------------------------------------


class EnhancedCache:
    """Enhanced Redis cache with single-flight protection and metrics."""

    def __init__(self, redis_client: redis.Redis):
        from jobs.manager import SingleFlightLock  # avoid top-level circular risk

        self.redis = redis_client
        self.lock_manager = SingleFlightLock(redis_client)
        self.cache_prefix = "cache:"
        self.metrics_prefix = "metrics:"
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.compute_times = []

    def _get_cache_key(self, key: str) -> str:
        return f"{self.cache_prefix}{key}"

    def _get_etag_key(self, key: str) -> str:
        return f"{self.cache_prefix}{key}:etag"

    def _get_metrics_key(self, key: str) -> str:
        return f"{self.metrics_prefix}{key}"

    async def get(self, key: str) -> Optional[Tuple[Any, str, datetime]]:
        cache_key = self._get_cache_key(key)
        etag_key = self._get_etag_key(key)

        try:
            cached_data = self.redis.get(cache_key)
            cached_etag = self.redis.get(etag_key)
            cached_timestamp = self.redis.hget(cache_key, "timestamp")

            if cached_data and cached_etag and cached_timestamp:
                data = json.loads(cached_data)
                etag = cached_etag.decode() if isinstance(cached_etag, bytes) else cached_etag
                last_modified = datetime.fromisoformat(
                    cached_timestamp.decode() if isinstance(cached_timestamp, bytes) else cached_timestamp
                )
                self.hits += 1
                return data, etag, last_modified
            else:
                self.misses += 1
                return None

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            self.errors += 1
            logger.error(f"Redis error getting cache key {key}: {e}")
            return None
        except (json.JSONDecodeError, ValueError) as e:
            self.errors += 1
            logger.error(f"Decode error for cache key {key}: {e}")
            return None

    async def get_updated_timestamp(self, key: str) -> Optional[datetime]:
        cache_key = self._get_cache_key(key)
        try:
            cached_timestamp = self.redis.hget(cache_key, "timestamp")
            if cached_timestamp:
                return datetime.fromisoformat(
                    cached_timestamp.decode() if isinstance(cached_timestamp, bytes) else cached_timestamp
                )
            return None
        except (redis.RedisError, redis.ConnectionError) as e:
            logger.error(f"Redis error getting timestamp for {key}: {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.error(f"Date parsing error for timestamp {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        data: Any,
        ttl: int = CACHE_TTL_DEFAULT,
        etag: Optional[str] = None,
        last_modified: Optional[datetime] = None,
    ) -> str:
        cache_key = self._get_cache_key(key)
        etag_key = self._get_etag_key(key)

        if etag is None:
            etag = ETagGenerator.generate_etag(data)
        if last_modified is None:
            last_modified = datetime.now(timezone.utc)

        try:
            json_data = json.dumps(data, sort_keys=True, separators=(",", ":"))
            self.redis.setex(cache_key, ttl, json_data)
            self.redis.setex(etag_key, ttl, etag)
            self.redis.hset(
                cache_key,
                mapping={
                    "timestamp": last_modified.isoformat(),
                    "ttl": ttl,
                    "size": len(json_data),
                },
            )
            self.redis.expire(cache_key, ttl)
            return etag

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            self.errors += 1
            logger.error(f"Redis error setting cache key {key}: {e}")
            raise
        except (json.JSONEncodeError, TypeError) as e:
            self.errors += 1
            logger.error(f"JSON encoding error for cache key {key}: {e}")
            raise

    async def mset(self, items: List[Tuple[str, Any, int, Optional[str]]]) -> List[str]:
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

                json_data = json.dumps(data, sort_keys=True, separators=(",", ":"))
                pipeline.setex(cache_key, ttl, json_data)
                pipeline.setex(etag_key, ttl, etag)
                pipeline.hset(
                    cache_key,
                    mapping={
                        "timestamp": last_modified.isoformat(),
                        "ttl": ttl,
                        "size": len(json_data),
                    },
                )
                pipeline.expire(cache_key, ttl)

            pipeline.execute()
            return etags

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            self.errors += len(items)
            logger.error(f"Redis error in batch set: {e}")
            raise
        except (json.JSONEncodeError, TypeError) as e:
            self.errors += len(items)
            logger.error(f"JSON encoding error in batch set: {e}")
            raise

    async def mget(self, keys: List[str]) -> Dict[str, Optional[Tuple[Any, str, datetime]]]:
        if not keys:
            return {}

        try:
            pipeline = self.redis.pipeline()
            for key in keys:
                pipeline.get(self._get_cache_key(key))
                pipeline.get(self._get_etag_key(key))
                pipeline.hget(self._get_cache_key(key), "timestamp")

            results = pipeline.execute()
            parsed: Dict[str, Optional[Tuple[Any, str, datetime]]] = {}

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
                        parsed[key] = (data, etag, last_modified)
                        self.hits += 1
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Parse error for key {key}: {e}")
                        parsed[key] = None
                        self.errors += 1
                else:
                    parsed[key] = None
                    self.misses += 1

            return parsed

        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            self.errors += len(keys)
            logger.error(f"Redis error in batch get: {e}")
            return {key: None for key in keys}

    async def get_or_compute(
        self,
        key: str,
        compute_func,
        ttl: int = CACHE_TTL_DEFAULT,
        *args,
        **kwargs,
    ) -> Tuple[Any, str, datetime]:
        cached_result = await self.get(key)
        if cached_result is not None:
            return cached_result

        lock_acquired = await self.lock_manager.acquire(key)
        try:
            if lock_acquired:
                start_time = time.time()
                data = (
                    await compute_func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(compute_func)
                    else compute_func(*args, **kwargs)
                )
                self.compute_times.append(time.time() - start_time)
                etag = await self.set(key, data, ttl)
                last_modified = datetime.now(timezone.utc)
                return data, etag, last_modified
            else:
                await asyncio.sleep(0.1)
                return await self.get_or_compute(key, compute_func, ttl, *args, **kwargs)
        finally:
            if lock_acquired:
                self.lock_manager.release(key)

    def get_metrics(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        avg_compute = sum(self.compute_times) / len(self.compute_times) if self.compute_times else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total,
            "avg_compute_time": round(avg_compute, 3),
            "compute_times_count": len(self.compute_times),
        }

    def reset_metrics(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.compute_times = []


# ---------------------------------------------------------------------------
# CacheInvalidator
# ---------------------------------------------------------------------------


class CacheInvalidator:
    """Manage cache invalidation with various strategies and patterns."""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.invalidation_prefix = "invalidation_"
        self.batch_size = CACHE_INVALIDATION_BATCH_SIZE

    def invalidate_by_key(self, cache_key: str, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        try:
            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                exists = self.redis_client.exists(cache_key)
                return {
                    "status": "dry_run",
                    "cache_key": cache_key,
                    "exists": bool(exists),
                    "action": "would_delete" if exists else "no_action",
                }
            else:
                deleted = self.redis_client.delete(cache_key)
                return {
                    "status": "success",
                    "cache_key": cache_key,
                    "deleted": bool(deleted),
                    "action": "deleted" if deleted else "not_found",
                }
        except Exception as e:
            return {"status": "error", "cache_key": cache_key, "error": str(e)}

    def invalidate_by_pattern(self, pattern: str, dry_run: bool = False, max_keys: int = 10000) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        if max_keys > 100000:
            logger.warning(f"Cache invalidation max_keys ({max_keys}) is very high, capping at 100000")
            max_keys = 100000

        try:
            matching_keys = []
            cursor = 0
            try:
                while cursor != 0 or len(matching_keys) == 0:
                    cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
                    matching_keys.extend(keys)
                    if len(matching_keys) > max_keys:
                        logger.warning(f"Pattern matches >{max_keys} keys, stopping scan")
                        return {
                            "status": "error",
                            "pattern": pattern,
                            "error": f"Pattern matches too many keys (>{max_keys}). Use a more specific pattern.",
                        }
                    if cursor == 0:
                        break
            except Exception as e:
                if "permissions" in str(e).lower() or "scan" in str(e).lower():
                    return {
                        "status": "error",
                        "pattern": pattern,
                        "error": "Redis SCAN command not permitted on this instance",
                        "message": "Cache invalidation by pattern is not available on managed Redis services",
                    }
                raise e

            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                return {
                    "status": "dry_run",
                    "pattern": pattern,
                    "matching_keys": [k.decode() if isinstance(k, bytes) else k for k in matching_keys],
                    "count": len(matching_keys),
                    "action": "would_delete",
                }
            else:
                deleted_count = 0
                for i in range(0, len(matching_keys), self.batch_size):
                    batch = matching_keys[i : i + self.batch_size]
                    if batch:
                        deleted_count += self.redis_client.delete(*batch)
                return {
                    "status": "success",
                    "pattern": pattern,
                    "matching_keys": [k.decode() if isinstance(k, bytes) else k for k in matching_keys],
                    "deleted_count": deleted_count,
                    "total_found": len(matching_keys),
                }
        except Exception as e:
            return {"status": "error", "pattern": pattern, "error": str(e)}

    def invalidate_by_endpoint(self, endpoint: str, location: str = None, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        pattern = f"{endpoint}_{location.lower()}_*" if location else f"{endpoint}_*"
        return self.invalidate_by_pattern(pattern, dry_run)

    def invalidate_by_location(self, location: str, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        return self.invalidate_by_pattern(f"*_{location.lower()}_*", dry_run)

    def invalidate_by_date(self, date: str, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        date_patterns = [
            f"*_{date}_*",
            f"*_{date.replace('-', '_')}_*",
            f"*_{date.replace('-', '_')}",
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
            "pattern_results": results,
        }

    def invalidate_forecast_data(self, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        return self.invalidate_by_pattern("forecast_*", dry_run)

    def invalidate_today_data(self, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        today_pattern = today.strftime("%m_%d")
        results = [self.invalidate_by_date(fmt, dry_run) for fmt in [today_str, today_pattern]]
        return {
            "status": "success" if not dry_run and not CACHE_INVALIDATION_DRY_RUN else "dry_run",
            "date": today_str,
            "results": results,
        }

    def invalidate_expired_keys(self, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        try:
            all_keys = []
            cursor = 0
            max_keys = 100000
            try:
                while cursor != 0 or len(all_keys) == 0:
                    cursor, keys = self.redis_client.scan(cursor, match="*", count=100)
                    all_keys.extend(keys)
                    if len(all_keys) > max_keys:
                        logger.warning(f"Found >{max_keys} keys, stopping scan")
                        break
                    if cursor == 0:
                        break
            except Exception as e:
                if "permissions" in str(e).lower() or "scan" in str(e).lower():
                    return {
                        "status": "error",
                        "error": "Redis SCAN command not permitted on this instance",
                        "message": "Expired key invalidation is not available on managed Redis services",
                    }
                raise e

            expired_keys = [k for k in all_keys if self.redis_client.ttl(k) == 0]
            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                return {
                    "status": "dry_run",
                    "expired_keys": [k.decode() if isinstance(k, bytes) else k for k in expired_keys],
                    "count": len(expired_keys),
                    "action": "would_delete",
                }
            else:
                deleted_count = self.redis_client.delete(*expired_keys) if expired_keys else 0
                return {
                    "status": "success",
                    "expired_keys": [k.decode() if isinstance(k, bytes) else k for k in expired_keys],
                    "deleted_count": deleted_count,
                    "total_found": len(expired_keys),
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_cache_info(self) -> Dict:
        try:
            info = self.redis_client.info()
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
                "v1_records_summary": "records:*:summary",
            }
            max_keys_per_pattern = 10000
            scan_count_hint = 500
            pattern_counts = {}
            for name, pattern in patterns.items():
                try:
                    matching_keys = []
                    cursor = 0
                    while cursor != 0 or len(matching_keys) == 0:
                        cursor, keys = self.redis_client.scan(
                            cursor, match=pattern, count=scan_count_hint
                        )
                        matching_keys.extend(keys)
                        if len(matching_keys) > max_keys_per_pattern or cursor == 0:
                            break
                    pattern_counts[name] = len(matching_keys)
                except Exception as e:
                    pattern_counts[name] = (
                        "N/A (permissions required)"
                        if "permissions" in str(e).lower() or "scan" in str(e).lower()
                        else f"N/A (error: {str(e)[:50]})"
                    )
            return {
                "redis_info": {
                    "used_memory": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                },
                "key_counts": pattern_counts,
                "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0,
                "scan_limits": {
                    "max_keys_per_pattern": max_keys_per_pattern,
                    "scan_count_hint": scan_count_hint,
                },
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def clear_all_cache(self, dry_run: bool = False) -> Dict:
        if not CACHE_INVALIDATION_ENABLED:
            return {"status": "disabled", "message": "Cache invalidation is not enabled"}
        try:
            if dry_run or CACHE_INVALIDATION_DRY_RUN:
                try:
                    all_keys = []
                    cursor = 0
                    while cursor != 0 or len(all_keys) == 0:
                        cursor, keys = self.redis_client.scan(cursor, match="*", count=100)
                        all_keys.extend(keys)
                        if len(all_keys) > 100000 or cursor == 0:
                            break
                    return {"status": "dry_run", "total_keys": len(all_keys), "action": "would_delete_all"}
                except Exception as e:
                    if "permissions" in str(e).lower() or "scan" in str(e).lower():
                        return {
                            "status": "error",
                            "error": "Redis SCAN command not permitted on this instance",
                            "message": "Cannot count keys on managed Redis service for dry run",
                        }
                    raise e
            else:
                self.redis_client.flushdb()
                return {
                    "status": "success",
                    "action": "cleared_all_cache",
                    "message": "All cache data has been cleared",
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Simple cache utility functions
# ---------------------------------------------------------------------------


def get_cache_value(
    cache_key,
    redis_client: redis.Redis,
    endpoint: str = None,
    location: str = None,
    cache_stats=None,
):
    """Get a value from the cache with optional statistics tracking."""
    if DEBUG:
        logger.debug(f"🔍 CACHE GET: {cache_key}")
    try:
        result = redis_client.get(cache_key)
        hit = result is not None
        if cache_stats:
            cache_stats.track_cache_request(cache_key, hit, endpoint, location)
        if DEBUG:
            logger.debug(f"{'✅ CACHE HIT' if hit else '❌ CACHE MISS'}: {cache_key}")
        return result
    except Exception as e:
        if cache_stats:
            cache_stats.track_cache_request(cache_key, False, endpoint, location, error=True)
        if DEBUG:
            logger.error(f"❌ CACHE ERROR: {cache_key} - {e}")
        return None


def mget_cache_values(
    cache_keys: List[str],
    redis_client: redis.Redis,
    endpoint: str = None,
    location: str = None,
    cache_stats=None,
) -> List[Optional[Any]]:
    """Get multiple cache values in a single round trip.

    Returns a list of raw redis values aligned with ``cache_keys`` (None for
    misses). Records one cache-stats event per key, matching the per-key
    hit/miss tracking that ``get_cache_value`` does. On any Redis error all
    entries are reported as misses with ``error=True``.
    """
    if not cache_keys:
        return []
    try:
        results = redis_client.mget(cache_keys)
        if cache_stats:
            for cache_key, result in zip(cache_keys, results):
                cache_stats.track_cache_request(cache_key, result is not None, endpoint, location)
        if DEBUG:
            for cache_key, result in zip(cache_keys, results):
                logger.debug(f"{'✅ CACHE HIT' if result is not None else '❌ CACHE MISS'}: {cache_key}")
        return results
    except Exception as e:
        if cache_stats:
            for cache_key in cache_keys:
                cache_stats.track_cache_request(cache_key, False, endpoint, location, error=True)
        if DEBUG:
            logger.error(f"❌ CACHE MGET ERROR for {len(cache_keys)} keys: {e}")
        return [None] * len(cache_keys)


def set_cache_value(cache_key, lifetime, value, redis_client: redis.Redis):
    """Set a value in the cache with specified lifetime."""
    if DEBUG:
        logger.debug(f"💾 CACHE SET: {cache_key} | TTL: {lifetime}")
    redis_client.setex(cache_key, lifetime, value)


async def get_cache_updated_timestamp(cache_key: str, redis_client: redis.Redis) -> Optional[datetime]:
    """Get the last updated timestamp for a cache key."""
    try:
        if not redis_client:
            return None
        if not redis_client.exists(cache_key):
            return None
        try:
            key_type = redis_client.type(cache_key)
            if key_type == "hash":
                timestamp_data = redis_client.hget(cache_key, "timestamp")
                if timestamp_data:
                    return datetime.fromisoformat(
                        timestamp_data.decode() if isinstance(timestamp_data, bytes) else timestamp_data
                    )
        except Exception as _ts_err:
            logger.debug("Could not read timestamp for cache key %s: %s", cache_key, _ts_err)
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                data = json.loads(cached_data)
                if "updated" in data and data["updated"]:
                    return datetime.fromisoformat(data["updated"])
            except (json.JSONDecodeError, ValueError):
                pass
        ttl = redis_client.ttl(cache_key)
        if ttl > 0:
            return datetime.now(timezone.utc) - timedelta(seconds=ttl)
        return None
    except Exception as e:
        logger.warning(f"Error getting cache timestamp for {cache_key}: {e}")
        return None
