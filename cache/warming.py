"""
Cache warming and statistics.

CacheWarmer proactively populates Redis for popular locations.
CacheStats tracks hit/miss/error metrics.
scheduled_cache_warming() is the background loop started at app startup.
"""

import asyncio
import json
import logging
import os
import time
from datetime import date as dt_date
from datetime import datetime, timedelta
from typing import Dict, List

import aiohttp
import redis

from cache.keys import normalize_location_for_cache
from config import (
    API_ACCESS_TOKEN,
    BASE_URL,
    DEBUG,
    POPULARITY_WINDOW_DAYS,
    USAGE_TRACKING_ENABLED,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache warming configuration
# ---------------------------------------------------------------------------

CACHE_WARMING_ENABLED = os.getenv("CACHE_WARMING_ENABLED", "true").lower() == "true"
CACHE_WARMING_INTERVAL_HOURS = int(os.getenv("CACHE_WARMING_INTERVAL_HOURS", "4"))
CACHE_WARMING_DAYS_BACK = int(os.getenv("CACHE_WARMING_DAYS_BACK", "7"))
CACHE_WARMING_CONCURRENT_REQUESTS = int(os.getenv("CACHE_WARMING_CONCURRENT_REQUESTS", "3"))
CACHE_WARMING_MAX_LOCATIONS = int(os.getenv("CACHE_WARMING_MAX_LOCATIONS", "200"))

# ---------------------------------------------------------------------------
# Cache statistics configuration
# ---------------------------------------------------------------------------

CACHE_STATS_ENABLED = os.getenv("CACHE_STATS_ENABLED", "true").lower() == "true"
CACHE_STATS_RETENTION_HOURS = int(os.getenv("CACHE_STATS_RETENTION_HOURS", "24"))
CACHE_HEALTH_THRESHOLD = float(os.getenv("CACHE_HEALTH_THRESHOLD", "0.7"))


# ---------------------------------------------------------------------------
# CacheWarmer
# ---------------------------------------------------------------------------


class CacheWarmer:
    """Proactive cache warming for popular locations and recent dates."""

    def __init__(self, redis_client: redis.Redis, usage_tracker=None):
        self.redis_client = redis_client
        self.usage_tracker = usage_tracker
        self.warming_in_progress = False
        self.last_warming_time = None
        self.warming_stats = {
            "total_warmed": 0,
            "successful_warmed": 0,
            "failed_warmed": 0,
            "last_warming_duration": 0,
        }

    def get_locations_to_warm(self) -> List[str]:
        locations: List[str] = []
        seen_normalized: set = set()

        def add_location(candidate: str) -> None:
            if not candidate:
                return
            normalized = normalize_location_for_cache(candidate)
            if normalized in seen_normalized:
                return
            seen_normalized.add(normalized)
            locations.append(candidate)

        # Primary source: selections-based popular list (includes non-preapproved
        # locations; ranked most-selected first).  Only used when there is enough
        # signal to trust the ranking.
        if self.usage_tracker and USAGE_TRACKING_ENABLED:
            strings = self.usage_tracker.get_popular_display_strings(
                limit=CACHE_WARMING_MAX_LOCATIONS * 2,
                days=POPULARITY_WINDOW_DAYS,
            )
            for loc in strings:
                add_location(loc)
            if DEBUG and strings:
                logger.info(f"🔥 CACHE WARMING: {len(locations)} locations from selections signal")

        # Pad with preapproved list when signal is absent or insufficient.
        if len(locations) < CACHE_WARMING_MAX_LOCATIONS:
            preapproved = self.get_preapproved_locations()
            for loc in preapproved:
                add_location(loc)
            if DEBUG:
                logger.info(f"🔥 CACHE WARMING: padded to {len(locations)} with preapproved list")

        final_locations = locations[:CACHE_WARMING_MAX_LOCATIONS]
        if DEBUG:
            logger.info(f"🔥 CACHE WARMING: warming {len(final_locations)} locations total")
        return final_locations

    def get_preapproved_locations(self) -> List[str]:
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != os.path.dirname(project_root):
                if os.path.exists(os.path.join(project_root, "pyproject.toml")):
                    break
                project_root = os.path.dirname(project_root)

            data_file = os.path.join(project_root, "data", "preapproved_locations.json")
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            locations = []
            seen: set = set()
            for item in data:
                if "name" in item and "admin1" in item and "country_name" in item:
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
        except json.JSONDecodeError as e:
            if DEBUG:
                logger.warning(f"⚠️  Invalid JSON in preapproved locations: {e}")
        except (IOError, PermissionError) as e:
            if DEBUG:
                logger.warning(f"⚠️  Cannot read preapproved locations file: {e}")
        return []

    def get_dates_to_warm(self) -> List[str]:
        dates = []
        today = datetime.now().date()
        for days_back in range(CACHE_WARMING_DAYS_BACK):
            d = today - timedelta(days=days_back)
            dates.append(d.strftime("%Y-%m-%d"))
        current_year = today.year
        current_month = today.month
        for day in range(1, 32):
            try:
                d = dt_date(current_year, current_month, day)
                if d <= today:
                    dates.append(d.strftime("%Y-%m-%d"))
            except ValueError:
                continue
        return dates

    def get_month_days_to_warm(self) -> List[str]:
        today = datetime.now()
        month_days = [f"{today.month:02d}-{today.day:02d}"]
        for days_back in range(1, min(CACHE_WARMING_DAYS_BACK, 30)):
            d = today - timedelta(days=days_back)
            month_days.append(f"{d.month:02d}-{d.day:02d}")
        return month_days

    async def warm_location_data(self, location: str) -> Dict:
        if DEBUG:
            logger.info(f"🔥 WARMING LOCATION: {location}")

        results: Dict = {"location": location, "warmed_endpoints": [], "errors": []}

        try:
            try:
                auth_token = API_ACCESS_TOKEN
                if not auth_token:
                    results["errors"].append("forecast: No authentication token available")
                else:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{BASE_URL}/forecast/{location}",
                            headers={"Authorization": f"Bearer {auth_token}"},
                        ) as resp:
                            if resp.status == 200:
                                results["warmed_endpoints"].append("forecast")
                            else:
                                results["errors"].append(f"forecast: {resp.status}")
            except (aiohttp.ClientError, aiohttp.ClientConnectorError) as e:
                results["errors"].append(f"forecast: HTTP error - {e}")
            except asyncio.TimeoutError:
                results["errors"].append("forecast: Request timeout")

            month_days = self.get_month_days_to_warm()
            for month_day in month_days:
                for period in ["daily", "weekly", "monthly"]:
                    for sub in [None, "average", "trend", "summary"]:
                        url_path = f"/v1/records/{period}/{location}/{month_day}"
                        if sub:
                            url_path += f"/{sub}"
                        label = url_path.lstrip("/")
                        try:
                            auth_token = API_ACCESS_TOKEN
                            if not auth_token:
                                results["errors"].append(f"{label}: No authentication token available")
                            else:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(
                                        f"{BASE_URL}{url_path}",
                                        headers={"Authorization": f"Bearer {auth_token}"},
                                    ) as resp:
                                        if resp.status == 200:
                                            results["warmed_endpoints"].append(label)
                                        else:
                                            results["errors"].append(f"{label}: {resp.status}")
                        except (aiohttp.ClientError, aiohttp.ClientConnectorError) as e:
                            results["errors"].append(f"{label}: HTTP error - {e}")
                        except asyncio.TimeoutError:
                            results["errors"].append(f"{label}: Request timeout")

            for date in self.get_dates_to_warm()[:5]:
                try:
                    auth_token = API_ACCESS_TOKEN
                    if not auth_token:
                        results["errors"].append(f"weather/{date}: No authentication token available")
                    else:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{BASE_URL}/weather/{location}/{date}",
                                headers={"Authorization": f"Bearer {auth_token}"},
                            ) as resp:
                                if resp.status == 200:
                                    results["warmed_endpoints"].append(f"weather/{date}")
                                else:
                                    results["errors"].append(f"weather/{date}: {resp.status}")
                except (aiohttp.ClientError, aiohttp.ClientConnectorError) as e:
                    results["errors"].append(f"weather/{date}: HTTP error - {e}")
                except asyncio.TimeoutError:
                    results["errors"].append(f"weather/{date}: Request timeout")

        except (aiohttp.ClientError, aiohttp.ClientConnectorError) as e:
            results["errors"].append(f"location_warming: HTTP error - {e}")
        except Exception as e:
            results["errors"].append(f"location_warming: Unexpected error - {e}")
            logger.error(f"Unexpected error warming location {location}: {e}", exc_info=True)

        return results

    async def warm_all_locations(self) -> Dict:
        if self.warming_in_progress:
            return {"status": "already_in_progress", "message": "Cache warming already in progress"}
        if not CACHE_WARMING_ENABLED:
            return {"status": "disabled", "message": "Cache warming is disabled"}
        try:
            self.redis_client.ping()
        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"⚠️ Cache warming skipped - Redis not available: {e}")
            return {"status": "skipped", "message": "Redis not available", "error": str(e)}

        self.warming_in_progress = True
        try:
            self.redis_client.setex("cache_warming:in_progress", 3600, "1")
        except Exception as _e:
            logger.debug("Could not set cache_warming:in_progress flag: %s", _e)
        start_time = time.time()

        try:
            locations = self.get_locations_to_warm()
            if DEBUG:
                logger.info(f"🔥 STARTING CACHE WARMING: {len(locations)} locations")
                logger.info(f"🏙️  LOCATIONS: {', '.join(locations)}")

            try:
                auth_token = API_ACCESS_TOKEN
                if auth_token:
                    timeout = aiohttp.ClientTimeout(total=10)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(
                            f"{BASE_URL}/v1/locations/preapproved",
                            headers={"Authorization": f"Bearer {auth_token}"},
                        ) as resp:
                            if DEBUG:
                                msg = (
                                    "✅ PREAPPROVED ENDPOINT: Warmed successfully"
                                    if resp.status == 200
                                    else f"⚠️  PREAPPROVED ENDPOINT: HTTP {resp.status}"
                                )
                                logger.info(msg)
                else:
                    logger.warning("⚠️  PREAPPROVED ENDPOINT: No authentication token available")
            except aiohttp.ClientConnectorError as e:
                logger.warning(f"⚠️  PREAPPROVED ENDPOINT: Cannot connect to {BASE_URL} - {e}")
                logger.info("💡  TIP: Set BASE_URL environment variable to your API server URL")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️  PREAPPROVED ENDPOINT: Request timeout to {BASE_URL}")
            except Exception as e:
                logger.warning(f"⚠️  PREAPPROVED ENDPOINT: {e}")

            semaphore = asyncio.Semaphore(CACHE_WARMING_CONCURRENT_REQUESTS)

            async def warm_with_semaphore(loc):
                async with semaphore:
                    return await self.warm_location_data(loc)

            results = await asyncio.gather(*[warm_with_semaphore(loc) for loc in locations], return_exceptions=True)

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

            duration = time.time() - start_time
            self.warming_stats.update(
                {
                    "total_warmed": len(locations),
                    "successful_warmed": successful_locations,
                    "failed_warmed": len(locations) - successful_locations,
                    "last_warming_duration": duration,
                }
            )
            self.last_warming_time = datetime.now()
            try:
                self.redis_client.set(
                    "cache_warming:stats",
                    json.dumps(
                        {
                            "last_warming_time": self.last_warming_time.isoformat(),
                            **self.warming_stats,
                        }
                    ),
                )
            except Exception as _e:
                logger.debug("Could not persist warming stats to Redis: %s", _e)

            if DEBUG:
                logger.info(
                    f"✅ CACHE WARMING COMPLETED: {successful_locations}/{len(locations)} locations | "
                    f"{total_endpoints} endpoints | {duration:.1f}s"
                )

            return {
                "status": "completed",
                "locations_processed": len(locations),
                "successful_locations": successful_locations,
                "total_endpoints_warmed": total_endpoints,
                "total_errors": total_errors,
                "duration_seconds": duration,
                "results": results,
            }

        except Exception as e:
            if DEBUG:
                logger.error(f"❌ CACHE WARMING FAILED: {e}")
            return {"status": "error", "message": str(e), "duration_seconds": time.time() - start_time}
        finally:
            self.warming_in_progress = False
            try:
                self.redis_client.delete("cache_warming:in_progress")
            except Exception as _e:
                logger.debug("Could not clear cache_warming:in_progress flag: %s", _e)

    def get_warming_stats(self) -> Dict:
        stats = dict(self.warming_stats)
        last_warming_time = self.last_warming_time.isoformat() if self.last_warming_time else None
        in_progress = self.warming_in_progress

        try:
            raw = self.redis_client.get("cache_warming:stats")
            if raw:
                persisted = json.loads(raw)
                last_warming_time = persisted.get("last_warming_time")
                stats = {
                    "total_warmed": persisted.get("total_warmed", 0),
                    "successful_warmed": persisted.get("successful_warmed", 0),
                    "failed_warmed": persisted.get("failed_warmed", 0),
                    "last_warming_duration": persisted.get("last_warming_duration", 0),
                }
            in_progress = bool(self.redis_client.exists("cache_warming:in_progress"))
        except Exception as _e:
            logger.debug("Could not read warming stats from Redis: %s", _e)

        return {
            "enabled": CACHE_WARMING_ENABLED,
            "in_progress": in_progress,
            "last_warming_time": last_warming_time,
            "stats": stats,
            "configuration": {
                "interval_hours": CACHE_WARMING_INTERVAL_HOURS,
                "days_back": CACHE_WARMING_DAYS_BACK,
                "concurrent_requests": CACHE_WARMING_CONCURRENT_REQUESTS,
                "max_locations": CACHE_WARMING_MAX_LOCATIONS,
            },
        }


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


class CacheStats:
    """Track and analyze cache performance statistics."""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.stats_prefix = "cache_stats_"
        self.retention_seconds = CACHE_STATS_RETENTION_HOURS * 3600
        self.stats: Dict = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_errors": 0,
            "endpoint_stats": {},
            "location_stats": {},
            "hourly_stats": {},
            "last_reset": time.time(),
        }

    def track_cache_request(
        self,
        cache_key: str,
        hit: bool,
        endpoint: str = None,
        location: str = None,
        error: bool = False,
    ):
        if not CACHE_STATS_ENABLED:
            return

        current_time = time.time()
        hour_key = int(current_time // 3600)

        self.stats["total_requests"] += 1
        if error:
            self.stats["cache_errors"] += 1
        elif hit:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1

        if endpoint:
            if endpoint not in self.stats["endpoint_stats"]:
                self.stats["endpoint_stats"][endpoint] = {"hits": 0, "misses": 0, "errors": 0, "total": 0}
            bucket = self.stats["endpoint_stats"][endpoint]
            bucket["total"] += 1
            if error:
                bucket["errors"] += 1
            elif hit:
                bucket["hits"] += 1
            else:
                bucket["misses"] += 1

        if location:
            location = normalize_location_for_cache(location)
            if location not in self.stats["location_stats"]:
                self.stats["location_stats"][location] = {"hits": 0, "misses": 0, "errors": 0, "total": 0}
            bucket = self.stats["location_stats"][location]
            bucket["total"] += 1
            if error:
                bucket["errors"] += 1
            elif hit:
                bucket["hits"] += 1
            else:
                bucket["misses"] += 1

        if hour_key not in self.stats["hourly_stats"]:
            self.stats["hourly_stats"][hour_key] = {"hits": 0, "misses": 0, "errors": 0, "total": 0}
        bucket = self.stats["hourly_stats"][hour_key]
        bucket["total"] += 1
        if error:
            bucket["errors"] += 1
        elif hit:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1

        self._store_stats_in_redis()

    def _store_stats_in_redis(self):
        if not CACHE_STATS_ENABLED:
            return
        try:
            self.redis_client.setex(
                f"{self.stats_prefix}current",
                self.retention_seconds,
                json.dumps(self.stats),
            )
        except Exception as e:
            if DEBUG:
                logger.error(f"Failed to store cache stats in Redis: {e}")

    def _load_stats_from_redis(self):
        if not CACHE_STATS_ENABLED:
            return
        try:
            cached_stats = self.redis_client.get(f"{self.stats_prefix}current")
            if cached_stats:
                loaded = json.loads(cached_stats)
                for key, value in loaded.items():
                    if key in self.stats and isinstance(value, dict) and isinstance(self.stats[key], dict):
                        self.stats[key].update(value)
                    else:
                        self.stats[key] = value
        except Exception as e:
            if DEBUG:
                logger.error(f"Failed to load cache stats from Redis: {e}")

    def get_hit_rate(self) -> float:
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        return self.stats["cache_hits"] / total if total > 0 else 0.0

    def get_error_rate(self) -> float:
        total = self.stats["total_requests"]
        return self.stats["cache_errors"] / total if total > 0 else 0.0

    def get_endpoint_stats(self) -> Dict:
        result = {}
        for endpoint, stats in self.stats["endpoint_stats"].items():
            total = stats["hits"] + stats["misses"]
            result[endpoint] = {
                "total_requests": stats["total"],
                "cache_hits": stats["hits"],
                "cache_misses": stats["misses"],
                "cache_errors": stats["errors"],
                "hit_rate": stats["hits"] / total if total > 0 else 0.0,
                "error_rate": stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0,
            }
        return result

    def get_location_stats(self) -> Dict:
        merged: Dict[str, Dict] = {}
        for location, stats in self.stats["location_stats"].items():
            key = normalize_location_for_cache(location)
            if key not in merged:
                merged[key] = {"hits": 0, "misses": 0, "errors": 0, "total": 0}
            for field in ("hits", "misses", "errors", "total"):
                merged[key][field] += stats[field]

        result = {}
        for location, stats in merged.items():
            total = stats["hits"] + stats["misses"]
            result[location] = {
                "total_requests": stats["total"],
                "cache_hits": stats["hits"],
                "cache_misses": stats["misses"],
                "cache_errors": stats["errors"],
                "hit_rate": stats["hits"] / total if total > 0 else 0.0,
                "error_rate": stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0,
            }
        return result

    def get_hourly_stats(self, hours: int = 24) -> List[Dict]:
        current_hour = int(time.time() // 3600)
        hourly_data = []
        for i in range(hours):
            hour_key = current_hour - i
            stats = self.stats["hourly_stats"].get(hour_key)
            if stats:
                total = stats["hits"] + stats["misses"]
                hourly_data.append(
                    {
                        "hour": hour_key,
                        "timestamp": hour_key * 3600,
                        "total_requests": stats["total"],
                        "cache_hits": stats["hits"],
                        "cache_misses": stats["misses"],
                        "cache_errors": stats["errors"],
                        "hit_rate": stats["hits"] / total if total > 0 else 0.0,
                    }
                )
            else:
                hourly_data.append(
                    {
                        "hour": hour_key,
                        "timestamp": hour_key * 3600,
                        "total_requests": 0,
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "cache_errors": 0,
                        "hit_rate": 0.0,
                    }
                )
        return list(reversed(hourly_data))

    def get_cache_health(self) -> Dict:
        hit_rate = self.get_hit_rate()
        error_rate = self.get_error_rate()
        if error_rate > 0.1:
            health_status = "unhealthy"
        elif hit_rate < CACHE_HEALTH_THRESHOLD:
            health_status = "degraded"
        else:
            health_status = "healthy"
        return {
            "status": health_status,
            "hit_rate": hit_rate,
            "error_rate": error_rate,
            "threshold": CACHE_HEALTH_THRESHOLD,
            "total_requests": self.stats["total_requests"],
            "uptime_hours": (time.time() - self.stats["last_reset"]) / 3600,
        }

    def get_comprehensive_stats(self) -> Dict:
        return {
            "overall": {
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_errors": self.stats["cache_errors"],
                "hit_rate": self.get_hit_rate(),
                "error_rate": self.get_error_rate(),
            },
            "by_endpoint": self.get_endpoint_stats(),
            "by_location": self.get_location_stats(),
            "hourly": self.get_hourly_stats(24),
            "health": self.get_cache_health(),
            "configuration": {
                "enabled": CACHE_STATS_ENABLED,
                "retention_hours": CACHE_STATS_RETENTION_HOURS,
                "health_threshold": CACHE_HEALTH_THRESHOLD,
            },
        }

    def reset_stats(self):
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_errors": 0,
            "endpoint_stats": {},
            "location_stats": {},
            "hourly_stats": {},
            "last_reset": time.time(),
        }
        self._store_stats_in_redis()
        if DEBUG:
            logger.info("📊 CACHE STATS RESET: All statistics cleared")


# ---------------------------------------------------------------------------
# Background warming loop
# ---------------------------------------------------------------------------


async def scheduled_cache_warming(cache_warmer: CacheWarmer):
    """Background task that schedules cache warming jobs periodically."""
    if not CACHE_WARMING_ENABLED or not cache_warmer:
        return

    while True:
        try:
            await asyncio.sleep(CACHE_WARMING_INTERVAL_HOURS * 3600)

            if not cache_warmer.warming_in_progress:
                if DEBUG:
                    logger.info("🕐 SCHEDULED CACHE WARMING: Creating cache warming job")
                from cache.accessors import get_job_manager  # lazy to avoid circular

                job_manager = get_job_manager()
                if job_manager:
                    job_id = job_manager.create_job(
                        "cache_warming",
                        {
                            "type": "all",
                            "locations": [],
                            "scheduled": True,
                            "scheduled_at": __import__("datetime")
                            .datetime.now(__import__("datetime").timezone.utc)
                            .isoformat(),
                        },
                    )
                    logger.info(f"✅ Cache warming job created: {job_id}")
                else:
                    logger.warning("⚠️  Job manager not available, falling back to direct warming")
                    asyncio.create_task(cache_warmer.warm_all_locations())
            else:
                if DEBUG:
                    logger.info("⏭️  SCHEDULED CACHE WARMING: Skipping - warming already in progress")

        except Exception as e:
            if DEBUG:
                logger.error(f"❌ SCHEDULED CACHE WARMING ERROR: {e}")
            await asyncio.sleep(300)
