import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import redis

from config import USAGE_TRACKING_ENABLED, DEBUG, POPULARITY_WINDOW_DAYS

logger = logging.getLogger(__name__)


class LocationUsageTracker:
    """Track location usage patterns for analytics and cache warming."""

    def __init__(self, redis_client: redis.Redis, retention_days: int = 7):
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

        usage_key = f"{self.usage_prefix}{location}"
        self.redis_client.incr(usage_key)
        self.redis_client.expire(usage_key, self.retention_seconds)

        timestamp_key = f"{self.timestamp_prefix}{location}"
        self.redis_client.lpush(timestamp_key, current_time)
        self.redis_client.expire(timestamp_key, self.retention_seconds)

        if endpoint:
            endpoint_key = f"{self.usage_prefix}{location}_{endpoint}"
            self.redis_client.incr(endpoint_key)
            self.redis_client.expire(endpoint_key, self.retention_seconds)

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

        tracked_locations_key = "tracked_locations"
        tracked_locations = self.redis_client.smembers(tracked_locations_key)

        for location_bytes in tracked_locations:
            location = location_bytes.decode() if isinstance(location_bytes, bytes) else location_bytes

            timestamp_key = f"{self.timestamp_prefix}{location}"
            timestamps = self.redis_client.lrange(timestamp_key, 0, -1)

            recent_count = sum(1 for ts in timestamps if float(ts) > cutoff_time)
            if recent_count > 0:
                location_counts.append((location, recent_count))

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

        now = time.time()
        last_hour = sum(1 for ts in timestamps if ts > now - 3600)
        last_24h = sum(1 for ts in timestamps if ts > now - 86400)

        return {
            "location": location,
            "total_requests": total_requests,
            "last_hour": last_hour,
            "last_24h": last_24h,
            "first_request": min(timestamps) if timestamps else None,
            "last_request": max(timestamps) if timestamps else None,
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

    def record_selection(self, location_id: str, user_uid: str) -> None:
        """Record a location selection from a user (for popularity ranking)."""
        if not USAGE_TRACKING_ENABLED:
            return

        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        dedup_key = f"selection_dedup:{user_uid}:{location_id}:{today}"
        if self.redis_client.exists(dedup_key):
            logger.debug("selection dedup hit — skipped: uid=%s location=%s key=%s", user_uid, location_id, dedup_key)
            return

        # Mark dedup with 25-hour TTL so it spans day boundaries safely
        self.redis_client.set(dedup_key, 1, ex=25 * 3600)

        daily_key = f"selections:{today}"
        self.redis_client.zincrby(daily_key, 1, location_id)
        self.redis_client.expire(daily_key, (POPULARITY_WINDOW_DAYS + 5) * 86400)
        logger.debug("selection recorded: uid=%s location=%s dedup_key=%s", user_uid, location_id, dedup_key)

    def get_popular_from_selections(self, limit: int = 20, days: int = 30) -> List[Tuple[str, int]]:
        """Return ranked location IDs from the rolling selection window."""
        daily_keys = []
        for i in range(days):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d")
            key = f"selections:{date}"
            if self.redis_client.exists(key):
                daily_keys.append(key)

        if not daily_keys:
            return []

        tmp_key = "selections:union:tmp"
        self.redis_client.zunionstore(tmp_key, daily_keys, aggregate="SUM")
        self.redis_client.expire(tmp_key, 60)

        raw = self.redis_client.zrevrange(tmp_key, 0, limit - 1, withscores=True)
        self.redis_client.delete(tmp_key)

        return [
            (item.decode() if isinstance(item, bytes) else item, int(score))
            for item, score in raw
        ]

    def store_location_display(self, location_id: str, display_string: str) -> None:
        """Persist a human-readable display string for a location ID.

        Stored so that non-preapproved locations (which have no entry in the
        preapproved JSON) can still be returned by get_popular_display_strings
        and therefore prewarmed.  TTL is 95 days — well beyond the longest
        popularity window (POPULARITY_WINDOW_DAYS default 30) so a location
        never disappears from the display map while it is still rankable.
        """
        if not USAGE_TRACKING_ENABLED or not display_string:
            return
        try:
            self.redis_client.set(
                f"loc_display:{location_id}",
                display_string,
                ex=95 * 86400,
            )
        except Exception:
            pass  # best-effort

    def get_popular_display_strings(self, limit: int = 20, days: int = 30) -> List[str]:
        """Return display strings for the top-ranked locations.

        Unlike get_popular_from_selections (which returns raw IDs), this
        resolves each ID to its stored display string so the result can be
        used directly for prewarming or returned to callers who only need the
        human-readable location name.

        Locations whose display string has expired or was never stored are
        silently skipped.
        """
        ranked = self.get_popular_from_selections(limit=limit * 2, days=days)
        results: List[str] = []
        for location_id, _score in ranked:
            raw = self.redis_client.get(f"loc_display:{location_id}")
            if raw:
                display = raw.decode() if isinstance(raw, bytes) else raw
                results.append(display)
            if len(results) >= limit:
                break
        return results

    def get_total_selections(self, days: int = 30) -> int:
        """Return total number of selections recorded in the rolling window."""
        daily_keys = []
        for i in range(days):
            date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d")
            key = f"selections:{date}"
            if self.redis_client.exists(key):
                daily_keys.append(key)

        if not daily_keys:
            return 0

        tmp_key = "selections:total:tmp"
        self.redis_client.zunionstore(tmp_key, daily_keys, aggregate="SUM")
        self.redis_client.expire(tmp_key, 60)

        total = sum(
            int(score) for _, score in self.redis_client.zrange(tmp_key, 0, -1, withscores=True)
        )
        self.redis_client.delete(tmp_key)
        return total
