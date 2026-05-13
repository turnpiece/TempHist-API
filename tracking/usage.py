import logging
import time
from typing import Dict, List

import redis

from config import USAGE_TRACKING_ENABLED, DEBUG

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
