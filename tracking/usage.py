import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import redis

from config import POPULARITY_WINDOW_DAYS, USAGE_TRACKING_ENABLED

logger = logging.getLogger(__name__)

# Sorted set tracking locations seen in roughly the last 24 hours. Score is the
# epoch-second timestamp of the most recent selection; entries are pruned by
# ZREMRANGEBYSCORE on each read, so no Redis TTL is needed.
RECENT_24H_KEY = "recent_selections:24h"
RECENT_24H_SECONDS = 24 * 3600


class LocationUsageTracker:
    """Track location usage patterns for analytics and cache warming."""

    def __init__(self, redis_client: redis.Redis, retention_days: int = 7):
        self.redis_client = redis_client
        self.retention_days = retention_days

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

        # Tier 2 (24h) recency set — score is current epoch seconds, so a more
        # recent selection always overwrites an older one. Pruned lazily on read.
        try:
            self.redis_client.zadd(RECENT_24H_KEY, {location_id: int(time.time())})
        except Exception as _e:
            logger.debug("Could not update 24h recency set for %s: %s", location_id, _e)

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

        return [(item.decode() if isinstance(item, bytes) else item, int(score)) for item, score in raw]

    def get_weighted_popular(self, limit: int = 150) -> List[Tuple[str, float]]:
        """Return ranked location IDs scored by recency-weighted selection counts.

        score = 0.5 * selections_last_7d
              + 0.3 * selections_last_30d
              + 0.2 * selections_last_90d

        Windows are cumulative (90d includes 30d includes 7d) to match the
        spec on issue #52 — recent activity is weighted strongly without
        eliminating long-tail signal.
        """
        if limit <= 0:
            return []

        windows = [(7, 0.5), (30, 0.3), (90, 0.2)]
        weighted_keys: dict = {}
        tmp_keys: List[str] = []

        try:
            for days, weight in windows:
                daily_keys = []
                for i in range(days):
                    date = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d")
                    key = f"selections:{date}"
                    if self.redis_client.exists(key):
                        daily_keys.append(key)
                if not daily_keys:
                    continue
                tmp_key = f"selections:weighted:window:{days}d:tmp"
                self.redis_client.zunionstore(tmp_key, daily_keys, aggregate="SUM")
                self.redis_client.expire(tmp_key, 60)
                weighted_keys[tmp_key] = weight
                tmp_keys.append(tmp_key)

            if not weighted_keys:
                return []

            # redis-py applies per-key weights when `keys` is a {key: weight} dict.
            final_key = "selections:weighted:tmp"
            self.redis_client.zunionstore(final_key, weighted_keys, aggregate="SUM")
            self.redis_client.expire(final_key, 60)
            tmp_keys.append(final_key)

            raw = self.redis_client.zrevrange(final_key, 0, limit - 1, withscores=True)
        finally:
            if tmp_keys:
                self.redis_client.delete(*tmp_keys)

        return [
            (item.decode() if isinstance(item, bytes) else item, float(score))
            for item, score in raw
        ]

    def get_weighted_popular_display_strings(self, limit: int = 150) -> List[Tuple[str, str, float]]:
        """Return (location_id, display_string, score) for top weighted-popular locations.

        IDs without a stored display string are skipped (same behaviour as
        get_popular_display_strings).
        """
        ranked = self.get_weighted_popular(limit=limit * 2)
        results: List[Tuple[str, str, float]] = []
        for location_id, score in ranked:
            raw = self.redis_client.get(f"loc_display:{location_id}")
            if raw:
                display = raw.decode() if isinstance(raw, bytes) else raw
                results.append((location_id, display, score))
            if len(results) >= limit:
                break
        return results

    def prune_recent_active(self) -> None:
        """Drop entries from the 24h recency set older than now - 24h."""
        if not USAGE_TRACKING_ENABLED:
            return
        try:
            cutoff = int(time.time()) - RECENT_24H_SECONDS
            self.redis_client.zremrangebyscore(RECENT_24H_KEY, "-inf", cutoff)
        except Exception as _e:
            logger.debug("Could not prune 24h recency set: %s", _e)

    def get_recent_active_locations(self) -> List[Tuple[str, int]]:
        """Return (location_id, last_seen_epoch) for locations active in last 24h.

        Prunes stale entries first so callers don't see anything older than 24h.
        """
        if not USAGE_TRACKING_ENABLED:
            return []
        self.prune_recent_active()
        try:
            raw = self.redis_client.zrevrange(RECENT_24H_KEY, 0, -1, withscores=True)
        except Exception as _e:
            logger.debug("Could not read 24h recency set: %s", _e)
            return []
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
        except Exception as _e:
            logger.debug("Could not cache display string for %s: %s", location_id, _e)

    def store_location_metadata(self, location_id: str, metadata: dict) -> None:
        """Persist minimal metadata for a non-preapproved location.

        Stored so that the location can be returned by the popular endpoint
        with enough fields to be rendered (name, country, etc.) even without
        a full preapproved LocationItem.  TTL matches store_location_display.
        """
        if not USAGE_TRACKING_ENABLED or not metadata:
            return
        try:
            self.redis_client.set(
                f"loc_meta:{location_id}",
                json.dumps(metadata, ensure_ascii=False),
                ex=95 * 86400,
            )
        except Exception as _e:
            logger.debug("Could not cache metadata for %s: %s", location_id, _e)

    # Geo-index used to canonicalize selections of the same physical place
    # submitted under different name strings / admin-level detail (e.g.
    # "Chennai" vs "Chennai, Tamil Nadu, India" both resolving to whichever
    # slug claimed that spot first). Populated lazily — only when a freshly
    # minted (non-explicit, non-preapproved) slug arrives with coordinates.
    geo_index_key = "loc_geo:v1"

    def find_nearby_canonical_id(self, latitude: float, longitude: float, radius_km: float) -> Optional[str]:
        """Return the closest canonical ID already registered within radius_km, or None.

        Backed by Redis GEOSEARCH. Used so that a newly-submitted location
        with coordinates converges onto an existing canonical ID for the same
        physical place rather than minting a new, signal-fragmenting slug.
        """
        if not USAGE_TRACKING_ENABLED:
            return None
        try:
            results = self.redis_client.geosearch(
                self.geo_index_key,
                longitude=longitude,
                latitude=latitude,
                radius=radius_km,
                unit="km",
                sort="ASC",
                count=1,
            )
            if results:
                member = results[0]
                return member.decode() if isinstance(member, bytes) else member
        except Exception as _e:
            logger.debug("Geo-index lookup failed for (%s, %s): %s", latitude, longitude, _e)
        return None

    def add_to_geo_index(self, canonical_id: str, latitude: float, longitude: float) -> None:
        """Register a canonical ID's coordinates so future nearby submissions converge onto it."""
        if not USAGE_TRACKING_ENABLED:
            return
        try:
            self.redis_client.geoadd(self.geo_index_key, [longitude, latitude, canonical_id])
        except Exception as _e:
            logger.debug("Could not add %s to geo-index: %s", canonical_id, _e)

    def seed_geo_index(self, anchors: List[Tuple[str, float, float]]) -> int:
        """Seed the geo index with known canonical anchors (id, latitude, longitude).

        Lets coordinate-bearing selections converge onto these IDs instead of
        minting fragment slugs. Used to register curated locations at startup so
        e.g. a "Greater London, …" selection with coordinates lands on `london`
        rather than a new slug. Idempotent — geoadd updates an existing member's
        position in place, so re-seeding on every boot is safe.

        Returns the number of anchors written.
        """
        if not USAGE_TRACKING_ENABLED or not anchors:
            return 0
        members: List = []
        for canonical_id, latitude, longitude in anchors:
            members.extend([longitude, latitude, canonical_id])
        try:
            self.redis_client.geoadd(self.geo_index_key, members)
            return len(anchors)
        except Exception as _e:
            logger.debug("Could not seed geo-index with %d anchors: %s", len(anchors), _e)
            return 0

    def get_location_metadata(self, location_id: str) -> Optional[dict]:
        """Retrieve stored minimal metadata for a location ID, or None."""
        if not USAGE_TRACKING_ENABLED:
            return None
        try:
            raw = self.redis_client.get(f"loc_meta:{location_id}")
            if raw:
                return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception as _e:
            logger.debug("Could not retrieve metadata for %s: %s", location_id, _e)
        return None

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

        total = sum(int(score) for _, score in self.redis_client.zrange(tmp_key, 0, -1, withscores=True))
        self.redis_client.delete(tmp_key)
        return total
