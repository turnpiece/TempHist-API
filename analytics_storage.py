"""Analytics storage and management."""

import json
import logging
import time
from datetime import datetime
from typing import List

import redis
from fastapi import HTTPException

from config import DEBUG
from models import AnalyticsData

_RT_MAX_SAMPLES = 10_000  # cap per Redis list
_SELECTION_METHODS = ("own_location", "carousel", "recent", "popular", "search")


def _compute_percentiles(values: List[int]) -> dict:
    """Return p50/p95/p99 and sample_size for a list of integers."""
    if not values:
        return {"p50": None, "p95": None, "p99": None, "sample_size": 0}
    s = sorted(values)
    n = len(s)

    def _pct(p: int) -> int:
        idx = (p / 100) * (n - 1)
        lo, frac = int(idx), idx - int(idx)
        if lo + 1 < n:
            return round(s[lo] + frac * (s[lo + 1] - s[lo]))
        return s[lo]

    return {"p50": _pct(50), "p95": _pct(95), "p99": _pct(99), "sample_size": n}


logger = logging.getLogger(__name__)


class AnalyticsStorage:
    """Store and manage client analytics data."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize analytics storage.

        Args:
            redis_client: Redis client for storing analytics data
        """
        self.redis = redis_client
        self.analytics_prefix = "analytics_"
        self.retention_seconds = 7 * 24 * 3600  # 7 days retention
        self.max_errors_per_session = 50  # Limit errors per session

    def store_analytics(self, analytics_data: AnalyticsData, client_ip: str) -> str:
        """Store analytics data and return unique ID."""
        analytics_id = f"analytics_{int(time.time() * 1000)}_{hash(client_ip) % 10000}"

        # Prepare data for storage
        analytics_record = {
            "id": analytics_id,
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip,
            "session_duration": analytics_data.session_duration,
            "api_calls": analytics_data.api_calls,
            "api_failure_rate": analytics_data.api_failure_rate,
            "retry_attempts": analytics_data.retry_attempts,
            "location_failures": analytics_data.location_failures,
            "error_count": analytics_data.error_count,
            "recent_errors": [
                error.model_dump() for error in analytics_data.recent_errors[: self.max_errors_per_session]
            ],
            "app_version": analytics_data.app_version,
            "platform": analytics_data.platform,
            "user_agent": analytics_data.user_agent,
            "session_id": analytics_data.session_id,
            "response_time_ms": analytics_data.response_time_ms,
            "cache_hit": analytics_data.cache_hit,
            "canonical_location": analytics_data.canonical_location,
            "requested_location": analytics_data.requested_location,
            "selection_method": analytics_data.selection_method,
        }

        try:
            # Store in Redis with expiration
            self.redis.setex(
                f"{self.analytics_prefix}{analytics_id}", self.retention_seconds, json.dumps(analytics_record)
            )

            # Add to analytics index for easy retrieval
            self.redis.lpush("analytics_index", analytics_id)
            self.redis.expire("analytics_index", self.retention_seconds)

            # Update analytics summary stats
            self._update_analytics_summary(analytics_record)

            if DEBUG:
                logger.debug(
                    f"📊 ANALYTICS STORED: {analytics_id} | Errors: {analytics_data.error_count} | Duration: {analytics_data.session_duration}s"
                )

            return analytics_id

        except Exception as e:
            logger.error(f"Failed to store analytics data: {e}")
            raise HTTPException(status_code=500, detail="Failed to store analytics data")

    def _update_analytics_summary(self, analytics_record: dict):
        """Update analytics summary statistics.

        Args:
            analytics_record: Dictionary containing analytics data to aggregate
        """
        try:
            # Get current summary
            summary_key = "analytics_summary"
            summary = self.redis.get(summary_key)
            if summary:
                summary_data = json.loads(summary)
            else:
                summary_data = {
                    "total_sessions": 0,
                    "total_api_calls": 0,
                    "total_errors": 0,
                    "avg_session_duration": 0,
                    "avg_failure_rate": 0,
                    "platforms": {},
                    "error_types": {},
                    "last_updated": datetime.now().isoformat(),
                }

            # Update counters
            summary_data["total_sessions"] += 1
            summary_data["total_api_calls"] += analytics_record["api_calls"]
            summary_data["total_errors"] += analytics_record["error_count"]

            # Update averages
            total_sessions = summary_data["total_sessions"]
            summary_data["avg_session_duration"] = (
                summary_data["avg_session_duration"] * (total_sessions - 1) + analytics_record["session_duration"]
            ) / total_sessions

            # Update avg_failure_rate running average
            raw_rate = analytics_record.get("api_failure_rate", "0%")
            if isinstance(raw_rate, str):
                raw_rate = raw_rate.rstrip("%")
            try:
                failure_rate = float(raw_rate)
            except (ValueError, TypeError):
                failure_rate = 0.0
            summary_data["avg_failure_rate"] = (
                summary_data["avg_failure_rate"] * (total_sessions - 1) + failure_rate
            ) / total_sessions

            # Update platform stats
            platform = analytics_record.get("platform", "unknown")
            summary_data["platforms"][platform] = summary_data["platforms"].get(platform, 0) + 1

            # Update error type stats
            for error in analytics_record["recent_errors"]:
                error_type = error.get("error_type", "unknown")
                summary_data["error_types"][error_type] = summary_data["error_types"].get(error_type, 0) + 1

            summary_data["last_updated"] = datetime.now().isoformat()

            # Store updated summary
            self.redis.setex(summary_key, self.retention_seconds, json.dumps(summary_data))

            # Accumulate response times in capped Redis lists for percentile queries
            rt = analytics_record.get("response_time_ms")
            if rt is not None:
                try:
                    rt_int = int(rt)
                    self._push_rt("analytics:rt:all", rt_int)

                    method = analytics_record.get("selection_method")
                    if method and method in _SELECTION_METHODS:
                        self._push_rt(f"analytics:rt:method:{method}", rt_int)

                    cache_hit = analytics_record.get("cache_hit")
                    if cache_hit is True:
                        self._push_rt("analytics:rt:cache:hit", rt_int)
                    elif cache_hit is False:
                        self._push_rt("analytics:rt:cache:miss", rt_int)
                except (TypeError, ValueError):
                    pass

        except Exception as e:
            logger.error(f"Failed to update analytics summary: {e}")

    def _push_rt(self, key: str, value: int) -> None:
        """Append a response time sample to a capped Redis list."""
        self.redis.lpush(key, value)
        self.redis.ltrim(key, 0, _RT_MAX_SAMPLES - 1)
        self.redis.expire(key, self.retention_seconds)

    def _load_rt_percentiles(self, key: str) -> dict:
        """Load a response-time list from Redis and return percentile stats."""
        try:
            raw = self.redis.lrange(key, 0, -1)
            values = [int(v) for v in raw if v is not None]
            return _compute_percentiles(values)
        except Exception:
            return {"p50": None, "p95": None, "p99": None, "sample_size": 0}

    def get_analytics_summary(self) -> dict:
        """Get analytics summary statistics."""
        try:
            summary_key = "analytics_summary"
            summary = self.redis.get(summary_key)
            if summary:
                summary_data = json.loads(summary)
            else:
                # No pre-aggregated summary — recompute from individual records if any exist
                analytics_ids = self.redis.lrange("analytics_index", 0, -1)
                if not analytics_ids:
                    summary_data = {
                        "total_sessions": 0,
                        "total_api_calls": 0,
                        "total_errors": 0,
                        "avg_session_duration": 0,
                        "avg_failure_rate": 0,
                        "platforms": {},
                        "error_types": {},
                        "last_updated": datetime.now().isoformat(),
                    }
                else:
                    summary_data = {
                        "total_sessions": 0,
                        "total_api_calls": 0,
                        "total_errors": 0,
                        "avg_session_duration": 0.0,
                        "avg_failure_rate": 0.0,
                        "platforms": {},
                        "error_types": {},
                        "last_updated": datetime.now().isoformat(),
                    }
                    for analytics_id in analytics_ids:
                        analytics_id = analytics_id.decode("utf-8") if isinstance(analytics_id, bytes) else analytics_id
                        record_raw = self.redis.get(f"{self.analytics_prefix}{analytics_id}")
                        if not record_raw:
                            continue
                        record = json.loads(record_raw.decode("utf-8") if isinstance(record_raw, bytes) else record_raw)
                        self._update_analytics_summary(record)

                    rebuilt = self.redis.get(summary_key)
                    summary_data = json.loads(rebuilt) if rebuilt else summary_data

            # Attach live response-time percentiles from the RT lists
            by_method = {}
            for method in _SELECTION_METHODS:
                stats = self._load_rt_percentiles(f"analytics:rt:method:{method}")
                if stats["sample_size"] > 0:
                    by_method[method] = stats

            summary_data["response_times"] = {
                "overall": self._load_rt_percentiles("analytics:rt:all"),
                "by_selection_method": by_method,
                "by_cache_hit": {
                    "hit": self._load_rt_percentiles("analytics:rt:cache:hit"),
                    "miss": self._load_rt_percentiles("analytics:rt:cache:miss"),
                },
            }

            return summary_data

        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {"error": "Failed to retrieve analytics summary"}

    def get_recent_analytics(self, limit: int = 100) -> List[dict]:
        """Get recent analytics records."""
        try:
            # Get recent analytics IDs
            analytics_ids = self.redis.lrange("analytics_index", 0, limit - 1)
            analytics_records = []

            for analytics_id in analytics_ids:
                analytics_id = analytics_id.decode("utf-8") if isinstance(analytics_id, bytes) else analytics_id
                record = self.redis.get(f"{self.analytics_prefix}{analytics_id}")
                if record:
                    record_str = record.decode("utf-8") if isinstance(record, bytes) else record
                    analytics_records.append(json.loads(record_str))

            return analytics_records

        except Exception as e:
            logger.error(f"Failed to get recent analytics: {e}")
            return []

    def get_analytics_by_session(self, session_id: str) -> List[dict]:
        """Get analytics records for a specific session.

        Args:
            session_id: Unique session identifier to search for

        Returns:
            List of analytics records matching the session ID
        """
        try:
            # This would require a more sophisticated indexing system
            # For now, we'll search through recent analytics
            recent_analytics = self.get_recent_analytics(1000)  # Get more records
            session_analytics = [record for record in recent_analytics if record.get("session_id") == session_id]
            return session_analytics

        except Exception as e:
            logger.error(f"Failed to get analytics by session: {e}")
            return []
