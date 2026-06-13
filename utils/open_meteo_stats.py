"""Open-Meteo production-traffic health statistics."""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import redis

from config import (
    OPEN_METEO_CONSECUTIVE_FAILURE_THRESHOLD,
    OPEN_METEO_DEGRADED_FAILURE_RATE,
    OPEN_METEO_MONITORING_ENABLED,
    OPEN_METEO_STATS_WINDOW_SECONDS,
    OPEN_METEO_UNHEALTHY_FAILURE_RATE,
)

logger = logging.getLogger(__name__)


class OpenMeteoStats:
    """Track Open-Meteo failures in Redis-backed rolling minute buckets."""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.prefix = "open_meteo:stats"
        self.window_seconds = OPEN_METEO_STATS_WINDOW_SECONDS
        self.bucket_seconds = 60
        self.retention_seconds = self.window_seconds + (2 * self.bucket_seconds)

    def record_call(self, endpoint: Optional[str] = None) -> None:
        self._increment("calls", endpoint=endpoint)

    def record_attempt(self, endpoint: Optional[str] = None) -> None:
        self._increment("attempts", endpoint=endpoint)

    def record_success(self, endpoint: Optional[str] = None) -> None:
        self._increment("successes", endpoint=endpoint)
        self._set_value("consecutive_failures", 0)

    def record_failure(
        self,
        reason: str,
        *,
        endpoint: Optional[str] = None,
        terminal: bool = True,
        timeout: bool = False,
    ) -> None:
        if timeout:
            self._increment("timeouts", endpoint=endpoint)
        if reason in {"rate_limited", "rate_limit_exceeded"}:
            self._increment("rate_limits", endpoint=endpoint)
        elif reason.startswith("http_"):
            self._increment("http_errors", endpoint=endpoint)
        elif reason in {"connection_timeout", "client_error"}:
            self._increment("transport_errors", endpoint=endpoint)

        if not terminal:
            return

        self._increment("failures", endpoint=endpoint)
        now = _utc_now_iso()
        self._set_value("last_failure", now)
        self._set_value("last_failure_reason", reason)
        self._increment_counter("consecutive_failures")

    def get_health(self, probe_status: Optional[str] = None) -> Dict:
        if not OPEN_METEO_MONITORING_ENABLED:
            return {
                "status": "disabled",
                "monitoring_enabled": False,
                "window_seconds": self.window_seconds,
            }

        try:
            totals = self._window_totals()
            calls = totals.get("calls", 0)
            failures = totals.get("failures", 0)
            failure_rate = failures / calls if calls else 0.0
            consecutive_failures = self._get_int("consecutive_failures")

            status = self._status_for(failure_rate, consecutive_failures)
            if probe_status == "unhealthy":
                status = "unhealthy"
            elif probe_status == "degraded" and status == "healthy":
                status = "degraded"

            return {
                "status": status,
                "monitoring_enabled": True,
                "window_seconds": self.window_seconds,
                "calls_last_5m": calls,
                "attempts_last_5m": totals.get("attempts", 0),
                "successes_last_5m": totals.get("successes", 0),
                "failures_last_5m": failures,
                "failure_rate": round(failure_rate, 4),
                "timeouts_last_5m": totals.get("timeouts", 0),
                "rate_limits_last_5m": totals.get("rate_limits", 0),
                "http_errors_last_5m": totals.get("http_errors", 0),
                "transport_errors_last_5m": totals.get("transport_errors", 0),
                "consecutive_failures": consecutive_failures,
                "last_failure": self._get_value("last_failure"),
                "last_failure_reason": self._get_value("last_failure_reason"),
                "thresholds": {
                    "degraded_failure_rate": OPEN_METEO_DEGRADED_FAILURE_RATE,
                    "unhealthy_failure_rate": OPEN_METEO_UNHEALTHY_FAILURE_RATE,
                    "consecutive_failures": OPEN_METEO_CONSECUTIVE_FAILURE_THRESHOLD,
                },
            }
        except Exception as exc:
            logger.warning("Could not read Open-Meteo stats: %s", exc)
            return {
                "status": "unknown",
                "monitoring_enabled": True,
                "window_seconds": self.window_seconds,
                "message": "Open-Meteo stats unavailable",
            }

    def _status_for(self, failure_rate: float, consecutive_failures: int) -> str:
        if (
            failure_rate >= OPEN_METEO_UNHEALTHY_FAILURE_RATE
            or consecutive_failures >= OPEN_METEO_CONSECUTIVE_FAILURE_THRESHOLD
        ):
            return "unhealthy"
        if failure_rate >= OPEN_METEO_DEGRADED_FAILURE_RATE:
            return "degraded"
        return "healthy"

    def _increment(self, field: str, *, endpoint: Optional[str] = None) -> None:
        if not OPEN_METEO_MONITORING_ENABLED:
            return

        try:
            bucket_key = self._bucket_key()
            pipe = self.redis_client.pipeline()
            pipe.hincrby(bucket_key, field, 1)
            if endpoint:
                pipe.hincrby(bucket_key, f"{field}_{endpoint}", 1)
            pipe.expire(bucket_key, self.retention_seconds)
            pipe.execute()
        except Exception as exc:
            logger.debug("Could not record Open-Meteo stat %s: %s", field, exc)

    def _increment_counter(self, key: str) -> None:
        if not OPEN_METEO_MONITORING_ENABLED:
            return

        try:
            self.redis_client.incr(self._meta_key(key))
            self.redis_client.expire(self._meta_key(key), self.retention_seconds)
        except Exception as exc:
            logger.debug("Could not increment Open-Meteo stat %s: %s", key, exc)

    def _set_value(self, key: str, value: object) -> None:
        if not OPEN_METEO_MONITORING_ENABLED:
            return

        try:
            self.redis_client.setex(self._meta_key(key), self.retention_seconds, value)
        except Exception as exc:
            logger.debug("Could not set Open-Meteo stat %s: %s", key, exc)

    def _get_value(self, key: str) -> Optional[str]:
        raw = self.redis_client.get(self._meta_key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    def _get_int(self, key: str) -> int:
        raw = self.redis_client.get(self._meta_key(key))
        if raw is None:
            return 0
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return int(raw)

    def _window_totals(self) -> Dict[str, int]:
        now = int(time.time())
        start = now - self.window_seconds
        totals: Dict[str, int] = {}
        bucket = (start // self.bucket_seconds) * self.bucket_seconds
        end_bucket = (now // self.bucket_seconds) * self.bucket_seconds

        while bucket <= end_bucket:
            for field, value in self.redis_client.hgetall(self._bucket_key(bucket)).items():
                name = field.decode("utf-8") if isinstance(field, bytes) else str(field)
                totals[name] = totals.get(name, 0) + _to_int(value)
            bucket += self.bucket_seconds

        return totals

    def _bucket_key(self, bucket_start: Optional[int] = None) -> str:
        if bucket_start is None:
            bucket_start = (int(time.time()) // self.bucket_seconds) * self.bucket_seconds
        return f"{self.prefix}:bucket:{bucket_start}"

    def _meta_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"


def _to_int(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return int(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
