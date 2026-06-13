"""Tests for Open-Meteo rolling health statistics."""

import pytest

from utils import open_meteo_stats
from utils.open_meteo_stats import OpenMeteoStats


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis

    def hincrby(self, key, field, amount):
        self.redis.hincrby(key, field, amount)
        return self

    def expire(self, key, ttl):
        self.redis.expire(key, ttl)
        return self

    def execute(self):
        return []


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.hashes = {}

    def pipeline(self):
        return FakePipeline(self)

    def hincrby(self, key, field, amount):
        bucket = self.hashes.setdefault(key, {})
        bucket[field] = bucket.get(field, 0) + amount

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def expire(self, key, ttl):
        return True

    def setex(self, key, ttl, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1


@pytest.fixture
def frozen_time(monkeypatch):
    monkeypatch.setattr(open_meteo_stats.time, "time", lambda: 1_700_000_000)


def test_records_rolling_counts(frozen_time):
    stats = OpenMeteoStats(FakeRedis())

    stats.record_call("archive")
    stats.record_attempt("archive")
    stats.record_success("archive")

    health = stats.get_health()

    assert health["status"] == "healthy"
    assert health["calls_last_5m"] == 1
    assert health["attempts_last_5m"] == 1
    assert health["successes_last_5m"] == 1
    assert health["failures_last_5m"] == 0


def test_degrades_on_failure_rate(frozen_time):
    stats = OpenMeteoStats(FakeRedis())

    for _ in range(20):
        stats.record_call("forecast")
    for _ in range(2):
        stats.record_failure("http_503", endpoint="forecast")

    health = stats.get_health()

    assert health["status"] == "degraded"
    assert health["failure_rate"] == pytest.approx(0.1)
    assert health["http_errors_last_5m"] == 2


def test_unhealthy_on_consecutive_failures(frozen_time):
    stats = OpenMeteoStats(FakeRedis())

    for _ in range(10):
        stats.record_failure("connection_timeout", endpoint="archive", timeout=True)

    health = stats.get_health()

    assert health["status"] == "unhealthy"
    assert health["consecutive_failures"] == 10
    assert health["timeouts_last_5m"] == 10
    assert health["last_failure_reason"] == "connection_timeout"


def test_success_resets_consecutive_failures(frozen_time):
    stats = OpenMeteoStats(FakeRedis())

    stats.record_failure("http_503", endpoint="archive")
    stats.record_success("archive")

    assert stats.get_health()["consecutive_failures"] == 0
