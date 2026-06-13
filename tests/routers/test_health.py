"""Tests for detailed health checks."""

import json
from unittest.mock import patch

import pytest

from routers import health


class FakeRedis:
    def __init__(self):
        self.values = {}

    def ping(self):
        return True

    def setex(self, key, ttl, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        self.values.pop(key, None)


class FakePostgresStore:
    async def ping(self):
        return {"status": "healthy"}


class FakeStats:
    def __init__(self, payload):
        self.payload = payload
        self.probe_status = None

    def get_health(self, probe_status=None):
        self.probe_status = probe_status
        return dict(self.payload)


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def fake_async_client(status_code=None, exc=None):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

        async def get(self, *args, **kwargs):
            if exc:
                raise exc
            return FakeResponse(status_code)

    return FakeAsyncClient


async def fake_daily_temperature_store():
    return FakePostgresStore()


async def call_detailed_health(monkeypatch, *, probe_status_code=200, probe_exc=None, stats_payload=None):
    stats = FakeStats(
        stats_payload
        or {
            "status": "healthy",
            "calls_last_5m": 0,
            "failures_last_5m": 0,
            "failure_rate": 0.0,
        }
    )

    monkeypatch.setattr(health, "WEATHER_PROVIDER", "open_meteo")
    monkeypatch.setattr(health.httpx, "AsyncClient", fake_async_client(probe_status_code, probe_exc))
    monkeypatch.setattr(health, "get_open_meteo_stats", lambda: stats)
    monkeypatch.setattr(health, "get_cache_stats", lambda: None)
    monkeypatch.setattr(health, "get_daily_temperature_store", fake_daily_temperature_store)

    with patch("firebase_admin.auth.verify_id_token", side_effect=Exception("invalid token")):
        response = await health.detailed_health_check(FakeRedis())

    return response, json.loads(response.body), stats


@pytest.mark.asyncio
async def test_detailed_health_includes_open_meteo_rolling_stats(monkeypatch):
    response, body, stats = await call_detailed_health(
        monkeypatch,
        stats_payload={
            "status": "healthy",
            "calls_last_5m": 142,
            "failures_last_5m": 3,
            "failure_rate": 0.0211,
            "timeouts_last_5m": 2,
            "consecutive_failures": 0,
        },
    )

    open_meteo = body["checks"]["open_meteo_api"]
    assert response.status_code == 200
    assert open_meteo["status"] == "healthy"
    assert open_meteo["probe_status"] == "healthy"
    assert open_meteo["probe_status_code"] == 200
    assert open_meteo["calls_last_5m"] == 142
    assert stats.probe_status == "healthy"


@pytest.mark.asyncio
async def test_detailed_health_treats_429_probe_as_degraded(monkeypatch):
    response, body, stats = await call_detailed_health(
        monkeypatch,
        probe_status_code=429,
        stats_payload={
            "status": "degraded",
            "calls_last_5m": 12,
            "failures_last_5m": 1,
            "failure_rate": 0.0833,
        },
    )

    open_meteo = body["checks"]["open_meteo_api"]
    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert open_meteo["status"] == "degraded"
    assert open_meteo["probe_status"] == "degraded"
    assert open_meteo["probe_status_code"] == 429
    assert stats.probe_status == "degraded"


@pytest.mark.asyncio
async def test_detailed_health_unhealthy_open_meteo_stats_mark_overall_unhealthy(monkeypatch):
    response, body, _stats = await call_detailed_health(
        monkeypatch,
        stats_payload={
            "status": "unhealthy",
            "calls_last_5m": 40,
            "failures_last_5m": 12,
            "failure_rate": 0.3,
            "consecutive_failures": 10,
        },
    )

    assert response.status_code == 503
    assert body["status"] == "unhealthy"
    assert body["checks"]["open_meteo_api"]["failure_rate"] == 0.3


@pytest.mark.asyncio
async def test_detailed_health_skips_open_meteo_when_provider_inactive(monkeypatch):
    monkeypatch.setattr(health, "WEATHER_PROVIDER", "visual_crossing")
    monkeypatch.setattr(health, "get_cache_stats", lambda: None)
    monkeypatch.setattr(health, "get_daily_temperature_store", fake_daily_temperature_store)

    with patch("firebase_admin.auth.verify_id_token", side_effect=Exception("invalid token")):
        response = await health.detailed_health_check(FakeRedis())

    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["checks"]["open_meteo_api"]["status"] == "skipped"
    assert body["checks"]["open_meteo_api"]["provider"] == "visual_crossing"
