"""Tests for utils.visual_crossing_client timezone caching."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils import visual_crossing_client


class _FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return ""


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def get(self, url, headers=None):
        return _FakeResponse(200, self._payload)


@pytest.fixture(autouse=True)
def _vc_api_key(monkeypatch):
    monkeypatch.setenv("VISUAL_CROSSING_API_KEY", "test-key")


@pytest.mark.asyncio
async def test_vc_fetch_timeline_caches_timezone_from_payload(monkeypatch):
    payload = {
        "resolvedAddress": "Swanage, Dorset, England, United Kingdom",
        "address": "Swanage, Dorset, England, United Kingdom",
        "timezone": "Europe/London",
        "latitude": 50.6,
        "longitude": -1.96,
        "days": [
            {"datetime": "2024-06-10", "temp": 60.0, "tempmax": 65.0, "tempmin": 55.0},
        ],
    }
    fake_session = _FakeSession(payload)
    monkeypatch.setattr(visual_crossing_client, "_get_client", AsyncMock(return_value=fake_session))

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    location = "Swanage, Dorset, England, United Kingdom"
    days, metadata = await visual_crossing_client.fetch_timeline_for_location(
        location, date(2024, 6, 10), date(2024, 6, 10)
    )

    assert metadata["timezone"] == "Europe/London"
    assert len(days) == 1
    store_mock.assert_called_once_with(location, "Europe/London")


@pytest.mark.asyncio
async def test_vc_fetch_single_date_caches_timezone_via_timeline(monkeypatch):
    payload = {
        "resolvedAddress": "Random Hamlet",
        "timezone": "America/Chicago",
        "latitude": 41.0,
        "longitude": -87.0,
        "days": [
            {"datetime": "2024-06-01", "temp": 70.0, "tempmax": 75.0, "tempmin": 65.0},
        ],
    }
    fake_session = _FakeSession(payload)
    monkeypatch.setattr(visual_crossing_client, "_get_client", AsyncMock(return_value=fake_session))

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    result = await visual_crossing_client.fetch_single_date("Random Hamlet", "2024-06-01")

    assert "days" in result
    store_mock.assert_called_once_with("Random Hamlet", "America/Chicago")


@pytest.mark.asyncio
async def test_vc_fetch_timeline_skips_cache_when_no_timezone(monkeypatch):
    payload = {
        "resolvedAddress": "Nowhere",
        "timezone": None,
        "latitude": 0.0,
        "longitude": 0.0,
        "days": [
            {"datetime": "2024-06-01", "temp": 50.0, "tempmax": 55.0, "tempmin": 45.0},
        ],
    }
    fake_session = _FakeSession(payload)
    monkeypatch.setattr(visual_crossing_client, "_get_client", AsyncMock(return_value=fake_session))

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    _days, metadata = await visual_crossing_client.fetch_timeline_for_location(
        "Nowhere", date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] is None
    store_mock.assert_not_called()
