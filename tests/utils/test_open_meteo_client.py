"""Tests for utils.open_meteo_client."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from utils import open_meteo_client


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_caches_timezone_from_metadata(monkeypatch):
    async def fake_geocode(location):
        return 40.0, -105.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 10.0, "tempmax": 11.0, "tempmin": 9.0}],
            {"resolvedAddress": None, "timezone": "America/Denver", "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    location = "Nowhere Town, USA"
    days, metadata = await open_meteo_client.fetch_timeline_for_location(
        location, date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] == "America/Denver"
    assert len(days) == 1
    store_mock.assert_called_once_with(location, "America/Denver")


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_uses_geocode_timezone_hint_when_metadata_missing(monkeypatch):
    async def fake_geocode(location):
        return 51.5, -0.1, "Europe/London"

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 12.0, "tempmax": 13.0, "tempmin": 11.0}],
            {"resolvedAddress": None, "timezone": None, "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    _days, metadata = await open_meteo_client.fetch_timeline_for_location(
        "London", date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] == "Europe/London"
    store_mock.assert_called_once_with("London", "Europe/London")


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_skips_cache_when_no_timezone_known(monkeypatch):
    async def fake_geocode(location):
        return 0.0, 0.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 20.0, "tempmax": 21.0, "tempmin": 19.0}],
            {"resolvedAddress": None, "timezone": None, "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    _days, metadata = await open_meteo_client.fetch_timeline_for_location(
        "Mystery Place", date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] is None
    store_mock.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_single_date_caches_timezone_from_metadata(monkeypatch):
    async def fake_geocode(location):
        return 41.0, -87.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 7.0, "tempmax": 8.0, "tempmin": 6.0}],
            {"resolvedAddress": None, "timezone": "America/Chicago", "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    result = await open_meteo_client.fetch_single_date("Random Hamlet", "2024-06-01")

    assert "days" in result
    store_mock.assert_called_once_with("Random Hamlet", "America/Chicago")


@pytest.mark.asyncio
async def test_fetch_single_date_uses_geocode_timezone_hint_when_metadata_missing(monkeypatch):
    async def fake_geocode(location):
        return 35.0, 139.0, "Asia/Tokyo"

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 22.0, "tempmax": 23.0, "tempmin": 21.0}],
            {"resolvedAddress": None, "timezone": None, "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    await open_meteo_client.fetch_single_date("Some Tokyo Suburb", "2024-06-01")

    store_mock.assert_called_once_with("Some Tokyo Suburb", "Asia/Tokyo")


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_swallows_cache_errors(monkeypatch):
    async def fake_geocode(location):
        return 0.0, 0.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 5.0, "tempmax": 6.0, "tempmin": 4.0}],
            {"resolvedAddress": None, "timezone": "UTC", "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    def boom(*args, **kwargs):
        raise RuntimeError("redis exploded")

    monkeypatch.setattr("cache.keys.store_location_timezone", boom)

    days, metadata = await open_meteo_client.fetch_timeline_for_location(
        "Anywhere", date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] == "UTC"
    assert len(days) == 1
