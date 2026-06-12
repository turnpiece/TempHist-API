"""Regression test for P1-126.

Confirms that fetching records for a non-preapproved location populates the
Redis `location_timezone:{normalized}` cache via the Open-Meteo provider,
so that `_get_location_timezone` no longer returns `None` for arbitrary
user-supplied locations.
"""

from datetime import date, timedelta
from typing import Dict
from unittest.mock import MagicMock

import pytest

from routers.v1_records import _collect_rolling_window_values
from utils.daily_temperature_store import DailyTemperatureRecord


def _normalize_location(text: str) -> str:
    return text.lower().replace(" ", "_").replace(",", "_")


class InMemoryDailyTemperatureStore:
    def __init__(self):
        self._data: Dict[tuple, DailyTemperatureRecord] = {}

    async def fetch(self, location: str, dates):
        normalized = _normalize_location(location)
        return {day: self._data[(normalized, day)] for day in dates if (normalized, day) in self._data}

    async def upsert(self, location: str, records, metadata=None):
        normalized = _normalize_location(location)
        for record in records:
            self._data[(normalized, record.date)] = record


def _timeline_days(start: date, end: date, temp: float = 12.0):
    days = (end - start).days + 1
    return [
        {
            "datetime": (start + timedelta(days=i)).isoformat(),
            "temp": temp,
            "tempmax": temp + 1,
            "tempmin": temp - 1,
        }
        for i in range(days)
    ]


@pytest.mark.asyncio
async def test_records_fetch_populates_timezone_cache_for_non_preapproved_location(monkeypatch):
    store = InMemoryDailyTemperatureStore()

    async def fake_get_store():
        return store

    monkeypatch.setattr("routers.v1_records.get_daily_temperature_store", fake_get_store)
    monkeypatch.setattr("routers.v1_records.get_job_manager", lambda: None)

    location = "Nowhere Town, USA"
    expected_tz = "America/Denver"

    async def fake_geocode(_location):
        return 40.0, -105.0, None

    async def fake_fetch_days(_lat, _lon, start, end):
        return (
            _timeline_days(start, end),
            {"resolvedAddress": None, "timezone": expected_tz, "latitude": 40.0, "longitude": -105.0},
        )

    monkeypatch.setattr("utils.open_meteo_client.geocode_location", fake_geocode)
    monkeypatch.setattr("utils.open_meteo_client.fetch_days", fake_fetch_days)
    monkeypatch.setattr("config.WEATHER_PROVIDER", "open_meteo", raising=False)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    values, _aggregated, _missing, _coverage = await _collect_rolling_window_values(
        location,
        "weekly",
        11,
        8,
        "celsius",
        [2022, 2023],
    )

    assert len(values) == 2
    assert store_mock.called, "store_location_timezone should have been called from the OM provider path"
    calls = [call.args for call in store_mock.call_args_list]
    assert any(args[0] == location and args[1] == expected_tz for args in calls), (
        f"expected store_location_timezone({location!r}, {expected_tz!r}); saw {calls}"
    )
