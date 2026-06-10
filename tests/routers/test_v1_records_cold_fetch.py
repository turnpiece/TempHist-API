import asyncio
from datetime import date, timedelta
from typing import Dict

import pytest

from routers import v1_records
from routers.v1_records import _collect_rolling_window_values
from utils.daily_temperature_store import DailyTemperatureRecord


def _normalize_location(text: str) -> str:
    return text.lower().replace(" ", "_").replace(",", "_")


class InMemoryDailyTemperatureStore:
    def __init__(self):
        self._data: Dict[tuple[str, date], DailyTemperatureRecord] = {}

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
async def test_cold_cache_fetches_missing_years_with_bounded_concurrency(monkeypatch):
    store = InMemoryDailyTemperatureStore()

    async def fake_get_store():
        return store

    monkeypatch.setattr("routers.v1_records.get_daily_temperature_store", fake_get_store)
    monkeypatch.setattr("routers.v1_records.get_job_manager", lambda: None)

    active_fetches = 0
    max_active_fetches = 0
    fetched_ranges = []

    async def fake_fetch(location, start, end):
        nonlocal active_fetches, max_active_fetches
        active_fetches += 1
        max_active_fetches = max(max_active_fetches, active_fetches)
        fetched_ranges.append((start, end))
        await asyncio.sleep(0.01)
        active_fetches -= 1
        return _timeline_days(start, end), {"resolvedAddress": "Test City"}

    monkeypatch.setattr("routers.v1_records.fetch_timeline_days", fake_fetch)

    values, _aggregated, missing, _coverage = await _collect_rolling_window_values(
        "Test City",
        "weekly",
        11,
        8,
        "celsius",
        [2020, 2021, 2022, 2023],
    )

    assert len(values) == 4
    assert missing == []
    assert len(fetched_ranges) == 4
    assert max_active_fetches == min(v1_records.MAX_CONCURRENT_REQUESTS, len(fetched_ranges))


@pytest.mark.asyncio
async def test_cold_cache_fetches_are_persisted_for_later_calls(monkeypatch):
    store = InMemoryDailyTemperatureStore()

    async def fake_get_store():
        return store

    monkeypatch.setattr("routers.v1_records.get_daily_temperature_store", fake_get_store)
    monkeypatch.setattr("routers.v1_records.get_job_manager", lambda: None)

    fetched_ranges = []

    async def fake_fetch(location, start, end):
        fetched_ranges.append((start, end))
        return _timeline_days(start, end, temp=8.0), {"resolvedAddress": "Test City"}

    monkeypatch.setattr("routers.v1_records.fetch_timeline_days", fake_fetch)

    values, aggregated, missing, coverage = await _collect_rolling_window_values(
        "Test City",
        "weekly",
        11,
        8,
        "celsius",
        [2022, 2023],
    )

    assert len(values) == 2
    assert aggregated == [8.0, 8.0]
    assert missing == []
    assert all(item["available_days"] == 7 for item in coverage)
    assert len(fetched_ranges) == 2

    fetched_ranges.clear()
    values_again, aggregated_again, missing_again, coverage_again = await _collect_rolling_window_values(
        "Test City",
        "weekly",
        11,
        8,
        "celsius",
        [2022, 2023],
    )

    assert fetched_ranges == []
    assert [value.model_dump() for value in values_again] == [value.model_dump() for value in values]
    assert aggregated_again == aggregated
    assert missing_again == missing
    assert coverage_again == coverage


@pytest.mark.asyncio
async def test_cold_cache_timeline_error_marks_only_failed_year(monkeypatch):
    store = InMemoryDailyTemperatureStore()

    async def fake_get_store():
        return store

    monkeypatch.setattr("routers.v1_records.get_daily_temperature_store", fake_get_store)
    monkeypatch.setattr("routers.v1_records.get_job_manager", lambda: None)

    async def fake_fetch(location, start, end):
        if end.year == 2021:
            raise RuntimeError("provider timeout")
        return _timeline_days(start, end), {"resolvedAddress": "Test City"}

    monkeypatch.setattr("routers.v1_records.fetch_timeline_days", fake_fetch)

    values, _aggregated, missing, coverage = await _collect_rolling_window_values(
        "Test City",
        "weekly",
        11,
        8,
        "celsius",
        [2020, 2021],
    )

    assert [value.year for value in values] == [2020]
    assert coverage[0]["year"] == 2020
    assert missing == [{"year": 2021, "reason": "timeline_error"}]
