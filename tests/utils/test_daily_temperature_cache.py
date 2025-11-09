from datetime import date, timedelta
from typing import Dict
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest

from routers.v1_records import _collect_rolling_window_values
from utils.daily_temperature_store import DailyTemperatureRecord


def _normalize_location(text: str) -> str:
    return text.lower().replace(" ", "_").replace(",", "_")


class InMemoryDailyTemperatureStore:
    def __init__(self):
        self._data: Dict[tuple[str, date], DailyTemperatureRecord] = {}

    async def fetch(self, location: str, dates):
        normalized = _normalize_location(location)
        result = {}
        for day in dates:
            key = (normalized, day)
            if key in self._data:
                result[day] = self._data[key]
        return result

    async def upsert(self, location: str, records, metadata=None):
        normalized = _normalize_location(location)
        for record in records:
            self._data[(normalized, record.date)] = record


@pytest.mark.asyncio
async def test_collect_rolling_window_weekly_caches(monkeypatch):
    store = InMemoryDailyTemperatureStore()

    async def fake_get_store():
        return store

    monkeypatch.setattr("routers.v1_records.get_daily_temperature_store", fake_get_store)

    fetched_ranges = []

    async def fake_fetch(location, start, end):
        fetched_ranges.append((start, end))
        days = (end - start).days + 1
        return (
            [
                {
                    "datetime": (start + timedelta(days=i)).isoformat(),
                    "temp": 10.0 + i,
                    "tempmax": 11.0 + i,
                    "tempmin": 9.0 + i,
                }
                for i in range(days)
            ],
            {
                "resolvedAddress": "Test City",
                "latitude": 53.0,
                "longitude": -2.0,
                "timezone": "Europe/London",
            },
        )

    monkeypatch.setattr("routers.v1_records.fetch_timeline_days", fake_fetch)

    values, aggregated, missing = await _collect_rolling_window_values(
        "Test City",
        "weekly",
        11,
        8,
        "celsius",
        [2024],
    )

    assert missing == []
    assert len(values) == 1
    assert values[0].year == 2024
    assert fetched_ranges == [(date(2024, 11, 2), date(2024, 11, 8))]
    assert pytest.approx(aggregated[0], rel=1e-6) == sum(10.0 + i for i in range(7)) / 7

    # Second call should hit the cache, avoiding new API fetches
    fetched_ranges.clear()
    values_again, aggregated_again, missing_again = await _collect_rolling_window_values(
        "Test City",
        "weekly",
        11,
        8,
        "celsius",
        [2024],
    )

    assert fetched_ranges == []
    assert missing_again == []
    assert values_again[0].model_dump() == values[0].model_dump()
    assert pytest.approx(aggregated_again[0], rel=1e-6) == aggregated[0]


@pytest.mark.asyncio
async def test_collect_rolling_window_respects_unit_conversion(monkeypatch):
    store = InMemoryDailyTemperatureStore()

    async def fake_get_store():
        return store

    monkeypatch.setattr("routers.v1_records.get_daily_temperature_store", fake_get_store)

    # Pre-populate store with Celsius readings so no timeline fetch is required
    anchor = date(2023, 11, 8)
    records = []
    for i in range(7):
        current_day = anchor - timedelta(days=6 - i)
        temp_c = 5.0 + i
        records.append(
            DailyTemperatureRecord(
                date=current_day,
                temp_c=temp_c,
                temp_max_c=temp_c + 1,
                temp_min_c=temp_c - 1,
                payload={
                    "datetime": current_day.isoformat(),
                    "temp": temp_c,
                    "tempmax": temp_c + 1,
                    "tempmin": temp_c - 1,
                },
                source="test",
            )
        )

    await store.upsert("Test City", records)

    async def failing_fetch(*args, **kwargs):
        raise AssertionError("Timeline fetch should not be called when cache is populated")

    monkeypatch.setattr("routers.v1_records.fetch_timeline_days", failing_fetch)

    values, aggregated, missing = await _collect_rolling_window_values(
        "Test City",
        "weekly",
        11,
        8,
        "fahrenheit",
        [2023],
    )

    assert missing == []
    assert len(values) == 1
    assert values[0].year == 2023

    avg_c = sum(5.0 + i for i in range(7)) / 7
    expected_f = (avg_c * 9.0 / 5.0) + 32.0
    assert pytest.approx(aggregated[0], rel=1e-6) == expected_f

