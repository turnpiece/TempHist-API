"""Tests for P1-117: narrow-path year recomputation on partial cache misses.

These tests verify that when only a subset of years is missing from cache,
the API fetches just those years instead of running the full 51-year
pipeline.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def fake_redis():
    """Minimal Redis stub usable by code that only calls setex / get."""
    r = MagicMock()
    r.setex.return_value = True
    r.get.return_value = None
    return r


class TestComputePerYearRecordsNarrowPath:
    """Direct unit tests on the narrow helper."""

    @pytest.mark.asyncio
    async def test_empty_years_returns_empty(self, fake_redis):
        from routers.v1_records import compute_per_year_records

        per_year, missing, coverage = await compute_per_year_records(
            "london", "monthly", "06-10", [], "celsius", fake_redis
        )
        assert per_year == {}
        assert missing == []
        assert coverage == []

    @pytest.mark.asyncio
    async def test_weekly_passes_only_requested_years_to_window_collector(self, fake_redis):
        """The narrow helper must hand _collect_rolling_window_values only the requested years."""
        from models import TemperatureValue
        from routers import v1_records

        captured_kwargs = {}

        async def fake_collect(**kwargs):
            captured_kwargs.update(kwargs)
            years = kwargs["years"]
            values = [
                TemperatureValue(date=f"{y}-06-10", year=y, temperature=15.0) for y in years
            ]
            return values, [15.0] * len(years), [], []

        with (
            patch.object(v1_records, "_collect_rolling_window_values", side_effect=fake_collect),
            patch.object(v1_records, "resolve_location_cache_slug", new=AsyncMock(return_value="london")),
        ):
            per_year, _, _ = await v1_records.compute_per_year_records(
                "london", "weekly", "06-10", [2024, 2026], "celsius", fake_redis
            )

        assert captured_kwargs["years"] == [2024, 2026]
        assert set(per_year.keys()) == {2024, 2026}

    @pytest.mark.asyncio
    async def test_daily_passes_years_to_series(self, fake_redis):
        """The daily branch must forward years= to get_temperature_series."""
        from routers import v1_records

        captured = {}

        async def fake_series(location, month, day, redis_client, years=None):
            captured["years"] = years
            return {
                "data": [{"x": y, "y": 14.0} for y in (years or [])],
                "metadata": {},
            }

        with (
            patch.object(v1_records, "get_temperature_series", side_effect=fake_series),
            patch.object(v1_records, "resolve_location_cache_slug", new=AsyncMock(return_value="london")),
        ):
            per_year, _, _ = await v1_records.compute_per_year_records(
                "london", "daily", "06-10", [2025], "celsius", fake_redis
            )

        assert captured["years"] == [2025]
        assert set(per_year.keys()) == {2025}


class TestProcessRecordJobNarrowPath:
    """The single-year job must not call the full 51-year pipeline."""

    @pytest.mark.asyncio
    async def test_single_year_job_skips_full_pipeline(self, fake_redis):
        from job_worker import JobWorker

        # DB pre-check fetch: return non-empty so the worker doesn't short-circuit.
        store = MagicMock()
        store.fetch = AsyncMock(return_value={datetime(2024, 6, 10).date(): MagicMock(temp_c=15.0)})

        narrow_calls = []
        full_pipeline_calls = []

        async def fake_compute(location, period, identifier, years, unit_group="celsius", redis_client=None):
            narrow_calls.append(list(years))
            year = years[0]
            return (
                {
                    year: {
                        "date": f"{year}-06-10",
                        "year": year,
                        "temperature": 15.0,
                    }
                },
                [],
                [{"year": year, "coverage_ratio": 0.95, "available_days": 28, "expected_days": 30}],
            )

        async def fake_full(*args, **kwargs):
            full_pipeline_calls.append(args)
            return {"values": []}

        def fake_rebuild(values, *args, **kwargs):
            return {"values": values}

        with (
            patch("routers.v1_records.compute_per_year_records", side_effect=fake_compute),
            patch("routers.v1_records.get_temperature_data_v1", side_effect=fake_full),
            patch("routers.v1_records._rebuild_full_response_from_values", side_effect=fake_rebuild),
            patch("utils.daily_temperature_store.resolve_location_cache_slug", new=AsyncMock(return_value="london")),
            patch("utils.daily_temperature_store.get_daily_temperature_store", new=AsyncMock(return_value=store)),
        ):
            worker = JobWorker(fake_redis)
            target_year = datetime.now().year - 1
            result = await worker.process_record_job(
                {
                    "scope": "monthly",
                    "location": "london",
                    "identifier": "06-10",
                    "year": target_year,
                }
            )

        assert narrow_calls == [[target_year]], "narrow helper should be called once with [year]"
        assert full_pipeline_calls == [], "full pipeline must NOT be called for single-year jobs"
        assert result["year"] == target_year
