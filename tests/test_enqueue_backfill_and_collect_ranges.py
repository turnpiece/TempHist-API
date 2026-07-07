"""Tests for _enqueue_backfill_and_collect_ranges (P1-153).

Verifies that a backfill job is only enqueued when the current request is
NOT already fetching that year's range inline — avoiding the redundant
per-year job fan-out that flooded the single-threaded job worker's queue
for cold (never-before-cached) locations.
"""

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from routers.v1_records import _enqueue_backfill_and_collect_ranges
from utils.daily_temperature_store import DailyTemperatureRecord


def _record(d: date, temp_c: float = 20.0) -> DailyTemperatureRecord:
    return DailyTemperatureRecord(
        date=d, temp_c=temp_c, temp_max_c=temp_c, temp_min_c=temp_c, payload={}, source="test"
    )


def _week_sequence(year: int) -> list[date]:
    # 7 consecutive days ending 07-07 of the given (non-current) year.
    return [date(year, 7, d) for d in range(1, 8)]


@pytest.mark.asyncio
async def test_partial_but_tolerable_gap_enqueues_backfill_and_skips_inline_fetch():
    """weekly tolerates 1 missing day (6/7) — should backfill in the background,
    not fetch inline, since the request can already be served from what's cached."""
    year = 2000
    sequence = _week_sequence(year)
    cache = {d: _record(d) for d in sequence[:-1]}  # 6 of 7 present

    with patch("routers.v1_records._enqueue_backfill_job", new=AsyncMock()) as mock_enqueue:
        ranges = await _enqueue_backfill_and_collect_ranges(
            location="Testville",
            period="weekly",
            month=7,
            day=7,
            slug="testville",
            year_to_date_sequence={year: sequence},
            cache=cache,
            current_year_now=2026,
        )

    mock_enqueue.assert_awaited_once()
    assert ranges == []


@pytest.mark.asyncio
async def test_coverage_below_threshold_fetches_inline_and_skips_backfill_enqueue():
    """weekly with 2+ days missing (5/7) is below tolerance — the request already
    fetches the range inline, so no duplicate backfill job should be created."""
    year = 2000
    sequence = _week_sequence(year)
    cache = {d: _record(d) for d in sequence[:-2]}  # 5 of 7 present

    with patch("routers.v1_records._enqueue_backfill_job", new=AsyncMock()) as mock_enqueue:
        ranges = await _enqueue_backfill_and_collect_ranges(
            location="Testville",
            period="weekly",
            month=7,
            day=7,
            slug="testville",
            year_to_date_sequence={year: sequence},
            cache=cache,
            current_year_now=2026,
        )

    mock_enqueue.assert_not_awaited()
    assert len(ranges) == 1
    assert ranges[0][0] == year


@pytest.mark.asyncio
async def test_no_missing_dates_skips_both_enqueue_and_fetch():
    year = 2000
    sequence = _week_sequence(year)
    cache = {d: _record(d) for d in sequence}  # fully covered

    with patch("routers.v1_records._enqueue_backfill_job", new=AsyncMock()) as mock_enqueue:
        ranges = await _enqueue_backfill_and_collect_ranges(
            location="Testville",
            period="weekly",
            month=7,
            day=7,
            slug="testville",
            year_to_date_sequence={year: sequence},
            cache=cache,
            current_year_now=2026,
        )

    mock_enqueue.assert_not_awaited()
    assert ranges == []
