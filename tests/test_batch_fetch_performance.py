#!/usr/bin/env python3
"""
Quick performance test to demonstrate the batch fetch optimization.
This test shows that we fetch all dates in a single query instead of 51 separate queries.
"""
import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routers.v1_records import _collect_rolling_window_values


@pytest.mark.asyncio
async def test_batch_fetch_reduces_database_calls():
    """Test that rolling window values are fetched in a single database call."""

    # Mock the DailyTemperatureStore
    mock_store = AsyncMock()
    mock_store.fetch = AsyncMock(return_value={})  # Return empty cache for simplicity
    mock_store.upsert = AsyncMock()

    # Track how many times fetch is called
    fetch_call_count = 0
    original_fetch = mock_store.fetch

    async def counting_fetch(*args, **kwargs):
        nonlocal fetch_call_count
        fetch_call_count += 1
        return await original_fetch(*args, **kwargs)

    mock_store.fetch = counting_fetch

    # Mock fetch_timeline_days to return empty results (avoid external API calls)
    with patch('routers.v1_records.get_daily_temperature_store', return_value=mock_store):
        with patch('routers.v1_records.fetch_timeline_days', return_value=([], {})):
            # Call the function with a yearly rolling window (365 days) across 51 years
            years = list(range(1975, 2026))  # 51 years
            await _collect_rolling_window_values(
                location="London",
                period="yearly",
                month=11,
                day=12,
                unit_group="celsius",
                years=years,
            )

    # The optimization should result in exactly 1 fetch call for all dates
    # Before optimization: 51 calls (one per year)
    # After optimization: 1 call (batch fetch all unique dates)
    assert fetch_call_count == 1, (
        f"Expected 1 database fetch call (batch), but got {fetch_call_count}. "
        "The optimization should fetch all unique dates in a single query!"
    )


@pytest.mark.asyncio
async def test_batch_fetch_calculates_unique_dates():
    """Test that the batch fetch includes all unique dates across years."""

    mock_store = AsyncMock()
    fetched_dates = None

    async def capture_fetch(location, dates):
        nonlocal fetched_dates
        fetched_dates = set(dates)
        # Return mock records for all dates
        return {d: MagicMock(temp_c=15.0) for d in dates}

    mock_store.fetch = capture_fetch
    mock_store.upsert = AsyncMock()

    with patch('routers.v1_records.get_daily_temperature_store', return_value=mock_store):
        with patch('routers.v1_records.fetch_timeline_days', return_value=([], {})):
            # Test with just 3 years to make verification easier
            years = [2023, 2024, 2025]
            await _collect_rolling_window_values(
                location="London",
                period="weekly",  # 7-day windows
                month=11,
                day=12,
                unit_group="celsius",
                years=years,
            )

    # For a 7-day window (weekly) ending on Nov 12 across 3 years:
    # 2023: Nov 6-12, 2023 (7 dates)
    # 2024: Nov 6-12, 2024 (7 dates)
    # 2025: Nov 6-12, 2025 (7 dates)
    # Total unique dates: 21 (7 days × 3 years, all unique)

    assert fetched_dates is not None, "fetch should have been called"
    assert len(fetched_dates) == 21, (
        f"Expected 21 unique dates for 7-day windows across 3 years, got {len(fetched_dates)}"
    )


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_batch_fetch_reduces_database_calls())
    asyncio.run(test_batch_fetch_calculates_unique_dates())
    print("✅ All performance tests passed!")
