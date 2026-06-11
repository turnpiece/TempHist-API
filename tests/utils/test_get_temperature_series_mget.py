"""Regression test for batched cache reads in get_temperature_series.

Issue: turnpiece/TempHist-API#68 — the per-year cache lookup used to call
GET once per year (up to ~51 sequential round trips on a cold cache). It
must now collapse into a single MGET.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.weather_data import get_temperature_series


@pytest.mark.asyncio
async def test_get_temperature_series_issues_single_mget():
    """All per-year cache reads must collapse into one MGET call."""
    years = list(range(1975, 2026))  # 51 years
    expected_keys = [f"weather_london_{y}_06_02" for y in years]

    mock_redis = MagicMock()
    # All keys miss → fallback to fetch_weather_batch.
    mock_redis.mget.return_value = [None] * len(years)
    mock_redis.get.return_value = None

    with patch("utils.weather_data.fetch_weather_batch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {}
        await get_temperature_series("london", 6, 2, mock_redis, years=years)

    # Exactly one MGET, with every expected key, in order.
    assert mock_redis.mget.call_count == 1
    called_keys = mock_redis.mget.call_args.args[0]
    assert called_keys == expected_keys
    # No per-year GETs.
    mock_redis.get.assert_not_called()


@pytest.mark.asyncio
async def test_get_temperature_series_uses_cached_values_from_mget():
    """A mix of hits and misses from one MGET must yield the right series."""
    years = [2022, 2023, 2024]
    cached_2023 = json.dumps({"days": [{"temp": 18.0}]})

    mock_redis = MagicMock()
    mock_redis.mget.return_value = [None, cached_2023, None]
    mock_redis.get.return_value = None

    with patch("utils.weather_data.fetch_weather_batch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {
            "2022-06-02": {"days": [{"temp": 17.0}]},
            "2024-06-02": {"days": [{"temp": 19.0}]},
        }
        result = await get_temperature_series("london", 6, 2, mock_redis, years=years)

    points = {p["x"]: p["y"] for p in result["data"]}
    assert points == {2022: 17.0, 2023: 18.0, 2024: 19.0}
    # The MGET happened first; only the uncached years were passed to fetch.
    assert sorted(mock_fetch.call_args.args[1]) == ["2022-06-02", "2024-06-02"]
