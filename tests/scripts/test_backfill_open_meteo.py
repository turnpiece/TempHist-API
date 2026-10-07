"""Unit tests for scripts/backfill_open_meteo.py (P1-173)."""

import json
from datetime import date, timedelta

import pytest

from scripts import backfill_open_meteo as backfill


def _daily(start: date, end: date, value_for) -> dict:
    out, day = {}, start
    while day <= end:
        out[day] = value_for(day)
        day += timedelta(days=1)
    return out


class TestArchiveEndDate:
    def test_stops_one_day_before_forecast_boundary(self):
        # fetch_days sends the last 7 days to the forecast endpoint, which cannot serve the pinned model.
        assert backfill.archive_end_date(date(2026, 10, 7)) == date(2026, 9, 29)


class TestEstimateCalls:
    def test_matches_pricing_page_examples(self):
        assert backfill.estimate_open_meteo_calls(14) == 1
        assert backfill.estimate_open_meteo_calls(28) == 3

    def test_full_history_is_thousands_of_calls(self):
        assert 2500 < backfill.estimate_open_meteo_calls(18628) < 2800

    def test_never_below_one(self):
        assert backfill.estimate_open_meteo_calls(1) == 1


class TestRollingYearMeans:
    def test_recovers_a_known_trend(self):
        anchor = date(2026, 10, 6)
        daily = _daily(date(1975, 1, 1), anchor, lambda d: 10 + 0.03 * (d.year - 1975))
        slope, r2, _err = backfill.calculate_trend_slope(backfill.rolling_year_means(daily, anchor))
        assert slope == pytest.approx(0.3, abs=0.02)
        assert r2 > 0.95

    def test_year_with_too_few_days_is_dropped(self):
        anchor = date(2026, 10, 6)
        daily = _daily(date(2000, 1, 1), anchor, lambda d: 10.0)
        for offset in range(100):  # leave only 265 days in the 2010 window
            daily.pop(date(2010, 10, 6) - timedelta(days=offset))
        years = [p["x"] for p in backfill.rolling_year_means(daily, anchor)]
        assert 2010 not in years
        assert 2009 in years

    def test_leap_day_anchor_does_not_raise(self):
        anchor = date(2024, 2, 29)
        daily = _daily(date(1990, 1, 1), anchor, lambda d: 5.0)
        assert backfill.rolling_year_means(daily, anchor)


class TestBuildRows:
    def test_skips_days_without_a_mean(self):
        days = [
            {"datetime": "2020-01-01", "temp": 3.5, "tempmax": 5.0, "tempmin": 1.0},
            {"datetime": "2020-01-02", "temp": None, "tempmax": 5.0, "tempmin": 1.0},
        ]
        rows = backfill.build_rows(7, days, "open-meteo:era5_land")
        assert len(rows) == 1
        location_id, day, temp, tmax, tmin, payload, source = rows[0]
        assert (location_id, day, temp, tmax, tmin) == (7, date(2020, 1, 1), 3.5, 5.0, 1.0)
        assert source == "open-meteo:era5_land"

    def test_payload_matches_the_shape_the_store_writes(self):
        rows = backfill.build_rows(1, [{"datetime": "2020-01-01", "temp": 3.5, "tempmax": 5.0, "tempmin": 1.0}], "s")
        assert json.loads(rows[0][5]) == {"datetime": "2020-01-01", "temp": 3.5, "tempmax": 5.0, "tempmin": 1.0}


class TestCheckCoverage:
    def test_complete_fetch_passes(self):
        backfill.check_coverage(366, date(2020, 1, 1), date(2020, 12, 31))

    def test_empty_fetch_is_an_error_because_fetch_days_swallows_failures(self):
        with pytest.raises(backfill.FetchIncomplete):
            backfill.check_coverage(0, date(2020, 1, 1), date(2020, 12, 31))

    def test_just_below_threshold_fails(self):
        with pytest.raises(backfill.FetchIncomplete):
            backfill.check_coverage(300, date(2020, 1, 1), date(2020, 12, 31))


class TestCompareSeries:
    def test_step_in_stored_series_is_removed(self):
        anchor = date(2026, 10, 6)
        start = date(1975, 1, 1)
        true = _daily(start, anchor, lambda d: 10 + 0.03 * (d.year - 1975))
        # Stored copy cools by 1 degC from 2017 on, like best_match in Hong Kong.
        stored = {d: v - (1.0 if d.year >= 2017 else 0.0) for d, v in true.items()}
        result = backfill.compare_series(stored, true, anchor)
        assert result["trend_before"] < 0.15
        assert result["trend_after"] == pytest.approx(0.3, abs=0.02)
        assert result["days_changed"] > 0

    def test_identical_series_changes_nothing(self):
        anchor = date(2026, 10, 6)
        series = _daily(date(1975, 1, 1), anchor, lambda d: 10 + 0.02 * (d.year - 1975))
        result = backfill.compare_series(series, series, anchor)
        assert result["trend_before"] == result["trend_after"]
        assert result["days_changed"] == 0
        assert result["mean_abs_diff"] == 0.0


class TestFetchSeries:
    async def test_raises_when_open_meteo_returns_too_little(self, monkeypatch):
        async def fake_fetch_days(lat, lon, start, end):
            return [{"datetime": "2020-01-01", "temp": 1.0, "tempmax": 2.0, "tempmin": 0.0}], {}

        monkeypatch.setattr("utils.open_meteo_client.fetch_days", fake_fetch_days)
        loc = backfill.LocationSummary(1, "x", 51.5, -0.1, 10, date(2020, 1, 1), date(2020, 12, 31), 0, 10, 0)
        with pytest.raises(backfill.FetchIncomplete):
            await backfill.fetch_series(loc, date(2020, 12, 31))


class TestParseArgs:
    def test_execute_requires_backup_dir(self):
        with pytest.raises(SystemExit):
            backfill.parse_args(["--execute"])

    def test_default_is_read_only(self):
        args = backfill.parse_args([])
        assert not args.execute and not args.compare

    def test_execute_with_backup_dir_is_accepted(self):
        assert backfill.parse_args(["--execute", "--backup-dir", "backups"]).execute
