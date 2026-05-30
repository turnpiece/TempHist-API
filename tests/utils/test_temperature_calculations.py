"""Unit tests for temperature calculation functions: mean, std dev, trend slope, slope error."""
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from datetime import datetime
from utils.temperature import calculate_standard_deviation, calculate_trend_slope, generate_summary


class TestCalculateStandardDeviation:
    def test_uniform_series_returns_zero(self):
        assert calculate_standard_deviation([10.0, 10.0, 10.0]) == 0.0

    def test_known_values(self):
        # pop std dev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        assert calculate_standard_deviation([2, 4, 4, 4, 5, 5, 7, 9]) == 2.0

    def test_two_values(self):
        # pop std dev of [0, 10] = 5.0
        assert calculate_standard_deviation([0.0, 10.0]) == 5.0

    def test_single_value_returns_none(self):
        assert calculate_standard_deviation([15.0]) is None

    def test_empty_list_returns_none(self):
        assert calculate_standard_deviation([]) is None

    def test_result_is_rounded_to_two_decimal_places(self):
        result = calculate_standard_deviation([1.0, 2.0, 3.0])
        assert result == round(result, 2)

    def test_negative_temperatures(self):
        result = calculate_standard_deviation([-10.0, -5.0, 0.0, 5.0, 10.0])
        assert result is not None
        assert result > 0


class TestCalculateTrendSlope:
    def _make_data(self, years, temps):
        return [{"x": y, "y": t} for y, t in zip(years, temps)]

    def test_perfect_positive_trend(self):
        # 1°C per year = 10°C/decade
        data = self._make_data([2000, 2001, 2002, 2003, 2004], [10, 11, 12, 13, 14])
        slope, r_squared, slope_error = calculate_trend_slope(data)
        assert slope == pytest.approx(10.0, abs=0.01)
        assert r_squared == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_trend(self):
        data = self._make_data([2000, 2001, 2002, 2003, 2004], [14, 13, 12, 11, 10])
        slope, r_squared, slope_error = calculate_trend_slope(data)
        assert slope == pytest.approx(-10.0, abs=0.01)
        assert r_squared == pytest.approx(1.0, abs=0.01)

    def test_flat_series_returns_zero_slope(self):
        data = self._make_data([2000, 2001, 2002, 2003], [15.0, 15.0, 15.0, 15.0])
        slope, r_squared, slope_error = calculate_trend_slope(data)
        assert slope == 0.0
        # r_squared is None when ss_tot == 0 (no variance)
        assert r_squared is None

    def test_single_point_returns_defaults(self):
        data = [{"x": 2020, "y": 15.0}]
        slope, r_squared, slope_error = calculate_trend_slope(data)
        assert slope == 0.0
        assert r_squared is None
        assert slope_error is None

    def test_two_points_no_slope_error(self):
        # slope_error requires n > 2
        data = self._make_data([2000, 2001], [10.0, 11.0])
        slope, r_squared, slope_error = calculate_trend_slope(data)
        assert slope == pytest.approx(10.0, abs=0.01)
        assert slope_error is None

    def test_slope_error_present_for_three_or_more_points(self):
        data = self._make_data([2000, 2001, 2002, 2003, 2004], [10, 11, 12, 13, 14])
        _, _, slope_error = calculate_trend_slope(data)
        # Perfect linear fit → residuals = 0 → slope_error = 0
        assert slope_error == pytest.approx(0.0, abs=0.01)

    def test_noisy_data_has_positive_slope_error(self):
        data = self._make_data(
            [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
            [10.0, 11.5, 11.0, 13.0, 12.5, 14.5, 14.0, 15.5, 15.0, 17.0],
        )
        slope, r_squared, slope_error = calculate_trend_slope(data)
        assert slope > 0
        assert r_squared is not None and 0 < r_squared <= 1.0
        assert slope_error is not None and slope_error > 0

    def test_none_temperatures_are_skipped(self):
        data = [{"x": 2000, "y": 10.0}, {"x": 2001, "y": None}, {"x": 2002, "y": 12.0}]
        slope, r_squared, slope_error = calculate_trend_slope(data)
        # Should work with the two valid points
        assert slope == pytest.approx(10.0, abs=0.01)

    def test_slope_is_per_decade(self):
        # 0.3°C per year = 3°C/decade
        data = self._make_data(
            list(range(2000, 2011)),
            [10.0 + i * 0.3 for i in range(11)],
        )
        slope, _, _ = calculate_trend_slope(data)
        assert slope == pytest.approx(3.0, abs=0.05)

    def test_r_squared_range(self):
        data = self._make_data(
            [2000, 2001, 2002, 2003, 2004, 2005],
            [10.0, 10.5, 11.3, 11.8, 12.6, 13.1],
        )
        _, r_squared, _ = calculate_trend_slope(data)
        assert r_squared is not None
        assert 0.0 <= r_squared <= 1.0

    def test_sparse_data_inflates_slope_error(self):
        # 3 points at 2000, 2005, 2010: year_span=11, n=3, inflation=sqrt(11/3)≈1.915
        # Manually verified: raw SE ≈ 1.73, inflated SE ≈ 3.32
        data = self._make_data([2000, 2005, 2010], [10.0, 11.0, 9.0])
        _, _, slope_error = calculate_trend_slope(data)
        assert slope_error == pytest.approx(3.32, abs=0.01)

    def test_complete_data_no_inflation(self):
        # When every year in the span has data, the inflation factor is 1 — SE unchanged.
        years = list(range(2000, 2010))
        temps = [10.0 + i * 0.2 + (0.3 if i % 3 == 0 else 0.0) for i in range(10)]
        data = self._make_data(years, temps)
        _, _, slope_error = calculate_trend_slope(data)
        # year_span == n == 10, so no inflation; error should still be positive
        assert slope_error is not None and slope_error >= 0


class TestGenerateSummaryAverageAwareWording:
    """generate_summary uses warm framing when above average, cold when below."""

    def _data(self, historical_temps, current_temp, current_year=2026):
        """Build a data list with historical years + current year."""
        start_year = current_year - len(historical_temps)
        return [{"x": start_year + i, "y": t} for i, t in enumerate(historical_temps)] + \
               [{"x": current_year, "y": current_temp}]

    def _date(self, year=2026):
        return datetime(year, 5, 15)

    def test_above_average_cooler_than_last_year_years_since_2(self):
        # avg ≈ 15, current=20 (above avg), last_year=22 (warmer), two years ago=18 (colder)
        data = self._data([14, 15, 16, 18, 22], 20)
        summary = generate_summary(data, self._date(), mean=15.0)
        assert "not as warm as last year but warmer than" in summary
        assert "colder than last year" not in summary

    def test_above_average_cooler_than_last_year_no_colder_year(self):
        # avg ≈ 15, current=20 (above avg), last_year=25 (warmer).
        # One historical year ties current (20) so is_coldest is False, but
        # no historical year is strictly colder, so last_colder is None.
        data = self._data([20, 22, 23, 24, 25], 20)
        summary = generate_summary(data, self._date(), mean=15.0)
        assert "not as warm as last year" in summary
        assert "colder than last year" not in summary

    def test_below_average_warmer_than_last_year_years_since_2(self):
        # avg ≈ 20, current=10 (below avg), last_year=8 (colder), two years ago=12 (warmer)
        data = self._data([21, 20, 19, 12, 8], 10)
        summary = generate_summary(data, self._date(), mean=20.0)
        assert "not as cold as last year but cooler than" in summary
        assert "warmer than last year" not in summary

    def test_below_average_warmer_than_last_year_no_warmer_year(self):
        # avg ≈ 20, current=10 (below avg), last_year=5 (colder).
        # One historical year ties current (10) so is_warmest is False, but
        # no historical year is strictly warmer, so last_warmer is None.
        data = self._data([10, 8, 7, 6, 5], 10)
        summary = generate_summary(data, self._date(), mean=20.0)
        assert "not as cold as last year" in summary
        assert "warmer than last year" not in summary

    def test_below_average_cooler_than_last_year_keeps_cold_framing(self):
        # avg ≈ 20, current=10 (below avg), last_year=12 (warmer), two years ago=8 (colder)
        data = self._data([21, 20, 19, 8, 12], 10)
        summary = generate_summary(data, self._date(), mean=20.0)
        assert "colder than last year but not as cold as" in summary

    def test_above_average_warmer_than_last_year_keeps_warm_framing(self):
        # avg ≈ 15, current=22 (above avg), last_year=20 (cooler), two years ago=24 (warmer)
        data = self._data([14, 15, 16, 24, 20], 22)
        summary = generate_summary(data, self._date(), mean=15.0)
        assert "warmer than last year but not as warm as" in summary
