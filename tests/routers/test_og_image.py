"""Tests for the OG preview image chart geometry (routers/og_image.py)."""

from unittest.mock import patch

import pytest
from matplotlib.axes import Axes

from routers.og_image import _render_chart, _temperature_axis_extents


class TestTemperatureAxisExtents:
    """Mirrors calculateTemperatureRange + getTemperatureLinearAxisExtents in the web app."""

    def test_mixed_sign_kirkenes_like(self):
        # min -1.15 floors to -2; max 3.0 is an integer so it gets +1 -> 4
        assert _temperature_axis_extents([-1.15, 0.5, 3.0]) == (-2, 4, 1)

    def test_integer_extremes_get_one_degree_of_padding(self):
        assert _temperature_axis_extents([2.0, 5.0]) == (1, 6, 1)

    def test_non_integer_extremes_are_floored_and_ceiled(self):
        assert _temperature_axis_extents([2.3, 5.6]) == (2, 6, 1)

    def test_step_is_two_for_ranges_up_to_twenty(self):
        lo, hi, step = _temperature_axis_extents([1.5, 14.5])  # 1..15 -> range 14
        assert step == 2
        assert (lo, hi) == (0, 16)

    def test_step_is_five_for_wide_ranges(self):
        lo, hi, step = _temperature_axis_extents([-10.5, 15.5])  # -11..16 -> range 27
        assert step == 5
        assert (lo, hi) == (-15, 20)

    def test_fahrenheit_sized_range(self):
        lo, hi, step = _temperature_axis_extents([23.2, 41.9])  # 23..42 -> range 19
        assert step == 2
        assert (lo, hi) == (22, 42)

    def test_extents_always_contain_the_data(self):
        temps = [-3.7, 0.2, 8.9]
        lo, hi, _ = _temperature_axis_extents(temps)
        assert lo <= min(temps)
        assert hi >= max(temps)


class TestBarGeometry:
    @pytest.mark.parametrize("unit", ["celsius", "fahrenheit"])
    def test_bars_start_at_axis_minimum_not_zero(self, unit):
        share = {"location": "Kirkenes", "period": "yearly", "identifier": "10-16", "ref_year": 2026, "unit": unit}
        records = [{"year": 2026 - i, "temperature": t} for i, t in enumerate([2.0, 3.0, -1.15, 0.5, -0.4, 1.2])]

        captured = {}
        real_barh = Axes.barh

        def spy(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return real_barh(self, *args, **kwargs)

        with patch.object(Axes, "barh", spy):
            png = _render_chart(share, records, show_title=False)

        assert png.startswith(b"\x89PNG")
        left = captured["kwargs"]["left"]
        lengths = captured["args"][1]
        temps = [r["temperature"] for r in records]
        if unit == "fahrenheit":
            temps = [t * 9 / 5 + 32 for t in temps]
        x_min, _x_max, _step = _temperature_axis_extents(temps)

        assert left == x_min
        assert all(length > 0 for length in lengths), "every bar extends right of the axis minimum"
        # bar end (left + length) is the real temperature
        assert sorted(left + length for length in lengths) == pytest.approx(sorted(temps))
