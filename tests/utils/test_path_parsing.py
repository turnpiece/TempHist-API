"""Tests for middleware location extraction from URL paths."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest  # noqa: E402

from utils.path_parsing import extract_location_from_path  # noqa: E402


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/weather/london/2024-01-15", ("london", "weather")),
        ("/forecast/berlin", ("berlin", "forecast")),
        ("/v1/records/daily/london/01-01", ("london", "daily")),
        ("/v1/records/yearly/berlin%2C%20germany/06-09", ("berlin, germany", "yearly")),
        ("/v1/records/daily/london/01-01/summary", ("london", "daily")),
        ("/v1/records/rolling-bundle/london/06-09/async", ("london", "rolling-bundle")),
        ("/health", (None, None)),
        ("/v1/records/daily", (None, None)),
    ],
)
def test_extract_location_from_path(path, expected):
    assert extract_location_from_path(path) == expected
