"""Tests for utils/weather.py's is_today / is_today_or_future timezone handling."""

import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cache.keys as cache_keys  # noqa: E402
from utils.weather import is_today, is_today_or_future  # noqa: E402


def test_is_today_without_location_uses_server_date():
    today = date.today()
    assert is_today(today.year, today.month, today.day) is True
    assert is_today(1999, 1, 1) is False


def test_is_today_or_future_without_location_uses_server_date():
    today = date.today()
    assert is_today_or_future(today.year, today.month, today.day) is True
    assert is_today_or_future(9999, 12, 31) is True
    assert is_today_or_future(1999, 1, 1) is False


def test_is_today_uses_location_local_date_when_provided(monkeypatch):
    # Server/UTC date differs from the location's local date near a day boundary.
    location_local_today = date(2026, 8, 15)
    monkeypatch.setattr(cache_keys, "get_local_today", lambda location, redis_client=None: location_local_today)

    # A date that would be "yesterday" by server/UTC time is "today" for this location.
    assert is_today(2026, 8, 15, "Auckland", None) is True
    assert is_today(2026, 8, 14, "Auckland", None) is False


def test_is_today_or_future_uses_location_local_date_when_provided(monkeypatch):
    location_local_today = date(2026, 8, 15)
    monkeypatch.setattr(cache_keys, "get_local_today", lambda location, redis_client=None: location_local_today)

    assert is_today_or_future(2026, 8, 15, "Auckland", None) is True
    assert is_today_or_future(2026, 8, 16, "Auckland", None) is True
    assert is_today_or_future(2026, 8, 14, "Auckland", None) is False
