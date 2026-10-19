"""Providers return climatological estimates for days that have not happened yet.

Those must never be written to daily_temperatures (nor served from the in-memory
cache as if they were observations). Boundary: the location's local today is kept,
the day after is dropped.
"""

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from routers import v1_records
from utils.daily_temperature_store import (
    DailyTemperatureRecord,
    DailyTemperatureStore,
    drop_future_records,
    local_today,
)

TZ_AHEAD = "Pacific/Kiritimati"  # UTC+14
TZ_BEHIND = "Etc/GMT+12"  # UTC-12


def _record(day: date, source: str = "timeline") -> DailyTemperatureRecord:
    return DailyTemperatureRecord(
        date=day, temp_c=7.0, temp_max_c=9.0, temp_min_c=5.0, payload={"datetime": day.isoformat()}, source=source
    )


def _timeline_days(start: date, end: date):
    return [
        {"datetime": (start + timedelta(days=i)).isoformat(), "temp": 7.0, "tempmax": 9.0, "tempmin": 5.0}
        for i in range((end - start).days + 1)
    ]


class TestLocalToday:
    def test_defaults_to_utc(self):
        assert local_today(None) == datetime.now(timezone.utc).date()

    def test_invalid_timezone_falls_back_to_utc(self):
        assert local_today("Not/AZone") == datetime.now(timezone.utc).date()

    @pytest.mark.parametrize("tz", [TZ_AHEAD, TZ_BEHIND])
    def test_uses_location_timezone(self, tz):
        assert local_today(tz) == datetime.now(ZoneInfo(tz)).date()


class TestDropFutureRecords:
    def test_keeps_today_and_past_drops_tomorrow_onwards(self):
        today = date(2026, 10, 19)
        records = [_record(today - timedelta(days=1)), _record(today), _record(today + timedelta(days=1)), _record(date(2026, 12, 31))]
        kept = drop_future_records(records, today)
        assert [r.date for r in kept] == [today - timedelta(days=1), today]

    def test_all_future_returns_empty(self):
        today = date(2026, 10, 19)
        assert drop_future_records([_record(today + timedelta(days=1))], today) == []


class _FakeConn:
    def __init__(self, db_timezone=None):
        self.db_timezone = db_timezone
        self.rows = []

    @asynccontextmanager
    async def transaction(self):
        yield

    async def fetchval(self, sql, *args):
        return self.db_timezone

    async def executemany(self, sql, rows):
        self.rows.extend(rows)


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


def _store_with(conn: _FakeConn, monkeypatch) -> DailyTemperatureStore:
    store = DailyTemperatureStore(dsn="postgresql://unused")

    async def fake_pool():
        return _FakePool(conn)

    async def fake_location_id(*_args, **_kwargs):
        return 1

    monkeypatch.setattr(store, "_ensure_pool", fake_pool)
    monkeypatch.setattr(store, "_get_or_create_location_id", fake_location_id)
    return store


def _stored_days(conn: _FakeConn) -> list[date]:
    return sorted(row[1] for row in conn.rows)  # params are (location_id, day, ...)


class TestUpsertGuard:
    async def test_boundary_today_stored_tomorrow_dropped(self, monkeypatch):
        conn = _FakeConn()
        store = _store_with(conn, monkeypatch)
        today = local_today("Europe/London")
        records = [_record(today - timedelta(days=1)), _record(today), _record(today + timedelta(days=1))]

        await store.upsert("London", records, metadata={"timezone": "Europe/London"})

        assert _stored_days(conn) == [today - timedelta(days=1), today]

    async def test_future_only_batch_writes_nothing(self, monkeypatch):
        conn = _FakeConn()
        store = _store_with(conn, monkeypatch)
        today = local_today("Europe/London")
        future = [_record(today + timedelta(days=n)) for n in range(1, 32)]

        await store.upsert("London", future, metadata={"timezone": "Europe/London"})

        assert conn.rows == []

    async def test_uses_stored_location_timezone_when_metadata_has_none(self, monkeypatch):
        # UTC+14 is a calendar day ahead of UTC-12, so their "tomorrow" differs.
        ahead_today = local_today(TZ_AHEAD)
        behind_today = local_today(TZ_BEHIND)
        assert ahead_today >= behind_today

        conn = _FakeConn(db_timezone=TZ_AHEAD)
        store = _store_with(conn, monkeypatch)
        await store.upsert("Kiritimati", [_record(ahead_today), _record(ahead_today + timedelta(days=1))])

        assert _stored_days(conn) == [ahead_today]

    async def test_falls_back_to_utc_when_timezone_unknown(self, monkeypatch):
        conn = _FakeConn(db_timezone=None)
        store = _store_with(conn, monkeypatch)
        today = datetime.now(timezone.utc).date()

        await store.upsert("Nowhere", [_record(today), _record(today + timedelta(days=1))])

        assert _stored_days(conn) == [today]


class _InMemoryStore:
    def __init__(self):
        self.data: dict[date, DailyTemperatureRecord] = {}

    async def upsert(self, location, records, metadata=None):
        for record in records:
            self.data[record.date] = record


class TestRouterGuard:
    async def test_future_days_are_not_stored_or_cached(self, monkeypatch):
        store = _InMemoryStore()

        async def fake_get_store():
            return store

        today = local_today("Europe/London")
        start, end = today - timedelta(days=3), today + timedelta(days=5)

        async def fake_fetch(location, range_start, range_end):
            return _timeline_days(range_start, range_end), {"timezone": "Europe/London"}

        monkeypatch.setattr(v1_records, "get_daily_temperature_store", fake_get_store)
        monkeypatch.setattr(v1_records, "fetch_timeline_days", fake_fetch)

        cache: dict[date, DailyTemperatureRecord] = {}
        await v1_records._fetch_and_store_timeline_ranges("London", [(today.year, start, end)], cache, [], set())

        expected = [start + timedelta(days=i) for i in range(4)]  # start .. today inclusive
        assert sorted(store.data) == expected
        assert sorted(cache) == expected
        assert today in cache and today + timedelta(days=1) not in cache

    async def test_future_window_stays_missing_for_aggregation(self, monkeypatch):
        """A window ending after today still yields the past days; future days count as missing."""
        store = _InMemoryStore()

        async def fake_get_store():
            return store

        async def fake_fetch(location, range_start, range_end):
            return _timeline_days(range_start, range_end), {"timezone": "Europe/London"}

        monkeypatch.setattr(v1_records, "get_daily_temperature_store", fake_get_store)
        monkeypatch.setattr(v1_records, "get_job_manager", lambda: None)
        monkeypatch.setattr(v1_records, "fetch_timeline_days", fake_fetch)

        today = local_today("Europe/London")
        cache: dict[date, DailyTemperatureRecord] = {}
        future_range = (today.year, today + timedelta(days=1), today + timedelta(days=30))
        await v1_records._fetch_and_store_timeline_ranges("London", [future_range], cache, [], set())

        assert store.data == {}
        assert cache == {}
