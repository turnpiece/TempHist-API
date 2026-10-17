"""Tests for ShareStore's SQL paths and the share country backfill script, using a fake asyncpg pool."""

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from scripts import backfill_share_country
from utils.share_store import ShareStore


class FakeConn:
    def __init__(self, *, fetchrow=None, fetch=None, execute="UPDATE 1"):
        self.fetchrow = AsyncMock(side_effect=fetchrow if isinstance(fetchrow, list) else None, return_value=fetchrow)
        self.fetch = AsyncMock(return_value=fetch or [])
        self.execute = AsyncMock(return_value=execute)


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return self

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *args):
        pass


def make_store(conn=None):
    store = ShareStore.__new__(ShareStore)
    store._disabled = conn is None
    store._pool = FakePool(conn) if conn is not None else None
    return store


class TestCreateShare:
    async def test_stores_country_code_and_returns_id_and_url(self):
        conn = FakeConn(fetchrow={"id": "aB3xY7qZ"})
        result = await make_store(conn).create_share("Kirkenes, Finnmark, Norway", "daily", "10-16", 2026, "celsius", 69.7, 30.0, "NO")
        assert result == {"id": "aB3xY7qZ", "url": "/s/aB3xY7qZ"}
        args = conn.fetchrow.await_args.args
        assert "country_code" in args[0]
        assert args[-1] == "NO"

    async def test_country_code_defaults_to_null(self):
        conn = FakeConn(fetchrow={"id": "aB3xY7qZ"})
        await make_store(conn).create_share("Madrid", "daily", "10-16", 2026, "celsius")
        assert conn.fetchrow.await_args.args[-1] is None

    async def test_retries_on_id_collision(self):
        conn = FakeConn(fetchrow=[asyncpg.UniqueViolationError(), {"id": "aB3xY7qZ"}])
        result = await make_store(conn).create_share("Madrid", "daily", "10-16", 2026, "celsius")
        assert result["id"] == "aB3xY7qZ"
        assert conn.fetchrow.await_count == 2

    async def test_gives_up_after_repeated_collisions(self):
        conn = FakeConn(fetchrow=[asyncpg.UniqueViolationError()] * 5)
        assert await make_store(conn).create_share("Madrid", "daily", "10-16", 2026, "celsius") is None

    async def test_disabled_store_returns_none(self):
        assert await make_store().create_share("Madrid", "daily", "10-16", 2026, "celsius") is None


class TestGetShare:
    ROW = {
        "id": "aB3xY7qZ",
        "location": "Kirkenes, Finnmark, Norway",
        "period": "daily",
        "identifier": "10-16",
        "ref_year": 2026,
        "unit": "celsius",
        "country_code": "NO",
        "created_at": datetime(2026, 10, 17, 16, 20, tzinfo=timezone.utc),
    }

    async def test_returns_country_code(self):
        share = await make_store(FakeConn(fetchrow=self.ROW)).get_share("aB3xY7qZ")
        assert share["country_code"] == "NO"
        assert share["created_at"] == "2026-10-17T16:20:00+00:00"

    async def test_unknown_share(self):
        assert await make_store(FakeConn(fetchrow=None)).get_share("nope0000") is None

    async def test_disabled_store_returns_none(self):
        assert await make_store().get_share("aB3xY7qZ") is None


class TestEnsureTable:
    async def test_adds_country_code_column(self):
        conn = FakeConn()
        store = make_store(conn)
        await store._ensure_table()
        assert "ADD COLUMN IF NOT EXISTS country_code" in conn.execute.await_args.args[0]

    async def test_no_pool_is_a_no_op(self):
        await make_store()._ensure_table()


class TestCountryBackfillQueries:
    async def test_list_shares_missing_country(self):
        rows = [{"id": "aB3xY7qZ", "location": "Madrid", "latitude": None, "longitude": None}]
        conn = FakeConn(fetch=rows)
        assert await make_store(conn).list_shares_missing_country() == rows
        assert "country_code IS NULL" in conn.fetch.await_args.args[0]

    async def test_list_shares_missing_country_disabled(self):
        assert await make_store().list_shares_missing_country() is None

    async def test_set_country_code_reports_whether_a_row_changed(self):
        assert await make_store(FakeConn(execute="UPDATE 1")).set_country_code("aB3xY7qZ", "ES") is True
        assert await make_store(FakeConn(execute="UPDATE 0")).set_country_code("aB3xY7qZ", "ES") is False

    async def test_set_country_code_only_touches_shares_without_one(self):
        conn = FakeConn()
        await make_store(conn).set_country_code("aB3xY7qZ", "ES")
        assert "country_code IS NULL" in conn.execute.await_args.args[0]

    async def test_set_country_code_disabled(self):
        assert await make_store().set_country_code("aB3xY7qZ", "ES") is False


class TestBackfillScript:
    async def test_no_database_configured_exits(self):
        store = MagicMock()
        store.list_shares_missing_country = AsyncMock(return_value=None)
        with pytest.raises(SystemExit):
            await backfill_share_country.backfill(store, None, execute=False)

    async def test_execute_without_redis_still_updates(self):
        store = MagicMock()
        store.list_shares_missing_country = AsyncMock(
            return_value=[{"id": "aB3xY7qZ", "location": "Kirkenes, Finnmark, Norway", "latitude": None, "longitude": None}]
        )
        store.set_country_code = AsyncMock(return_value=True)
        summary = await backfill_share_country.backfill(store, None, execute=True)
        assert summary["updated"] == 1

    @pytest.mark.parametrize("argv,redis_url,expect_warning", [
        (["prog"], None, False),
        (["prog", "--execute"], None, True),
        (["prog", "--execute", "--redis-url", "redis://x"], "redis://x", False),
    ])
    def test_main(self, argv, redis_url, expect_warning, capsys, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        summary = {"missing": 1, "resolved": 1, "unresolved": 0, "updated": 1}
        with (
            patch.object(sys, "argv", argv),
            patch.object(backfill_share_country.redis, "from_url") as from_url,
            patch.object(backfill_share_country, "ShareStore"),
            patch.object(backfill_share_country, "backfill", new=AsyncMock(return_value=summary)),
        ):
            backfill_share_country.main()
        out = capsys.readouterr().out
        assert ("no REDIS_URL" in out) is expect_warning
        assert "1 shares without a country" in out
        assert from_url.called is (redis_url is not None)
