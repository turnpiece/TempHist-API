"""Tests for resolve_cache_slug (Phase 2)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.daily_temperature_store import (
    DailyTemperatureStore,
    _canonical_preapproved_cache_slug,
    resolve_location_cache_slug,
)


class TestCanonicalPreapprovedCacheSlug:
    def test_london_short_name_maps_to_preapproved_slug(self):
        assert _canonical_preapproved_cache_slug("London") == "london"

    def test_london_full_name_maps_to_preapproved_slug(self):
        assert _canonical_preapproved_cache_slug("London, England, United Kingdom") == "london"

    def test_unknown_location_returns_none(self):
        assert _canonical_preapproved_cache_slug("Not A Real Place XYZ") is None


class TestResolveCacheSlug:
    def _make_store(self, pool=None, disabled=False):
        store = DailyTemperatureStore.__new__(DailyTemperatureStore)
        store._disabled = disabled
        store._dsn = None if disabled else "postgresql://test"
        store._pool = pool
        store._pool_lock = None
        store._schema_lock = None
        store._schema_initialized = False
        store._pool_retry_after = 0.0
        return store

    @pytest.mark.asyncio
    async def test_without_postgres_uses_preapproved_slug(self):
        store = self._make_store(disabled=True)
        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=None)):
            slug = await store.resolve_cache_slug("London")
        assert slug == "london"

    @pytest.mark.asyncio
    async def test_without_postgres_falls_back_to_lexical_normalize(self):
        store = self._make_store(disabled=True)
        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=None)):
            slug = await store.resolve_cache_slug("Not A Real Place XYZ")
        assert slug == "not_a_real_place_xyz"

    @pytest.mark.asyncio
    async def test_alias_lookup_returns_canonical_normalized_name(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"normalized_name": "greater_london__england__united_kingdom"})
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=pool)):
            slug = await store.resolve_cache_slug("London")

        assert slug == "greater_london__england__united_kingdom"
        conn.fetchrow.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creates_alias_via_geo_snap_when_no_existing_alias(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        conn.fetchval = AsyncMock(return_value="london__england__united_kingdom")
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=pool)):
            with patch.object(
                store, "get_coordinates", new=AsyncMock(return_value=(51.5074, -0.1278, "Europe/London"))
            ):
                with patch.object(
                    store,
                    "_get_or_create_location_id",
                    new=AsyncMock(return_value=42),
                ) as mock_create:
                    slug = await store.resolve_cache_slug("London")

        assert slug == "london__england__united_kingdom"
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_module_helper_delegates_to_store(self):
        mock_store = MagicMock()
        mock_store.resolve_cache_slug = AsyncMock(return_value="london")
        with patch(
            "utils.daily_temperature_store.get_daily_temperature_store",
            new=AsyncMock(return_value=mock_store),
        ):
            slug = await resolve_location_cache_slug("London")
        assert slug == "london"
        mock_store.resolve_cache_slug.assert_awaited_once_with("London", None)
