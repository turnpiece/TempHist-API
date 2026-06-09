"""Tests for resolve_cache_slug (Phase 2)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.daily_temperature_store import (
    DailyTemperatureStore,
    LocationCacheIdentity,
    _canonical_preapproved_cache_slug,
    resolve_location_cache_identity,
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
        conn.fetchrow = AsyncMock(
            return_value={
                "id": 42,
                "normalized_name": "greater_london__england__united_kingdom",
                "canonical_name": "Greater London, England, United Kingdom",
            }
        )
        conn.fetch = AsyncMock(
            return_value=[
                {"slug": "greater_london__england__united_kingdom"},
                {"slug": "london"},
            ]
        )
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
        conn.fetchrow = AsyncMock(
            side_effect=[
                None,
                {
                    "normalized_name": "london__england__united_kingdom",
                    "canonical_name": "London, England, United Kingdom",
                },
            ]
        )
        conn.fetch = AsyncMock(
            return_value=[
                {"slug": "london__england__united_kingdom"},
                {"slug": "london"},
            ]
        )
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
    async def test_module_slug_helper_returns_redis_slug(self):
        expected = LocationCacheIdentity(
            redis_slug="london",
            canonical_name="London, England, United Kingdom",
            lookup_slugs=("london",),
        )
        mock_store = MagicMock()
        mock_store.resolve_location_cache_identity = AsyncMock(return_value=expected)
        with patch(
            "utils.daily_temperature_store.get_daily_temperature_store",
            new=AsyncMock(return_value=mock_store),
        ):
            slug = await resolve_location_cache_slug("London")
        assert slug == "london"
        mock_store.resolve_location_cache_identity.assert_awaited_once_with("London", None)

    @pytest.mark.asyncio
    async def test_identity_without_postgres_uses_preapproved(self):
        store = self._make_store(disabled=True)
        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=None)):
            identity = await store.resolve_location_cache_identity("London")
        assert identity == LocationCacheIdentity(
            redis_slug="london",
            canonical_name="London, England, United Kingdom",
            lookup_slugs=("london",),
        )

    @pytest.mark.asyncio
    async def test_identity_alias_lookup_returns_both_fields(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "id": 42,
                "normalized_name": "greater_london__england__united_kingdom",
                "canonical_name": "Greater London, England, United Kingdom",
            }
        )
        conn.fetch = AsyncMock(
            return_value=[
                {"slug": "greater_london__england__united_kingdom"},
                {"slug": "london"},
            ]
        )
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=pool)):
            identity = await store.resolve_location_cache_identity("London")

        assert identity.redis_slug == "greater_london__england__united_kingdom"
        assert identity.canonical_name == "Greater London, England, United Kingdom"
        assert "london" in identity.lookup_slugs

    @pytest.mark.asyncio
    async def test_module_identity_helper_delegates_to_store(self):
        expected = LocationCacheIdentity(
            redis_slug="london",
            canonical_name="London, England, United Kingdom",
            lookup_slugs=("london",),
        )
        mock_store = MagicMock()
        mock_store.resolve_location_cache_identity = AsyncMock(return_value=expected)
        with patch(
            "utils.daily_temperature_store.get_daily_temperature_store",
            new=AsyncMock(return_value=mock_store),
        ):
            identity = await resolve_location_cache_identity("London")
        assert identity == expected
