"""Tests for metro-area cache snapping."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.daily_temperature_store import (
    DailyTemperatureStore,
    coords_to_metro_slug,
)


class TestCoordsToMetroSlug:
    def test_london_center_and_lambeth_share_grid_cell(self):
        london = coords_to_metro_slug(51.5074, -0.1278)
        lambeth = coords_to_metro_slug(51.4613, -0.1217)
        assert london == lambeth

    def test_paris_and_asnieres_share_grid_cell(self):
        paris = coords_to_metro_slug(48.8566, 2.3522)
        asnieres = coords_to_metro_slug(48.9142, 2.2858)
        assert paris == asnieres

    def test_distant_cities_differ(self):
        london = coords_to_metro_slug(51.5074, -0.1278)
        new_york = coords_to_metro_slug(40.7128, -74.0060)
        assert london != new_york


class TestMetroAnchorResolution:
    def _make_store(self, disabled: bool = False):
        store = DailyTemperatureStore.__new__(DailyTemperatureStore)
        store._disabled = disabled
        store._dsn = None if disabled else "postgresql://test"
        store._pool = None
        store._pool_lock = None
        store._schema_lock = None
        store._schema_initialized = False
        store._pool_retry_after = 0.0
        return store

    @pytest.mark.asyncio
    async def test_snaps_to_nearby_anchor_slug(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            side_effect=[
                {
                    "id": 10,
                    "normalized_name": "lambeth__england__united_kingdom",
                    "canonical_name": "Lambeth, England, United Kingdom",
                },
                {
                    "normalized_name": "lambeth__england__united_kingdom",
                    "canonical_name": "Lambeth, England, United Kingdom",
                },
            ]
        )
        conn.fetch = AsyncMock(
            return_value=[
                {"slug": "greater_london__england__united_kingdom"},
                {"slug": "lambeth__england__united_kingdom"},
                {"slug": "london"},
            ]
        )
        conn.execute = AsyncMock()
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        anchor = MagicMock()
        anchor.__getitem__ = lambda self, key: {
            "id": 5,
            "normalized_name": "greater_london__england__united_kingdom",
            "resolved_name": "Greater London, England, United Kingdom",
            "original_name": "Greater London",
        }[key]

        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=pool)):
            with patch.object(
                store,
                "_resolve_coords_for_cache",
                new=AsyncMock(return_value=(51.4613, -0.1217, "Europe/London")),
            ):
                with patch.object(store, "_find_nearby_metro_anchor", new=AsyncMock(return_value=anchor)):
                    identity = await store.resolve_location_cache_identity("Lambeth")

        assert identity.redis_slug == "greater_london__england__united_kingdom"
        conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_without_postgres_uses_geo_grid_when_coords_available(self):
        store = self._make_store(disabled=True)
        with patch.object(store, "_ensure_pool", new=AsyncMock(return_value=None)):
            with patch.object(
                store,
                "_resolve_coords_for_cache",
                new=AsyncMock(return_value=(51.5074, -0.1278, "Europe/London")),
            ):
                identity = await store.resolve_location_cache_identity("London")

        assert identity.redis_slug == coords_to_metro_slug(51.5074, -0.1278)
        assert "london" in identity.lookup_slugs
