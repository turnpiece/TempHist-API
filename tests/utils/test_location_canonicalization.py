"""Tests for location canonicalization radius in DailyTemperatureStore."""

import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest

import config
from utils.daily_temperature_store import DailyTemperatureStore


class TestCanonicalizationRadius:
    def test_default_radius_is_45km(self):
        """_find_nearby_location default arg must match CANONICALIZATION_RADIUS_KM (45.0)."""
        sig = inspect.signature(DailyTemperatureStore._find_nearby_location)
        default = sig.parameters["max_distance_km"].default
        assert default == config.CANONICALIZATION_RADIUS_KM
        assert default == 45

    def test_config_default_is_45(self):
        assert config.CANONICALIZATION_RADIUS_KM == 45

    def test_env_var_override(self, monkeypatch):
        """CANONICALIZATION_RADIUS_KM env var is read by config at import time;
        verify the config module reads it correctly when patched."""
        import importlib

        monkeypatch.setenv("CANONICALIZATION_RADIUS_KM", "30")
        import config as cfg_module

        importlib.reload(cfg_module)
        assert cfg_module.CANONICALIZATION_RADIUS_KM == 30
        # Restore
        monkeypatch.delenv("CANONICALIZATION_RADIUS_KM", raising=False)
        importlib.reload(cfg_module)


class TestFindNearbyLocation:
    """Unit tests for DailyTemperatureStore._find_nearby_location."""

    def _make_store(self):
        store = DailyTemperatureStore.__new__(DailyTemperatureStore)
        return store

    def _make_row(self, distance_km: float) -> MagicMock:
        row = MagicMock()
        row.__getitem__ = lambda self, key: distance_km if key == "distance_km" else 1
        row.__bool__ = lambda self: True
        return row

    @pytest.mark.asyncio
    async def test_location_within_45km_is_snapped(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=self._make_row(44.9))

        result = await store._find_nearby_location(conn, 51.5, -0.1)
        assert result is not None

    @pytest.mark.asyncio
    async def test_location_beyond_45km_is_not_snapped(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=self._make_row(45.1))

        result = await store._find_nearby_location(conn, 51.5, -0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_boundary_is_snapped(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=self._make_row(45.0))

        result = await store._find_nearby_location(conn, 51.5, -0.1)
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_coordinates_returns_none(self):
        store = self._make_store()
        conn = AsyncMock()

        result = await store._find_nearby_location(conn, None, None)
        assert result is None
        conn.fetchrow.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_db_row_returns_none(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)

        result = await store._find_nearby_location(conn, 51.5, -0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_radius_respected(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=self._make_row(30.0))

        # Within 35km custom radius
        result = await store._find_nearby_location(conn, 51.5, -0.1, max_distance_km=35.0)
        assert result is not None

        conn.fetchrow = AsyncMock(return_value=self._make_row(36.0))
        result = await store._find_nearby_location(conn, 51.5, -0.1, max_distance_km=35.0)
        assert result is None
