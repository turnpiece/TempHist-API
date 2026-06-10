"""Tests for location canonicalization radius in DailyTemperatureStore."""

import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest  # noqa: E402

import config  # noqa: E402
from utils.daily_temperature_store import DailyTemperatureStore  # noqa: E402


class TestSameLocationRadius:
    def test_default_radius_matches_config(self):
        """_find_nearby_location defaults to the identity-dedup radius (same point)."""
        sig = inspect.signature(DailyTemperatureStore._find_nearby_location)
        default = sig.parameters["max_distance_km"].default
        assert default == config.SAME_LOCATION_RADIUS_KM
        assert default == 2.0

    def test_config_default_is_2km(self):
        assert config.SAME_LOCATION_RADIUS_KM == 2.0

    def test_env_var_override(self, monkeypatch):
        """SAME_LOCATION_RADIUS_KM env var is read by config at import time;
        verify the config module reads it correctly when patched."""
        import importlib

        monkeypatch.setenv("SAME_LOCATION_RADIUS_KM", "1.5")
        import config as cfg_module

        importlib.reload(cfg_module)
        assert cfg_module.SAME_LOCATION_RADIUS_KM == 1.5
        # Restore
        monkeypatch.delenv("SAME_LOCATION_RADIUS_KM", raising=False)
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
    async def test_location_within_identity_radius_is_snapped(self):
        """Same point (under the 2 km identity radius) folds onto the existing row."""
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=self._make_row(1.9))

        result = await store._find_nearby_location(conn, 51.5, -0.1)
        assert result is not None

    @pytest.mark.asyncio
    async def test_location_beyond_identity_radius_is_not_snapped(self):
        """A distinct nearby place (beyond 2 km) keeps its own identity."""
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=self._make_row(2.1))

        result = await store._find_nearby_location(conn, 51.5, -0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_exact_boundary_is_snapped(self):
        store = self._make_store()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=self._make_row(2.0))

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
