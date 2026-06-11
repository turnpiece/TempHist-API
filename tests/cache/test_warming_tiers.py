"""Unit tests for CacheWarmer two-tier location selection (issue #52)."""

from unittest.mock import MagicMock

import pytest

from cache import warming as warming_module
from cache.warming import CacheWarmer


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def mock_tracker():
    tracker = MagicMock()
    tracker.get_weighted_popular_display_strings.return_value = []
    tracker.get_recent_active_locations.return_value = []
    return tracker


@pytest.fixture
def warmer(mock_redis, mock_tracker):
    return CacheWarmer(mock_redis, usage_tracker=mock_tracker)


@pytest.fixture
def enable_tracking(monkeypatch):
    """USAGE_TRACKING_ENABLED is imported into cache.warming at module load — patch it there."""
    monkeypatch.setattr(warming_module, "USAGE_TRACKING_ENABLED", True)


class TestTierComposition:
    def test_tier1_from_weighted_signal(self, warmer, mock_tracker, enable_tracking, monkeypatch):
        # Limit Tier 1 to 2 so the test is small and self-contained.
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER1_SIZE", 2)
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER2_MAX", 0)
        mock_tracker.get_weighted_popular_display_strings.return_value = [
            ("london", "London, England, UK", 50.0),
            ("paris", "Paris, Île-de-France, France", 25.0),
        ]

        locs = warmer.get_locations_to_warm()

        assert locs == ["London, England, UK", "Paris, Île-de-France, France"]
        assert len(warmer.last_tier1) == 2
        assert warmer.last_tier1[0]["id"] == "london"
        assert warmer.last_tier1[0]["score"] == 50.0
        assert warmer.last_tier2 == []

    def test_tier1_padded_with_preapproved_when_signal_sparse(self, warmer, mock_tracker, enable_tracking, monkeypatch):
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER1_SIZE", 3)
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER2_MAX", 0)
        mock_tracker.get_weighted_popular_display_strings.return_value = [
            ("london", "London, England, United Kingdom", 50.0),
        ]
        # Stub preapproved list — must include London (de-duplicated) plus extras.
        monkeypatch.setattr(
            warmer,
            "get_preapproved_locations",
            lambda: [
                "London, England, United Kingdom",
                "Manchester, England, United Kingdom",
                "Birmingham, England, United Kingdom",
            ],
        )

        locs = warmer.get_locations_to_warm()

        assert len(locs) == 3
        assert locs[0] == "London, England, United Kingdom"
        # Preapproved padding appended without duplicating London.
        assert "Manchester, England, United Kingdom" in locs
        assert "Birmingham, England, United Kingdom" in locs

    def test_tier2_excludes_tier1_locations(self, warmer, mock_tracker, mock_redis, enable_tracking, monkeypatch):
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER1_SIZE", 1)
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER2_MAX", 10)
        mock_tracker.get_weighted_popular_display_strings.return_value = [
            ("london", "London, England, UK", 50.0),
        ]
        # 24h set returns london (already in Tier 1) and tokyo (new).
        mock_tracker.get_recent_active_locations.return_value = [
            ("london", 1700000000),
            ("tokyo", 1700000500),
        ]
        # Redis serves display strings for both IDs.
        mock_redis.get.side_effect = lambda key: {
            "loc_display:london": b"London, England, UK",
            "loc_display:tokyo": b"Tokyo, Tokyo, Japan",
        }.get(key)

        locs = warmer.get_locations_to_warm()

        assert "London, England, UK" in locs
        assert "Tokyo, Tokyo, Japan" in locs
        # Tier 2 should contain only tokyo — london was excluded.
        assert len(warmer.last_tier2) == 1
        assert warmer.last_tier2[0]["id"] == "tokyo"

    def test_tier2_respects_cap(self, warmer, mock_tracker, mock_redis, enable_tracking, monkeypatch):
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER1_SIZE", 0)
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER2_MAX", 2)
        monkeypatch.setattr(warmer, "get_preapproved_locations", lambda: [])
        mock_tracker.get_recent_active_locations.return_value = [
            ("a", 1), ("b", 2), ("c", 3), ("d", 4),
        ]
        mock_redis.get.side_effect = lambda key: f"display-{key.split(':')[-1]}".encode()

        warmer.get_locations_to_warm()

        assert len(warmer.last_tier2) == 2

    def test_tier2_skips_ids_without_display_string(self, warmer, mock_tracker, mock_redis, enable_tracking, monkeypatch):
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER1_SIZE", 0)
        monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER2_MAX", 10)
        monkeypatch.setattr(warmer, "get_preapproved_locations", lambda: [])
        mock_tracker.get_recent_active_locations.return_value = [("ghost", 1)]
        mock_redis.get.return_value = None

        warmer.get_locations_to_warm()

        assert warmer.last_tier2 == []


class TestClassifyLocationTier:
    def test_returns_tier1_for_snapshot_match(self, warmer):
        warmer.last_tier1 = [{"id": "london", "display": "London, England, UK", "score": 50.0}]
        warmer.last_tier2 = []

        assert warmer.classify_location_tier("London, England, UK") == "tier1"

    def test_returns_tier2_for_snapshot_match(self, warmer):
        warmer.last_tier1 = []
        warmer.last_tier2 = [{"id": "tokyo", "display": "Tokyo, Tokyo, Japan", "last_seen": 1}]

        assert warmer.classify_location_tier("Tokyo, Tokyo, Japan") == "tier2"

    def test_returns_cold_for_unknown(self, warmer):
        warmer.last_tier1 = []
        warmer.last_tier2 = []

        assert warmer.classify_location_tier("Atlantis, , ") == "cold"

    def test_returns_cold_for_empty(self, warmer):
        assert warmer.classify_location_tier("") == "cold"
