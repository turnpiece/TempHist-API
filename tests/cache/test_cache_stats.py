"""Tests for CacheStats location normalization (Phase 1)."""

from unittest.mock import MagicMock

import pytest

from cache.warming import CacheStats


@pytest.fixture
def stats():
    return CacheStats(MagicMock())


class TestTrackCacheRequestLocationNormalization:
    def test_case_variants_share_one_bucket(self, stats):
        stats.track_cache_request("key1", hit=True, endpoint="/weather", location="London")
        stats.track_cache_request("key2", hit=False, endpoint="/weather", location="london")

        location_stats = stats.stats["location_stats"]
        assert set(location_stats.keys()) == {"london"}
        assert location_stats["london"]["hits"] == 1
        assert location_stats["london"]["misses"] == 1
        assert location_stats["london"]["total"] == 2

    def test_whitespace_and_commas_normalized(self, stats):
        stats.track_cache_request("key1", hit=True, location="London, UK")
        stats.track_cache_request("key2", hit=True, location="london, uk")

        location_stats = stats.stats["location_stats"]
        assert set(location_stats.keys()) == {"london__uk"}
        assert location_stats["london__uk"]["total"] == 2

    def test_different_locations_stay_separate(self, stats):
        stats.track_cache_request("key1", hit=True, location="London")
        stats.track_cache_request("key2", hit=True, location="Paris")

        assert set(stats.stats["location_stats"].keys()) == {"london", "paris"}


class TestGetLocationStatsMergesLegacyBuckets:
    def test_merges_pre_normalization_buckets_on_read(self, stats):
        # Simulate legacy stats written before normalization was applied.
        stats.stats["location_stats"] = {
            "London": {"hits": 10, "misses": 2, "errors": 0, "total": 12},
            "london": {"hits": 3, "misses": 1, "errors": 0, "total": 4},
        }

        result = stats.get_location_stats()

        assert set(result.keys()) == {"london"}
        assert result["london"]["total_requests"] == 16
        assert result["london"]["cache_hits"] == 13
        assert result["london"]["cache_misses"] == 3
        assert result["london"]["hit_rate"] == pytest.approx(13 / 16)

    def test_long_form_names_remain_separate_for_now(self, stats):
        stats.track_cache_request("key1", hit=True, location="London")
        stats.track_cache_request(
            "key2",
            hit=True,
            location="Greater London, England, United Kingdom",
        )

        result = stats.get_location_stats()

        assert "london" in result
        assert "greater_london__england__united_kingdom" in result
        assert result["london"]["total_requests"] == 1
        assert result["greater_london__england__united_kingdom"]["total_requests"] == 1
