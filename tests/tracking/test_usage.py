"""Unit tests for LocationUsageTracker selection methods."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from tracking.usage import RECENT_24H_KEY, LocationUsageTracker


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def tracker(mock_redis):
    return LocationUsageTracker(mock_redis)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# record_selection
# ---------------------------------------------------------------------------


class TestRecordSelection:
    def test_increments_daily_key(self, tracker, mock_redis):
        mock_redis.exists.return_value = 0  # dedup key absent

        tracker.record_selection("london", "user1")

        daily_key = f"selections:{_today()}"
        mock_redis.zincrby.assert_called_once_with(daily_key, 1, "london")

    def test_dedup_same_user_same_day(self, tracker, mock_redis):
        """Second call for same user+location+day is a no-op."""
        mock_redis.exists.return_value = 1  # dedup key present

        tracker.record_selection("london", "user1")

        mock_redis.zincrby.assert_not_called()

    def test_different_users_both_count(self, tracker, mock_redis):
        """Different users can each contribute one count."""
        mock_redis.exists.return_value = 0

        tracker.record_selection("london", "user1")
        tracker.record_selection("london", "user2")

        assert mock_redis.zincrby.call_count == 2

    def test_sets_dedup_key_with_ttl(self, tracker, mock_redis):
        mock_redis.exists.return_value = 0

        tracker.record_selection("london", "user1")

        dedup_key = f"selection_dedup:user1:london:{_today()}"
        mock_redis.set.assert_called_once_with(dedup_key, 1, ex=25 * 3600)

    def test_sets_expire_on_daily_key(self, tracker, mock_redis):
        from config import POPULARITY_WINDOW_DAYS

        mock_redis.exists.return_value = 0

        tracker.record_selection("london", "user1")

        daily_key = f"selections:{_today()}"
        mock_redis.expire.assert_called_with(daily_key, (POPULARITY_WINDOW_DAYS + 5) * 86400)


# ---------------------------------------------------------------------------
# get_popular_from_selections
# ---------------------------------------------------------------------------


class TestGetPopularFromSelections:
    def test_returns_empty_when_no_keys(self, tracker, mock_redis):
        mock_redis.exists.return_value = 0  # no daily keys

        result = tracker.get_popular_from_selections(limit=10, days=30)

        assert result == []
        mock_redis.zunionstore.assert_not_called()

    def test_ranks_correctly(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1  # every key present
        mock_redis.zrevrange.return_value = [
            (b"london", 50.0),
            (b"paris", 30.0),
            (b"new_york", 10.0),
        ]

        result = tracker.get_popular_from_selections(limit=3, days=1)

        assert result[0] == ("london", 50)
        assert result[1] == ("paris", 30)
        assert result[2] == ("new_york", 10)

    def test_cleans_up_tmp_key(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1
        mock_redis.zrevrange.return_value = []

        tracker.get_popular_from_selections(limit=5, days=1)

        mock_redis.delete.assert_called_once_with("selections:union:tmp")

    def test_spans_multiple_days(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1
        mock_redis.zrevrange.return_value = []

        tracker.get_popular_from_selections(limit=5, days=3)

        # 3 daily keys should be passed to zunionstore
        args, kwargs = mock_redis.zunionstore.call_args
        assert len(args[1]) == 3


# ---------------------------------------------------------------------------
# get_total_selections
# ---------------------------------------------------------------------------


class TestGetTotalSelections:
    def test_returns_zero_when_no_keys(self, tracker, mock_redis):
        mock_redis.exists.return_value = 0

        total = tracker.get_total_selections(days=30)

        assert total == 0

    def test_sums_scores(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1
        mock_redis.zrange.return_value = [
            (b"london", 50.0),
            (b"paris", 30.0),
        ]

        total = tracker.get_total_selections(days=1)

        assert total == 80

    def test_cleans_up_tmp_key(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1
        mock_redis.zrange.return_value = []

        tracker.get_total_selections(days=1)

        mock_redis.delete.assert_called_once_with("selections:total:tmp")


# ---------------------------------------------------------------------------
# geo-index: find_nearby_canonical_id / add_to_geo_index
#
# Backs canonicalization/dedup — converging differently-spelled submissions
# of the same physical place ("Chennai" vs "Chennai, Tamil Nadu, India") onto
# a single canonical ID so selection signal isn't fragmented.
# ---------------------------------------------------------------------------


class TestGeoIndex:
    def test_find_nearby_returns_closest_member(self, tracker, mock_redis):
        mock_redis.geosearch.return_value = [b"new_delhi"]

        result = tracker.find_nearby_canonical_id(28.7041, 77.1025, 45)

        assert result == "new_delhi"
        mock_redis.geosearch.assert_called_once_with(
            tracker.geo_index_key,
            longitude=77.1025,
            latitude=28.7041,
            radius=45,
            unit="km",
            sort="ASC",
            count=1,
        )

    def test_find_nearby_returns_none_when_empty(self, tracker, mock_redis):
        mock_redis.geosearch.return_value = []

        result = tracker.find_nearby_canonical_id(0.0, 0.0, 45)

        assert result is None

    def test_find_nearby_decodes_string_members(self, tracker, mock_redis):
        mock_redis.geosearch.return_value = ["new_delhi"]

        result = tracker.find_nearby_canonical_id(28.7041, 77.1025, 45)

        assert result == "new_delhi"

    def test_find_nearby_returns_none_on_redis_error(self, tracker, mock_redis):
        mock_redis.geosearch.side_effect = Exception("boom")

        result = tracker.find_nearby_canonical_id(28.7041, 77.1025, 45)

        assert result is None

    def test_add_to_geo_index_calls_geoadd_with_lon_lat_member(self, tracker, mock_redis):
        tracker.add_to_geo_index("chennai", 13.0827, 80.2707)

        mock_redis.geoadd.assert_called_once_with(tracker.geo_index_key, [80.2707, 13.0827, "chennai"])

    def test_add_to_geo_index_swallows_redis_error(self, tracker, mock_redis):
        mock_redis.geoadd.side_effect = Exception("boom")

        # Should not raise
        tracker.add_to_geo_index("chennai", 13.0827, 80.2707)

    def test_seed_geo_index_writes_flat_lon_lat_member_list(self, tracker, mock_redis):
        seeded = tracker.seed_geo_index([("london", 51.5072, -0.1276), ("dublin", 53.3498, -6.2603)])

        assert seeded == 2
        mock_redis.geoadd.assert_called_once_with(
            tracker.geo_index_key,
            [-0.1276, 51.5072, "london", -6.2603, 53.3498, "dublin"],
        )

    def test_seed_geo_index_empty_is_noop(self, tracker, mock_redis):
        assert tracker.seed_geo_index([]) == 0
        mock_redis.geoadd.assert_not_called()

    def test_seed_geo_index_swallows_redis_error(self, tracker, mock_redis):
        mock_redis.geoadd.side_effect = Exception("boom")

        # Should not raise; returns 0 on failure
        assert tracker.seed_geo_index([("london", 51.5072, -0.1276)]) == 0


# ---------------------------------------------------------------------------
# Weighted popular scoring + 24h recency set (issue #52)
# ---------------------------------------------------------------------------


class TestGetWeightedPopular:
    def test_returns_empty_when_no_keys(self, tracker, mock_redis):
        mock_redis.exists.return_value = 0  # no daily keys for any window

        result = tracker.get_weighted_popular(limit=10)

        assert result == []
        mock_redis.zunionstore.assert_not_called()

    def test_uses_recency_weights(self, tracker, mock_redis):
        """Final zunionstore must apply weights [0.5, 0.3, 0.2] via a {key: weight} dict."""
        mock_redis.exists.return_value = 1
        mock_redis.zrevrange.return_value = []

        tracker.get_weighted_popular(limit=10)

        # The final call is the one whose second arg is a dict (weighted union).
        weighted_call = next(
            call for call in mock_redis.zunionstore.call_args_list if isinstance(call.args[1], dict)
        )
        weights_by_window = {k: v for k, v in weighted_call.args[1].items()}
        assert sorted(weights_by_window.values()) == [0.2, 0.3, 0.5]
        assert weighted_call.kwargs["aggregate"] == "SUM"

    def test_ranks_by_weighted_score(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1
        mock_redis.zrevrange.return_value = [
            (b"london", 50.0),
            (b"paris", 20.0),
        ]

        result = tracker.get_weighted_popular(limit=2)

        assert result[0] == ("london", 50.0)
        assert result[1] == ("paris", 20.0)

    def test_cleans_up_tmp_keys(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1
        mock_redis.zrevrange.return_value = []

        tracker.get_weighted_popular(limit=5)

        # All four tmp keys should be deleted (3 window unions + final weighted union).
        mock_redis.delete.assert_called_once()
        deleted_keys = mock_redis.delete.call_args[0]
        assert "selections:weighted:tmp" in deleted_keys
        assert any("window:7d" in k for k in deleted_keys)
        assert any("window:30d" in k for k in deleted_keys)
        assert any("window:90d" in k for k in deleted_keys)

    def test_zero_limit_short_circuits(self, tracker, mock_redis):
        result = tracker.get_weighted_popular(limit=0)

        assert result == []
        mock_redis.zunionstore.assert_not_called()


class TestRecent24hSet:
    def test_record_selection_updates_24h_set(self, tracker, mock_redis):
        mock_redis.exists.return_value = 0

        tracker.record_selection("london", "user1")

        # zadd call shape: zadd(key, {member: score})
        mock_redis.zadd.assert_called_once()
        args, _ = mock_redis.zadd.call_args
        assert args[0] == RECENT_24H_KEY
        assert "london" in args[1]
        # Score should be epoch seconds — a positive int reasonably close to now.
        assert isinstance(args[1]["london"], int)
        assert args[1]["london"] > 0

    def test_record_selection_swallows_zadd_error(self, tracker, mock_redis):
        mock_redis.exists.return_value = 0
        mock_redis.zadd.side_effect = Exception("boom")

        # Must not raise — the primary ZINCRBY still succeeded.
        tracker.record_selection("london", "user1")

    def test_get_recent_active_locations_prunes_first(self, tracker, mock_redis):
        mock_redis.zrevrange.return_value = [(b"london", 1700000000.0)]

        tracker.get_recent_active_locations()

        mock_redis.zremrangebyscore.assert_called_once()
        args = mock_redis.zremrangebyscore.call_args[0]
        assert args[0] == RECENT_24H_KEY
        assert args[1] == "-inf"
        # cutoff is now - 24h, so a sensible positive number.
        assert args[2] > 0

    def test_get_recent_active_locations_returns_id_and_ts(self, tracker, mock_redis):
        mock_redis.zrevrange.return_value = [
            (b"london", 1700000000.0),
            ("paris", 1699999000.0),  # already a str — also supported
        ]

        result = tracker.get_recent_active_locations()

        assert result == [("london", 1700000000), ("paris", 1699999000)]

    def test_get_recent_active_locations_swallows_redis_error(self, tracker, mock_redis):
        mock_redis.zrevrange.side_effect = Exception("boom")

        result = tracker.get_recent_active_locations()

        assert result == []


class TestGetWeightedPopularDisplayStrings:
    def test_returns_id_display_score_tuples(self, tracker, mock_redis):
        mock_redis.exists.return_value = 1
        mock_redis.zrevrange.return_value = [
            (b"london", 50.0),
            (b"paris", 20.0),
        ]
        # loc_display:london present, loc_display:paris missing → paris skipped
        mock_redis.get.side_effect = lambda key: b"London, England, UK" if key == "loc_display:london" else None

        result = tracker.get_weighted_popular_display_strings(limit=10)

        assert result == [("london", "London, England, UK", 50.0)]
