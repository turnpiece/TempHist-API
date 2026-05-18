"""Unit tests for LocationUsageTracker selection methods."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from tracking.usage import LocationUsageTracker


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
