"""Unit tests for scripts/merge_duplicate_locations.py (API#103 cleanup)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from scripts.merge_duplicate_locations import RECENT_24H_KEY, merge_location
from tracking.usage import LocationUsageTracker


def _daily_key(offset: int = 0) -> str:
    day = datetime.now(timezone.utc) - timedelta(days=offset)
    return f"selections:{day.strftime('%Y%m%d')}"


@pytest.fixture
def mock_redis():
    return MagicMock()


class TestMergeLocationDryRun:
    def test_reports_selection_scores_without_writing(self, mock_redis):
        today_key = _daily_key(0)

        def zscore(key, member):
            return 3.0 if (key, member) == (today_key, "bad_id") else None

        mock_redis.zscore.side_effect = zscore
        mock_redis.exists.return_value = False

        summary = merge_location(mock_redis, "bad_id", "good_id", days=2, dry_run=True)

        assert summary["selections_merged"] == {today_key: 3.0}
        mock_redis.zincrby.assert_not_called()
        mock_redis.zrem.assert_not_called()
        mock_redis.zadd.assert_not_called()
        mock_redis.delete.assert_not_called()

    def test_reports_nothing_found_when_bad_id_absent_everywhere(self, mock_redis):
        mock_redis.zscore.return_value = None
        mock_redis.exists.return_value = False

        summary = merge_location(mock_redis, "bad_id", "good_id", days=2, dry_run=True)

        assert summary["selections_merged"] == {}
        assert summary["recent_24h_merged"] is False
        assert summary["geo_index_removed"] is False
        assert summary["meta_removed"] is False
        assert summary["display_removed"] is False


class TestMergeLocationExecute:
    def test_merges_daily_selection_scores_into_canonical(self, mock_redis):
        today_key = _daily_key(0)

        def zscore(key, member):
            return 3.0 if (key, member) == (today_key, "bad_id") else None

        mock_redis.zscore.side_effect = zscore
        mock_redis.exists.return_value = False

        merge_location(mock_redis, "bad_id", "good_id", days=2, dry_run=False)

        mock_redis.zincrby.assert_called_once_with(today_key, 3.0, "good_id")
        mock_redis.zrem.assert_any_call(today_key, "bad_id")

    def test_merges_recent_24h_score_keeping_the_max(self, mock_redis):
        def zscore(key, member):
            if key == RECENT_24H_KEY and member == "bad_id":
                return 500.0
            if key == RECENT_24H_KEY and member == "good_id":
                return 200.0
            return None

        mock_redis.zscore.side_effect = zscore
        mock_redis.exists.return_value = False

        merge_location(mock_redis, "bad_id", "good_id", days=1, dry_run=False)

        mock_redis.zadd.assert_called_once_with(RECENT_24H_KEY, {"good_id": 500.0})
        mock_redis.zrem.assert_any_call(RECENT_24H_KEY, "bad_id")

    def test_keeps_canonical_recency_when_it_is_already_more_recent(self, mock_redis):
        def zscore(key, member):
            if key == RECENT_24H_KEY and member == "bad_id":
                return 100.0
            if key == RECENT_24H_KEY and member == "good_id":
                return 900.0
            return None

        mock_redis.zscore.side_effect = zscore
        mock_redis.exists.return_value = False

        merge_location(mock_redis, "bad_id", "good_id", days=1, dry_run=False)

        mock_redis.zadd.assert_not_called()
        mock_redis.zrem.assert_any_call(RECENT_24H_KEY, "bad_id")

    def test_removes_bad_id_from_geo_index(self, mock_redis):
        def zscore(key, member):
            if key == LocationUsageTracker.geo_index_key and member == "bad_id":
                return 1.0
            return None

        mock_redis.zscore.side_effect = zscore
        mock_redis.exists.return_value = False

        merge_location(mock_redis, "bad_id", "good_id", days=1, dry_run=False)

        mock_redis.zrem.assert_any_call(LocationUsageTracker.geo_index_key, "bad_id")

    def test_deletes_meta_and_display_keys_for_bad_id(self, mock_redis):
        mock_redis.zscore.return_value = None
        mock_redis.exists.side_effect = lambda key: key in {"loc_meta:bad_id", "loc_display:bad_id"}

        merge_location(mock_redis, "bad_id", "good_id", days=1, dry_run=False)

        mock_redis.delete.assert_any_call("loc_meta:bad_id")
        mock_redis.delete.assert_any_call("loc_display:bad_id")

    def test_no_writes_when_bad_id_absent_everywhere(self, mock_redis):
        mock_redis.zscore.return_value = None
        mock_redis.exists.return_value = False

        merge_location(mock_redis, "bad_id", "good_id", days=1, dry_run=False)

        mock_redis.zincrby.assert_not_called()
        mock_redis.zrem.assert_not_called()
        mock_redis.zadd.assert_not_called()
        mock_redis.delete.assert_not_called()
