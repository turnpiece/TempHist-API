"""Regression tests for batched Redis reads in AnalyticsStorage.

Issue: turnpiece/TempHist-API#68 — get_recent_analytics used to issue N
sequential GET calls after LRANGE; it must now issue exactly one MGET.
"""

import json
from unittest.mock import MagicMock

import pytest

from analytics_storage import AnalyticsStorage


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def storage(mock_redis):
    return AnalyticsStorage(mock_redis)


def test_get_recent_analytics_uses_single_mget(storage, mock_redis):
    ids = [b"a1", b"a2", b"a3"]
    mock_redis.lrange.return_value = ids
    mock_redis.mget.return_value = [
        json.dumps({"id": "a1"}),
        json.dumps({"id": "a2"}),
        json.dumps({"id": "a3"}),
    ]

    result = storage.get_recent_analytics(limit=3)

    mock_redis.lrange.assert_called_once_with("analytics_index", 0, 2)
    mock_redis.mget.assert_called_once_with(
        ["analytics_a1", "analytics_a2", "analytics_a3"]
    )
    mock_redis.get.assert_not_called()
    assert [r["id"] for r in result] == ["a1", "a2", "a3"]


def test_get_recent_analytics_skips_missing_records(storage, mock_redis):
    """An id still in the index list but TTL'd out of the per-id key is skipped."""
    mock_redis.lrange.return_value = [b"a1", b"a2", b"a3"]
    mock_redis.mget.return_value = [
        json.dumps({"id": "a1"}),
        None,
        json.dumps({"id": "a3"}),
    ]

    result = storage.get_recent_analytics(limit=3)

    assert [r["id"] for r in result] == ["a1", "a3"]


def test_get_recent_analytics_empty_index_short_circuits(storage, mock_redis):
    mock_redis.lrange.return_value = []

    result = storage.get_recent_analytics(limit=5)

    assert result == []
    mock_redis.mget.assert_not_called()
