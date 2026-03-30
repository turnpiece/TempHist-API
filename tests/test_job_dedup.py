"""Tests for job deduplication and backfill cooldown logic.

Verifies that:
- Jobs are deduplicated across all statuses (PENDING, PROCESSING, READY, ERROR)
- Completed/failed jobs keep their dedup keys alive with appropriate TTLs
- The router-level backfill cooldown prevents rapid re-enqueuing
"""

import hashlib
import json
import time
import pytest
from unittest.mock import MagicMock, patch, call

from cache_utils import JobManager, JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> JobManager:
    """Create a JobManager backed by a mock Redis client."""
    mock_redis = MagicMock()
    mock_redis.llen.return_value = 0  # queue is empty by default
    manager = JobManager(mock_redis)
    return manager


def _params_hash(params: dict) -> str:
    return hashlib.sha256(str(params).encode()).hexdigest()[:16]


TEST_PARAMS = {
    "scope": "weekly",
    "slug": "singapore__singapore",
    "identifier": "03-26",
    "year": 2024,
    "location": "Singapore, Singapore",
}

TEST_JOB_TYPE = "record_computation"
TEST_HASH = _params_hash(TEST_PARAMS)
TEST_DEDUP_KEY = f"job:dedup:{TEST_JOB_TYPE}:{TEST_HASH}"


# ---------------------------------------------------------------------------
# Dedup: PENDING / PROCESSING (existing behaviour, kept as regression test)
# ---------------------------------------------------------------------------

class TestDedupPendingProcessing:

    @pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.PROCESSING])
    def test_returns_existing_job_id(self, status):
        mgr = _make_manager()
        existing_id = "record_computation_111_abcd1234"
        mgr.redis.get.side_effect = lambda key: (
            existing_id.encode() if key == TEST_DEDUP_KEY
            else json.dumps({"status": status}).encode()
        )

        result = mgr.create_job(TEST_JOB_TYPE, TEST_PARAMS)

        assert result == existing_id
        # Should NOT push to queue
        mgr.redis.lpush.assert_not_called()


# ---------------------------------------------------------------------------
# Dedup: READY (completed jobs should block re-enqueue)
# ---------------------------------------------------------------------------

class TestDedupReady:

    def test_completed_job_blocks_reenqueue(self):
        """A recently completed job should prevent creating a duplicate."""
        mgr = _make_manager()
        existing_id = "record_computation_222_abcd1234"
        mgr.redis.get.side_effect = lambda key: (
            existing_id.encode() if key == TEST_DEDUP_KEY
            else json.dumps({"status": JobStatus.READY}).encode()
        )

        result = mgr.create_job(TEST_JOB_TYPE, TEST_PARAMS)

        assert result == existing_id
        mgr.redis.lpush.assert_not_called()


# ---------------------------------------------------------------------------
# Dedup: ERROR (failed jobs should block re-enqueue during cooldown)
# ---------------------------------------------------------------------------

class TestDedupError:

    def test_failed_job_blocks_reenqueue(self):
        """A recently failed job should prevent creating a duplicate."""
        mgr = _make_manager()
        existing_id = "record_computation_333_abcd1234"
        mgr.redis.get.side_effect = lambda key: (
            existing_id.encode() if key == TEST_DEDUP_KEY
            else json.dumps({"status": JobStatus.ERROR, "error": "No data found for year 1982"}).encode()
        )

        result = mgr.create_job(TEST_JOB_TYPE, TEST_PARAMS)

        assert result == existing_id
        mgr.redis.lpush.assert_not_called()


# ---------------------------------------------------------------------------
# Dedup key NOT deleted on completion — kept alive with TTL
# ---------------------------------------------------------------------------

class TestDedupKeyRetainedOnCompletion:

    def test_ready_status_refreshes_dedup_key(self):
        """update_job_status(READY) should refresh dedup key, not delete it."""
        mgr = _make_manager()
        job_id = "record_computation_444_abcd1234"
        job_data = {
            "id": job_id,
            "type": TEST_JOB_TYPE,
            "status": JobStatus.PROCESSING,
            "params": TEST_PARAMS,
        }
        mgr.redis.get.return_value = json.dumps(job_data).encode()

        mgr.update_job_status(job_id, JobStatus.READY, result={"data": "ok"})

        # Should NOT delete the dedup key
        mgr.redis.delete.assert_not_called()
        # Should set the dedup key with a 300s TTL
        dedup_key = f"job:dedup:{TEST_JOB_TYPE}:{TEST_HASH}"
        mgr.redis.setex.assert_any_call(dedup_key, 300, job_id)

    def test_error_status_refreshes_dedup_key_with_shorter_ttl(self):
        """update_job_status(ERROR) should refresh dedup key with 120s TTL."""
        mgr = _make_manager()
        job_id = "record_computation_555_abcd1234"
        job_data = {
            "id": job_id,
            "type": TEST_JOB_TYPE,
            "status": JobStatus.PROCESSING,
            "params": TEST_PARAMS,
        }
        mgr.redis.get.return_value = json.dumps(job_data).encode()

        mgr.update_job_status(job_id, JobStatus.ERROR, error="No data found for year 1982")

        mgr.redis.delete.assert_not_called()
        dedup_key = f"job:dedup:{TEST_JOB_TYPE}:{TEST_HASH}"
        mgr.redis.setex.assert_any_call(dedup_key, 120, job_id)


# ---------------------------------------------------------------------------
# New job is created when no dedup key exists
# ---------------------------------------------------------------------------

class TestNewJobCreation:

    def test_creates_job_when_no_dedup_key(self):
        """Without a dedup key, a new job should be created and pushed to queue."""
        mgr = _make_manager()
        mgr.redis.get.return_value = None  # No dedup key, no existing job

        job_id = mgr.create_job(TEST_JOB_TYPE, TEST_PARAMS)

        assert job_id.startswith("record_computation_")
        mgr.redis.lpush.assert_called_once()
        # Dedup key should be set
        dedup_calls = [c for c in mgr.redis.setex.call_args_list if "dedup" in str(c)]
        assert len(dedup_calls) >= 1


# ---------------------------------------------------------------------------
# Backfill cooldown in _enqueue_backfill_job
# ---------------------------------------------------------------------------

class TestBackfillCooldown:

    @patch("routers.v1_records.get_job_manager")
    def test_first_call_enqueues(self, mock_get_jm):
        """First backfill request should set cooldown and create job."""
        from routers.v1_records import _enqueue_backfill_job

        mock_jm = MagicMock()
        mock_get_jm.return_value = mock_jm
        # Redis SET NX returns True (key was newly created)
        mock_jm.redis.set.return_value = True

        _enqueue_backfill_job("weekly", "Singapore, Singapore", "03-26", 2024)

        mock_jm.redis.set.assert_called_once()
        mock_jm.create_job.assert_called_once()

    @patch("routers.v1_records.get_job_manager")
    def test_second_call_within_cooldown_skips(self, mock_get_jm):
        """Repeated backfill request within cooldown should skip enqueue."""
        from routers.v1_records import _enqueue_backfill_job

        mock_jm = MagicMock()
        mock_get_jm.return_value = mock_jm
        # Redis SET NX returns False (key already exists — cooldown active)
        mock_jm.redis.set.return_value = False

        _enqueue_backfill_job("weekly", "Singapore, Singapore", "03-26", 2024)

        mock_jm.create_job.assert_not_called()

    @patch("routers.v1_records.get_job_manager")
    def test_cooldown_key_format(self, mock_get_jm):
        """Cooldown key should encode period, slug, identifier, and year."""
        from routers.v1_records import _enqueue_backfill_job

        mock_jm = MagicMock()
        mock_get_jm.return_value = mock_jm
        mock_jm.redis.set.return_value = True

        _enqueue_backfill_job("yearly", "London, United Kingdom", "06-15", 1990)

        set_call = mock_jm.redis.set.call_args
        cooldown_key = set_call[0][0]
        assert cooldown_key.startswith("backfill:cd:")
        assert "yearly" in cooldown_key
        assert "1990" in cooldown_key
        assert "06-15" in cooldown_key
        # Should use nx=True and ex=60
        assert set_call[1]["nx"] is True
        assert set_call[1]["ex"] == 60

    @patch("routers.v1_records.get_job_manager")
    def test_no_job_manager_does_not_raise(self, mock_get_jm):
        """If job manager is unavailable, should silently return."""
        from routers.v1_records import _enqueue_backfill_job

        mock_get_jm.return_value = None

        # Should not raise
        _enqueue_backfill_job("weekly", "Singapore", "03-26", 2024)
