"""Tests for JobWorker helper methods extracted in P1-144."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis

from job_worker import JobWorker
from jobs.manager import JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker() -> JobWorker:
    mock_redis = MagicMock()
    mock_redis.pipeline.return_value = MagicMock()
    return JobWorker(mock_redis)


# ---------------------------------------------------------------------------
# _dispatch_job_coro
# ---------------------------------------------------------------------------


class TestDispatchJobCoro:
    def test_record_computation_returns_coro(self):
        w = _make_worker()
        coro = w._dispatch_job_coro(job_type="record_computation", params={})
        assert asyncio.iscoroutine(coro)
        coro.close()

    def test_cache_warming_returns_coro(self):
        w = _make_worker()
        coro = w._dispatch_job_coro(job_type="cache_warming", params={})
        assert asyncio.iscoroutine(coro)
        coro.close()

    def test_unknown_type_raises_value_error(self):
        w = _make_worker()
        with pytest.raises(ValueError, match="Unknown job type"):
            w._dispatch_job_coro(job_type="bogus", params={})


# ---------------------------------------------------------------------------
# _run_job_with_timeout
# ---------------------------------------------------------------------------


class TestRunJobWithTimeout:
    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        w = _make_worker()
        expected = {"data": "ok"}
        with patch.object(w, "_dispatch_job_coro", return_value=_coro(expected)):
            result = await w._run_job_with_timeout(job_id="j1", job_type="record_computation", params={})
        assert result == expected

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self):
        w = _make_worker()
        with patch.object(w, "_dispatch_job_coro", return_value=_slow_coro()):
            with patch("job_worker.MAX_JOB_EXECUTION_SECONDS", 0):
                with pytest.raises(TimeoutError, match="execution limit"):
                    await w._run_job_with_timeout(job_id="j1", job_type="record_computation", params={})

    @pytest.mark.asyncio
    async def test_value_error_reraises(self):
        w = _make_worker()
        with patch.object(w, "_dispatch_job_coro", return_value=_raising_coro(ValueError("bad param"))):
            with pytest.raises(ValueError, match="bad param"):
                await w._run_job_with_timeout(job_id="j1", job_type="record_computation", params={})

    @pytest.mark.asyncio
    async def test_unexpected_error_reraises(self):
        w = _make_worker()
        with patch.object(w, "_dispatch_job_coro", return_value=_raising_coro(RuntimeError("boom"))):
            with pytest.raises(RuntimeError, match="boom"):
                await w._run_job_with_timeout(job_id="j1", job_type="record_computation", params={})


# ---------------------------------------------------------------------------
# _db_precheck_year
# ---------------------------------------------------------------------------


class TestDbPrecheckYear:
    @pytest.mark.asyncio
    async def test_returns_true_when_data_exists(self):
        w = _make_worker()
        mock_store = AsyncMock()
        mock_store.fetch = AsyncMock(return_value={"2024-01-01": [1.0]})
        with _patch_precheck_imports(mock_store):
            result = await w._db_precheck_year(
                location="London, UK", scope="weekly", identifier="01-01", year=2023, slug="london__uk"
            )
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_and_sets_skip_key_when_no_data(self):
        w = _make_worker()
        mock_store = AsyncMock()
        mock_store.fetch = AsyncMock(return_value={})
        with _patch_precheck_imports(mock_store):
            result = await w._db_precheck_year(
                location="London, UK", scope="weekly", identifier="01-01", year=2023, slug="london__uk"
            )
        assert result is False
        w.redis.setex.assert_called_once()
        call_args = w.redis.setex.call_args[0]
        assert call_args[0].startswith("backfill:skip:")
        assert call_args[1] == 86400

    @pytest.mark.asyncio
    async def test_exception_returns_true_fail_open(self):
        w = _make_worker()
        mock_store = AsyncMock()
        mock_store.fetch = AsyncMock(side_effect=Exception("db down"))
        with _patch_precheck_imports(mock_store):
            result = await w._db_precheck_year(
                location="London, UK", scope="weekly", identifier="01-01", year=2023, slug="london__uk"
            )
        assert result is True


# ---------------------------------------------------------------------------
# _evaluate_single_year_coverage
# ---------------------------------------------------------------------------


class TestEvaluateSingleYearCoverage:
    def test_current_year_low_coverage_returns_true(self):
        w = _make_worker()
        details = [{"year": 2026, "coverage_ratio": 0.5}]
        assert w._evaluate_single_year_coverage(year=2026, current_year=2026, coverage_details=details) is True

    def test_current_year_high_coverage_returns_false(self):
        w = _make_worker()
        details = [{"year": 2026, "coverage_ratio": 0.9}]
        assert w._evaluate_single_year_coverage(year=2026, current_year=2026, coverage_details=details) is False

    def test_historical_year_always_false(self):
        w = _make_worker()
        details = [{"year": 2020, "coverage_ratio": 0.1}]
        assert w._evaluate_single_year_coverage(year=2020, current_year=2026, coverage_details=details) is False

    def test_no_coverage_detail_returns_false(self):
        w = _make_worker()
        assert w._evaluate_single_year_coverage(year=2026, current_year=2026, coverage_details=[]) is False

    def test_exactly_at_threshold_returns_false(self):
        w = _make_worker()
        details = [{"year": 2026, "coverage_ratio": 0.8}]
        assert w._evaluate_single_year_coverage(year=2026, current_year=2026, coverage_details=details) is False


# ---------------------------------------------------------------------------
# _cache_single_year_record
# ---------------------------------------------------------------------------


class TestCacheSingleYearRecord:
    def test_writes_both_keys(self):
        w = _make_worker()
        w._cache_single_year_record(
            year_key="rec:y:2020",
            etag_key="etag:y:2020",
            year_record={"avg": 20.0},
            etag="abc123",
            ttl=3600,
        )
        assert w.redis.setex.call_count == 2
        calls = {c[0][0]: c[0][1] for c in w.redis.setex.call_args_list}
        assert calls["rec:y:2020"] == 3600
        assert calls["etag:y:2020"] == 3600

    def test_redis_error_reraises(self):
        w = _make_worker()
        w.redis.setex.side_effect = redis.RedisError("conn refused")
        with pytest.raises(redis.RedisError):
            w._cache_single_year_record(
                year_key="k", etag_key="ek", year_record={}, etag="e", ttl=60
            )

    def test_type_error_reraises(self):
        w = _make_worker()
        with pytest.raises(TypeError):
            # json.dumps will raise TypeError on a non-serialisable value
            w._cache_single_year_record(
                year_key="k", etag_key="ek", year_record={"v": object()}, etag="e", ttl=60
            )


# ---------------------------------------------------------------------------
# _build_pipeline_cache_all_years
# ---------------------------------------------------------------------------


class TestBuildPipelineCacheAllYears:
    def _make_records(self, years):
        return {y: {"avg": float(y)} for y in years}

    def test_caches_historical_years(self):
        w = _make_worker()
        pipeline = MagicMock()
        w.redis.pipeline.return_value = pipeline
        records = self._make_records([2020, 2021])

        with _patch_pipeline_imports():
            w._build_pipeline_cache_all_years(
                scope="weekly",
                slug="london__uk",
                identifier="01-01",
                per_year_records=records,
                current_year=2026,
                coverage_by_year={},
            )

        pipeline.execute.assert_called_once()
        assert pipeline.setex.call_count == 4  # data + etag per year

    def test_skips_current_year_low_coverage(self):
        w = _make_worker()
        pipeline = MagicMock()
        w.redis.pipeline.return_value = pipeline
        records = self._make_records([2025, 2026])
        coverage = {2026: {"coverage_ratio": 0.5}}

        with _patch_pipeline_imports():
            w._build_pipeline_cache_all_years(
                scope="weekly",
                slug="london__uk",
                identifier="01-01",
                per_year_records=records,
                current_year=2026,
                coverage_by_year=coverage,
            )

        # Only 2025 should be cached (2 setex calls: data + etag)
        assert pipeline.setex.call_count == 2

    def test_includes_current_year_high_coverage(self):
        w = _make_worker()
        pipeline = MagicMock()
        w.redis.pipeline.return_value = pipeline
        records = self._make_records([2025, 2026])
        coverage = {2026: {"coverage_ratio": 0.9}}

        with _patch_pipeline_imports():
            w._build_pipeline_cache_all_years(
                scope="weekly",
                slug="london__uk",
                identifier="01-01",
                per_year_records=records,
                current_year=2026,
                coverage_by_year=coverage,
            )

        assert pipeline.setex.call_count == 4

    def test_redis_error_swallowed(self):
        w = _make_worker()
        pipeline = MagicMock()
        pipeline.execute.side_effect = redis.RedisError("gone")
        w.redis.pipeline.return_value = pipeline
        records = self._make_records([2020])

        with _patch_pipeline_imports():
            # Must not raise
            w._build_pipeline_cache_all_years(
                scope="weekly",
                slug="london__uk",
                identifier="01-01",
                per_year_records=records,
                current_year=2026,
                coverage_by_year={},
            )


# ---------------------------------------------------------------------------
# _calculate_warming_summary
# ---------------------------------------------------------------------------


class TestCalculateWarmingSummary:
    def test_counts_endpoints_and_errors(self):
        w = _make_worker()
        results = {
            "london": {"warmed_endpoints": ["/a", "/b"], "errors": ["/c"]},
            "paris": {"warmed_endpoints": ["/d"], "errors": []},
        }
        summary = w._calculate_warming_summary(location_results=results)
        assert summary["total_locations"] == 2
        assert summary["successful_locations"] == 2
        assert summary["total_endpoints_warmed"] == 3
        assert summary["total_errors"] == 1

    def test_skips_non_dict_entries(self):
        w = _make_worker()
        results = {"all_locations": "some-string-result", "paris": {"warmed_endpoints": ["/d"], "errors": []}}
        summary = w._calculate_warming_summary(location_results=results)
        assert summary["total_locations"] == 2
        assert summary["successful_locations"] == 1

    def test_empty_results(self):
        w = _make_worker()
        summary = w._calculate_warming_summary(location_results={})
        assert summary == {
            "total_locations": 0,
            "successful_locations": 0,
            "total_endpoints_warmed": 0,
            "total_errors": 0,
        }

    def test_location_with_no_warmed_endpoints_not_counted_as_successful(self):
        w = _make_worker()
        results = {"london": {"warmed_endpoints": [], "errors": ["/a"]}}
        summary = w._calculate_warming_summary(location_results=results)
        assert summary["successful_locations"] == 0


# ---------------------------------------------------------------------------
# process_job integration
# ---------------------------------------------------------------------------


class TestProcessJobIntegration:
    @pytest.mark.asyncio
    async def test_job_not_found_removes_from_queue(self):
        w = _make_worker()
        job_manager = MagicMock()
        job_manager.get_job_status.return_value = None
        await w.process_job("j1", job_manager)
        w.redis.lrem.assert_called_once_with(w.job_queue_key, 1, "j1")

    @pytest.mark.asyncio
    async def test_success_updates_ready_and_removes_from_queue(self):
        w = _make_worker()
        job_manager = MagicMock()
        job_manager.get_job_status.return_value = {"type": "record_computation", "params": {}}
        result = {"data": "ok"}
        with patch.object(w, "_run_job_with_timeout", new=AsyncMock(return_value=result)):
            await w.process_job("j1", job_manager)
        job_manager.update_job_status.assert_called_with("j1", JobStatus.READY, result)
        w.redis.lrem.assert_called_with(w.job_queue_key, 1, "j1")

    @pytest.mark.asyncio
    async def test_data_error_updates_error_and_removes_from_queue(self):
        w = _make_worker()
        job_manager = MagicMock()
        job_manager.get_job_status.return_value = {"type": "record_computation", "params": {}}
        with patch.object(w, "_run_job_with_timeout", new=AsyncMock(side_effect=ValueError("bad"))):
            await w.process_job("j1", job_manager)
        update_calls = job_manager.update_job_status.call_args_list
        statuses = [c[0][1] for c in update_calls]
        assert JobStatus.ERROR in statuses
        w.redis.lrem.assert_called_with(w.job_queue_key, 1, "j1")

    @pytest.mark.asyncio
    async def test_redis_error_does_not_update_status(self):
        w = _make_worker()
        job_manager = MagicMock()
        job_manager.update_job_status.side_effect = redis.RedisError("gone")
        job_manager.get_job_status.return_value = {"type": "record_computation", "params": {}}
        # Should not raise; Redis errors are swallowed at the outer level
        await w.process_job("j1", job_manager)
        # After the RedisError on update_job_status(PROCESSING), no further status updates possible
        assert job_manager.update_job_status.call_count == 1


# ---------------------------------------------------------------------------
# process_cache_warming_job integration
# ---------------------------------------------------------------------------


class TestProcessCacheWarmingJobIntegration:
    @pytest.mark.asyncio
    async def test_disabled_returns_skipped(self):
        w = _make_worker()
        with patch("job_worker.JobWorker.process_cache_warming_job", wraps=w.process_cache_warming_job):
            with patch("cache.accessors.get_cache_warmer", return_value=None):
                with patch("cache.warming.CACHE_WARMING_ENABLED", False):
                    result = await w.process_cache_warming_job({})
        assert result["skipped"] is True

    @pytest.mark.asyncio
    async def test_all_type_calls_warm_all_locations(self):
        w = _make_worker()
        cache_warmer = AsyncMock()
        cache_warmer.warm_all_locations = AsyncMock(return_value={"warmed_endpoints": ["/a"], "errors": []})
        with patch("cache.accessors.get_cache_warmer", return_value=cache_warmer):
            with patch("cache.warming.CACHE_WARMING_ENABLED", True):
                result = await w.process_cache_warming_job({"type": "all"})
        cache_warmer.warm_all_locations.assert_called_once()
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_specific_with_no_locations_raises(self):
        w = _make_worker()
        cache_warmer = MagicMock()
        with patch("cache.accessors.get_cache_warmer", return_value=cache_warmer):
            with patch("cache.warming.CACHE_WARMING_ENABLED", True):
                with pytest.raises(ValueError, match="No locations specified"):
                    await w.process_cache_warming_job({"type": "specific", "locations": []})

    @pytest.mark.asyncio
    async def test_unknown_warming_type_raises(self):
        w = _make_worker()
        cache_warmer = MagicMock()
        with patch("cache.accessors.get_cache_warmer", return_value=cache_warmer):
            with patch("cache.warming.CACHE_WARMING_ENABLED", True):
                with pytest.raises(ValueError, match="Unknown warming type"):
                    await w.process_cache_warming_job({"type": "bogus"})

    @pytest.mark.asyncio
    async def test_summary_present_after_completion(self):
        w = _make_worker()
        cache_warmer = AsyncMock()
        cache_warmer.warm_all_locations = AsyncMock(return_value={"warmed_endpoints": ["/a"], "errors": []})
        with patch("cache.accessors.get_cache_warmer", return_value=cache_warmer):
            with patch("cache.warming.CACHE_WARMING_ENABLED", True):
                result = await w.process_cache_warming_job({"type": "all"})
        assert "summary" in result
        assert "total_locations" in result["summary"]


# ---------------------------------------------------------------------------
# Coroutine helpers
# ---------------------------------------------------------------------------


async def _coro(value):
    return value


async def _slow_coro():
    await asyncio.sleep(9999)


async def _raising_coro(exc):
    raise exc


# ---------------------------------------------------------------------------
# Import-patching context managers
# ---------------------------------------------------------------------------


def _patch_precheck_imports(mock_store):
    """Patch the imports used by _db_precheck_year."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with (
            patch("routers.v1_records.parse_identifier", return_value=(1, 1, None)),
            patch("routers.v1_records._resolve_anchor_date", return_value=MagicMock()),
            patch("routers.v1_records.WINDOW_DAYS", {"weekly": 7}),
            patch("utils.daily_temperature_store.get_daily_temperature_store", new=AsyncMock(return_value=mock_store)),
        ):
            yield

    return _ctx()


def _patch_pipeline_imports():
    """Patch the imports used by _build_pipeline_cache_all_years."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with (
            patch("cache.core.ETagGenerator.generate_etag", return_value="etag-abc"),
            patch("cache.core.get_ttl_for_year", return_value=86400),
            patch("cache.keys.rec_key", side_effect=lambda s, sl, i, y: f"rec:{y}"),
            patch("cache.keys.rec_etag_key", side_effect=lambda s, sl, i, y: f"etag:{y}"),
            patch("routers.v1_records._get_ttl_for_current_year", return_value=3600),
        ):
            yield

    return _ctx()
