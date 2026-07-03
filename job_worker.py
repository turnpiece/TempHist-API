"""
Background job worker for processing async API requests.

This worker processes jobs from the job queue and stores results in Redis cache.
It runs as a background task in the FastAPI application lifecycle.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis

from cache.accessors import get_job_manager
from jobs.manager import CACHE_TTL_JOB, JobStatus

logger = logging.getLogger(__name__)

# Job worker constants
JOB_PROCESSING_BATCH_SIZE = 10  # Maximum jobs to process in one cycle
WORKER_POLL_INTERVAL_SECONDS = 1  # Time between job queue polls
WORKER_HEARTBEAT_INTERVAL_CYCLES = 60  # Update heartbeat every 60 poll cycles (60 seconds)
WORKER_DEEP_CLEANUP_INTERVAL_CYCLES = 300  # Deep queue cleanup every 300 cycles (~5 minutes)
WORKER_DEEP_CLEANUP_BATCH_SIZE = 1000  # Max entries to scan during deep cleanup
WORKER_HEARTBEAT_LOG_INTERVAL_CYCLES = 300  # Log heartbeat every 300 cycles (5 minutes)
MAX_PENDING_AGE_SECONDS = 300  # Jobs pending longer than 5 minutes are expired unprocessed
MAX_JOB_EXECUTION_SECONDS = int(os.environ.get("MAX_JOB_EXECUTION_SECONDS", "600"))  # Hard wall-clock cap per job


class JobWorker:
    """Background worker for processing async jobs."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.running = False
        self.job_queue_key = "job_queue"
        self.processing_key = "processing_jobs"

    async def start(self):
        """Start the job worker."""
        self.running = True
        logger.info("🚀 Job worker started")
        self._refresh_heartbeat()

        poll_count = 0
        try:
            while self.running:
                await self.process_jobs()
                poll_count += 1

                # Update heartbeat periodically
                if poll_count % WORKER_HEARTBEAT_INTERVAL_CYCLES == 0:
                    try:
                        self.redis.setex("worker:heartbeat", 180, datetime.now(timezone.utc).isoformat())
                        # Log heartbeat with queue depth
                        if poll_count % WORKER_HEARTBEAT_LOG_INTERVAL_CYCLES == 0:
                            queue_depth = self.redis.llen(self.job_queue_key)
                            logger.info(f"💓 Worker heartbeat active (poll #{poll_count}, queue depth: {queue_depth})")
                    except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
                        logger.warning(f"⚠️  Redis error updating heartbeat: {e}")

                # Periodic deep cleanup of stale queue entries
                if poll_count % WORKER_DEEP_CLEANUP_INTERVAL_CYCLES == 0:
                    self.cleanup_stale_queue_entries()

                await asyncio.sleep(WORKER_POLL_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"❌ Job worker error: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
        finally:
            logger.info(f"🛑 Job worker stopped (processed {poll_count} poll cycles)")

    def stop(self):
        """Stop the job worker."""
        self.running = False
        logger.info("📴 Job worker stop requested")

    def _refresh_heartbeat(self) -> None:
        """Write a fresh heartbeat to Redis. Called at startup and before each job."""
        try:
            self.redis.setex("worker:heartbeat", 180, datetime.now(timezone.utc).isoformat())
        except Exception as _e:
            logger.debug("Could not update worker heartbeat: %s", _e)

    async def process_jobs(self):
        """Process pending jobs from the queue."""
        try:
            # Get all pending jobs
            job_manager = get_job_manager()

            # Find pending jobs (this is a simple implementation)
            # In production, you might want to use Redis Streams or a proper queue
            pending_jobs = await self.get_pending_jobs()

            if pending_jobs:
                logger.info(f"📋 Found {len(pending_jobs)} pending jobs")

            for job_id in pending_jobs:
                logger.info(f"🔄 Processing job: {job_id}")
                await self.process_job(job_id, job_manager)

        except Exception as e:
            logger.error(f"❌ Error processing jobs: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")

    async def get_pending_jobs(self) -> list:
        """Get list of pending job IDs from the job queue.

        Also cleans up orphaned entries (expired job data) and
        non-pending jobs that are still in the queue.
        """
        try:
            pending_jobs = []
            stale_job_ids = []

            queue_length = self.redis.llen(self.job_queue_key)

            if queue_length > 0:
                for i in range(min(queue_length, JOB_PROCESSING_BATCH_SIZE)):
                    job_id = self.redis.lindex(self.job_queue_key, i)
                    if job_id:
                        if isinstance(job_id, bytes):
                            job_id = job_id.decode("utf-8")

                        job_key = f"job:{job_id}"
                        job_data = self.redis.get(job_key)
                        if job_data:
                            job = json.loads(job_data)
                            job_status = job.get("status")
                            if job_status == JobStatus.PENDING:
                                pending_jobs.append(job_id)
                            else:
                                # Job already processed/failed but still in queue
                                stale_job_ids.append(job_id)
                        else:
                            # Job data expired — orphaned queue entry
                            stale_job_ids.append(job_id)

            # Batch-remove stale entries after iteration to avoid index shifting
            if stale_job_ids:
                for job_id in stale_job_ids:
                    self.redis.lrem(self.job_queue_key, 1, job_id)
                logger.info(f"🧹 Cleaned {len(stale_job_ids)} stale entries from job queue")

            return pending_jobs

        except Exception as e:
            logger.error(f"❌ Error getting pending jobs: {e}")
            import traceback

            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []

    def cleanup_stale_queue_entries(self):
        """Deep scan the queue and remove orphaned/completed entries.

        Called periodically to clean entries beyond the normal batch window.
        """
        try:
            queue_length = self.redis.llen(self.job_queue_key)
            if queue_length == 0:
                return

            scan_size = min(queue_length, WORKER_DEEP_CLEANUP_BATCH_SIZE)
            stale_ids = []

            for i in range(scan_size):
                job_id = self.redis.lindex(self.job_queue_key, i)
                if not job_id:
                    continue
                if isinstance(job_id, bytes):
                    job_id = job_id.decode("utf-8")

                job_data = self.redis.get(f"job:{job_id}")
                if not job_data:
                    stale_ids.append(job_id)
                else:
                    job = json.loads(job_data)
                    status = job.get("status")
                    if status == JobStatus.PROCESSING:
                        updated_at_str = job.get("updated_at")
                        if updated_at_str:
                            age_seconds = (
                                datetime.now(timezone.utc) - datetime.fromisoformat(updated_at_str)
                            ).total_seconds()
                            if age_seconds > 600:
                                logger.warning(
                                    f"⏰ Job {job_id} stuck in PROCESSING for {age_seconds:.0f}s, timing out"
                                )
                                job["status"] = JobStatus.ERROR
                                job["error"] = "Job timed out after 10 minutes in PROCESSING state"
                                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                                self.redis.setex(f"job:{job_id}", CACHE_TTL_JOB, json.dumps(job))
                                stale_ids.append(job_id)
                    elif status == JobStatus.PENDING:
                        created_at_str = job.get("created_at")
                        if created_at_str:
                            age_seconds = (
                                datetime.now(timezone.utc) - datetime.fromisoformat(created_at_str)
                            ).total_seconds()
                            if age_seconds > MAX_PENDING_AGE_SECONDS:
                                logger.warning(f"⏰ Job {job_id} expired after {age_seconds:.0f}s in PENDING state")
                                job["status"] = JobStatus.ERROR
                                job["error"] = "Job expired: too long waiting in queue"
                                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                                self.redis.setex(f"job:{job_id}", CACHE_TTL_JOB, json.dumps(job))
                                stale_ids.append(job_id)
                    else:
                        stale_ids.append(job_id)

            for job_id in stale_ids:
                self.redis.lrem(self.job_queue_key, 1, job_id)

            remaining = self.redis.llen(self.job_queue_key)
            if stale_ids:
                logger.info(
                    f"🧹 Deep cleanup: removed {len(stale_ids)} stale entries "
                    f"(scanned {scan_size}, queue now {remaining})"
                )
            else:
                logger.info(f"🧹 Deep cleanup: queue healthy ({remaining} entries, scanned {scan_size})")

        except Exception as e:
            logger.error(f"❌ Error during deep queue cleanup: {e}")

    def _log_job_metric(
        self,
        *,
        job_id: str,
        job_type: Optional[str],
        params: Dict[str, Any],
        status: str,
        duration: float,
        queue_wait_s: Optional[float] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Emit one structured, greppable summary line per finished job.

        Distinct from the surrounding traceback logs (which are easy to lose in
        Railway's stream) so a failing/slow job — job_id, period, location,
        queue wait, execution duration, error reason — can be found or aggregated
        on its own. Added for P1-153 to confirm whether async jobs are actually
        failing/timing out server-side, rather than inferring it from client-side
        fallback behavior. queue_wait_s separates "sat pending too long behind
        other queued jobs" from "was slow to execute once picked up" — the worker
        processes jobs one at a time, so a burst of jobs (e.g. cold-location
        backfills) can bury a single job well past the client's poll budget even
        though that job's own duration_s looks fine once it finally runs.
        """
        from utils.sanitization import sanitize_for_logging

        location = params.get("location")
        log_fn = logger.error if status == "error" else logger.info
        log_fn(
            "JOB_METRIC job_id=%s type=%s status=%s period=%s location=%s identifier=%s year=%s "
            "queue_wait_s=%s duration_s=%.2f error_type=%s error=%s",
            job_id,
            job_type,
            status,
            params.get("scope") or params.get("period"),
            sanitize_for_logging(location) if location else None,
            params.get("identifier"),
            params.get("year"),
            f"{queue_wait_s:.2f}" if queue_wait_s is not None else None,
            duration,
            error_type,
            sanitize_for_logging(error_message) if error_message else None,
        )

    async def process_job(self, job_id: str, job_manager):
        """Process a single job."""
        import traceback

        start_time = datetime.now(timezone.utc)
        job_type: Optional[str] = None
        params: Dict[str, Any] = {}
        queue_wait_s: Optional[float] = None
        # Keep heartbeat alive so the worker-status endpoint sees us as healthy
        # even when a single long-running job would otherwise starve the poll loop.
        self._refresh_heartbeat()
        try:
            job_manager.update_job_status(job_id, JobStatus.PROCESSING)
            logger.info(f"🔄 Processing job: {job_id}")

            job_data = job_manager.get_job_status(job_id)
            if not job_data:
                logger.error(f"❌ Job not found: {job_id}")
                self.redis.lrem(self.job_queue_key, 1, job_id)
                return

            job_type = job_data.get("type")
            params = job_data.get("params", {})
            queue_wait_s = self._compute_queue_wait(job_data, start_time)
            logger.info(f"📋 Job type: {job_type}, Params: {params}")

            result = await self._run_job_with_timeout(job_id=job_id, job_type=job_type, params=params)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            job_manager.update_job_status(job_id, JobStatus.READY, result)
            logger.info(f"✅ Job completed: {job_id} (took {duration:.2f}s)")
            self._log_job_metric(
                job_id=job_id,
                job_type=job_type,
                params=params,
                status="success",
                duration=duration,
                queue_wait_s=queue_wait_s,
            )
            self.redis.lrem(self.job_queue_key, 1, job_id)

        except redis.RedisError as redis_error:
            # Redis errors - don't update job status if Redis is down
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_traceback = traceback.format_exc()
            logger.error(f"❌ Redis error processing job {job_id}: {type(redis_error).__name__}: {str(redis_error)}")
            logger.error(f"❌ Full traceback:\n{error_traceback}")
            self._log_job_metric(
                job_id=job_id,
                job_type=job_type,
                params=params,
                status="error",
                duration=duration,
                queue_wait_s=queue_wait_s,
                error_type=type(redis_error).__name__,
                error_message=str(redis_error),
            )
            # Cannot update job status or remove from queue if Redis is failing

        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as data_error:
            # Data/validation errors - store in job status
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_traceback = traceback.format_exc()
            error_details = {
                "error_type": type(data_error).__name__,
                "error_message": str(data_error),
                "traceback": error_traceback,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.error(f"❌ Data error processing job {job_id}: {type(data_error).__name__}: {str(data_error)}")
            logger.error(f"❌ Full traceback:\n{error_traceback}")
            job_manager.update_job_status(job_id, JobStatus.ERROR, error=str(data_error), error_details=error_details)
            self._log_job_metric(
                job_id=job_id,
                job_type=job_type,
                params=params,
                status="error",
                duration=duration,
                queue_wait_s=queue_wait_s,
                error_type=type(data_error).__name__,
                error_message=str(data_error),
            )
            self.redis.lrem(self.job_queue_key, 1, job_id)

        except Exception as e:
            # Unexpected errors - log and store
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_traceback = traceback.format_exc()
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_traceback,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.error(f"❌ Unexpected error processing job {job_id}: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ Full traceback:\n{error_traceback}")
            job_manager.update_job_status(job_id, JobStatus.ERROR, error=str(e), error_details=error_details)
            self._log_job_metric(
                job_id=job_id,
                job_type=job_type,
                params=params,
                status="error",
                duration=duration,
                queue_wait_s=queue_wait_s,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            self.redis.lrem(self.job_queue_key, 1, job_id)

    @staticmethod
    def _compute_queue_wait(job_data: Dict[str, Any], reference_time: datetime) -> Optional[float]:
        """Seconds between job creation and when this worker picked it up, or None if unavailable."""
        created_at_str = job_data.get("created_at")
        if not created_at_str:
            return None
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except ValueError:
            return None
        return (reference_time - created_at).total_seconds()

    def _dispatch_job_coro(self, *, job_type: str, params: dict):
        """Return the coroutine for the given job type, or raise ValueError."""
        if job_type == "record_computation":
            logger.info("🔢 Starting record computation...")
            return self.process_record_job(params)
        elif job_type == "cache_warming":
            logger.info("🔥 Starting cache warming...")
            return self.process_cache_warming_job(params)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    async def _run_job_with_timeout(self, *, job_id: str, job_type: str, params: dict) -> Any:
        """Dispatch to the right handler and enforce the wall-clock timeout."""
        import traceback

        coro = self._dispatch_job_coro(job_type=job_type, params=params)
        try:
            return await asyncio.wait_for(coro, timeout=MAX_JOB_EXECUTION_SECONDS)
        except asyncio.TimeoutError:
            logger.error(f"⏰ Job {job_id} ({job_type}) timed out after {MAX_JOB_EXECUTION_SECONDS}s")
            raise TimeoutError(f"Job exceeded {MAX_JOB_EXECUTION_SECONDS}s execution limit")
        except (ValueError, KeyError, TypeError) as data_error:
            logger.error(f"❌ Data error for job {job_id}: {type(data_error).__name__}")
            logger.error(f"❌ Error message: {str(data_error)}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            raise
        except (redis.RedisError, json.JSONDecodeError) as cache_error:
            logger.error(f"❌ Cache error for job {job_id}: {type(cache_error).__name__}")
            logger.error(f"❌ Error message: {str(cache_error)}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            raise
        except Exception as unexpected_error:
            # Catch any other unexpected errors but log them distinctly
            logger.error(f"❌ Unexpected error for job {job_id}: {type(unexpected_error).__name__}")
            logger.error(f"❌ Error message: {str(unexpected_error)}")
            logger.error(f"❌ Error repr: {repr(unexpected_error)}")
            logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
            raise

    async def process_record_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a record computation job with per-year granularity."""
        from fastapi import HTTPException

        from cache.keys import rec_key
        from routers.v1_records import (
            _extract_per_year_records,
            _rebuild_full_response_from_values,
            get_temperature_data_v1,
        )
        from utils.daily_temperature_store import resolve_location_cache_slug
        from utils.weather import get_year_range

        logger.info(f"🔍 Processing record job with params: {params}")
        scope = params.get("scope") or params.get("period")  # Support both for backward compatibility
        location = params.get("location")
        identifier = params.get("identifier")
        year = params.get("year")
        unit_group = params.get("unit_group", "celsius")

        if not all([scope, location, identifier]):
            raise ValueError(
                f"Missing required params - scope: {scope}, location: {location}, identifier: {identifier}"
            )

        # Resolve canonical cache slug (may differ from slug stored in older jobs)
        slug = await resolve_location_cache_slug(location)
        current_year = datetime.now(timezone.utc).year
        oldest_year = current_year - 50

        if year is not None:
            # Discard jobs for years that have rolled off the 50-year window.
            if year < oldest_year or year > current_year:
                logger.warning(f"⏭️ Skipping out-of-window year {year} (valid range {oldest_year}–{current_year})")
                return {"skipped": True, "reason": f"year {year} outside window {oldest_year}–{current_year}"}
            return await self._process_single_year(
                location=location,
                scope=scope,
                identifier=identifier,
                year=year,
                slug=slug,
                unit_group=unit_group,
                current_year=current_year,
            )

        # Default behavior: if no year specified, fetch all available years for the identifier
        logger.info("ℹ️ No year specified; returning the full historical range")
        try:
            data = await get_temperature_data_v1(location, scope, identifier, "celsius", self.redis)
        except HTTPException as http_err:
            raise ValueError(f"{http_err.detail}") from http_err

        per_year_records = _extract_per_year_records(data)
        coverage_by_year = {d["year"]: d for d in data.get("coverage", {}).get("per_year", [])}
        self._build_pipeline_cache_all_years(
            scope=scope,
            slug=slug,
            identifier=identifier,
            per_year_records=per_year_records,
            current_year=current_year,
            coverage_by_year=coverage_by_year,
        )

        month, day = int(identifier.split("-")[0]), int(identifier.split("-")[1])
        years = get_year_range(current_year)
        values_list = [per_year_records[y] for y in sorted(per_year_records.keys())]
        if values_list:
            result_data = _rebuild_full_response_from_values(
                values_list, scope, location, identifier, month, day, current_year, years, self.redis, unit_group
            )
        else:
            result_data = data

        return {
            "cache_keys": [rec_key(scope, slug, identifier, y) for y in per_year_records.keys()],
            "years_cached": list(per_year_records.keys()),
            "data": result_data,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _process_single_year(
        self,
        *,
        location: str,
        scope: str,
        identifier: str,
        year: int,
        slug: str,
        unit_group: str,
        current_year: int,
    ) -> Dict[str, Any]:
        """Fetch and cache a single year's record data (Path A of process_record_job)."""
        from fastapi import HTTPException

        from cache.core import ETagGenerator, get_ttl_for_year
        from cache.keys import rec_etag_key, rec_key
        from routers.v1_records import (
            _get_ttl_for_current_year,
            _rebuild_full_response_from_values,
            compute_per_year_records,
        )
        from utils.weather import get_year_range

        logger.info(f"📅 Fetching data for year {year} only")

        # Quick DB pre-check for historical years: if the DB has no records for this
        # year's date window, skip now rather than running the full 51-year computation
        # which also triggers inline VC API calls for every other missing year.
        # Current year is exempt — it may have sparse data still being collected.
        if year < current_year and not await self._db_precheck_year(
            location=location, scope=scope, identifier=identifier, year=year, slug=slug
        ):
            return {"skipped": True, "reason": f"no data available for year {year}"}

        # Narrow path: fetch only the requested year, not the full 51-year window.
        try:
            per_year_records, _missing, coverage_details = await compute_per_year_records(
                location, scope, identifier, [year], "celsius", self.redis
            )
        except HTTPException as http_err:
            raise ValueError(f"{http_err.detail}") from http_err

        if year not in per_year_records:
            logger.warning(f"⏭️ No data available for year {year} at {location}, skipping")
            # VC confirmed no data for this year. Persist a skip key so
            # future _enqueue_backfill_job calls for this year skip fast.
            try:
                skip_key = f"backfill:skip:{scope}:{slug}:{year}"
                self.redis.setex(skip_key, 86400, "1")
            except Exception as _e:
                logger.debug("Could not set backfill skip key: %s", _e)
            return {"skipped": True, "reason": f"no data available for year {year}"}

        year_record = per_year_records[year]
        ttl = get_ttl_for_year(year) if year < current_year else _get_ttl_for_current_year(scope)
        year_key = rec_key(scope, slug, identifier, year)
        etag_key = rec_etag_key(scope, slug, identifier, year)
        etag = ETagGenerator.generate_etag(year_record)

        # Skip caching current-year results when coverage is below 80% to avoid poisoning
        # the cache with a warm-biased average computed before cold months are backfilled.
        if not self._evaluate_single_year_coverage(
            year=year, current_year=current_year, coverage_details=coverage_details
        ):
            self._cache_single_year_record(
                year_key=year_key, etag_key=etag_key, year_record=year_record, etag=etag, ttl=ttl
            )

        month, day = int(identifier.split("-")[0]), int(identifier.split("-")[1])
        years = get_year_range(current_year)
        result_data = _rebuild_full_response_from_values(
            [year_record], scope, location, identifier, month, day, current_year, years, self.redis, unit_group
        )
        return {
            "cache_key": year_key,
            "etag": etag,
            "year": year,
            "data": result_data,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _db_precheck_year(
        self,
        *,
        location: str,
        scope: str,
        identifier: str,
        year: int,
        slug: str,
    ) -> bool:
        """Check the DB for data in the year's date window.

        Returns True if data was found (proceed) or False if the year should be
        skipped (also writes a backfill:skip key). Swallows exceptions (fail-open).
        """
        from datetime import timedelta as _td

        from routers.v1_records import WINDOW_DAYS, _resolve_anchor_date
        from routers.v1_records import parse_identifier as _parse_id
        from utils.daily_temperature_store import get_daily_temperature_store

        try:
            _month, _day, _ = _parse_id(scope, identifier)
            _anchor = _resolve_anchor_date(year, _month, _day)
            if _anchor is not None:
                _window = 1 if scope == "daily" else WINDOW_DAYS.get(scope, 1)
                _year_dates = [_anchor - _td(days=_window - 1) + _td(days=i) for i in range(_window)]
                _store = await get_daily_temperature_store()
                _year_cache = await _store.fetch(location, _year_dates)
                if not _year_cache:
                    logger.info(f"⏭️ No DB data for year {year} at {location}, skipping")
                    try:
                        skip_key = f"backfill:skip:{scope}:{slug}:{year}"
                        self.redis.setex(skip_key, 86400, "1")
                    except Exception as _e:
                        logger.debug("Could not set backfill skip key: %s", _e)
                    return False
        except Exception as _pre_err:
            logger.debug(f"DB pre-check skipped for {location} year {year}: {_pre_err}")
        return True

    def _evaluate_single_year_coverage(
        self,
        *,
        year: int,
        current_year: int,
        coverage_details: list,
    ) -> bool:
        """Return True (skip cache) if current-year coverage is below 80%."""
        if year != current_year:
            return False
        year_detail = next((d for d in coverage_details if d.get("year") == year), None)
        if year_detail and year_detail.get("coverage_ratio", 1.0) < 0.8:
            logger.info(
                f"⏭️ Skipping cache for current year {year}: coverage {year_detail['coverage_ratio']:.1%} < 80%"
            )
            return True
        return False

    def _cache_single_year_record(
        self,
        *,
        year_key: str,
        etag_key: str,
        year_record: dict,
        etag: str,
        ttl: int,
    ) -> None:
        """Write a per-year record and its ETag to Redis."""
        try:
            json_data = json.dumps(year_record, sort_keys=True, separators=(",", ":"))
            self.redis.setex(year_key, ttl, json_data)
            self.redis.setex(etag_key, ttl, etag)
            logger.info(f"✅ Cached per-year record: {year_key} (TTL: {ttl}s)")
        except (redis.RedisError, redis.ConnectionError) as redis_error:
            logger.warning(f"Redis error caching per-year record {year_key}: {redis_error}")
            raise
        except (ValueError, TypeError) as serialize_error:
            logger.warning(f"Serialization error caching per-year record {year_key}: {serialize_error}")
            raise

    def _build_pipeline_cache_all_years(
        self,
        *,
        scope: str,
        slug: str,
        identifier: str,
        per_year_records: dict,
        current_year: int,
        coverage_by_year: dict,
    ) -> None:
        """Batch-cache all per-year records using a Redis pipeline."""
        from cache.core import ETagGenerator, get_ttl_for_year
        from cache.keys import rec_etag_key, rec_key
        from routers.v1_records import _get_ttl_for_current_year

        try:
            pipeline = self.redis.pipeline()

            for y, record_data in per_year_records.items():
                if y == current_year:
                    detail = coverage_by_year.get(y, {})
                    ratio = detail.get("coverage_ratio", 1.0)
                    if ratio < 0.8:
                        logger.info(f"⏭️ Skipping cache for current year {y}: coverage {ratio:.1%} < 80%")
                        continue

                year_key = rec_key(scope, slug, identifier, y)
                etag_key = rec_etag_key(scope, slug, identifier, y)
                ttl = get_ttl_for_year(y) if y < current_year else _get_ttl_for_current_year(scope)
                etag = ETagGenerator.generate_etag(record_data)
                json_data = json.dumps(record_data, sort_keys=True, separators=(",", ":"))

                pipeline.setex(year_key, ttl, json_data)
                pipeline.setex(etag_key, ttl, etag)

            pipeline.execute()
            logger.info(f"✅ Cached {len(per_year_records)} years using pipeline")

        except (redis.RedisError, ValueError, TypeError) as e:
            logger.warning(f"Error caching years with pipeline: {type(e).__name__}: {e}")

    async def process_cache_warming_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a cache warming job."""
        from cache.accessors import get_cache_warmer
        from cache.warming import CACHE_WARMING_ENABLED

        logger.info(f"🔥 Processing cache warming job with params: {params}")

        # If warming was disabled after this job was enqueued, drop it quietly
        # rather than failing — the queue is durable, so a flipped flag must not
        # produce ongoing error noise from stale jobs.
        cache_warmer = get_cache_warmer()
        if not CACHE_WARMING_ENABLED or not cache_warmer:
            logger.info(
                "⏭️  Skipping cache_warming job — warming disabled "
                "(CACHE_WARMING_ENABLED=%s, warmer=%s)",
                CACHE_WARMING_ENABLED,
                bool(cache_warmer),
            )
            return {
                "job_type": "cache_warming",
                "skipped": True,
                "reason": "cache warming disabled",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        locations = params.get("locations", [])
        warming_type = params.get("type", "all")  # "all", "popular", "specific"

        results = {
            "job_type": "cache_warming",
            "warming_type": warming_type,
            "locations_requested": locations,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "results": {},
        }

        try:
            if warming_type == "all":
                logger.info("🔥 Warming all popular locations...")
                results["results"]["all_locations"] = await cache_warmer.warm_all_locations()
            elif warming_type == "popular":
                logger.info("🔥 Warming popular locations...")
                popular_locations = cache_warmer.get_locations_to_warm()
                await self._warm_popular_or_specific(
                    cache_warmer=cache_warmer, locations=popular_locations, results=results
                )
            elif warming_type == "specific":
                if not locations:
                    raise ValueError("No locations specified for specific warming")
                logger.info(f"🔥 Warming specific locations: {locations}")
                await self._warm_popular_or_specific(
                    cache_warmer=cache_warmer, locations=locations, results=results
                )
            else:
                raise ValueError(f"Unknown warming type: {warming_type}")

            results["status"] = "completed"
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            results["summary"] = self._calculate_warming_summary(location_results=results["results"])
            logger.info(
                f"✅ Cache warming completed: {results['summary']['successful_locations']} locations, "
                f"{results['summary']['total_endpoints_warmed']} endpoints"
            )

        except (ValueError, KeyError, TypeError) as data_error:
            logger.error(f"❌ Data/validation error during cache warming: {data_error}")
            results["status"] = "failed"
            results["error"] = f"Data error: {str(data_error)}"
            results["failed_at"] = datetime.now(timezone.utc).isoformat()
            raise
        except (redis.RedisError, redis.ConnectionError) as cache_error:
            logger.error(f"❌ Redis error during cache warming: {cache_error}")
            results["status"] = "failed"
            results["error"] = f"Cache error: {str(cache_error)}"
            results["failed_at"] = datetime.now(timezone.utc).isoformat()
            raise
        except Exception as e:
            # Catch any HTTP or unexpected errors
            logger.error(f"❌ Unexpected error during cache warming: {type(e).__name__}: {e}")
            results["status"] = "failed"
            results["error"] = f"Unexpected error: {str(e)}"
            results["failed_at"] = datetime.now(timezone.utc).isoformat()
            raise

        return results

    async def _warm_popular_or_specific(
        self,
        *,
        cache_warmer,
        locations: list,
        results: dict,
    ) -> None:
        """Warm a list of locations using an authenticated session."""
        auth_token = os.getenv("API_ACCESS_TOKEN")
        if not auth_token:
            raise RuntimeError("API_ACCESS_TOKEN not set; cannot warm locations")
        async with cache_warmer._create_warming_session(auth_token) as session:
            for location in locations:
                location_result = await cache_warmer.warm_location_data(location, session)
                results["results"][location] = location_result

    def _calculate_warming_summary(self, *, location_results: dict) -> dict:
        """Build summary statistics from per-location warming results."""
        total_endpoints = 0
        total_errors = 0
        successful_locations = 0

        for result in location_results.values():
            if isinstance(result, dict) and "warmed_endpoints" in result:
                total_endpoints += len(result.get("warmed_endpoints", []))
                total_errors += len(result.get("errors", []))
                if result.get("warmed_endpoints"):
                    successful_locations += 1

        return {
            "total_locations": len(location_results),
            "successful_locations": successful_locations,
            "total_endpoints_warmed": total_endpoints,
            "total_errors": total_errors,
        }

    def _store_cache_data(self, cache_key: str, data: dict, data_type: str = "data") -> str:
        """Helper function to store data in cache and generate ETag."""
        import hashlib
        from datetime import timedelta

        from cache.core import CACHE_TTL_LONG, set_cache_value

        try:
            # Convert TTL to timedelta (CACHE_TTL_LONG is in seconds)
            cache_duration = timedelta(seconds=CACHE_TTL_LONG)

            # Use the same cache storage function as the main endpoints
            set_cache_value(cache_key, cache_duration, json.dumps(data), self.redis)

            # Generate ETag using SHA256 (same as main endpoints)
            etag = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]

            logger.info(f"✅ Cached {data_type} for {cache_key}")
            return etag

        except (redis.RedisError, redis.ConnectionError) as redis_error:
            logger.warning(f"Redis error storing cache for {cache_key}: {redis_error}")
            # Generate a simple ETag even if cache fails (use SHA256, not MD5)
            etag = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]
            return etag
        except (json.JSONEncodeError, TypeError) as encode_error:
            logger.warning(f"JSON encoding error for {cache_key}: {encode_error}")
            # Generate a simple ETag even if cache fails
            etag = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]
            return etag


# Global worker instance
worker: JobWorker = None


def initialize_worker(redis_client):
    """Initialize the job worker."""
    global worker
    worker = JobWorker(redis_client)
    logger.info("Job worker initialized")


def get_worker() -> JobWorker:
    """Get the global worker instance."""
    if worker is None:
        raise RuntimeError("Worker not initialized. Call initialize_worker() first.")
    return worker


async def start_background_worker():
    """Start the background job worker."""
    try:
        worker = get_worker()
        await worker.start()
    except Exception as e:
        logger.error(f"Background worker error: {e}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, _frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        if worker:
            worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


async def main():
    """Main function to run the job worker."""
    import os

    import redis

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Reduce verbosity of noisy third-party loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Connect to Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)

    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("✅ Connected to Redis")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        return

    # Initialize cache system first (required for job manager)
    from cache.accessors import initialize_cache

    initialize_cache(redis_client)
    logger.info("✅ Cache system initialized")

    # Initialize worker
    initialize_worker(redis_client)

    # Setup signal handlers
    setup_signal_handlers()

    # Start worker
    logger.info("🚀 Starting job worker...")
    await start_background_worker()


if __name__ == "__main__":
    asyncio.run(main())
