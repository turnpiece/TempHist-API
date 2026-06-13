"""Job management endpoints."""

import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Literal

import redis
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse

from cache.accessors import get_job_manager
from jobs.manager import JobQueueFullError, JobStatus
from routers.dependencies import get_redis_client
from utils.daily_temperature_store import get_daily_temperature_store

logger = logging.getLogger(__name__)
router = APIRouter()

# Job diagnostics constants
STUCK_JOB_THRESHOLD_SECONDS = 300  # 5 minutes - jobs older than this are considered stuck
WORKER_HEARTBEAT_TIMEOUT_SECONDS = 60  # 1 minute - worker considered unhealthy if heartbeat older
MAX_STUCK_JOBS_TO_DISPLAY = 10  # Limit number of stuck jobs shown in diagnostics


def get_diagnostics_recommendations(worker_alive, heartbeat_age, jobs_by_status, stuck_count, long_pending_count=0):
    """Generate diagnostic recommendations based on worker and job status."""
    recommendations = []

    if not worker_alive:
        recommendations.append(
            {
                "severity": "critical",
                "issue": "Background worker is not running",
                "actions": [
                    "Check server logs for worker startup errors",
                    "Restart the API server",
                    "Verify Redis connection is available",
                ],
            }
        )
    elif heartbeat_age and heartbeat_age > WORKER_HEARTBEAT_TIMEOUT_SECONDS:
        recommendations.append(
            {
                "severity": "warning",
                "issue": f"Worker heartbeat is stale ({int(heartbeat_age)}s old)",
                "actions": [
                    "Worker may be stuck or crashed",
                    "Check server logs for errors",
                    "Consider restarting the API server",
                ],
            }
        )

    if jobs_by_status.get("pending", 0) > 0 and worker_alive:
        recommendations.append(
            {
                "severity": "info",
                "issue": f"{jobs_by_status['pending']} jobs in pending state",
                "actions": [
                    "Jobs are waiting to be processed",
                    "Worker should process these shortly",
                    "If they remain pending for >1 minute, check worker logs",
                ],
            }
        )

    if stuck_count > 0:
        recommendations.append(
            {
                "severity": "warning",
                "issue": f"{stuck_count} job(s) stuck in PROCESSING for >5 minutes",
                "actions": [
                    "Check server logs for processing errors",
                    "These jobs may need to be manually cleared",
                    "Use diagnose_jobs.py --clear-stuck to clear them",
                ],
            }
        )

    if long_pending_count > 0:
        recommendations.append(
            {
                "severity": "info",
                "issue": f"{long_pending_count} job(s) have been pending for >5 minutes",
                "actions": [
                    "Jobs are queued but not yet processed — this is normal for a long queue",
                    "Worker is processing them serially; they will clear in time",
                    "If count is not decreasing, check worker logs for errors",
                ],
            }
        )

    if jobs_by_status.get("error", 0) > 0:
        recommendations.append(
            {
                "severity": "warning",
                "issue": f"{jobs_by_status['error']} jobs failed with errors",
                "actions": [
                    "Check individual job status for error details",
                    "Common causes: API errors, timeouts, invalid parameters",
                ],
            }
        )

    if not recommendations:
        recommendations.append({"severity": "success", "issue": "System is healthy", "actions": ["No action needed"]})

    return recommendations


@router.post("/v1/records/{period}/{location}/{identifier}/async")
async def create_record_job(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius", description="Temperature unit for response"),
    response: Response = None,
):
    """Create an async job to compute heavy record data."""
    try:
        logger.info(f"Creating async job: period={period}, location={location}, identifier={identifier}")
        job_manager = get_job_manager()
        logger.info("Job manager retrieved successfully")

        # Create job
        job_id = job_manager.create_job(
            "record_computation",
            {"period": period, "location": location, "identifier": identifier, "unit_group": unit_group},
        )
        logger.info(f"Job created successfully: {job_id}")

        # Return 202 Accepted with job info
        response.status_code = 202
        response.headers["Retry-After"] = "3"

        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Job created successfully",
            "retry_after": 3,
            "status_url": f"/v1/jobs/{job_id}",
        }

    except JobQueueFullError as e:
        logger.warning(f"Job queue full, returning 503: {e}")
        response.status_code = 503
        response.headers["Retry-After"] = "10"
        return {"error": "service_unavailable", "message": str(e), "retry_after": 10}
    except Exception as e:
        logger.error(f"Error creating record job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an async job."""
    try:
        job_manager = get_job_manager()
        job_status = job_manager.get_job_status(job_id)

        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return job_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job status")


@router.post("/v1/records/rolling-bundle/{location}/{anchor}/async")
async def create_rolling_bundle_job(
    request: Request,
    location: str = Path(..., description="Location name"),
    anchor: str = Path(..., description="Anchor date"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius"),
):
    """This endpoint has been removed. Use individual period endpoints instead."""
    from urllib.parse import quote

    base_url = str(request.base_url).rstrip("/")
    try:
        from datetime import datetime as _dt

        mmdd = _dt.strptime(anchor, "%Y-%m-%d").strftime("%m-%d")
    except ValueError:
        mmdd = "01-15"
    enc = quote(location, safe="")
    links = {
        period: f"{base_url}/v1/records/{period}/{enc}/{mmdd}?unit_group={unit_group}"
        for period in ["daily", "weekly", "monthly", "yearly"]
    }
    return JSONResponse(
        status_code=410,
        content={
            "error": "GONE",
            "message": "The rolling-bundle async job endpoint has been removed",
            "code": "GONE",
            "details": "This endpoint is no longer available. Please use the individual period endpoints instead.",
            "links": links,
        },
    )


@router.get("/v1/jobs/diagnostics/worker-status")
async def get_worker_diagnostics(redis_client: Annotated[redis.Redis, Depends(get_redis_client)]):
    """Get diagnostic information about the background worker and job queue."""
    try:
        # Check worker heartbeat
        heartbeat = redis_client.get("worker:heartbeat")
        worker_alive = heartbeat is not None

        # Get queue status
        queue_length = redis_client.llen("job_queue")

        # Get jobs from the queue (without KEYS command)
        # We'll examine jobs in the queue since we can't scan all keys
        jobs_by_status = {"pending": 0, "processing": 0, "ready": 0, "error": 0}

        stuck_jobs = []
        long_pending_jobs = []
        jobs_examined = []

        # Get all job IDs from the queue
        if queue_length > 0:
            # Get up to 100 jobs from the queue
            for i in range(min(queue_length, 100)):
                job_id = redis_client.lindex("job_queue", i)
                if job_id:
                    if isinstance(job_id, bytes):
                        job_id = job_id.decode("utf-8")
                    jobs_examined.append(job_id)

        # Examine each job in the queue
        for job_id in jobs_examined:
            try:
                job_key = f"job:{job_id}"
                job_data = redis_client.get(job_key)
                if job_data:
                    if isinstance(job_data, bytes):
                        job_data = job_data.decode("utf-8")
                    job = json.loads(job_data)
                    status = job.get("status", "unknown")

                    if status in jobs_by_status:
                        jobs_by_status[status] += 1

                    # Check for genuinely stuck jobs: PROCESSING jobs older than 5 minutes.
                    # Pending jobs are just waiting their turn — long wait ≠ stuck.
                    created = job.get("created_at")
                    if created:
                        try:
                            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                            entry = {
                                "job_id": job.get("id"),
                                "status": status,
                                "age_seconds": int(age),
                                "type": job.get("type"),
                                "params": job.get("params", {}),
                            }
                            if status == "processing" and age > STUCK_JOB_THRESHOLD_SECONDS:
                                stuck_jobs.append(entry)
                            elif status == "pending" and age > STUCK_JOB_THRESHOLD_SECONDS:
                                long_pending_jobs.append(entry)
                        except (ValueError, TypeError, KeyError):
                            pass
            except Exception as job_error:
                logger.warning(f"Error examining job {job_id}: {job_error}")

        # Parse heartbeat time if available
        heartbeat_age = None
        if heartbeat:
            try:
                if isinstance(heartbeat, bytes):
                    heartbeat = heartbeat.decode("utf-8")
                heartbeat_dt = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
                heartbeat_age = (datetime.now(timezone.utc) - heartbeat_dt).total_seconds()
            except (ValueError, TypeError, AttributeError):
                # Invalid heartbeat timestamp format
                pass

        return {
            "worker": {
                "alive": worker_alive,
                "heartbeat": heartbeat,
                "heartbeat_age_seconds": heartbeat_age,
                "status": "healthy"
                if worker_alive and (heartbeat_age is None or heartbeat_age < WORKER_HEARTBEAT_TIMEOUT_SECONDS)
                else "unhealthy",
            },
            "queue": {"length": queue_length, "jobs_examined": len(jobs_examined)},
            "jobs": {
                "by_status": jobs_by_status,
                "stuck_count": len(stuck_jobs),
                "stuck_jobs": stuck_jobs[:MAX_STUCK_JOBS_TO_DISPLAY],
                "long_pending_count": len(long_pending_jobs),
                "long_pending_jobs": long_pending_jobs[:MAX_STUCK_JOBS_TO_DISPLAY],
            },
            "recommendations": get_diagnostics_recommendations(
                worker_alive, heartbeat_age, jobs_by_status, len(stuck_jobs), len(long_pending_jobs)
            ),
            "note": "Only examining jobs in the queue (Redis KEYS command not available)",
            "database": await (await get_daily_temperature_store()).ping(),
        }
    except Exception as e:
        logger.error(f"Error getting worker diagnostics: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get diagnostics: {str(e)}")


@router.get("/debug/jobs")
async def debug_jobs_endpoint(redis_client: Annotated[redis.Redis, Depends(get_redis_client)]):
    """Debug endpoint to check job queue and job data in Redis."""
    try:
        now = datetime.now(timezone.utc)
        debug_info = {
            "queue_length": 0,
            "showing": 0,
            "longest_running_seconds": None,
            "jobs_in_queue": [],
            "job_data_status": {},
            "redis_connection": "unknown",
            "timestamp": now.isoformat(),
        }

        # Test Redis connection
        try:
            redis_client.ping()
            debug_info["redis_connection"] = "OK"
        except Exception as e:
            debug_info["redis_connection"] = f"FAILED: {e}"
            return debug_info

        # Check job queue
        queue_key = "job_queue"
        queue_length = redis_client.llen(queue_key)
        debug_info["queue_length"] = queue_length

        show_count = min(queue_length, 10)
        debug_info["showing"] = show_count

        if queue_length > 0:
            jobs_in_queue = []
            longest_seconds = None

            for i in range(show_count):
                job_id = redis_client.lindex(queue_key, i)
                if not job_id:
                    continue
                if isinstance(job_id, bytes):
                    job_id = job_id.decode("utf-8")

                jobs_in_queue.append(job_id)

                job_key = f"job:{job_id}"
                job_data = redis_client.get(job_key)

                if job_data:
                    try:
                        if isinstance(job_data, bytes):
                            job_data = job_data.decode("utf-8")
                        job = json.loads(job_data)
                        status = job.get("status", "unknown")
                        created_str = job.get("created_at")
                        params = job.get("params", {})

                        elapsed_seconds = None
                        if created_str:
                            try:
                                created_at = datetime.fromisoformat(created_str)
                                if created_at.tzinfo is None:
                                    created_at = created_at.replace(tzinfo=timezone.utc)
                                elapsed_seconds = round((now - created_at).total_seconds())
                                if longest_seconds is None or elapsed_seconds > longest_seconds:
                                    longest_seconds = elapsed_seconds
                            except (ValueError, TypeError):
                                pass

                        debug_info["job_data_status"][job_id] = {
                            "exists": True,
                            "status": status,
                            "created_at": created_str,
                            "elapsed_seconds": elapsed_seconds,
                            "params": params,
                        }
                    except (json.JSONDecodeError, ValueError, TypeError, UnicodeDecodeError):
                        debug_info["job_data_status"][job_id] = {"exists": True, "error": "invalid JSON"}
                else:
                    debug_info["job_data_status"][job_id] = {"exists": False}

            debug_info["jobs_in_queue"] = jobs_in_queue
            debug_info["longest_running_seconds"] = longest_seconds

        return debug_info

    except Exception as e:
        logger.error(f"Error in debug jobs endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
