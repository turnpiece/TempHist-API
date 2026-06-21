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


def _parse_heartbeat_age(heartbeat) -> float | None:
    try:
        if isinstance(heartbeat, bytes):
            heartbeat = heartbeat.decode("utf-8")
        heartbeat_dt = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - heartbeat_dt).total_seconds()
    except (ValueError, TypeError, AttributeError):
        return None


def _examine_job(redis_client: redis.Redis, job_id: str, now: datetime) -> dict | None:
    """Return a job-age entry dict for diagnostics, or None on failure."""
    try:
        job_data = redis_client.get(f"job:{job_id}")
        if not job_data:
            return None
        if isinstance(job_data, bytes):
            job_data = job_data.decode("utf-8")
        job = json.loads(job_data)
        created = job.get("created_at")
        if not created:
            return None
        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        age = (now - created_dt).total_seconds()
        return {
            "job_id": job.get("id"),
            "status": job.get("status", "unknown"),
            "age_seconds": int(age),
            "type": job.get("type"),
            "params": job.get("params", {}),
        }
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _get_job_debug_info(redis_client: redis.Redis, job_id: str, now: datetime) -> dict:
    """Return debug info dict for a single job."""
    job_data = redis_client.get(f"job:{job_id}")
    if not job_data:
        return {"exists": False}
    try:
        if isinstance(job_data, bytes):
            job_data = job_data.decode("utf-8")
        job = json.loads(job_data)
        created_str = job.get("created_at")
        elapsed_seconds = None
        if created_str:
            try:
                created_at = datetime.fromisoformat(created_str)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                elapsed_seconds = round((now - created_at).total_seconds())
            except (ValueError, TypeError):
                pass
        return {
            "exists": True,
            "status": job.get("status", "unknown"),
            "created_at": created_str,
            "elapsed_seconds": elapsed_seconds,
            "params": job.get("params", {}),
        }
    except (json.JSONDecodeError, ValueError, TypeError, UnicodeDecodeError):
        return {"exists": True, "error": "invalid JSON"}


@router.get("/v1/jobs/diagnostics/worker-status")
async def get_worker_diagnostics(redis_client: Annotated[redis.Redis, Depends(get_redis_client)]):
    """Get diagnostic information about the background worker and job queue."""
    try:
        heartbeat = redis_client.get("worker:heartbeat")
        worker_alive = heartbeat is not None
        queue_length = redis_client.llen("job_queue")
        now = datetime.now(timezone.utc)

        jobs_by_status = {"pending": 0, "processing": 0, "ready": 0, "error": 0}
        stuck_jobs: list = []
        long_pending_jobs: list = []
        jobs_examined: list = []

        for i in range(min(queue_length, 100)):
            raw_id = redis_client.lindex("job_queue", i)
            if raw_id:
                jobs_examined.append(raw_id.decode("utf-8") if isinstance(raw_id, bytes) else raw_id)

        for job_id in jobs_examined:
            try:
                entry = _examine_job(redis_client, job_id, now)
                if entry:
                    status = entry["status"]
                    if status in jobs_by_status:
                        jobs_by_status[status] += 1
                    if entry["age_seconds"] > STUCK_JOB_THRESHOLD_SECONDS:
                        (stuck_jobs if status == "processing" else long_pending_jobs if status == "pending" else []).append(entry)
            except Exception as job_error:
                logger.warning(f"Error examining job {job_id}: {job_error}")

        heartbeat_age = _parse_heartbeat_age(heartbeat) if heartbeat else None
        worker_healthy = worker_alive and (heartbeat_age is None or heartbeat_age < WORKER_HEARTBEAT_TIMEOUT_SECONDS)

        return {
            "worker": {
                "alive": worker_alive,
                "heartbeat": heartbeat,
                "heartbeat_age_seconds": heartbeat_age,
                "status": "healthy" if worker_healthy else "unhealthy",
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

        try:
            redis_client.ping()
            debug_info["redis_connection"] = "OK"
        except Exception as e:
            debug_info["redis_connection"] = f"FAILED: {e}"
            return debug_info

        queue_key = "job_queue"
        queue_length = redis_client.llen(queue_key)
        debug_info["queue_length"] = queue_length
        show_count = min(queue_length, 10)
        debug_info["showing"] = show_count

        if queue_length > 0:
            jobs_in_queue = []
            longest_seconds = None

            for i in range(show_count):
                raw_id = redis_client.lindex(queue_key, i)
                if not raw_id:
                    continue
                job_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else raw_id
                jobs_in_queue.append(job_id)
                info = _get_job_debug_info(redis_client, job_id, now)
                debug_info["job_data_status"][job_id] = info
                elapsed = info.get("elapsed_seconds")
                if elapsed is not None and (longest_seconds is None or elapsed > longest_seconds):
                    longest_seconds = elapsed

            debug_info["jobs_in_queue"] = jobs_in_queue
            debug_info["longest_running_seconds"] = longest_seconds

        return debug_info

    except Exception as e:
        logger.error(f"Error in debug jobs endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
