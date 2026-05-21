import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis

from config import DEBUG

logger = logging.getLogger(__name__)

MAX_JOB_QUEUE_SIZE = int(os.getenv("MAX_JOB_QUEUE_SIZE", "1000"))
CACHE_TTL_JOB = int(os.getenv("CACHE_TTL_JOB", "7200"))
_CACHE_TTL_JOB = CACHE_TTL_JOB


class JobQueueFullError(Exception):
    """Raised when the job queue exceeds MAX_JOB_QUEUE_SIZE."""
    pass


class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class SingleFlightLock:
    """Prevent cache stampedes with Redis-based locks."""

    def __init__(self, redis_client: redis.Redis, lock_ttl: int = 30):
        self.redis = redis_client
        self.lock_ttl = lock_ttl
        self.lock_prefix = "lock:"

    async def acquire(self, key: str) -> bool:
        """Acquire a lock for the given key."""
        lock_key = f"{self.lock_prefix}{key}"
        try:
            result = self.redis.set(lock_key, "1", nx=True, ex=self.lock_ttl)
            return result is not None
        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis error acquiring lock for {key}: {e}")
            return False

    def release(self, key: str):
        """Release a lock for the given key."""
        lock_key = f"{self.lock_prefix}{key}"
        try:
            self.redis.delete(lock_key)
        except (redis.RedisError, redis.ConnectionError) as e:
            logger.warning(f"Redis error releasing lock for {key}: {e}")


class JobManager:
    """Manage async job processing with Redis storage."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.job_prefix = "job:"
        self.result_prefix = "result:"
        self.job_ttl = _CACHE_TTL_JOB

    def create_job(self, job_type: str, params: Dict[str, Any]) -> str:
        """
        Create a new job and return job ID.
        Implements deduplication — if an identical job is already pending, processing,
        recently completed, or recently failed, returns the existing job_id.
        """
        try:
            params_hash = hashlib.sha256(str(params).encode()).hexdigest()[:16]
            dedup_key = f"job:dedup:{job_type}:{params_hash}"

            existing_job_id = self.redis.get(dedup_key)
            if existing_job_id:
                existing_job_id = existing_job_id.decode() if isinstance(existing_job_id, bytes) else existing_job_id
                job_key = f"{self.job_prefix}{existing_job_id}"
                job_data = self.redis.get(job_key)
                if job_data:
                    job = json.loads(job_data)
                    status = job.get("status")
                    if status in [JobStatus.PENDING, JobStatus.PROCESSING]:
                        logger.info(f"Deduplicated job {existing_job_id}: identical job already {status}")
                        return existing_job_id
                    if status == JobStatus.READY:
                        logger.info(f"Deduplicated job {existing_job_id}: identical job already completed")
                        return existing_job_id
                    if status == JobStatus.ERROR:
                        logger.info(f"Deduplicated job {existing_job_id}: identical job recently failed, skipping re-enqueue")
                        return existing_job_id

            queue_length = self.redis.llen("job_queue")
            if queue_length >= MAX_JOB_QUEUE_SIZE:
                logger.warning(f"Job queue full ({queue_length} >= {MAX_JOB_QUEUE_SIZE}), rejecting new job")
                raise JobQueueFullError(f"Job queue is full ({queue_length} jobs). Try again later.")

            job_id = f"{job_type}_{int(time.time() * 1000)}_{params_hash[:8]}"
            job_key = f"{self.job_prefix}{job_id}"

            job_data = {
                "id": job_id,
                "type": job_type,
                "status": JobStatus.PENDING,
                "params": params,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"Creating job with ID: {job_id}")
            logger.info(f"Job params: {params}")

            self.redis.setex(job_key, self.job_ttl, json.dumps(job_data))
            logger.info(f"Job data stored in Redis with key: {job_key}")

            self.redis.setex(dedup_key, 300, job_id)

            job_queue_key = "job_queue"
            self.redis.lpush(job_queue_key, job_id)
            logger.info(f"Job {job_id} added to queue")

            return job_id
        except JobQueueFullError:
            raise  # already logged as WARNING above
        except Exception as e:
            logger.error(f"Error in create_job: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Job type: {job_type}, Params: {params}")
            raise

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result if ready."""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.redis.get(job_key)

        if not job_data:
            return None

        job = json.loads(job_data)

        if job["status"] == JobStatus.READY:
            result_key = f"{self.result_prefix}{job_id}"
            result_data = self.redis.get(result_key)
            if result_data:
                job["result"] = json.loads(result_data)

        return job

    def update_job_status(
        self,
        job_id: str,
        status: str,
        result: Any = None,
        error: str = None,
        error_details: Dict = None,
    ):
        """Update job status and optionally store result."""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.redis.get(job_key)

        if not job_data:
            return

        job = json.loads(job_data)
        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

        if error:
            job["error"] = error
        if error_details:
            job["error_details"] = error_details

        self.redis.setex(job_key, self.job_ttl, json.dumps(job))

        if result is not None:
            result_key = f"{self.result_prefix}{job_id}"
            self.redis.setex(result_key, self.job_ttl, json.dumps(result))

        # Keep deduplication key alive after completion so client retries don't
        # re-enqueue the same work.
        if status in [JobStatus.READY, JobStatus.ERROR]:
            job_type = job.get("type")
            params = job.get("params")
            if job_type and params:
                params_hash = hashlib.sha256(str(params).encode()).hexdigest()[:16]
                dedup_key = f"job:dedup:{job_type}:{params_hash}"
                cooldown_ttl = 300 if status == JobStatus.READY else 120
                self.redis.setex(dedup_key, cooldown_ttl, job_id)

    def cleanup_expired_jobs(self) -> int:
        """Clean up expired jobs (Redis TTL handles this automatically)."""
        return 0
