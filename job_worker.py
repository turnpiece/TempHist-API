"""
Background job worker for processing async API requests.

This worker processes jobs from the job queue and stores results in Redis cache.
It runs as a background task in the FastAPI application lifecycle.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, Any

from cache_utils import get_job_manager, get_cache, JobStatus
from routers.records_agg import rolling_bundle as rolling_bundle_func

logger = logging.getLogger(__name__)

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
        
        try:
            while self.running:
                await self.process_jobs()
                await asyncio.sleep(1)  # Poll every second
        except Exception as e:
            logger.error(f"Job worker error: {e}")
        finally:
            logger.info("🛑 Job worker stopped")
    
    def stop(self):
        """Stop the job worker."""
        self.running = False
        logger.info("📴 Job worker stop requested")
    
    async def process_jobs(self):
        """Process pending jobs from the queue."""
        try:
            # Get all pending jobs
            job_manager = get_job_manager()
            cache = get_cache()
            
            # Find pending jobs (this is a simple implementation)
            # In production, you might want to use Redis Streams or a proper queue
            pending_jobs = await self.get_pending_jobs()
            
            for job_id in pending_jobs:
                await self.process_job(job_id, job_manager, cache)
                
        except Exception as e:
            logger.error(f"Error processing jobs: {e}")
    
    async def get_pending_jobs(self) -> list:
        """Get list of pending job IDs from the job queue."""
        try:
            # Use Redis LIST operations instead of KEYS command
            # This is more efficient and doesn't require KEYS permission
            pending_jobs = []
            
            # Check if we have a job queue
            queue_length = self.redis.llen(self.job_queue_key)
            if queue_length > 0:
                # Get jobs from the queue (without removing them)
                for i in range(min(queue_length, 10)):  # Process up to 10 jobs at a time
                    job_id = self.redis.lindex(self.job_queue_key, i)
                    if job_id:
                        # Check if job is still pending
                        job_key = f"job:{job_id}"
                        job_data = self.redis.get(job_key)
                        if job_data:
                            job = json.loads(job_data)
                            if job.get("status") == JobStatus.PENDING:
                                pending_jobs.append(job_id)
            
            return pending_jobs
            
        except Exception as e:
            logger.error(f"Error getting pending jobs: {e}")
            return []
    
    async def process_job(self, job_id: str, job_manager, cache):
        """Process a single job."""
        try:
            # Update job status to processing
            job_manager.update_job_status(job_id, JobStatus.PROCESSING)
            logger.info(f"🔄 Processing job: {job_id}")
            
            # Get job details
            job_data = job_manager.get_job_status(job_id)
            if not job_data:
                logger.error(f"Job not found: {job_id}")
                # Remove from queue if job doesn't exist
                self.redis.lrem(self.job_queue_key, 1, job_id)
                return
            
            job_type = job_data.get("type")
            params = job_data.get("params", {})
            
            # Process based on job type
            if job_type == "record_computation":
                result = await self.process_record_job(params, cache)
            elif job_type == "rolling_bundle":
                result = await self.process_rolling_bundle_job(params, cache)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            
            # Store result and mark job as ready
            job_manager.update_job_status(job_id, JobStatus.READY, result)
            logger.info(f"✅ Job completed: {job_id}")
            
            # Remove job from queue after successful completion
            self.redis.lrem(self.job_queue_key, 1, job_id)
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            job_manager.update_job_status(job_id, JobStatus.ERROR, error=str(e))
            # Remove failed job from queue
            self.redis.lrem(self.job_queue_key, 1, job_id)
    
    async def process_record_job(self, params: Dict[str, Any], cache) -> Dict[str, Any]:
        """Process a record computation job."""
        from main import get_temperature_data_v1
        
        period = params["period"]
        location = params["location"]
        identifier = params["identifier"]
        
        # Build cache key
        from cache_utils import CacheKeyBuilder
        cache_key = CacheKeyBuilder.build_cache_key(
            "v1/records",
            {"period": period, "location": location, "identifier": identifier}
        )
        
        # Compute the data
        data = await get_temperature_data_v1(location, period, identifier)
        
        # Store in cache using simple Redis string operations (avoid type conflicts)
        from cache_utils import CACHE_TTL_LONG
        try:
            # Use simple Redis string operations to avoid hash/string conflicts
            cache_data = json.dumps(data)
            self.redis.setex(cache_key, CACHE_TTL_LONG, cache_data)
            
            # Generate ETag
            import hashlib
            etag = hashlib.md5(cache_data.encode()).hexdigest()
            
            # Store ETag separately
            etag_key = f"{cache_key}:etag"
            self.redis.setex(etag_key, CACHE_TTL_LONG, etag)
            
            logger.info(f"✅ Cached data for {cache_key}")
        except Exception as cache_error:
            logger.warning(f"Cache storage failed for {cache_key}: {cache_error}")
            # Generate a simple ETag even if cache fails
            import hashlib
            etag = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        return {
            "cache_key": cache_key,
            "etag": etag,
            "data": data,
            "computed_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def process_rolling_bundle_job(self, params: Dict[str, Any], cache) -> Dict[str, Any]:
        """Process a rolling bundle job."""
        from routers.records_agg import _rolling_bundle_impl
        
        location = params["location"]
        anchor = params["anchor"]
        unit_group = params.get("unit_group", "celsius")
        month_mode = params.get("month_mode", "rolling1m")
        days_back = params.get("days_back", 7)
        include = params.get("include")
        exclude = params.get("exclude")
        
        # Compute the data
        data = await _rolling_bundle_impl(
            location, anchor, unit_group, month_mode, days_back, include, exclude
        )
        
        # Build cache key
        from cache_utils import CacheKeyBuilder
        cache_key = CacheKeyBuilder.build_cache_key(
            "v1/records/rolling-bundle",
            {"location": location, "anchor": anchor},
            {
                "unit_group": unit_group,
                "month_mode": month_mode,
                "days_back": days_back,
                "include": include,
                "exclude": exclude
            }
        )
        
        # Store in cache using simple Redis string operations (avoid type conflicts)
        from cache_utils import CACHE_TTL_LONG
        try:
            # Use simple Redis string operations to avoid hash/string conflicts
            cache_data = json.dumps(data)
            self.redis.setex(cache_key, CACHE_TTL_LONG, cache_data)
            
            # Generate ETag
            import hashlib
            etag = hashlib.md5(cache_data.encode()).hexdigest()
            
            # Store ETag separately
            etag_key = f"{cache_key}:etag"
            self.redis.setex(etag_key, CACHE_TTL_LONG, etag)
            
            logger.info(f"✅ Cached rolling bundle data for {cache_key}")
        except Exception as cache_error:
            logger.warning(f"Cache storage failed for {cache_key}: {cache_error}")
            # Generate a simple ETag even if cache fails
            import hashlib
            etag = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        return {
            "cache_key": cache_key,
            "etag": etag,
            "data": data,
            "computed_at": datetime.now(timezone.utc).isoformat()
        }

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
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        if worker:
            worker.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

async def main():
    """Main function to run the job worker."""
    import redis
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
    from cache_utils import initialize_cache
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
