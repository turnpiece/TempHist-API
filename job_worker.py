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
        
        poll_count = 0
        try:
            while self.running:
                await self.process_jobs()
                poll_count += 1
                
                # Update heartbeat every 60 seconds (log every 5 minutes for less noise)
                if poll_count % 60 == 0:
                    try:
                        self.redis.setex("worker:heartbeat", 180, datetime.now(timezone.utc).isoformat())
                        # Only log heartbeat every 5 minutes to reduce log volume
                        if poll_count % 300 == 0:
                            logger.info(f"💓 Worker heartbeat active (poll #{poll_count})")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not update heartbeat: {e}")
                
                await asyncio.sleep(1)  # Poll every second
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
    
    async def process_jobs(self):
        """Process pending jobs from the queue."""
        try:
            # Get all pending jobs
            job_manager = get_job_manager()
            cache = get_cache()
            
            # Find pending jobs (this is a simple implementation)
            # In production, you might want to use Redis Streams or a proper queue
            pending_jobs = await self.get_pending_jobs()
            
            if pending_jobs:
                logger.info(f"📋 Found {len(pending_jobs)} pending jobs")
            
            for job_id in pending_jobs:
                logger.info(f"🔄 Processing job: {job_id}")
                await self.process_job(job_id, job_manager, cache)
                
        except Exception as e:
            logger.error(f"❌ Error processing jobs: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
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
                        # Convert byte string to string if needed
                        if isinstance(job_id, bytes):
                            job_id = job_id.decode('utf-8')
                        
                        # Check if job is still pending
                        job_key = f"job:{job_id}"
                        job_data = self.redis.get(job_key)
                        if job_data:
                            job = json.loads(job_data)
                            job_status = job.get("status")
                            if job_status == JobStatus.PENDING:
                                pending_jobs.append(job_id)
                        else:
                            logger.warning(f"⚠️ No job data found for: {job_id}")
            
            return pending_jobs
            
        except Exception as e:
            logger.error(f"❌ Error getting pending jobs: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []
    
    async def process_job(self, job_id: str, job_manager, cache):
        """Process a single job."""
        start_time = datetime.now(timezone.utc)
        try:
            # Update job status to processing
            job_manager.update_job_status(job_id, JobStatus.PROCESSING)
            logger.info(f"🔄 Processing job: {job_id}")
            
            # Get job details
            job_data = job_manager.get_job_status(job_id)
            if not job_data:
                logger.error(f"❌ Job not found: {job_id}")
                # Remove from queue if job doesn't exist
                self.redis.lrem(self.job_queue_key, 1, job_id)
                return
            
            job_type = job_data.get("type")
            params = job_data.get("params", {})
            logger.info(f"📋 Job type: {job_type}, Params: {params}")
            
            # Process based on job type
            try:
                if job_type == "record_computation":
                    logger.info(f"🔢 Starting record computation...")
                    result = await self.process_record_job(params, cache)
                elif job_type == "rolling_bundle":
                    logger.info(f"📦 Starting rolling bundle computation...")
                    result = await self.process_rolling_bundle_job(params, cache)
                elif job_type == "cache_warming":
                    logger.info(f"🔥 Starting cache warming...")
                    result = await self.process_cache_warming_job(params, cache)
                else:
                    raise ValueError(f"Unknown job type: {job_type}")
            except Exception as compute_error:
                logger.error(f"❌ Computation error for job {job_id}")
                logger.error(f"❌ Error type: {type(compute_error).__name__}")
                logger.error(f"❌ Error message: {str(compute_error)}")
                logger.error(f"❌ Error repr: {repr(compute_error)}")
                import traceback
                logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
                raise
            
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Store result and mark job as ready
            job_manager.update_job_status(job_id, JobStatus.READY, result)
            logger.info(f"✅ Job completed: {job_id} (took {duration:.2f}s)")
            
            # Remove job from queue after successful completion
            self.redis.lrem(self.job_queue_key, 1, job_id)
            
        except Exception as e:
            logger.error(f"❌ Error processing job {job_id}: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            job_manager.update_job_status(job_id, JobStatus.ERROR, error=str(e))
            # Remove failed job from queue
            self.redis.lrem(self.job_queue_key, 1, job_id)
    
    async def process_record_job(self, params: Dict[str, Any], cache) -> Dict[str, Any]:
        """Process a record computation job."""
        from main import get_temperature_data_v1
        from fastapi import HTTPException
        
        logger.info(f"🔍 Processing record job with params: {params}")
        period = params.get("period")
        location = params.get("location")
        identifier = params.get("identifier")
        
        if not all([period, location, identifier]):
            raise ValueError(f"Missing required params - period: {period}, location: {location}, identifier: {identifier}")
        
        # Build cache key using the same format as the main endpoint
        from cache_utils import normalize_location_for_cache
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:values,average,trend,summary"
        
        # Compute the data - catch HTTPException and convert to regular exception
        try:
            data = await get_temperature_data_v1(location, period, identifier)
        except HTTPException as http_err:
            # Convert HTTPException to regular exception for job error handling
            raise ValueError(f"{http_err.detail}") from http_err
        
        # Store in cache using the same utilities as the main endpoint
        from cache_utils import set_cache_value, LONG_CACHE_DURATION
        try:
            # Use the same cache storage function as the main endpoint
            set_cache_value(cache_key, LONG_CACHE_DURATION, json.dumps(data), self.redis)
            
            # Generate ETag using the same method as the main endpoint
            import hashlib
            etag = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
            
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
        from fastapi import HTTPException
        
        location = params["location"]
        anchor = params["anchor"]
        unit_group = params.get("unit_group", "celsius")
        month_mode = params.get("month_mode", "rolling1m")
        days_back = params.get("days_back", 7)
        include = params.get("include")
        exclude = params.get("exclude")
        
        # Compute the data - catch HTTPException and convert to regular exception
        try:
            data = await _rolling_bundle_impl(
                location, anchor, unit_group, month_mode, days_back, include, exclude
            )
        except HTTPException as http_err:
            # Convert HTTPException to regular exception for job error handling
            raise ValueError(f"{http_err.detail}") from http_err
        
        # Build cache key using the same format as the rolling bundle endpoint
        cache_key = f"rolling_bundle:{location}:{anchor}:{unit_group}"
        
        # Store in cache using the same utilities as the rolling bundle endpoint
        from cache_utils import set_cache_value, LONG_CACHE_DURATION
        try:
            # Use the same cache storage function as the rolling bundle endpoint
            set_cache_value(cache_key, LONG_CACHE_DURATION, json.dumps(data), self.redis)
            
            # Generate ETag using the same method as the rolling bundle endpoint
            import hashlib
            etag = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
            
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
    
    async def process_cache_warming_job(self, params: Dict[str, Any], cache) -> Dict[str, Any]:
        """Process a cache warming job."""
        from cache_utils import get_cache_warmer
        
        logger.info(f"🔥 Processing cache warming job with params: {params}")
        
        # Get cache warmer instance
        cache_warmer = get_cache_warmer()
        if not cache_warmer:
            raise ValueError("Cache warmer not available")
        
        # Determine warming scope
        locations = params.get("locations", [])
        warming_type = params.get("type", "all")  # "all", "popular", "specific"
        
        results = {
            "job_type": "cache_warming",
            "warming_type": warming_type,
            "locations_requested": locations,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "results": {}
        }
        
        try:
            if warming_type == "all":
                # Warm all popular locations
                logger.info("🔥 Warming all popular locations...")
                warming_result = await cache_warmer.warm_all_locations()
                results["results"]["all_locations"] = warming_result
                
            elif warming_type == "popular":
                # Warm only popular locations (from usage tracking)
                logger.info("🔥 Warming popular locations...")
                popular_locations = cache_warmer.get_locations_to_warm()
                for location in popular_locations:
                    location_result = await cache_warmer.warm_location_data(location)
                    results["results"][location] = location_result
                    
            elif warming_type == "specific":
                # Warm specific locations
                if not locations:
                    raise ValueError("No locations specified for specific warming")
                
                logger.info(f"🔥 Warming specific locations: {locations}")
                for location in locations:
                    location_result = await cache_warmer.warm_location_data(location)
                    results["results"][location] = location_result
            else:
                raise ValueError(f"Unknown warming type: {warming_type}")
            
            results["status"] = "completed"
            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            # Calculate summary statistics
            total_endpoints = 0
            total_errors = 0
            successful_locations = 0
            
            for location, result in results["results"].items():
                if isinstance(result, dict):
                    if "warmed_endpoints" in result:
                        total_endpoints += len(result.get("warmed_endpoints", []))
                        total_errors += len(result.get("errors", []))
                        if result.get("warmed_endpoints"):
                            successful_locations += 1
            
            results["summary"] = {
                "total_locations": len(results["results"]),
                "successful_locations": successful_locations,
                "total_endpoints_warmed": total_endpoints,
                "total_errors": total_errors
            }
            
            logger.info(f"✅ Cache warming completed: {successful_locations} locations, {total_endpoints} endpoints")
            
        except Exception as e:
            logger.error(f"❌ Cache warming failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            results["failed_at"] = datetime.now(timezone.utc).isoformat()
            raise
        
        return results

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
    
    # Reduce verbosity of noisy third-party loggers
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
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
