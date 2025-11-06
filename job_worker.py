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
    
    def _store_cache_data(self, cache_key: str, data: dict, data_type: str = "data") -> str:
        """Helper function to store data in cache and generate ETag."""
        from cache_utils import set_cache_value, CACHE_TTL_LONG
        from datetime import timedelta
        import hashlib
        
        try:
            # Convert TTL to timedelta (CACHE_TTL_LONG is in seconds)
            cache_duration = timedelta(seconds=CACHE_TTL_LONG)
            
            # Use the same cache storage function as the main endpoints
            set_cache_value(cache_key, cache_duration, json.dumps(data), self.redis)
            
            # Generate ETag using SHA256 (same as main endpoints)
            etag = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]
            
            logger.info(f"✅ Cached {data_type} for {cache_key}")
            return etag
            
        except Exception as cache_error:
            logger.warning(f"Cache storage failed for {cache_key}: {cache_error}")
            # Generate a simple ETag even if cache fails (use SHA256, not MD5)
            etag = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]
            return etag
    
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
            # MED-006: Improved error handling with detailed logging and error storage
            import traceback
            error_traceback = traceback.format_exc()
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_traceback,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.error(f"❌ Error processing job {job_id}: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ Full traceback:\n{error_traceback}")
            
            # Store detailed error in job status
            job_manager.update_job_status(
                job_id, 
                JobStatus.ERROR, 
                error=str(e),
                error_details=error_details  # Store full details for diagnostics
            )
            # Remove failed job from queue
            self.redis.lrem(self.job_queue_key, 1, job_id)
    
    async def process_record_job(self, params: Dict[str, Any], cache) -> Dict[str, Any]:
        """Process a record computation job with per-year granularity."""
        from routers.v1_records import get_temperature_data_v1, _extract_per_year_records, _get_ttl_for_current_year
        from cache_utils import (
            normalize_location_for_cache, rec_key, rec_etag_key,
            TTL_STABLE, ETagGenerator
        )
        from fastapi import HTTPException
        
        logger.info(f"🔍 Processing record job with params: {params}")
        scope = params.get("scope") or params.get("period")  # Support both for backward compatibility
        location = params.get("location")
        identifier = params.get("identifier")
        year = params.get("year")
        
        if not all([scope, location, identifier]):
            raise ValueError(f"Missing required params - scope: {scope}, location: {location}, identifier: {identifier}")
        
        # Normalize location to slug
        slug = normalize_location_for_cache(location)
        current_year = datetime.now(timezone.utc).year
        
        # If year is specified, fetch only that year's data
        if year is not None:
            logger.info(f"📅 Fetching data for year {year} only")
            
            # Fetch full data (get_temperature_data_v1 fetches all years, but we'll extract just the one we need)
            try:
                full_data = await get_temperature_data_v1(location, scope, identifier, "celsius", self.redis)
            except HTTPException as http_err:
                raise ValueError(f"{http_err.detail}") from http_err
            
            # Extract per-year records
            per_year_records = _extract_per_year_records(full_data)
            
            if year not in per_year_records:
                raise ValueError(f"No data found for year {year}")
            
            # Get the specific year's record
            year_record = per_year_records[year]
            
            # Determine TTL
            if year < current_year:
                ttl = TTL_STABLE
            else:
                ttl = _get_ttl_for_current_year(scope)
            
            # Generate cache keys
            year_key = rec_key(scope, slug, identifier, year)
            etag_key = rec_etag_key(scope, slug, identifier, year)
            
            # Generate ETag
            etag = ETagGenerator.generate_etag(year_record)
            
            # Store per-year record
            try:
                json_data = json.dumps(year_record, sort_keys=True, separators=(',', ':'))
                self.redis.setex(year_key, ttl, json_data)
                self.redis.setex(etag_key, ttl, etag)
                logger.info(f"✅ Cached per-year record: {year_key} (TTL: {ttl}s)")
            except Exception as e:
                logger.warning(f"Error caching per-year record {year_key}: {e}")
                raise
            
            return {
                "cache_key": year_key,
                "etag": etag,
                "year": year,
                "data": year_record,
                "computed_at": datetime.now(timezone.utc).isoformat()
            }
        else:
            # Backward compatibility: if no year specified, fetch all years (old behavior)
            logger.info(f"⚠️  No year specified, fetching all years (backward compatibility mode)")
            try:
                data = await get_temperature_data_v1(location, scope, identifier, "celsius", self.redis)
            except HTTPException as http_err:
                raise ValueError(f"{http_err.detail}") from http_err
            
            # Extract and store all per-year records
            per_year_records = _extract_per_year_records(data)
            for y, record_data in per_year_records.items():
                year_key = rec_key(scope, slug, identifier, y)
                etag_key = rec_etag_key(scope, slug, identifier, y)
                
                if y < current_year:
                    ttl = TTL_STABLE
                else:
                    ttl = _get_ttl_for_current_year(scope)
                
                etag = ETagGenerator.generate_etag(record_data)
                
                try:
                    json_data = json.dumps(record_data, sort_keys=True, separators=(',', ':'))
                    self.redis.setex(year_key, ttl, json_data)
                    self.redis.setex(etag_key, ttl, etag)
                except Exception as e:
                    logger.warning(f"Error caching year {y}: {e}")
            
            # Return summary
            return {
                "cache_keys": [rec_key(scope, slug, identifier, y) for y in per_year_records.keys()],
                "years_cached": list(per_year_records.keys()),
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
