"""
Background job worker for processing async API requests.

This worker processes jobs from the job queue and stores results in Redis cache.
It runs as a background task in the FastAPI application lifecycle.
"""

import asyncio
import json
import logging
import redis
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, Any

from cache_utils import get_job_manager, JobStatus

logger = logging.getLogger(__name__)

# Job worker constants
JOB_PROCESSING_BATCH_SIZE = 10  # Maximum jobs to process in one cycle
WORKER_POLL_INTERVAL_SECONDS = 1  # Time between job queue polls
WORKER_HEARTBEAT_INTERVAL_CYCLES = 60  # Update heartbeat every 60 poll cycles (60 seconds)
WORKER_DEEP_CLEANUP_INTERVAL_CYCLES = 300  # Deep queue cleanup every 300 cycles (~5 minutes)
WORKER_DEEP_CLEANUP_BATCH_SIZE = 1000  # Max entries to scan during deep cleanup
WORKER_HEARTBEAT_LOG_INTERVAL_CYCLES = 300  # Log heartbeat every 300 cycles (5 minutes)

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
                            job_id = job_id.decode('utf-8')

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
                    job_id = job_id.decode('utf-8')

                job_data = self.redis.get(f"job:{job_id}")
                if not job_data:
                    stale_ids.append(job_id)
                else:
                    status = json.loads(job_data).get("status")
                    if status not in (JobStatus.PENDING, JobStatus.PROCESSING):
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

    async def process_job(self, job_id: str, job_manager):
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
                    result = await self.process_record_job(params)
                elif job_type == "cache_warming":
                    logger.info(f"🔥 Starting cache warming...")
                    result = await self.process_cache_warming_job(params)
                else:
                    raise ValueError(f"Unknown job type: {job_type}")
            except (ValueError, KeyError, TypeError) as data_error:
                logger.error(f"❌ Data error for job {job_id}: {type(data_error).__name__}")
                logger.error(f"❌ Error message: {str(data_error)}")
                import traceback
                logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
                raise
            except (redis.RedisError, json.JSONDecodeError) as cache_error:
                logger.error(f"❌ Cache error for job {job_id}: {type(cache_error).__name__}")
                logger.error(f"❌ Error message: {str(cache_error)}")
                import traceback
                logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
                raise
            except Exception as unexpected_error:
                # Catch any other unexpected errors but log them distinctly
                logger.error(f"❌ Unexpected error for job {job_id}: {type(unexpected_error).__name__}")
                logger.error(f"❌ Error message: {str(unexpected_error)}")
                logger.error(f"❌ Error repr: {repr(unexpected_error)}")
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
            
        except redis.RedisError as redis_error:
            # Redis errors - don't update job status if Redis is down
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"❌ Redis error processing job {job_id}: {type(redis_error).__name__}: {str(redis_error)}")
            logger.error(f"❌ Full traceback:\n{error_traceback}")
            # Cannot update job status or remove from queue if Redis is failing

        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as data_error:
            # Data/validation errors - store in job status
            import traceback
            error_traceback = traceback.format_exc()
            error_details = {
                "error_type": type(data_error).__name__,
                "error_message": str(data_error),
                "traceback": error_traceback,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.error(f"❌ Data error processing job {job_id}: {type(data_error).__name__}: {str(data_error)}")
            logger.error(f"❌ Full traceback:\n{error_traceback}")

            # Store detailed error in job status
            job_manager.update_job_status(
                job_id,
                JobStatus.ERROR,
                error=str(data_error),
                error_details=error_details
            )
            # Remove failed job from queue
            self.redis.lrem(self.job_queue_key, 1, job_id)

        except Exception as e:
            # Unexpected errors - log and store
            import traceback
            error_traceback = traceback.format_exc()
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": error_traceback,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            logger.error(f"❌ Unexpected error processing job {job_id}: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ Full traceback:\n{error_traceback}")

            # Store detailed error in job status
            job_manager.update_job_status(
                job_id,
                JobStatus.ERROR,
                error=str(e),
                error_details=error_details
            )
            # Remove failed job from queue
            self.redis.lrem(self.job_queue_key, 1, job_id)
    
    async def process_record_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a record computation job with per-year granularity."""
        from routers.v1_records import get_temperature_data_v1, _extract_per_year_records, _get_ttl_for_current_year, _rebuild_full_response_from_values
        from cache_utils import (
            normalize_location_for_cache, rec_key, rec_etag_key,
            get_ttl_for_year, ETagGenerator
        )
        from utils.weather import get_year_range
        from fastapi import HTTPException

        logger.info(f"🔍 Processing record job with params: {params}")
        scope = params.get("scope") or params.get("period")  # Support both for backward compatibility
        location = params.get("location")
        identifier = params.get("identifier")
        year = params.get("year")
        unit_group = params.get("unit_group", "celsius")

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
            
            # Determine TTL based on data age
            if year < current_year:
                ttl = get_ttl_for_year(year)
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
            except (redis.RedisError, redis.ConnectionError) as redis_error:
                logger.warning(f"Redis error caching per-year record {year_key}: {redis_error}")
                raise
            except (json.JSONEncodeError, TypeError) as serialize_error:
                logger.warning(f"Serialization error caching per-year record {year_key}: {serialize_error}")
                raise
            
            # Rebuild single-year result in the requested unit for the response
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
                "computed_at": datetime.now(timezone.utc).isoformat()
            }
        else:
            # Default behavior: if no year specified, fetch all available years for the identifier
            logger.info("ℹ️ No year specified; returning the full historical range")
            try:
                data = await get_temperature_data_v1(location, scope, identifier, "celsius", self.redis)
            except HTTPException as http_err:
                raise ValueError(f"{http_err.detail}") from http_err

            # Extract and store all per-year records (celsius) using Redis pipeline
            per_year_records = _extract_per_year_records(data)

            try:
                # Use pipeline to batch all cache operations
                pipeline = self.redis.pipeline()

                for y, record_data in per_year_records.items():
                    year_key = rec_key(scope, slug, identifier, y)
                    etag_key = rec_etag_key(scope, slug, identifier, y)

                    if y < current_year:
                        ttl = get_ttl_for_year(y)
                    else:
                        ttl = _get_ttl_for_current_year(scope)

                    etag = ETagGenerator.generate_etag(record_data)
                    json_data = json.dumps(record_data, sort_keys=True, separators=(',', ':'))

                    # Add to pipeline instead of executing immediately
                    pipeline.setex(year_key, ttl, json_data)
                    pipeline.setex(etag_key, ttl, etag)

                # Execute all operations in a single round trip
                pipeline.execute()
                logger.info(f"✅ Cached {len(per_year_records)} years using pipeline")

            except (redis.RedisError, json.JSONEncodeError, TypeError) as e:
                logger.warning(f"Error caching years with pipeline: {type(e).__name__}: {e}")

            # Rebuild full response in the requested unit
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
                "computed_at": datetime.now(timezone.utc).isoformat()
            }
    
    async def process_cache_warming_job(self, params: Dict[str, Any]) -> Dict[str, Any]:
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
