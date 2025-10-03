#!/usr/bin/env python3
"""
Debug version of job worker to diagnose deployment issues.
This version has extensive logging and error handling to help identify problems.
"""

import os
import sys
import logging
import redis
import asyncio
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check environment variables and configuration."""
    logger.info("🔍 Checking environment configuration...")
    
    # Check required environment variables
    env_vars = {
        'REDIS_URL': os.getenv('REDIS_URL'),
        'VISUAL_CROSSING_API_KEY': os.getenv('VISUAL_CROSSING_API_KEY'),
        'OPENWEATHER_API_KEY': os.getenv('OPENWEATHER_API_KEY'),
        'API_ACCESS_TOKEN': os.getenv('API_ACCESS_TOKEN'),
        'PYTHON_VERSION': os.getenv('PYTHON_VERSION'),
        'PORT': os.getenv('PORT'),
    }
    
    for key, value in env_vars.items():
        if value:
            # Mask sensitive values
            if 'KEY' in key or 'TOKEN' in key:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                logger.info(f"  {key}: {masked_value}")
            else:
                logger.info(f"  {key}: {value}")
        else:
            logger.warning(f"  {key}: NOT SET")
    
    # Check Python version and paths
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path[:3]}...")  # First 3 entries

def check_redis_connection():
    """Check Redis connection and configuration."""
    logger.info("🔗 Testing Redis connection...")
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    logger.info(f"Redis URL: {redis_url}")
    
    try:
        # Try to connect to Redis
        redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Test basic operations
        redis_client.ping()
        logger.info("✅ Redis connection successful")
        
        # Test basic operations
        test_key = f"worker_test_{datetime.now().timestamp()}"
        redis_client.set(test_key, "test_value", ex=10)
        value = redis_client.get(test_key)
        redis_client.delete(test_key)
        
        if value == "test_value":
            logger.info("✅ Redis read/write operations successful")
        else:
            logger.error("❌ Redis read/write operations failed")
            
        return redis_client
        
    except redis.ConnectionError as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.error("  This usually means Redis is not running or not accessible")
        return None
    except Exception as e:
        logger.error(f"❌ Redis error: {e}")
        return None

def check_imports():
    """Check if all required modules can be imported."""
    logger.info("📦 Checking module imports...")
    
    modules_to_test = [
        'main',
        'cache_utils', 
        'routers.records_agg',
        'routers.locations_preapproved'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"  ✅ {module_name}")
        except ImportError as e:
            logger.error(f"  ❌ {module_name}: {e}")
        except Exception as e:
            logger.error(f"  ❌ {module_name}: {e}")

def check_job_queue():
    """Check if there are any pending jobs."""
    logger.info("📋 Checking job queue...")
    
    redis_client = check_redis_connection()
    if not redis_client:
        logger.error("Cannot check job queue without Redis connection")
        return
    
    try:
        # Check for pending jobs
        job_queue_key = "job_queue"
        pending_jobs = redis_client.llen(job_queue_key)
        logger.info(f"Pending jobs in queue: {pending_jobs}")
        
        # Check for processing jobs
        processing_key = "processing_jobs"
        processing_jobs = redis_client.llen(processing_key)
        logger.info(f"Jobs currently processing: {processing_jobs}")
        
        # List some recent job IDs
        job_keys = redis_client.keys("job:*")
        logger.info(f"Total job records in Redis: {len(job_keys)}")
        
        if job_keys:
            # Show a few recent job keys
            recent_keys = sorted(job_keys)[-5:]
            logger.info(f"Recent job keys: {recent_keys}")
            
            # Check status of a recent job
            if recent_keys:
                recent_job_key = recent_keys[-1]
                job_data = redis_client.hgetall(recent_job_key)
                logger.info(f"Recent job ({recent_job_key}): {job_data}")
        
    except Exception as e:
        logger.error(f"Error checking job queue: {e}")

async def test_job_processing():
    """Test basic job processing functionality."""
    logger.info("🧪 Testing job processing...")
    
    try:
        # Try to import job worker components
        from cache_utils import get_job_manager, get_cache
        
        logger.info("✅ Successfully imported job worker components")
        
        # Try to get job manager
        job_manager = get_job_manager()
        logger.info("✅ Job manager available")
        
        # Try to get cache
        cache = get_cache()
        logger.info("✅ Cache system available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Job processing test failed: {e}")
        return False

async def main():
    """Main diagnostic function."""
    logger.info("🚀 Starting job worker diagnostics...")
    logger.info("=" * 60)
    
    # Run all diagnostic checks
    check_environment()
    logger.info("-" * 40)
    
    check_imports()
    logger.info("-" * 40)
    
    redis_client = check_redis_connection()
    logger.info("-" * 40)
    
    if redis_client:
        check_job_queue()
        logger.info("-" * 40)
    
    await test_job_processing()
    logger.info("-" * 40)
    
    logger.info("🏁 Diagnostics complete")
    
    # If everything looks good, try to start the actual worker
    if redis_client:
        logger.info("🚀 Attempting to start job worker...")
        try:
            from job_worker import main as worker_main
            await worker_main()
        except Exception as e:
            logger.error(f"❌ Failed to start job worker: {e}")
    else:
        logger.error("❌ Cannot start job worker without Redis connection")

if __name__ == "__main__":
    asyncio.run(main())
