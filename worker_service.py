#!/usr/bin/env python3
"""
Standalone worker service for processing async jobs.

This can run as a separate service (e.g., on Railway) independent of the main API.
It continuously polls Redis for jobs and processes them.

Usage:
    python worker_service.py

Environment Variables:
    REDIS_URL: Redis connection URL
    VISUAL_CROSSING_API_KEY: API key for Visual Crossing
    DEBUG: Set to 'true' for debug logging
"""

import asyncio
import logging
import os
import signal
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('worker.log')
    ]
)

logger = logging.getLogger(__name__)

# Reduce verbosity of noisy third-party loggers
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

def validate_environment():
    """Validate required environment variables are set."""
    required_vars = {
        "REDIS_URL": os.getenv("REDIS_URL"),
        "VISUAL_CROSSING_API_KEY": os.getenv("VISUAL_CROSSING_API_KEY"),
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    
    logger.info("✅ Environment variables validated")
    return required_vars

async def main():
    """Main entry point for the worker service."""
    import redis
    from cache_utils import initialize_cache
    from job_worker import JobWorker
    
    logger.info("=" * 60)
    logger.info("🚀 ASYNC JOB WORKER SERVICE STARTING")
    logger.info("=" * 60)
    
    # Validate environment
    env_vars = validate_environment()
    
    # Connect to Redis
    redis_url = env_vars["REDIS_URL"]
    logger.info(f"📡 Connecting to Redis: {redis_url[:30]}...")
    
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Connected to Redis successfully")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        sys.exit(1)
    
    # Initialize cache system (required for job manager)
    logger.info("🔧 Initializing cache system...")
    try:
        initialize_cache(redis_client)
        logger.info("✅ Cache system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize cache system: {e}")
        sys.exit(1)
    
    # Create worker instance
    logger.info("🔧 Creating job worker instance...")
    worker = JobWorker(redis_client)
    logger.info("✅ Job worker instance created")
    
    # Setup graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info(f"📴 Received signal {signum}, initiating graceful shutdown...")
        worker.stop()
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("=" * 60)
    logger.info("✅ WORKER SERVICE READY")
    logger.info("=" * 60)
    
    # Start worker (will run until stopped)
    try:
        await worker.start()
    except Exception as e:
        logger.error(f"❌ Worker crashed: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        logger.info("🛑 Worker service stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Worker service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

