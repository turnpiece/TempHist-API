"""
Background worker that runs as a thread within the FastAPI app.
This is an alternative to a separate worker service for environments
that don't support worker services (like Render free tier).
"""

import asyncio
import logging
import os
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class BackgroundWorker:
    """Background worker that runs in a separate thread."""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.running = False
        self.thread = None
        self.loop = None
        
    def start(self):
        """Start the background worker thread."""
        if self.running:
            logger.warning("Background worker is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_worker, daemon=True)
        self.thread.start()
        logger.info("🚀 Background worker thread started")
        logger.info(f"📊 Thread info: {self.thread.name}, daemon={self.thread.daemon}, alive={self.thread.is_alive()}")
        
    def stop(self):
        """Stop the background worker thread."""
        if not self.running:
            return
            
        self.running = False
        if self.loop and self.loop.is_running():
            # Schedule the loop to stop
            asyncio.run_coroutine_threadsafe(self._stop_loop(), self.loop)
            
        if self.thread:
            self.thread.join(timeout=5)
            
        logger.info("🛑 Background worker thread stopped")
        
    def _run_worker(self):
        """Run the worker in a separate thread with its own event loop."""
        # Create a new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            # Run the worker
            logger.info("🚀 Starting worker main loop...")
            self.loop.run_until_complete(self._worker_main())
        except Exception as e:
            logger.error(f"❌ Background worker error: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
        finally:
            logger.info("🛑 Background worker thread ending")
            self.loop.close()
            
    async def _stop_loop(self):
        """Stop the event loop gracefully."""
        # Cancel all tasks
        tasks = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
        for task in tasks:
            task.cancel()
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self.loop.stop()
        
    async def _worker_main(self):
        """Main worker loop."""
        logger.info("🚀 Background worker main loop started")
        
        try:
            # Ensure Redis client is configured for string responses
            if not hasattr(self.redis_client, 'decode_responses') or not self.redis_client.decode_responses:
                # Recreate Redis client with proper configuration
                import redis
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info("🔄 Recreated Redis client with decode_responses=True")
            
            # Test Redis connection
            try:
                self.redis_client.ping()
                logger.info("✅ Worker Redis connection verified")
            except Exception as e:
                logger.error(f"❌ Worker cannot connect to Redis: {e}")
                logger.warning("⚠️ Background worker will exit - Redis not available")
                return  # Exit gracefully instead of crashing
            
            # Set initial heartbeat
            try:
                self.redis_client.setex("worker:heartbeat", 60, datetime.now(timezone.utc).isoformat())
                logger.info("💓 Worker heartbeat initialized")
            except Exception as e:
                logger.warning(f"⚠️  Could not set heartbeat: {e}")
            
            # Import and start the job worker
            from job_worker import JobWorker
            worker = JobWorker(self.redis_client)
            await worker.start()
            
        except Exception as e:
            logger.error(f"❌ Background worker main loop error: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
        logger.info("🛑 Background worker main loop stopped")

# Global worker instance
background_worker = None

def start_background_worker(redis_client):
    """Start the background worker."""
    global background_worker
    
    if background_worker:
        logger.warning("Background worker is already initialized")
        return
        
    background_worker = BackgroundWorker(redis_client)
    background_worker.start()
    logger.info("✅ Background worker started")

def stop_background_worker():
    """Stop the background worker."""
    global background_worker
    
    if background_worker:
        background_worker.stop()
        background_worker = None
        logger.info("✅ Background worker stopped")
