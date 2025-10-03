"""
Background worker that runs as a thread within the FastAPI app.
This is an alternative to a separate worker service for environments
that don't support worker services (like Render free tier).
"""

import asyncio
import logging
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
            self.loop.run_until_complete(self._worker_main())
        except Exception as e:
            logger.error(f"Background worker error: {e}")
        finally:
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
            # Import job worker components
            from job_worker import JobWorker
            
            # Create and start the job worker
            worker = JobWorker(self.redis_client)
            await worker.start()
            
        except Exception as e:
            logger.error(f"Background worker main loop error: {e}")
            
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
