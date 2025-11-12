#!/usr/bin/env python3
"""
Admin script to clear the job queue.
Can be run directly or imported as a module.
"""
import redis
import os
import sys


def clear_job_queue(redis_url: str = None) -> dict:
    """
    Clear the job queue in Redis.

    Args:
        redis_url: Redis connection URL. If None, uses REDIS_URL env var or localhost.

    Returns:
        dict with status and counts
    """
    if redis_url is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    try:
        # Connect to Redis
        r = redis.from_url(redis_url, decode_responses=True)
        r.ping()

        # Get current queue length
        queue_length = r.llen("job_queue")

        if queue_length == 0:
            return {
                "status": "success",
                "message": "Queue was already empty",
                "jobs_cleared": 0
            }

        # Clear the queue
        r.delete("job_queue")

        # Verify
        new_length = r.llen("job_queue")

        return {
            "status": "success",
            "message": f"Cleared {queue_length} jobs from queue",
            "jobs_cleared": queue_length,
            "remaining_jobs": new_length
        }

    except redis.ConnectionError as e:
        return {
            "status": "error",
            "message": f"Redis connection error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }


if __name__ == "__main__":
    # When run as a script
    print("=== Job Queue Cleanup ===\n")

    redis_url = os.getenv("REDIS_URL")
    if not redis_url and len(sys.argv) > 1:
        redis_url = sys.argv[1]

    result = clear_job_queue(redis_url)

    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")

    if result['status'] == 'error':
        sys.exit(1)
