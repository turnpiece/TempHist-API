#!/usr/bin/env python3
"""
Debug script to check job queue and job data in Redis.
"""

import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def debug_jobs():
    """Debug job queue and job data."""
    
    print("🔍 DEBUGGING JOB QUEUE AND DATA")
    print("=" * 50)
    
    try:
        # Connect to Redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        print("✅ Redis connection: OK")
    except Exception as e:
        print(f"❌ Redis connection: FAILED - {e}")
        return
    
    # Check job queue
    queue_key = "job_queue"
    queue_length = r.llen(queue_key)
    print(f"\n📋 Job Queue Status:")
    print(f"  Queue length: {queue_length}")
    
    if queue_length > 0:
        print(f"\n📋 Jobs in queue:")
        for i in range(min(queue_length, 10)):
            job_id = r.lindex(queue_key, i)
            print(f"  {i+1}. {job_id}")
            
            # Check if job data exists
            job_key = f"job:{job_id}"
            job_data = r.get(job_key)
            
            if job_data:
                try:
                    job = json.loads(job_data)
                    status = job.get("status", "unknown")
                    created = job.get("created_at", "unknown")
                    print(f"     ✅ Job data exists - Status: {status}, Created: {created}")
                except:
                    print(f"     ⚠️ Job data exists but invalid JSON")
            else:
                print(f"     ❌ No job data found for {job_id}")
    else:
        print("  (empty queue)")
    
    # Check for any job keys in Redis
    print(f"\n🔍 All job keys in Redis:")
    try:
        # Note: This requires KEYS permission which might not be available
        job_keys = r.keys("job:*")
        print(f"  Found {len(job_keys)} job keys")
        
        for key in job_keys[:5]:  # Show first 5
            job_id = key.replace("job:", "")
            job_data = r.get(key)
            if job_data:
                try:
                    job = json.loads(job_data)
                    status = job.get("status", "unknown")
                    print(f"  - {job_id}: {status}")
                except:
                    print(f"  - {job_id}: invalid JSON")
            else:
                print(f"  - {job_id}: no data")
                
    except Exception as e:
        print(f"  ❌ Cannot list job keys: {e}")
    
    # Check job queue operations
    print(f"\n🔍 Job Queue Operations:")
    print(f"  Queue key: {queue_key}")
    print(f"  Queue TTL: {r.ttl(queue_key)} seconds (-1 = no expiry, -2 = doesn't exist)")

if __name__ == "__main__":
    debug_jobs()
