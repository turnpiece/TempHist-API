#!/usr/bin/env python3
"""
Quick script to check if jobs are being created but not processed.
"""

import redis
import json
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def check_worker_status():
    """Check if jobs are being created but not processed."""
    
    print("🔍 CHECKING WORKER STATUS")
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
    queue_length = r.llen("job_queue")
    print(f"📋 Jobs in queue: {queue_length}")
    
    if queue_length > 0:
        print("\n🔍 Examining jobs in queue:")
        for i in range(min(queue_length, 5)):
            job_id = r.lindex("job_queue", i)
            if job_id:
                job_key = f"job:{job_id}"
                job_data = r.get(job_key)
                if job_data:
                    job = json.loads(job_data)
                    status = job.get("status", "unknown")
                    created = job.get("created_at", "unknown")
                    job_type = job.get("type", "unknown")
                    
                    # Calculate age
                    try:
                        created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                        age_str = f"{age:.0f}s ago"
                    except:
                        age_str = "unknown age"
                    
                    print(f"  - {job_id}")
                    print(f"    Status: {status}")
                    print(f"    Type: {job_type}")
                    print(f"    Created: {age_str}")
                    
                    if age > 300:  # 5 minutes
                        print(f"    ⚠️  STUCK JOB (older than 5 minutes)")
    
    # Check worker heartbeat
    heartbeat = r.get("worker:heartbeat")
    if heartbeat:
        try:
            heartbeat_dt = datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - heartbeat_dt).total_seconds()
            print(f"\n💓 Worker heartbeat: {age:.0f}s ago")
            if age > 180:  # 3 minutes
                print("⚠️  Worker appears to be down (no recent heartbeat)")
            else:
                print("✅ Worker appears to be running")
        except:
            print(f"\n💓 Worker heartbeat: {heartbeat} (could not parse)")
    else:
        print("\n💓 Worker heartbeat: NOT FOUND")
        print("⚠️  Worker service is not running")
    
    print("\n" + "=" * 50)
    if queue_length > 0:
        print("🚨 ISSUE: Jobs are being created but not processed")
        print("   This indicates the worker service is not running")
        print("\n💡 SOLUTION:")
        print("   1. Check Railway dashboard for worker service")
        print("   2. If no worker service exists, Railway didn't create it from Procfile")
        print("   3. You may need to manually create a worker service in Railway")
    else:
        print("✅ No jobs in queue - system appears healthy")

if __name__ == "__main__":
    check_worker_status()
