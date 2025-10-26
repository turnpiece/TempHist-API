#!/usr/bin/env python3
"""
Monitor job creation and processing in real-time.
"""

import redis
import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def monitor_jobs():
    """Monitor job creation and processing."""
    
    print("🔍 MONITORING JOB CREATION AND PROCESSING")
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        # Connect to Redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        print("✅ Redis connection: OK")
    except Exception as e:
        print(f"❌ Redis connection: FAILED - {e}")
        return
    
    # Track previous state
    prev_queue_length = 0
    prev_job_keys = set()
    
    try:
        while True:
            current_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
            
            # Check queue length
            queue_length = r.llen("job_queue")
            if queue_length != prev_queue_length:
                if queue_length > prev_queue_length:
                    print(f"[{current_time}] 📥 New job(s) added to queue (total: {queue_length})")
                else:
                    print(f"[{current_time}] 📤 Job(s) removed from queue (total: {queue_length})")
                prev_queue_length = queue_length
            
            # Check for new job keys
            try:
                current_job_keys = set(r.keys("job:*"))
                new_keys = current_job_keys - prev_job_keys
                removed_keys = prev_job_keys - current_job_keys
                
                if new_keys:
                    for key in new_keys:
                        print(f"[{current_time}] 🆕 New job key: {key}")
                        # Check if job data exists immediately
                        job_data = r.get(key)
                        if job_data:
                            try:
                                job = json.loads(job_data)
                                status = job.get("status", "unknown")
                                print(f"    Status: {status}")
                            except:
                                print("    Invalid JSON data")
                        else:
                            print("    ❌ No job data found!")
                
                if removed_keys:
                    for key in removed_keys:
                        print(f"[{current_time}] 🗑️ Job key removed: {key}")
                
                prev_job_keys = current_job_keys
                
            except Exception as e:
                print(f"[{current_time}] ⚠️ Error checking job keys: {e}")
            
            # Check for jobs in queue without data
            if queue_length > 0:
                missing_data_jobs = []
                for i in range(min(queue_length, 10)):
                    job_id = r.lindex("job_queue", i)
                    if job_id:
                        job_key = f"job:{job_id}"
                        job_data = r.get(job_key)
                        if not job_data:
                            missing_data_jobs.append(job_id)
                
                if missing_data_jobs:
                    print(f"[{current_time}] ⚠️ Jobs in queue without data: {len(missing_data_jobs)}")
                    for job_id in missing_data_jobs[:3]:  # Show first 3
                        print(f"    - {job_id}")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_jobs()
