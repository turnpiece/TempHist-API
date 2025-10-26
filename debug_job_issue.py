#!/usr/bin/env python3
"""
Debug script to diagnose job data missing issue.
"""

import redis
import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def debug_job_issue():
    """Debug the job data missing issue."""
    
    print("🔍 DEBUGGING JOB DATA MISSING ISSUE")
    print("=" * 60)
    
    try:
        # Connect to Redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        print("✅ Redis connection: OK")
    except Exception as e:
        print(f"❌ Redis connection: FAILED - {e}")
        return
    
    # Check Redis info
    print(f"\n📊 Redis Info:")
    try:
        info = r.info()
        print(f"  Redis version: {info.get('redis_version', 'unknown')}")
        print(f"  Used memory: {info.get('used_memory_human', 'unknown')}")
        print(f"  Max memory: {info.get('maxmemory_human', 'unlimited')}")
        print(f"  Eviction policy: {info.get('maxmemory_policy', 'noeviction')}")
        print(f"  Connected clients: {info.get('connected_clients', 'unknown')}")
    except Exception as e:
        print(f"  Could not get Redis info: {e}")
    
    # Check job queue
    queue_key = "job_queue"
    queue_length = r.llen(queue_key)
    print(f"\n📋 Job Queue Status:")
    print(f"  Queue length: {queue_length}")
    
    if queue_length > 0:
        print(f"\n📋 Jobs in queue:")
        missing_data_count = 0
        for i in range(min(queue_length, 20)):  # Check up to 20 jobs
            job_id = r.lindex(queue_key, i)
            if job_id:
                print(f"  {i+1}. {job_id}")
                
                # Check if job data exists
                job_key = f"job:{job_id}"
                job_data = r.get(job_key)
                
                if job_data:
                    try:
                        job = json.loads(job_data)
                        status = job.get("status", "unknown")
                        created = job.get("created_at", "unknown")
                        
                        # Calculate age
                        try:
                            created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            age_seconds = (datetime.now(timezone.utc) - created_dt).total_seconds()
                            age_str = f"{age_seconds:.0f}s ago"
                        except:
                            age_str = "unknown age"
                        
                        print(f"     ✅ Job data exists - Status: {status}, Created: {age_str}")
                        
                        # Check TTL
                        ttl = r.ttl(job_key)
                        if ttl > 0:
                            print(f"     ⏰ TTL: {ttl}s remaining")
                        elif ttl == -1:
                            print(f"     ⏰ TTL: No expiration set")
                        else:
                            print(f"     ⏰ TTL: Expired")
                            
                    except Exception as e:
                        print(f"     ⚠️ Job data exists but invalid JSON: {e}")
                else:
                    print(f"     ❌ No job data found for {job_id}")
                    missing_data_count += 1
                    
                    # Check if there are any similar keys
                    try:
                        # Look for keys that might be related
                        similar_keys = r.keys(f"*{job_id}*")
                        if similar_keys:
                            print(f"     🔍 Found related keys: {similar_keys}")
                    except:
                        pass
        else:
            print(f"  (empty queue)")
        
        print(f"\n📊 Summary:")
        print(f"  Total jobs in queue: {queue_length}")
        print(f"  Jobs missing data: {missing_data_count}")
        if missing_data_count > 0:
            print(f"  Missing data rate: {(missing_data_count/queue_length)*100:.1f}%")
    
    # Check for any job keys in Redis
    print(f"\n🔍 All job keys in Redis:")
    try:
        job_keys = r.keys("job:*")
        print(f"  Found {len(job_keys)} job keys")
        
        if job_keys:
            # Show recent jobs
            recent_jobs = []
            for key in job_keys:
                try:
                    job_data = r.get(key)
                    if job_data:
                        job = json.loads(job_data)
                        created = job.get("created_at", "")
                        if created:
                            recent_jobs.append((key, created, job.get("status", "unknown")))
                except:
                    pass
            
            # Sort by creation time
            recent_jobs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Recent jobs:")
            for key, created, status in recent_jobs[:10]:  # Show last 10
                try:
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    age_seconds = (datetime.now(timezone.utc) - created_dt).total_seconds()
                    age_str = f"{age_seconds:.0f}s ago"
                except:
                    age_str = "unknown"
                print(f"    {key} - {status} - {age_str}")
                
    except Exception as e:
        print(f"  Could not scan job keys: {e}")
    
    # Check result keys
    print(f"\n🔍 Result keys in Redis:")
    try:
        result_keys = r.keys("result:*")
        print(f"  Found {len(result_keys)} result keys")
    except Exception as e:
        print(f"  Could not scan result keys: {e}")
    
    # Test Redis operations
    print(f"\n🧪 Testing Redis operations:")
    test_key = "test_job_debug"
    test_data = {"test": "data", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    try:
        # Test setex
        r.setex(test_key, 60, json.dumps(test_data))
        print("  ✅ setex operation: OK")
        
        # Test get
        retrieved = r.get(test_key)
        if retrieved:
            parsed = json.loads(retrieved)
            print("  ✅ get operation: OK")
        else:
            print("  ❌ get operation: FAILED")
        
        # Test ttl
        ttl = r.ttl(test_key)
        print(f"  ✅ ttl operation: {ttl}s")
        
        # Cleanup
        r.delete(test_key)
        print("  ✅ delete operation: OK")
        
    except Exception as e:
        print(f"  ❌ Redis operations test failed: {e}")

if __name__ == "__main__":
    debug_job_issue()
