#!/usr/bin/env python3
"""
Diagnostic tool for async job processing.
Use this to troubleshoot job timeouts and background worker issues.
"""

import redis
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load environment
from dotenv import load_dotenv
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def diagnose_job_system():
    """Run comprehensive diagnostics on the async job system."""
    
    print("🔍 ASYNC JOB SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    try:
        # Connect to Redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        print("✅ Redis connection: OK")
        print(f"   URL: {REDIS_URL}")
    except Exception as e:
        print(f"❌ Redis connection: FAILED - {e}")
        return
    
    print("\n" + "=" * 60)
    print("📊 JOB QUEUE STATUS")
    print("=" * 60)
    
    # Check job queue
    try:
        queue_key = "job_queue"
        queue_length = r.llen(queue_key)
        print(f"Queue length: {queue_length}")
        
        if queue_length > 0:
            print(f"\n📋 Jobs in queue (first 10):")
            for i in range(min(queue_length, 10)):
                job_id = r.lindex(queue_key, i)
                print(f"  {i+1}. {job_id}")
        else:
            print("  (empty)")
    except Exception as e:
        print(f"❌ Error checking queue: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 JOBS IN QUEUE")
    print("=" * 60)
    
    try:
        # Get jobs from queue instead of scanning all keys (KEYS command may not be available)
        job_ids = []
        if queue_length > 0:
            for i in range(min(queue_length, 100)):
                job_id = r.lindex(queue_key, i)
                if job_id:
                    job_ids.append(job_id)
        
        print(f"Total jobs in queue: {len(job_ids)}")
        
        # Try to count all jobs using KEYS (may fail on restricted Redis)
        try:
            job_keys = r.keys("job:*")
            result_keys = r.keys("result:*")
            # Filter out etag keys
            job_keys = [k for k in job_keys if not k.endswith(':etag')]
            print(f"Total job keys found (via KEYS): {len(job_keys)}")
            print(f"Total result keys found (via KEYS): {len(result_keys)}")
        except Exception as keys_error:
            print(f"⚠️  Cannot scan all keys (KEYS command restricted): {keys_error}")
            print(f"   Will only examine jobs in the queue")
            job_keys = [f"job:{job_id}" for job_id in job_ids]
        
        if job_keys:
            print(f"\n📝 Job details:")
            jobs_by_status = {"pending": [], "processing": [], "ready": [], "error": []}
            
            for job_key in sorted(job_keys, reverse=True)[:20]:  # Show latest 20
                try:
                    job_data = r.get(job_key)
                    if job_data:
                        job = json.loads(job_data)
                        status = job.get("status", "unknown")
                        job_id = job.get("id", job_key.replace("job:", ""))
                        job_type = job.get("type", "unknown")
                        created = job.get("created_at", "unknown")
                        updated = job.get("updated_at", "unknown")
                        params = job.get("params", {})
                        error = job.get("error")
                        
                        # Calculate age
                        try:
                            created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                            age_str = f"{age:.0f}s"
                        except:
                            age_str = "unknown"
                        
                        jobs_by_status[status].append({
                            "id": job_id,
                            "type": job_type,
                            "status": status,
                            "age": age_str,
                            "params": params,
                            "error": error
                        })
                except Exception as e:
                    print(f"  ❌ Error reading {job_key}: {e}")
            
            # Print by status
            for status in ["pending", "processing", "ready", "error"]:
                jobs = jobs_by_status[status]
                if jobs:
                    status_emoji = {
                        "pending": "⏳",
                        "processing": "🔄",
                        "ready": "✅",
                        "error": "❌"
                    }
                    print(f"\n{status_emoji.get(status, '📄')} {status.upper()} ({len(jobs)}):")
                    for job in jobs[:10]:  # Show first 10 of each status
                        loc = job['params'].get('location', 'N/A')
                        print(f"  • {job['id'][:50]}...")
                        print(f"    Type: {job['type']}, Age: {job['age']}, Location: {loc}")
                        if job.get('error'):
                            print(f"    Error: {job['error']}")
    
    except Exception as e:
        print(f"❌ Error scanning jobs: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🔧 BACKGROUND WORKER STATUS")
    print("=" * 60)
    
    # Check for worker heartbeat (if implemented)
    try:
        heartbeat = r.get("worker:heartbeat")
        if heartbeat:
            print(f"✅ Worker heartbeat: {heartbeat}")
        else:
            print("⚠️  No worker heartbeat found")
            print("   This could mean:")
            print("   1. Background worker is not running")
            print("   2. Worker heartbeat is not implemented")
    except Exception as e:
        print(f"❌ Error checking heartbeat: {e}")
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    
    # Provide recommendations based on findings
    if queue_length > 0 and len(jobs_by_status["pending"]) > 0:
        print("⚠️  ISSUE DETECTED: Jobs stuck in PENDING state")
        print("\n   Possible causes:")
        print("   1. Background worker is not running")
        print("   2. Worker crashed or failed to start")
        print("   3. Worker cannot connect to Redis")
        print("   4. Worker is in an error loop")
        print("\n   Troubleshooting steps:")
        print("   1. Check server logs for background worker startup:")
        print("      grep 'Background worker' /path/to/logs")
        print("   2. Check for worker errors:")
        print("      grep 'Job worker error' /path/to/logs")
        print("   3. Restart the API server to restart the worker")
        print("   4. Check if worker thread is alive (see logs)")
    
    if len(jobs_by_status["processing"]) > 0:
        print("⚠️  ISSUE DETECTED: Jobs stuck in PROCESSING state")
        print("\n   Possible causes:")
        print("   1. Worker crashed while processing")
        print("   2. Job computation is taking too long")
        print("   3. Job computation threw an unhandled exception")
        print("\n   Troubleshooting steps:")
        print("   1. Check server logs for job processing errors")
        print("   2. Manually mark stuck jobs as error:")
        print("      python diagnose_jobs.py --cleanup-stuck")
    
    if len(jobs_by_status["error"]) > 0:
        print("⚠️  Jobs with errors detected")
        print("   Check the error messages above for details")
    
    print("\n" + "=" * 60)
    print("📋 QUICK ACTIONS")
    print("=" * 60)
    print("Clear all stuck jobs:")
    print("  python diagnose_jobs.py --clear-stuck")
    print("\nClear all jobs:")
    print("  python diagnose_jobs.py --clear-all")
    print("\nWatch job processing in real-time:")
    print("  watch -n 2 'python diagnose_jobs.py'")

def clear_stuck_jobs():
    """Clear jobs that are stuck in pending or processing state."""
    print("🧹 Clearing stuck jobs...")
    
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        
        # Get jobs from queue instead of scanning all keys
        queue_length = r.llen("job_queue")
        job_ids = []
        if queue_length > 0:
            for i in range(queue_length):
                job_id = r.lindex("job_queue", i)
                if job_id:
                    job_ids.append(job_id)
        
        stuck_count = 0
        
        for job_id in job_ids:
            job_key = f"job:{job_id}"
            job_data = r.get(job_key)
            if job_data:
                job = json.loads(job_data)
                status = job.get("status")
                created = job.get("created_at")
                
                # Check if job is stuck (older than 5 minutes and still pending/processing)
                try:
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                    
                    if age > 300 and status in ["pending", "processing"]:  # 5 minutes
                        print(f"  Marking as error: {job.get('id')}")
                        job["status"] = "error"
                        job["error"] = f"Job stuck in {status} state for {age:.0f}s"
                        job["updated_at"] = datetime.now(timezone.utc).isoformat()
                        r.setex(job_key, 3600, json.dumps(job))
                        
                        # Remove from queue
                        r.lrem("job_queue", 1, job.get("id"))
                        stuck_count += 1
                except:
                    pass
        
        print(f"✅ Cleared {stuck_count} stuck jobs")
        
    except Exception as e:
        print(f"❌ Error clearing stuck jobs: {e}")

def clear_all_jobs():
    """Clear all jobs and results."""
    print("🧹 Clearing ALL jobs and results...")
    print("⚠️  This will delete all job data!")
    
    response = input("Are you sure? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled")
        return
    
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        
        # Get all job IDs from queue
        queue_length = r.llen("job_queue")
        job_ids = []
        if queue_length > 0:
            for i in range(queue_length):
                job_id = r.lindex("job_queue", i)
                if job_id:
                    job_ids.append(job_id)
        
        count = 0
        
        # Delete job and result keys
        for job_id in job_ids:
            job_key = f"job:{job_id}"
            result_key = f"result:{job_id}"
            etag_key = f"{job_key}:etag"
            
            r.delete(job_key)
            r.delete(result_key)
            r.delete(etag_key)
            count += 3
        
        # Try to delete using KEYS if available
        try:
            job_keys = r.keys("job:*")
            result_keys = r.keys("result:*")
            for key in job_keys + result_keys:
                r.delete(key)
                count += 1
        except:
            print("⚠️  Could not use KEYS command, only cleared jobs in queue")
        
        # Clear queue
        r.delete("job_queue")
        
        print(f"✅ Cleared {count} keys and emptied job queue")
        
    except Exception as e:
        print(f"❌ Error clearing jobs: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clear-stuck":
            clear_stuck_jobs()
        elif sys.argv[1] == "--clear-all":
            clear_all_jobs()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage:")
            print("  python diagnose_jobs.py              # Run diagnostics")
            print("  python diagnose_jobs.py --clear-stuck  # Clear stuck jobs")
            print("  python diagnose_jobs.py --clear-all    # Clear all jobs")
    else:
        diagnose_job_system()

