#!/usr/bin/env python3
"""
Test script to verify job deduplication is working correctly.
Creates the same job multiple times and checks if deduplication prevents duplicates.
"""
import os
import sys
import redis
import json
import time
from cache_utils import initialize_cache, get_job_manager

def main():
    print("=== Job Deduplication Test ===\n")

    # Initialize cache system first
    print("Initializing cache system...")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    initialize_cache(redis_client)

    # Get job manager
    job_manager = get_job_manager()
    if not job_manager:
        print("❌ ERROR: Could not initialize job manager")
        sys.exit(1)

    # Clear any existing deduplication keys for this test
    redis_client = job_manager.redis
    test_keys = redis_client.keys("job:dedup:record_computation:*")
    if test_keys:
        print(f"Cleaning up {len(test_keys)} existing deduplication keys...")
        for key in test_keys:
            redis_client.delete(key)

    # Get initial queue length
    initial_queue_length = redis_client.llen("job_queue")
    print(f"Initial job queue length: {initial_queue_length}\n")

    # Test parameters
    test_params = {
        "scope": "yearly",
        "slug": "test-location",
        "identifier": "11-12",
        "year": 2024,
        "location": {"name": "Test Location", "lat": 40.7128, "lon": -74.0060}
    }

    # Create the same job 10 times
    print("Creating the same job 10 times...")
    job_ids = []
    for i in range(10):
        job_id = job_manager.create_job("record_computation", test_params)
        job_ids.append(job_id)
        print(f"  Attempt {i+1}: job_id = {job_id}")
        time.sleep(0.1)  # Small delay to ensure timestamps would differ

    # Check if all job IDs are the same (deduplication working)
    unique_job_ids = set(job_ids)
    print(f"\nUnique job IDs created: {len(unique_job_ids)}")

    if len(unique_job_ids) == 1:
        print("✅ SUCCESS: All 10 attempts returned the same job_id (deduplication working!)")
    else:
        print(f"❌ FAILURE: Created {len(unique_job_ids)} different jobs instead of 1")
        print(f"Job IDs: {unique_job_ids}")

    # Check final queue length
    final_queue_length = redis_client.llen("job_queue")
    jobs_added = final_queue_length - initial_queue_length
    print(f"\nJobs added to queue: {jobs_added}")

    if jobs_added == 1:
        print("✅ SUCCESS: Only 1 job added to queue (deduplication prevented 9 duplicates)")
    else:
        print(f"❌ FAILURE: {jobs_added} jobs added to queue instead of 1")

    # Check the deduplication key exists
    import hashlib
    params_hash = hashlib.sha256(str(test_params).encode()).hexdigest()[:16]
    dedup_key = f"job:dedup:record_computation:{params_hash}"
    dedup_value = redis_client.get(dedup_key)

    if dedup_value:
        print(f"\n✅ Deduplication key exists: {dedup_key}")
        print(f"   Points to job: {dedup_value.decode() if isinstance(dedup_value, bytes) else dedup_value}")
    else:
        print(f"\n❌ WARNING: Deduplication key not found: {dedup_key}")

    # Test with different parameters
    print("\n--- Testing with different parameters ---")
    different_params = test_params.copy()
    different_params["year"] = 2025  # Different year

    job_id_different = job_manager.create_job("record_computation", different_params)
    print(f"Different params job_id: {job_id_different}")

    if job_id_different != job_ids[0]:
        print("✅ SUCCESS: Different parameters created a new job")
    else:
        print("❌ FAILURE: Different parameters returned same job_id")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
