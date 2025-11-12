#!/usr/bin/env python3
"""
Realistic test of job deduplication simulating cache warming scenario.
"""
import os
import redis
import json
import time
from cache_utils import initialize_cache, get_job_manager

def main():
    print("=== Realistic Job Deduplication Test ===\n")
    print("Simulating cache warming creating the same jobs repeatedly...\n")

    # Initialize
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    initialize_cache(redis_client)
    job_manager = get_job_manager()

    # Clear job queue
    print("Clearing job queue...")
    initial_length = redis_client.llen("job_queue")
    if initial_length > 0:
        redis_client.delete("job_queue")
    print(f"Cleared {initial_length} jobs\n")

    # Simulate cache warming for Auckland - creating same job 100 times
    # (like what happens when cache warming triggers for missing cache data)
    test_params = {
        "scope": "yearly",
        "slug": "auckland",
        "identifier": "11-12",
        "year": 2024,
        "location": {"name": "Auckland", "lat": -36.8485, "lon": 174.7633}
    }

    print("Simulating 100 cache warming requests for Auckland (same parameters)...")
    job_ids_created = []
    start_time = time.time()

    for i in range(100):
        job_id = job_manager.create_job("record_computation", test_params)
        job_ids_created.append(job_id)
        if i % 10 == 0:
            print(f"  Request {i}: job_id = {job_id}")

    elapsed = time.time() - start_time

    # Analysis
    unique_jobs = set(job_ids_created)
    final_queue_length = redis_client.llen("job_queue")

    print(f"\n📊 Results:")
    print(f"   Total job creation attempts: 100")
    print(f"   Unique job IDs returned: {len(unique_jobs)}")
    print(f"   Jobs actually in queue: {final_queue_length}")
    print(f"   Time elapsed: {elapsed:.2f}s")

    if len(unique_jobs) == 1:
        print(f"\n✅ EXCELLENT: Deduplication working perfectly!")
        print(f"   All 100 attempts returned the same job_id")
        print(f"   Prevented {100 - final_queue_length} duplicate jobs")
    elif len(unique_jobs) < 10:
        print(f"\n⚠️  PARTIAL SUCCESS: Only {len(unique_jobs)} unique jobs created")
        print(f"   Deduplication working but not perfect")
    else:
        print(f"\n❌ FAILURE: {len(unique_jobs)} different jobs created")
        print(f"   Deduplication not working effectively")

    # Test with 4 different locations (simulating real cache warming scenario)
    print(f"\n--- Testing with 4 different locations (25 requests each) ---")
    redis_client.delete("job_queue")

    locations = [
        ("berlin", {"name": "Berlin", "lat": 52.52, "lon": 13.405}),
        ("london", {"name": "London", "lat": 51.5074, "lon": -0.1278}),
        ("paris", {"name": "Paris", "lat": 48.8566, "lon": 2.3522}),
        ("tokyo", {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503})
    ]

    job_counts = {}
    for slug, location in locations:
        params = {
            "scope": "yearly",
            "slug": slug,
            "identifier": "11-12",
            "year": 2024,
            "location": location
        }
        job_ids = []
        for _ in range(25):
            job_id = job_manager.create_job("record_computation", params)
            job_ids.append(job_id)

        unique = set(job_ids)
        job_counts[slug] = len(unique)
        print(f"   {slug}: 25 attempts → {len(unique)} unique job(s)")

    total_unique = sum(job_counts.values())
    queue_length = redis_client.llen("job_queue")

    print(f"\n📊 Multi-location results:")
    print(f"   Total attempts: 100 (25 per location)")
    print(f"   Expected unique jobs: 4 (one per location)")
    print(f"   Actual unique jobs: {total_unique}")
    print(f"   Jobs in queue: {queue_length}")

    if total_unique == 4 and queue_length == 4:
        print(f"\n✅ PERFECT: Deduplication working correctly across multiple locations!")
    elif total_unique <= 10:
        print(f"\n⚠️  GOOD: Deduplication mostly working ({total_unique} jobs instead of 100)")
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: Created {total_unique} jobs instead of 4")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
