#!/usr/bin/env python3
"""
Test script for async job processing functionality.
This script helps debug async job issues by testing the complete flow.
"""

import requests
import time
import json
import os
from typing import Dict, Any

class AsyncJobTester:
    def __init__(self, base_url: str = "http://localhost:8000", api_token: str = None):
        if api_token is None:
            api_token = os.getenv("TEST_TOKEN")
            if not api_token:
                raise ValueError("TEST_TOKEN environment variable must be set")
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        })

    def test_redis_connection(self) -> bool:
        """Test if Redis is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/test-redis")
            if response.status_code == 200:
                print("✅ Redis connection: OK")
                return True
            else:
                print(f"❌ Redis connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Redis connection error: {e}")
            return False

    def test_cache_stats(self) -> bool:
        """Test cache statistics endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/cache-stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Cache stats: {stats}")
                return True
            else:
                print(f"❌ Cache stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cache stats error: {e}")
            return False

    def create_async_job(self, period: str, location: str, identifier: str) -> Dict[str, Any]:
        """Create an async job and return job info."""
        try:
            response = self.session.post(
                f"{self.base_url}/v1/records/{period}/{location}/{identifier}/async"
            )
            
            print(f"Job creation response: {response.status_code}")
            if response.status_code == 202:
                job_info = response.json()
                print(f"✅ Job created: {job_info}")
                return job_info
            else:
                print(f"❌ Job creation failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Job creation error: {e}")
            return {}

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a job."""
        try:
            response = self.session.get(f"{self.base_url}/v1/jobs/{job_id}")
            
            if response.status_code == 200:
                status = response.json()
                print(f"Job status: {status}")
                return status
            elif response.status_code == 404:
                print(f"❌ Job not found: {job_id}")
                return {}
            else:
                print(f"❌ Job status check failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Job status error: {e}")
            return {}

    def poll_job_completion(self, job_id: str, max_wait: int = 30) -> Dict[str, Any]:
        """Poll job until completion or timeout."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.check_job_status(job_id)
            
            if not status:
                return {}
            
            job_status = status.get('status', 'unknown')
            print(f"Job {job_id} status: {job_status}")
            
            if job_status == 'ready':
                print(f"✅ Job completed successfully!")
                return status
            elif job_status == 'error':
                print(f"❌ Job failed: {status.get('error', 'Unknown error')}")
                return status
            elif job_status in ['pending', 'processing']:
                print(f"⏳ Job still {job_status}, waiting...")
                time.sleep(3)
            else:
                print(f"❓ Unknown job status: {job_status}")
                time.sleep(3)
        
        print(f"⏰ Job timed out after {max_wait} seconds")
        return status

    def test_complete_async_flow(self):
        """Test the complete async job flow."""
        print("🚀 Starting async job test...")
        
        # Test Redis connection
        if not self.test_redis_connection():
            print("❌ Cannot proceed without Redis connection")
            return False
        
        # Test cache stats
        self.test_cache_stats()
        
        # Create async job
        print("\n📝 Creating async job...")
        job_info = self.create_async_job("daily", "london", "01-15")
        
        if not job_info:
            print("❌ Failed to create job")
            return False
        
        job_id = job_info.get('job_id')
        if not job_id:
            print("❌ No job_id returned")
            return False
        
        # Poll for completion
        print(f"\n⏳ Polling job {job_id} for completion...")
        final_status = self.poll_job_completion(job_id)
        
        if final_status.get('status') == 'ready':
            print("✅ Async job test completed successfully!")
            print(f"Result keys: {list(final_status.get('result', {}).keys())}")
            return True
        else:
            print("❌ Async job test failed")
            return False

def main():
    """Main test function."""
    print("🧪 Async Job Processing Test")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ API server is not running. Please start with: uvicorn main:app --reload")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("Please start the API server with: uvicorn main:app --reload")
        return
    
    # Run tests
    tester = AsyncJobTester()
    success = tester.test_complete_async_flow()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed. Check the output above for details.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure Redis is running: redis-server")
        print("2. Make sure job worker is running: python job_worker.py")
        print("3. Make sure API server is running: uvicorn main:app --reload")
        print("4. Check Redis connection: curl http://localhost:8000/test-redis")

if __name__ == "__main__":
    main()
