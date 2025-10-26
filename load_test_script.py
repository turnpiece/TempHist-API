#!/usr/bin/env python3
"""
Load testing script for TempHist API to verify Cloudflare optimization.

Tests response times, cache hit rates, and system resilience under load.
"""

import asyncio
import argparse
import json
import logging
import statistics
import time
from typing import List, Dict, Any

import aiohttp
# Use statistics module instead of numpy for better compatibility
# import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadTestStats:
    """Collect and analyze load test statistics."""
    
    def __init__(self):
        self.requests = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.start_time = None
        self.end_time = None
    
    def add_request(self, duration: float, status_code: int, cache_hit: bool = None, error: bool = False):
        """Add a request result to statistics."""
        self.requests.append({
            "duration": duration,
            "status_code": status_code,
            "cache_hit": cache_hit,
            "timestamp": time.time()
        })
        
        if error:
            self.errors += 1
        elif cache_hit is not None:
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile without numpy dependency."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary statistics."""
        if not self.requests:
            return {"error": "No requests recorded"}
        
        durations = [r["duration"] for r in self.requests]
        successful_requests = [r for r in self.requests if r["status_code"] < 400]
        successful_durations = [r["duration"] for r in successful_requests]
        
        total_requests = len(self.requests)
        successful_count = len(successful_requests)
        error_rate = (self.errors / total_requests * 100) if total_requests > 0 else 0
        
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total * 100) if cache_total > 0 else 0
        
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        rps = total_requests / total_time if total_time > 0 else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_count,
            "error_rate": round(error_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "requests_per_second": round(rps, 2),
            "response_times": {
                "min": round(min(durations), 3),
                "max": round(max(durations), 3),
                "mean": round(statistics.mean(durations), 3),
                "median": round(statistics.median(durations), 3),
                "p95": round(self._percentile(durations, 95), 3),
                "p99": round(self._percentile(durations, 99), 3)
            },
            "successful_response_times": {
                "min": round(min(successful_durations), 3) if successful_durations else 0,
                "max": round(max(successful_durations), 3) if successful_durations else 0,
                "mean": round(statistics.mean(successful_durations), 3) if successful_durations else 0,
                "median": round(statistics.median(successful_durations), 3) if successful_durations else 0,
                "p95": round(self._percentile(successful_durations, 95), 3) if successful_durations else 0,
                "p99": round(self._percentile(successful_durations, 99), 3) if successful_durations else 0
            }
        }

class LoadTester:
    """Load tester for TempHist API."""
    
    def __init__(self, base_url: str, concurrent_requests: int = 10):
        self.base_url = base_url.rstrip('/')
        self.concurrent_requests = concurrent_requests
        self.stats = LoadTestStats()
        
        # Test endpoints
        self.endpoints = [
            "/v1/records/daily/New York, NY/01-15",
            "/v1/records/weekly/New York, NY/2024-01-15",
            "/v1/records/monthly/New York, NY/2024-01",
            "/v1/records/yearly/New York, NY/2024",
            "/v1/records/rolling-bundle/New York, NY/2024-01-15",
            "/v1/records/daily/Los Angeles, CA/01-15",
            "/v1/records/daily/Chicago, IL/01-15",
            "/v1/records/daily/Houston, TX/01-15",
            "/v1/records/daily/Phoenix, AZ/01-15"
        ]
    
    async def make_request(self, session: aiohttp.ClientSession, url: str) -> tuple[bool, float, int, bool]:
        """Make a single HTTP request and return results."""
        start_time = time.time()
        
        try:
            async with session.get(url) as response:
                duration = time.time() - start_time
                
                # Check if response was cached
                cache_hit = (
                    response.headers.get("X-Cache-Status") == "HIT" or
                    "cache" in response.headers.get("X-Cache", "").lower() or
                    response.status == 304
                )
                
                return True, duration, response.status, cache_hit
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Request failed for {url}: {e}")
            return False, duration, 500, False
    
    async def run_concurrent_requests(self, session: aiohttp.ClientSession, num_requests: int):
        """Run concurrent requests."""
        tasks = []
        
        for _ in range(num_requests):
            # Select random endpoint
            import random
            endpoint = random.choice(self.endpoints)
            url = f"{self.base_url}{endpoint}"
            
            task = self.make_request(session, url)
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.stats.add_request(0, 500, error=True)
            else:
                success, duration, status_code, cache_hit = result
                self.stats.add_request(duration, status_code, cache_hit, not success)
    
    async def run_load_test(self, total_requests: int, duration_seconds: int = None):
        """Run the load test."""
        logger.info(f"🚀 Starting load test: {total_requests} requests, {self.concurrent_requests} concurrent")
        
        self.stats.start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            if duration_seconds:
                # Run for specified duration
                end_time = time.time() + duration_seconds
                request_count = 0
                
                while time.time() < end_time and request_count < total_requests:
                    batch_size = min(self.concurrent_requests, total_requests - request_count)
                    await self.run_concurrent_requests(session, batch_size)
                    request_count += batch_size
                    
                    # Small delay between batches
                    await asyncio.sleep(0.1)
            else:
                # Run specified number of requests
                batches = total_requests // self.concurrent_requests
                remainder = total_requests % self.concurrent_requests
                
                for _ in range(batches):
                    await self.run_concurrent_requests(session, self.concurrent_requests)
                    await asyncio.sleep(0.1)
                
                if remainder > 0:
                    await self.run_concurrent_requests(session, remainder)
        
        self.stats.end_time = time.time()
        
        # Print results
        summary = self.stats.get_summary()
        logger.info("📊 Load test completed!")
        logger.info(f"Total requests: {summary['total_requests']}")
        logger.info(f"Success rate: {100 - summary['error_rate']:.1f}%")
        logger.info(f"Cache hit rate: {summary['cache_hit_rate']:.1f}%")
        logger.info(f"Requests per second: {summary['requests_per_second']:.1f}")
        logger.info(f"Response times (ms):")
        logger.info(f"  Mean: {summary['response_times']['mean'] * 1000:.1f}")
        logger.info(f"  Median: {summary['response_times']['median'] * 1000:.1f}")
        logger.info(f"  P95: {summary['response_times']['p95'] * 1000:.1f}")
        logger.info(f"  P99: {summary['response_times']['p99'] * 1000:.1f}")
        
        return summary
    
    async def run_warmup(self):
        """Run warmup requests to populate cache."""
        logger.info("🔥 Running cache warmup...")
        
        async with aiohttp.ClientSession() as session:
            for endpoint in self.endpoints:
                url = f"{self.base_url}{endpoint}"
                await self.make_request(session, url)
                await asyncio.sleep(0.1)
        
        logger.info("✅ Cache warmup completed")

async def main():
    """Main load testing function."""
    parser = argparse.ArgumentParser(description="Load test TempHist API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--requests", type=int, default=1000, help="Total number of requests")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--duration", type=int, help="Test duration in seconds (overrides --requests)")
    parser.add_argument("--warmup", action="store_true", help="Run cache warmup before test")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create load tester
    tester = LoadTester(args.base_url, args.concurrent)
    
    try:
        # Run warmup if requested
        if args.warmup:
            await tester.run_warmup()
        
        # Run load test
        summary = await tester.run_load_test(args.requests, args.duration)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"📄 Results saved to: {args.output}")
        
        # Check performance targets
        successful_rt = summary.get('successful_response_times', {})
        p95_ms = successful_rt.get('p95', 0) * 1000
        
        if p95_ms < 500:
            logger.info("✅ Performance target met: P95 < 500ms")
        else:
            logger.warning(f"⚠️  Performance target missed: P95 = {p95_ms:.1f}ms")
        
        cache_hit_rate = summary.get('cache_hit_rate', 0)
        if cache_hit_rate > 80:
            logger.info("✅ Cache hit rate target met: >80%")
        else:
            logger.warning(f"⚠️  Cache hit rate target missed: {cache_hit_rate:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Load test interrupted by user")
    except Exception as e:
        logger.error(f"💥 Load test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
