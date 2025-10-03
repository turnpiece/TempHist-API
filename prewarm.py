#!/usr/bin/env python3
"""
Cache prewarming script for TempHist API.

This script prewarms the cache for popular locations and common date patterns
to ensure fast response times for typical API usage.

Usage:
    python prewarm.py [--locations N] [--days N] [--endpoints ENDPOINTS] [--verbose]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import aiohttp
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
DEFAULT_LOCATIONS = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
    "Philadelphia, PA", "San Antonio, TX", "San Diego, CA", "Dallas, TX", "San Jose, CA",
    "Austin, TX", "Jacksonville, FL", "Fort Worth, TX", "Columbus, OH", "Charlotte, NC",
    "San Francisco, CA", "Indianapolis, IN", "Seattle, WA", "Denver, CO", "Washington, DC"
]

DEFAULT_ENDPOINTS = [
    "v1/records/daily",
    "v1/records/weekly", 
    "v1/records/monthly",
    "v1/records/yearly",
    "v1/records/rolling-bundle"
]

class PrewarmStats:
    """Track prewarming statistics."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_time = 0
        self.request_times = []
        
    def add_request(self, success: bool, duration: float, cache_hit: bool = None):
        """Add a request result to statistics."""
        self.total_requests += 1
        self.total_time += duration
        self.request_times.append(duration)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        if cache_hit is not None:
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
        avg_time = (self.total_time / self.total_requests) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": hit_rate,
            "avg_request_time": round(avg_time, 3),
            "total_time": round(self.total_time, 3)
        }

class CachePrewarmer:
    """Cache prewarmer for popular locations and endpoints."""
    
    def __init__(self, base_url: str, redis_url: str = None):
        self.base_url = base_url.rstrip('/')
        self.redis = redis.from_url(redis_url) if redis_url else None
        self.stats = PrewarmStats()
        
    async def prewarm_location(self, location: str, endpoints: List[str], days: int = 7) -> Dict[str, Any]:
        """Prewarm cache for a specific location."""
        logger.info(f"🔥 Prewarming cache for: {location}")
        
        location_stats = {
            "location": location,
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "cache_misses": 0,
            "total_time": 0
        }
        
        # Generate date patterns to prewarm
        date_patterns = self._generate_date_patterns(days)
        
        for endpoint in endpoints:
            for date_pattern in date_patterns:
                url = self._build_url(endpoint, location, date_pattern)
                
                start_time = time.time()
                success, cache_hit = await self._make_request(url)
                duration = time.time() - start_time
                
                location_stats["requests"] += 1
                location_stats["total_time"] += duration
                
                if success:
                    location_stats["successes"] += 1
                else:
                    location_stats["failures"] += 1
                    
                if not cache_hit:
                    location_stats["cache_misses"] += 1
                
                self.stats.add_request(success, duration, cache_hit)
                
                # Small delay to avoid overwhelming the server
                await asyncio.sleep(0.1)
        
        return location_stats
    
    def _generate_date_patterns(self, days: int) -> List[Dict[str, str]]:
        """Generate date patterns for prewarming."""
        patterns = []
        today = datetime.now()
        
        # Today's date patterns
        today_str = today.strftime("%m-%d")
        patterns.append({"identifier": today_str, "anchor": today.strftime("%Y-%m-%d")})
        
        # Recent days
        for i in range(1, min(days + 1, 30)):
            date = today - timedelta(days=i)
            date_str = date.strftime("%m-%d")
            patterns.append({"identifier": date_str, "anchor": date.strftime("%Y-%m-%d")})
        
        # Monthly patterns (same month-day across years)
        current_month = today.month
        current_day = today.day
        patterns.append({"identifier": f"{current_month:02d}-{current_day:02d}"})
        
        # Weekly patterns (recent weeks)
        for i in range(4):  # Last 4 weeks
            week_start = today - timedelta(weeks=i, days=today.weekday())
            patterns.append({"identifier": week_start.strftime("%Y-%m-%d")})
        
        return patterns
    
    def _build_url(self, endpoint: str, location: str, date_pattern: Dict[str, str]) -> str:
        """Build URL for the given endpoint and parameters."""
        if endpoint == "v1/records/daily":
            return f"{self.base_url}/v1/records/daily/{location}/{date_pattern['identifier']}"
        elif endpoint == "v1/records/weekly":
            return f"{self.base_url}/v1/records/weekly/{location}/{date_pattern['identifier']}"
        elif endpoint == "v1/records/monthly":
            return f"{self.base_url}/v1/records/monthly/{location}/{date_pattern['identifier']}"
        elif endpoint == "v1/records/yearly":
            return f"{self.base_url}/v1/records/yearly/{location}/{date_pattern['identifier']}"
        elif endpoint == "v1/records/rolling-bundle":
            return f"{self.base_url}/v1/records/rolling-bundle/{location}/{date_pattern['anchor']}"
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
    
    async def _make_request(self, url: str) -> tuple[bool, bool]:
        """Make a request and return (success, cache_hit)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # Check if response was cached
                        cache_hit = response.headers.get("X-Cache-Status") == "HIT" or \
                                   "cache" in response.headers.get("X-Cache", "").lower()
                        return True, cache_hit
                    elif response.status == 304:
                        # Not Modified - this is a cache hit
                        return True, True
                    else:
                        logger.warning(f"Request failed with status {response.status}: {url}")
                        return False, False
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            return False, False
    
    async def prewarm_popular_locations(self, locations: List[str], endpoints: List[str], days: int = 7):
        """Prewarm cache for multiple popular locations."""
        logger.info(f"🚀 Starting cache prewarming for {len(locations)} locations")
        
        start_time = time.time()
        results = []
        
        for location in locations:
            try:
                result = await self.prewarm_location(location, endpoints, days)
                results.append(result)
                logger.info(f"✅ Completed {location}: {result['successes']}/{result['requests']} requests, {result['cache_misses']} cache misses")
            except Exception as e:
                logger.error(f"❌ Failed to prewarm {location}: {e}")
                results.append({
                    "location": location,
                    "error": str(e),
                    "requests": 0,
                    "successes": 0,
                    "failures": 0
                })
        
        total_time = time.time() - start_time
        stats = self.stats.get_stats()
        
        logger.info(f"🏁 Prewarming completed in {total_time:.2f} seconds")
        logger.info(f"📊 Stats: {stats['successful_requests']}/{stats['total_requests']} requests successful")
        logger.info(f"📊 Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        logger.info(f"📊 Average request time: {stats['avg_request_time']:.3f}s")
        
        return {
            "total_time": total_time,
            "stats": stats,
            "location_results": results
        }

async def main():
    """Main prewarming function."""
    parser = argparse.ArgumentParser(description="Prewarm TempHist API cache")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the API")
    parser.add_argument("--locations", type=int, default=10, help="Number of locations to prewarm")
    parser.add_argument("--days", type=int, default=7, help="Number of days to prewarm")
    parser.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS, help="Endpoints to prewarm")
    parser.add_argument("--redis-url", help="Redis URL for cache inspection")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--locations-file", help="JSON file with custom locations list")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load locations
    if args.locations_file:
        with open(args.locations_file, 'r') as f:
            custom_locations = json.load(f)
        locations = custom_locations[:args.locations]
    else:
        locations = DEFAULT_LOCATIONS[:args.locations]
    
    # Create prewarmer
    prewarmer = CachePrewarmer(args.base_url, args.redis_url)
    
    # Run prewarming
    try:
        results = await prewarmer.prewarm_popular_locations(locations, args.endpoints, args.days)
        
        # Save results to file
        results_file = f"prewarm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 Results saved to: {results_file}")
        
        # Exit with appropriate code
        if results["stats"]["failed_requests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Prewarming interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Prewarming failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
