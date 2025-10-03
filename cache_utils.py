"""
Enhanced caching utilities for Cloudflare-friendly API responses.

This module provides:
- Strong cache headers (Cache-Control, ETag, Last-Modified)
- Canonical cache key generation with parameter normalization
- Single-flight protection to prevent cache stampedes
- Cache hit/miss metrics and health monitoring
- Deterministic JSON serialization for consistent ETags
"""

import hashlib
import json
import time
import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from collections import OrderedDict

import redis
from fastapi import Request, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL_DEFAULT = 86400  # 24 hours
CACHE_TTL_SHORT = 3600    # 1 hour  
CACHE_TTL_LONG = 604800   # 7 days
CACHE_TTL_JOB = 7200      # 2 hours for job results

# Coordinate precision for cache key normalization
COORD_PRECISION = 4  # 4 decimal places (~11m precision)

# Cache header configuration
CACHE_CONTROL_PUBLIC = "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800"
CACHE_CONTROL_PRIVATE = "private, max-age=300, stale-while-revalidate=3600"

class CacheKeyBuilder:
    """Build canonical cache keys with normalized parameters."""
    
    @staticmethod
    def normalize_params(params: Dict[str, Any]) -> Dict[str, str]:
        """Normalize query parameters for consistent cache keys."""
        normalized = {}
        
        for key, value in sorted(params.items()):
            # Skip None values and empty strings
            if value is None or value == "":
                continue
                
            # Convert to string and strip whitespace
            str_value = str(value).strip()
            if not str_value:
                continue
            
            # Normalize location names
            if key in ['location']:
                str_value = str_value.lower().replace(" ", "_").replace(",", "_")
                
            # Normalize coordinate precision
            if key in ['lat', 'lon', 'latitude', 'longitude']:
                try:
                    coord = float(str_value)
                    str_value = f"{coord:.{COORD_PRECISION}f}"
                except (ValueError, TypeError):
                    pass
            
            # Skip default values to reduce cache key variations
            if key == 'unit_group' and str_value == 'celsius':
                continue
            if key == 'month_mode' and str_value == 'rolling1m':
                continue
            if key == 'days_back' and str_value in ['7', '0']:
                continue
                
            normalized[key] = str_value
            
        return normalized
    
    @staticmethod
    def build_cache_key(
        endpoint: str, 
        path_params: Dict[str, str] = None,
        query_params: Dict[str, Any] = None,
        prefix: str = "temphist"
    ) -> str:
        """Build a canonical cache key from endpoint and parameters."""
        path_params = path_params or {}
        query_params = query_params or {}
        
        # Normalize path parameters
        path_parts = []
        for key in sorted(path_params.keys()):
            value = str(path_params[key]).strip().lower()
            # Apply same normalization as query params for location
            if key == 'location':
                value = value.replace(" ", "_").replace(",", "_")
            path_parts.append(f"{key}={value}")
        
        # Normalize query parameters
        normalized_query = CacheKeyBuilder.normalize_params(query_params)
        query_parts = []
        for key, value in sorted(normalized_query.items()):
            query_parts.append(f"{key}={value}")
        
        # Build the key
        key_parts = [prefix, endpoint]
        if path_parts:
            key_parts.extend(path_parts)
        if query_parts:
            key_parts.extend(query_parts)
            
        return ":".join(key_parts)

class ETagGenerator:
    """Generate and validate ETags for responses."""
    
    @staticmethod
    def generate_etag(data: Any) -> str:
        """Generate ETag from response data."""
        # Ensure deterministic JSON serialization
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return f'"{hashlib.md5(json_str.encode()).hexdigest()}"'
    
    @staticmethod
    def parse_etag(etag: str) -> Optional[str]:
        """Parse ETag from If-None-Match header."""
        if not etag:
            return None
        # Remove quotes if present
        return etag.strip('"\'')
    
    @staticmethod
    def matches_etag(response_etag: str, request_etag: str) -> bool:
        """Check if request ETag matches response ETag."""
        if not response_etag or not request_etag:
            return False
        return response_etag.strip('"\'') == request_etag.strip('"\'')

class CacheHeaders:
    """Manage cache headers for responses."""
    
    @staticmethod
    def set_cache_headers(
        response: Response,
        etag: str,
        last_modified: datetime,
        cache_control: str = CACHE_CONTROL_PUBLIC,
        vary: str = "Accept-Encoding"
    ):
        """Set comprehensive cache headers."""
        response.headers["Cache-Control"] = cache_control
        response.headers["ETag"] = etag
        response.headers["Last-Modified"] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
        if vary:
            response.headers["Vary"] = vary
    
    @staticmethod
    def check_conditional_headers(request: Request, etag: str, last_modified: datetime) -> bool:
        """Check if request has conditional headers that match cached response."""
        # Check If-None-Match (ETag)
        if_none_match = request.headers.get("If-None-Match")
        if if_none_match and ETagGenerator.matches_etag(etag, if_none_match):
            return True
            
        # Check If-Modified-Since
        if_modified_since = request.headers.get("If-Modified-Since")
        if if_modified_since:
            try:
                # Parse the If-Modified-Since header
                request_time = datetime.strptime(if_modified_since, "%a, %d %b %Y %H:%M:%S GMT")
                # Add 1 second tolerance for clock skew
                if last_modified <= request_time + timedelta(seconds=1):
                    return True
            except ValueError:
                # Invalid date format, ignore
                pass
                
        return False

class SingleFlightLock:
    """Prevent cache stampedes with Redis-based locks."""
    
    def __init__(self, redis_client: redis.Redis, lock_ttl: int = 30):
        self.redis = redis_client
        self.lock_ttl = lock_ttl
        self.lock_prefix = "lock:"
    
    async def acquire(self, key: str) -> bool:
        """Acquire a lock for the given key."""
        lock_key = f"{self.lock_prefix}{key}"
        try:
            # Use SET with NX and EX for atomic lock acquisition
            result = self.redis.set(lock_key, "1", nx=True, ex=self.lock_ttl)
            return result is not None
        except Exception as e:
            logger.warning(f"Failed to acquire lock for {key}: {e}")
            return False
    
    def release(self, key: str):
        """Release a lock for the given key."""
        lock_key = f"{self.lock_prefix}{key}"
        try:
            self.redis.delete(lock_key)
        except Exception as e:
            logger.warning(f"Failed to release lock for {key}: {e}")

class EnhancedCache:
    """Enhanced Redis cache with single-flight protection and metrics."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.lock_manager = SingleFlightLock(redis_client)
        self.cache_prefix = "cache:"
        self.metrics_prefix = "metrics:"
        
        # Metrics tracking
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.compute_times = []
    
    def _get_cache_key(self, key: str) -> str:
        """Get full cache key with prefix."""
        return f"{self.cache_prefix}{key}"
    
    def _get_etag_key(self, key: str) -> str:
        """Get ETag key for cache entry."""
        return f"{self.cache_prefix}{key}:etag"
    
    def _get_metrics_key(self, key: str) -> str:
        """Get metrics key for cache entry."""
        return f"{self.metrics_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Tuple[Any, str, datetime]]:
        """Get cached value with ETag and Last-Modified timestamp."""
        cache_key = self._get_cache_key(key)
        etag_key = self._get_etag_key(key)
        
        try:
            # Get cached data and metadata
            cached_data = self.redis.get(cache_key)
            cached_etag = self.redis.get(etag_key)
            cached_timestamp = self.redis.hget(cache_key, "timestamp")
            
            if cached_data and cached_etag and cached_timestamp:
                # Parse the data
                data = json.loads(cached_data)
                etag = cached_etag.decode()
                last_modified = datetime.fromisoformat(cached_timestamp.decode())
                
                self.hits += 1
                return data, etag, last_modified
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache get error for {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        data: Any, 
        ttl: int = CACHE_TTL_DEFAULT,
        etag: Optional[str] = None,
        last_modified: Optional[datetime] = None
    ) -> str:
        """Set cached value with ETag and metadata."""
        cache_key = self._get_cache_key(key)
        etag_key = self._get_etag_key(key)
        
        if etag is None:
            etag = ETagGenerator.generate_etag(data)
        if last_modified is None:
            last_modified = datetime.now(timezone.utc)
        
        try:
            # Store data and ETag
            json_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
            self.redis.setex(cache_key, ttl, json_data)
            self.redis.setex(etag_key, ttl, etag)
            
            # Store metadata
            self.redis.hset(cache_key, mapping={
                "timestamp": last_modified.isoformat(),
                "ttl": ttl,
                "size": len(json_data)
            })
            self.redis.expire(cache_key, ttl)
            
            return etag
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache set error for {key}: {e}")
            raise
    
    async def get_or_compute(
        self,
        key: str,
        compute_func,
        ttl: int = CACHE_TTL_DEFAULT,
        *args,
        **kwargs
    ) -> Tuple[Any, str, datetime]:
        """Get from cache or compute with single-flight protection."""
        # Try to get from cache first
        cached_result = await self.get(key)
        if cached_result is not None:
            return cached_result
        
        # Try to acquire lock to prevent stampedes
        lock_acquired = await self.lock_manager.acquire(key)
        
        try:
            if lock_acquired:
                # We got the lock, compute the value
                start_time = time.time()
                data = await compute_func(*args, **kwargs) if asyncio.iscoroutinefunction(compute_func) else compute_func(*args, **kwargs)
                compute_time = time.time() - start_time
                
                # Store in cache
                etag = await self.set(key, data, ttl)
                last_modified = datetime.now(timezone.utc)
                
                # Track compute time
                self.compute_times.append(compute_time)
                
                return data, etag, last_modified
            else:
                # Another process is computing, wait and retry
                await asyncio.sleep(0.1)
                return await self.get_or_compute(key, compute_func, ttl, *args, **kwargs)
                
        finally:
            if lock_acquired:
                self.lock_manager.release(key)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        avg_compute_time = sum(self.compute_times) / len(self.compute_times) if self.compute_times else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "avg_compute_time": round(avg_compute_time, 3),
            "compute_times_count": len(self.compute_times)
        }
    
    def reset_metrics(self):
        """Reset cache metrics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.compute_times = []

# Job management for async processing
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class JobManager:
    """Manage async job processing with Redis storage."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.job_prefix = "job:"
        self.result_prefix = "result:"
        self.job_ttl = CACHE_TTL_JOB
    
    def create_job(self, job_type: str, params: Dict[str, Any]) -> str:
        """Create a new job and return job ID."""
        job_id = f"{job_type}_{int(time.time() * 1000)}_{hashlib.md5(str(params).encode()).hexdigest()[:8]}"
        job_key = f"{self.job_prefix}{job_id}"
        
        job_data = {
            "id": job_id,
            "type": job_type,
            "status": JobStatus.PENDING,
            "params": params,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store job data
        self.redis.setex(job_key, self.job_ttl, json.dumps(job_data))
        
        # Add job to queue for worker processing
        job_queue_key = "job_queue"
        self.redis.lpush(job_queue_key, job_id)
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result if ready."""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.redis.get(job_key)
        
        if not job_data:
            return None
        
        job = json.loads(job_data)
        
        # If job is ready, include result
        if job["status"] == JobStatus.READY:
            result_key = f"{self.result_prefix}{job_id}"
            result_data = self.redis.get(result_key)
            if result_data:
                job["result"] = json.loads(result_data)
        
        return job
    
    def update_job_status(self, job_id: str, status: str, result: Any = None, error: str = None):
        """Update job status and optionally store result."""
        job_key = f"{self.job_prefix}{job_id}"
        job_data = self.redis.get(job_key)
        
        if not job_data:
            return
        
        job = json.loads(job_data)
        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        if error:
            job["error"] = error
        
        self.redis.setex(job_key, self.job_ttl, json.dumps(job))
        
        # Store result if provided
        if result is not None:
            result_key = f"{self.result_prefix}{job_id}"
            self.redis.setex(result_key, self.job_ttl, json.dumps(result))
    
    def cleanup_expired_jobs(self) -> int:
        """Clean up expired jobs (Redis TTL should handle this automatically)."""
        # This is mainly for manual cleanup if needed
        # Redis TTL should handle automatic cleanup
        return 0

# Global cache instance (will be initialized in main.py)
enhanced_cache: Optional[EnhancedCache] = None
job_manager: Optional[JobManager] = None

def initialize_cache(redis_client: redis.Redis):
    """Initialize global cache instances."""
    global enhanced_cache, job_manager
    enhanced_cache = EnhancedCache(redis_client)
    job_manager = JobManager(redis_client)
    logger.info("Enhanced cache and job manager initialized")

def get_cache() -> EnhancedCache:
    """Get the global cache instance."""
    if enhanced_cache is None:
        raise RuntimeError("Cache not initialized. Call initialize_cache() first.")
    return enhanced_cache

def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    if job_manager is None:
        raise RuntimeError("Job manager not initialized. Call initialize_cache() first.")
    return job_manager
