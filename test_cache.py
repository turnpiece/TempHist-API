"""
Comprehensive tests for the enhanced caching system.

Tests cover:
- Cache key normalization and generation
- ETag generation and validation
- Cache headers and conditional requests
- Redis caching with single-flight protection
- Job lifecycle and async processing
- Performance and metrics
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock

from cache_utils import (
    CacheKeyBuilder, ETagGenerator, CacheHeaders, SingleFlightLock,
    EnhancedCache, JobManager, JobStatus, initialize_cache
)
from job_worker import JobWorker, initialize_worker

class TestCacheKeyBuilder:
    """Test cache key building and normalization."""
    
    def test_normalize_params_basic(self):
        """Test basic parameter normalization."""
        params = {
            "location": "New York, NY",
            "unit_group": "celsius",
            "month_mode": "rolling1m",
            "days_back": "7"
        }
        
        normalized = CacheKeyBuilder.normalize_params(params)
        
        # Should normalize location (commas become underscores)
        assert "location=new_york__ny" in " ".join(f"{k}={v}" for k, v in normalized.items())
        
        # Should skip default values
        assert "unit_group" not in normalized
        assert "month_mode" not in normalized
        assert "days_back" not in normalized
    
    def test_normalize_params_coordinates(self):
        """Test coordinate precision normalization."""
        params = {
            "lat": "40.7128",
            "lon": "-74.0060",
            "latitude": "40.71281234",
            "longitude": "-74.00601234"
        }
        
        normalized = CacheKeyBuilder.normalize_params(params)
        
        # Should round to 4 decimal places
        assert normalized["lat"] == "40.7128"
        assert normalized["lon"] == "-74.0060"
        assert normalized["latitude"] == "40.7128"
        assert normalized["longitude"] == "-74.0060"
    
    def test_normalize_params_empty_values(self):
        """Test that empty values are filtered out."""
        params = {
            "location": "New York",
            "unit_group": "",
            "month_mode": None,
            "days_back": "0"
        }
        
        normalized = CacheKeyBuilder.normalize_params(params)
        
        assert "location" in normalized
        assert "unit_group" not in normalized
        assert "month_mode" not in normalized
        assert "days_back" not in normalized
    
    def test_build_cache_key_basic(self):
        """Test basic cache key building."""
        key = CacheKeyBuilder.build_cache_key(
            "v1/records",
            {"period": "daily", "location": "New York"},
            {"unit_group": "celsius"}
        )
        
        # Should be deterministic and sorted
        assert key.startswith("temphist:v1/records")
        assert "period=daily" in key
        assert "location=new_york" in key
    
    def test_build_cache_key_with_query_params(self):
        """Test cache key building with query parameters."""
        key = CacheKeyBuilder.build_cache_key(
            "v1/records/rolling-bundle",
            {"location": "New York", "anchor": "2024-01-15"},
            {"unit_group": "fahrenheit", "days_back": "5"}
        )
        
        assert "location=new_york" in key
        assert "anchor=2024-01-15" in key
        assert "unit_group=fahrenheit" in key
        assert "days_back=5" in key

class TestETagGenerator:
    """Test ETag generation and validation."""
    
    def test_generate_etag_deterministic(self):
        """Test that ETags are deterministic."""
        data = {"temperature": 25.5, "location": "New York", "date": "2024-01-15"}
        
        etag1 = ETagGenerator.generate_etag(data)
        etag2 = ETagGenerator.generate_etag(data)
        
        assert etag1 == etag2
        assert etag1.startswith('"')
        assert etag1.endswith('"')
    
    def test_generate_etag_different_data(self):
        """Test that different data produces different ETags."""
        data1 = {"temperature": 25.5}
        data2 = {"temperature": 26.0}
        
        etag1 = ETagGenerator.generate_etag(data1)
        etag2 = ETagGenerator.generate_etag(data2)
        
        assert etag1 != etag2
    
    def test_parse_etag(self):
        """Test ETag parsing."""
        etag = '"abc123def456"'
        parsed = ETagGenerator.parse_etag(etag)
        assert parsed == "abc123def456"
        
        # Test without quotes
        etag2 = "abc123def456"
        parsed2 = ETagGenerator.parse_etag(etag2)
        assert parsed2 == "abc123def456"
    
    def test_matches_etag(self):
        """Test ETag matching."""
        etag1 = '"abc123"'
        etag2 = '"abc123"'
        etag3 = '"def456"'
        
        assert ETagGenerator.matches_etag(etag1, etag2)
        assert not ETagGenerator.matches_etag(etag1, etag3)

class TestCacheHeaders:
    """Test cache header management."""
    
    def test_set_cache_headers(self):
        """Test setting cache headers."""
        response = Mock()
        response.headers = {}
        etag = '"abc123"'
        last_modified = datetime(2024, 1, 15, 12, 0, 0)
        
        CacheHeaders.set_cache_headers(response, etag, last_modified)
        
        assert response.headers["Cache-Control"] == "public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800"
        assert response.headers["ETag"] == etag
        assert response.headers["Last-Modified"] == "Mon, 15 Jan 2024 12:00:00 GMT"
    
    def test_check_conditional_headers_etag_match(self):
        """Test conditional header checking with ETag match."""
        request = Mock()
        request.headers = {"If-None-Match": '"abc123"'}
        
        etag = '"abc123"'
        last_modified = datetime(2024, 1, 15, 12, 0, 0)
        
        result = CacheHeaders.check_conditional_headers(request, etag, last_modified)
        assert result is True
    
    def test_check_conditional_headers_etag_no_match(self):
        """Test conditional header checking with ETag mismatch."""
        request = Mock()
        request.headers = {"If-None-Match": '"def456"'}
        
        etag = '"abc123"'
        last_modified = datetime(2024, 1, 15, 12, 0, 0)
        
        result = CacheHeaders.check_conditional_headers(request, etag, last_modified)
        assert result is False
    
    def test_check_conditional_headers_if_modified_since(self):
        """Test conditional header checking with If-Modified-Since."""
        request = Mock()
        request.headers = {"If-Modified-Since": "Mon, 15 Jan 2024 12:00:00 GMT"}
        
        etag = '"abc123"'
        last_modified = datetime(2024, 1, 15, 12, 0, 0)
        
        result = CacheHeaders.check_conditional_headers(request, etag, last_modified)
        assert result is True

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = Mock()
    redis_mock.get = Mock(return_value=None)
    redis_mock.set = Mock(return_value=True)
    redis_mock.setex = Mock(return_value=True)
    redis_mock.delete = Mock(return_value=1)
    redis_mock.exists = Mock(return_value=False)
    redis_mock.hget = Mock(return_value=None)
    redis_mock.hset = Mock(return_value=1)
    redis_mock.expire = Mock(return_value=True)
    return redis_mock

class TestSingleFlightLock:
    """Test single-flight locking mechanism."""
    
    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, mock_redis):
        """Test successful lock acquisition."""
        mock_redis.set.return_value = True
        
        lock = SingleFlightLock(mock_redis)
        result = await lock.acquire("test_key")
        
        assert result is True
        mock_redis.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_acquire_lock_failure(self, mock_redis):
        """Test failed lock acquisition."""
        mock_redis.set.return_value = None  # Lock already exists
        
        lock = SingleFlightLock(mock_redis)
        result = await lock.acquire("test_key")
        
        assert result is False
    
    def test_release_lock(self, mock_redis):
        """Test lock release."""
        lock = SingleFlightLock(mock_redis)
        lock.release("test_key")
        
        mock_redis.delete.assert_called_once_with("lock:test_key")

class TestEnhancedCache:
    """Test enhanced caching functionality."""
    
    @pytest.fixture
    def cache(self, mock_redis):
        """Create enhanced cache instance."""
        return EnhancedCache(mock_redis)
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache, mock_redis):
        """Test cache miss scenario."""
        mock_redis.get.return_value = None
        
        result = await cache.get("test_key")
        
        assert result is None
        assert cache.misses == 1
    
    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache, mock_redis):
        """Test cache hit scenario."""
        cached_data = json.dumps({"temperature": 25.5})
        cached_etag = '"abc123"'
        cached_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Mock Redis responses - first call returns data, second returns etag
        mock_redis.get.side_effect = [
            cached_data.encode(),  # First call - cached data as bytes
            cached_etag.encode()   # Second call - etag as bytes
        ]
        mock_redis.hget.return_value = cached_timestamp.encode()
        
        result = await cache.get("test_key")
        
        assert result is not None
        data, etag, last_modified = result
        assert data == {"temperature": 25.5}
        assert etag == cached_etag
        assert cache.hits == 1
    
    @pytest.mark.asyncio
    async def test_set_cache(self, cache, mock_redis):
        """Test setting cache data."""
        data = {"temperature": 25.5}
        
        etag = await cache.set("test_key", data, 3600)
        
        assert etag is not None
        mock_redis.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self, cache, mock_redis):
        """Test get_or_compute with cache hit."""
        cached_data = json.dumps({"temperature": 25.5})
        cached_etag = '"abc123"'
        cached_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Mock Redis responses for cache hit
        mock_redis.get.side_effect = [
            cached_data.encode(),  # First call - cached data as bytes
            cached_etag.encode()   # Second call - etag as bytes
        ]
        mock_redis.hget.return_value = cached_timestamp.encode()
        
        def compute_func():
            return {"temperature": 25.5}
        
        result = await cache.get_or_compute("test_key", compute_func, 3600)
        
        assert result is not None
        # Should not call compute function since we have a cache hit
    
    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self, cache, mock_redis):
        """Test get_or_compute with cache miss."""
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True  # Lock acquired
        
        def compute_func():
            return {"temperature": 25.5}
        
        result = await cache.get_or_compute("test_key", compute_func, 3600)
        
        assert result is not None
        data, etag, last_modified = result
        assert data == {"temperature": 25.5}
    
    def test_get_metrics(self, cache):
        """Test metrics collection."""
        cache.hits = 5
        cache.misses = 3
        cache.errors = 1
        cache.compute_times = [0.1, 0.2, 0.15]
        
        metrics = cache.get_metrics()
        
        assert metrics["hits"] == 5
        assert metrics["misses"] == 3
        assert metrics["errors"] == 1
        assert metrics["hit_rate"] == 62.5
        assert metrics["avg_compute_time"] == 0.15

class TestJobManager:
    """Test job management functionality."""
    
    @pytest.fixture
    def job_manager(self, mock_redis):
        """Create job manager instance."""
        return JobManager(mock_redis)
    
    def test_create_job(self, job_manager, mock_redis):
        """Test job creation."""
        mock_redis.setex.return_value = True
        
        job_id = job_manager.create_job("test_job", {"param": "value"})
        
        assert job_id is not None
        assert job_id.startswith("test_job_")
        mock_redis.setex.assert_called_once()
    
    def test_get_job_status_not_found(self, job_manager, mock_redis):
        """Test getting status of non-existent job."""
        mock_redis.get.return_value = None
        
        result = job_manager.get_job_status("nonexistent_job")
        
        assert result is None
    
    def test_get_job_status_found(self, job_manager, mock_redis):
        """Test getting status of existing job."""
        job_data = {
            "id": "test_job_123",
            "type": "test_job",
            "status": JobStatus.PENDING,
            "params": {"param": "value"}
        }
        
        mock_redis.get.return_value = json.dumps(job_data)
        
        result = job_manager.get_job_status("test_job_123")
        
        assert result is not None
        assert result["status"] == JobStatus.PENDING
    
    def test_update_job_status(self, job_manager, mock_redis):
        """Test updating job status."""
        existing_job = {
            "id": "test_job_123",
            "type": "test_job",
            "status": JobStatus.PENDING,
            "params": {"param": "value"}
        }
        
        mock_redis.get.return_value = json.dumps(existing_job)
        mock_redis.setex.return_value = True
        
        job_manager.update_job_status("test_job_123", JobStatus.READY, {"result": "data"})
        
        # Should update job status
        mock_redis.setex.assert_called()
        
        # Should store result
        calls = mock_redis.setex.call_args_list
        assert len(calls) >= 2  # At least job update and result storage

class TestJobWorker:
    """Test job worker functionality."""
    
    @pytest.fixture
    def worker(self, mock_redis):
        """Create job worker instance."""
        return JobWorker(mock_redis)
    
    @pytest.mark.asyncio
    async def test_get_pending_jobs(self, worker, mock_redis):
        """Test getting pending jobs."""
        job_data = {
            "id": "test_job_123",
            "status": JobStatus.PENDING
        }
        
        mock_redis.keys.return_value = ["job:test_job_123"]
        mock_redis.get.return_value = json.dumps(job_data)
        
        pending_jobs = await worker.get_pending_jobs()
        
        assert "test_job_123" in pending_jobs
    
    @pytest.mark.asyncio
    async def test_process_job_record_computation(self, worker, mock_redis):
        """Test processing a record computation job."""
        job_data = {
            "id": "test_job_123",
            "type": "record_computation",
            "status": JobStatus.PENDING,
            "params": {
                "period": "daily",
                "location": "New York",
                "identifier": "01-15"
            }
        }
        
        mock_redis.get.return_value = json.dumps(job_data)
        mock_redis.setex.return_value = True
        
        # Mock the job manager and cache
        job_manager = Mock()
        cache = Mock()
        cache.set = AsyncMock(return_value='"etag123"')
        
        # Mock the temperature data function
        async def mock_get_temperature_data_v1(location, period, identifier):
            return {"temperature": 25.5}
        
        with patch('main.get_temperature_data_v1', side_effect=mock_get_temperature_data_v1):
            await worker.process_job("test_job_123", job_manager, cache)
            
            # Should update job status to processing and then ready
            assert job_manager.update_job_status.call_count >= 2

class TestIntegration:
    """Integration tests for the caching system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_caching_flow(self, mock_redis):
        """Test complete caching flow from request to response."""
        # Initialize cache system
        initialize_cache(mock_redis)
        cache = EnhancedCache(mock_redis)
        
        # Mock Redis responses for cache miss
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True  # Lock acquired
        mock_redis.setex.return_value = True
        
        def compute_func():
            return {"temperature": 25.5, "location": "New York"}
        
        # Test get_or_compute
        result = await cache.get_or_compute("test_key", compute_func, 3600)
        
        assert result is not None
        data, etag, last_modified = result
        assert data["temperature"] == 25.5
        
        # Test metrics
        metrics = cache.get_metrics()
        assert metrics["misses"] == 1
        assert metrics["total_requests"] == 1

class TestPerformance:
    """Performance tests for the caching system."""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, mock_redis):
        """Test cache performance under load."""
        cache = EnhancedCache(mock_redis)
        
        # Mock cache hit
        cached_data = json.dumps({"temperature": 25.5})
        cached_etag = '"abc123"'
        cached_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Mock Redis responses - cycle through data and etag
        def mock_get(key):
            if "test_key" in key:
                return cached_data.encode()
            elif "etag" in key:
                return cached_etag.encode()
            return None
        
        mock_redis.get.side_effect = mock_get
        mock_redis.hget.return_value = cached_timestamp.encode()
        
        # Measure cache hit performance
        start_time = time.time()
        
        for _ in range(100):
            result = await cache.get("test_key")
            assert result is not None
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should be very fast for cache hits
        assert avg_time < 0.01  # Less than 10ms per request
    
    @pytest.mark.asyncio
    async def test_etag_generation_performance(self):
        """Test ETag generation performance."""
        data = {"temperature": 25.5, "location": "New York", "date": "2024-01-15"}
        
        start_time = time.time()
        
        for _ in range(1000):
            ETagGenerator.generate_etag(data)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        
        # Should be very fast
        assert avg_time < 0.001  # Less than 1ms per ETag

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
