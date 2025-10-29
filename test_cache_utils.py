"""
Comprehensive tests for the improved caching system with canonicalized location keys and temporal tolerance.

Tests cover:
- Location canonicalization produces consistent keys
- Cache hit and fallback work as expected
- Metadata fields correctly reflect approximate usage
- Temporal tolerance logic for different aggregation periods
- Redis sorted set operations
"""

import pytest
import json
import redis
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch, AsyncMock
import asyncio

# Import the modules to test
from app.cache_utils import (
    canonicalize_location,
    get_temporal_tolerance,
    date_to_timestamp,
    timestamp_to_date,
    calculate_temporal_delta,
    ImprovedCache,
    initialize_improved_cache,
    get_improved_cache,
    cache_get,
    cache_set
)


class TestLocationCanonicalization:
    """Test location canonicalization logic."""
    
    def test_basic_canonicalization(self):
        """Test basic location canonicalization."""
        assert canonicalize_location("London, England, United Kingdom") == "london_england"
        assert canonicalize_location("New York, NY, USA") == "new_york_ny"
        assert canonicalize_location("Paris, France") == "paris"
    
    def test_case_insensitive(self):
        """Test that canonicalization is case insensitive."""
        assert canonicalize_location("LONDON, ENGLAND") == canonicalize_location("london, england")
        assert canonicalize_location("NEW YORK") == canonicalize_location("new york")
    
    def test_whitespace_handling(self):
        """Test handling of various whitespace patterns."""
        assert canonicalize_location("  London  ,  England  ") == "london_england"
        assert canonicalize_location("New\nYork,\tNY") == "new_york_ny"
    
    def test_common_suffixes(self):
        """Test removal of common country suffixes."""
        assert canonicalize_location("London, England, United Kingdom") == "london_england"
        assert canonicalize_location("Sydney, New South Wales, Australia") == "sydney_new_south_wales"
        assert canonicalize_location("Toronto, Ontario, Canada") == "toronto_ontario"
    
    def test_special_characters(self):
        """Test handling of special characters."""
        assert canonicalize_location("São Paulo, Brazil") == "são_paulo"
        assert canonicalize_location("München, Germany") == "münchen"
        assert canonicalize_location("Zürich, Switzerland") == "zürich"
    
    def test_empty_and_edge_cases(self):
        """Test edge cases and empty inputs."""
        assert canonicalize_location("") == ""
        assert canonicalize_location("   ") == ""
        assert canonicalize_location("London") == "london"
        assert canonicalize_location("London,") == "london"


class TestTemporalTolerance:
    """Test temporal tolerance logic."""
    
    def test_temporal_tolerance_values(self):
        """Test that temporal tolerance returns correct values."""
        assert get_temporal_tolerance("yearly") == 7
        assert get_temporal_tolerance("monthly") == 2
        assert get_temporal_tolerance("daily") == 0
        assert get_temporal_tolerance("weekly") == 0  # Default case
        assert get_temporal_tolerance("invalid") == 0  # Default case
    
    def test_case_insensitive_tolerance(self):
        """Test that tolerance is case insensitive."""
        assert get_temporal_tolerance("YEARLY") == 7
        assert get_temporal_tolerance("Monthly") == 2
        assert get_temporal_tolerance("DAILY") == 0


class TestDateOperations:
    """Test date conversion and calculation functions."""
    
    def test_date_to_timestamp(self):
        """Test date string to timestamp conversion."""
        # Test YYYY-MM-DD format
        timestamp = date_to_timestamp("2024-01-15")
        expected = datetime(2024, 1, 15).timestamp()
        assert abs(timestamp - expected) < 1  # Allow 1 second tolerance
        
        # Test MM-DD format (should use current year)
        timestamp = date_to_timestamp("01-15")
        current_year = datetime.now().year
        expected = datetime(current_year, 1, 15).timestamp()
        assert abs(timestamp - expected) < 1
    
    def test_timestamp_to_date(self):
        """Test timestamp to date string conversion."""
        dt = datetime(2024, 1, 15)
        timestamp = dt.timestamp()
        result = timestamp_to_date(timestamp)
        assert result == "2024-01-15"
    
    def test_calculate_temporal_delta(self):
        """Test temporal delta calculation."""
        assert calculate_temporal_delta("2024-01-15", "2024-01-15") == 0
        assert calculate_temporal_delta("2024-01-15", "2024-01-16") == 1
        assert calculate_temporal_delta("2024-01-15", "2024-01-13") == 2
        assert calculate_temporal_delta("2024-01-15", "2024-01-22") == 7
        
        # Test invalid date formats
        assert calculate_temporal_delta("invalid", "2024-01-15") == 0
        assert calculate_temporal_delta("2024-01-15", "invalid") == 0


class TestImprovedCache:
    """Test the ImprovedCache class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = Mock(spec=redis.Redis)
        mock.get = Mock(return_value=None)
        mock.setex = Mock(return_value=True)
        mock.zadd = Mock(return_value=1)
        mock.zrangebyscore = Mock(return_value=[])
        mock.zscore = Mock(return_value=None)
        mock.expire = Mock(return_value=True)
        mock.delete = Mock(return_value=1)
        mock.zrem = Mock(return_value=1)
        mock.keys = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def cache(self, mock_redis):
        """Create an ImprovedCache instance with mock Redis."""
        return ImprovedCache(mock_redis)
    
    def test_cache_set_success(self, cache, mock_redis):
        """Test successful cache set operation."""
        payload = {"test": "data"}
        result = asyncio.run(cache.cache_set(mock_redis, "daily", "London", "2024-01-15", payload))
        
        assert result is True
        mock_redis.setex.assert_called_once()
        mock_redis.zadd.assert_called_once()
        mock_redis.expire.assert_called_once()
    
    def test_cache_get_exact_match(self, cache, mock_redis):
        """Test cache get with exact match (daily data)."""
        # Mock Redis responses for exact match
        mock_redis.zscore.return_value = datetime(2024, 1, 15).timestamp()
        mock_redis.get.return_value = json.dumps({"test": "data"})
        
        result = asyncio.run(cache.cache_get(mock_redis, "daily", "London", "2024-01-15"))
        
        assert result is not None
        payload, meta = result
        assert payload == {"test": "data"}
        assert meta["approximate"]["temporal"] is False
        assert meta["served_from"]["temporal_delta_days"] == 0
    
    def test_cache_get_temporal_tolerance(self, cache, mock_redis):
        """Test cache get with temporal tolerance (monthly data)."""
        # Mock Redis responses for temporal tolerance
        target_timestamp = datetime(2024, 1, 15).timestamp()
        cached_timestamp = datetime(2024, 1, 16).timestamp()  # 1 day difference
        
        mock_redis.zrangebyscore.return_value = [
            (b"2024-01-16", cached_timestamp)
        ]
        mock_redis.get.return_value = json.dumps({"test": "data"})
        
        result = asyncio.run(cache.cache_get(mock_redis, "monthly", "London", "2024-01-15"))
        
        assert result is not None
        payload, meta = result
        assert payload == {"test": "data"}
        assert meta["approximate"]["temporal"] is True
        assert meta["served_from"]["temporal_delta_days"] == 1
        assert meta["served_from"]["end_date"] == "2024-01-16"
    
    def test_cache_get_miss(self, cache, mock_redis):
        """Test cache get with no match."""
        mock_redis.zrangebyscore.return_value = []
        mock_redis.zscore.return_value = None
        
        result = asyncio.run(cache.cache_get(mock_redis, "daily", "London", "2024-01-15"))
        
        assert result is None
    
    def test_cache_invalidate_specific_date(self, cache, mock_redis):
        """Test cache invalidation for specific date."""
        result = asyncio.run(cache.cache_invalidate(mock_redis, "daily", "London", "2024-01-15"))
        
        assert result is True
        mock_redis.delete.assert_called_once()
        mock_redis.zrem.assert_called_once()
    
    def test_cache_invalidate_all(self, cache, mock_redis):
        """Test cache invalidation for all dates."""
        mock_redis.zrange.return_value = [b"2024-01-15", b"2024-01-16"]
        
        result = asyncio.run(cache.cache_invalidate(mock_redis, "daily", "London"))
        
        assert result is True
        assert mock_redis.delete.call_count == 3  # 2 data keys + 1 sorted set


class TestIntegration:
    """Integration tests for the caching system."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client for integration tests."""
        mock = Mock(spec=redis.Redis)
        mock.get = Mock(return_value=None)
        mock.setex = Mock(return_value=True)
        mock.zadd = Mock(return_value=1)
        mock.zrangebyscore = Mock(return_value=[])
        mock.zscore = Mock(return_value=None)
        mock.expire = Mock(return_value=True)
        return mock
    
    def test_cache_get_cache_set_integration(self, mock_redis):
        """Test integration between cache_get and cache_set."""
        # First, set some data
        payload = {"temperature": 15.5, "location": "London"}
        result = asyncio.run(cache_set(mock_redis, "daily", "London, England", "2024-01-15", payload))
        assert result is True
        
        # Then, mock Redis to return the cached data
        mock_redis.zscore.return_value = datetime(2024, 1, 15).timestamp()
        mock_redis.get.return_value = json.dumps(payload)
        
        # Try to get the data
        cached_result = asyncio.run(cache_get(mock_redis, "daily", "London, England", "2024-01-15"))
        
        assert cached_result is not None
        cached_payload, meta = cached_result
        assert cached_payload == payload
        assert meta["requested"]["location"] == "London, England"
        assert meta["served_from"]["canonical_location"] == "london_england"
    
    def test_temporal_tolerance_integration(self, mock_redis):
        """Test temporal tolerance integration."""
        # Set data for 2024-01-16
        payload = {"temperature": 15.5}
        asyncio.run(cache_set(mock_redis, "monthly", "London", "2024-01-16", payload))
        
        # Mock Redis to return cached data within tolerance
        target_timestamp = datetime(2024, 1, 15).timestamp()
        cached_timestamp = datetime(2024, 1, 16).timestamp()
        
        mock_redis.zrangebyscore.return_value = [
            (b"2024-01-16", cached_timestamp)
        ]
        mock_redis.get.return_value = json.dumps(payload)
        
        # Try to get data for 2024-01-15 (should find 2024-01-16 within tolerance)
        result = asyncio.run(cache_get(mock_redis, "monthly", "London", "2024-01-15"))
        
        assert result is not None
        cached_payload, meta = result
        assert cached_payload == payload
        assert meta["approximate"]["temporal"] is True
        assert meta["served_from"]["temporal_delta_days"] == 1
    
    def test_canonicalization_integration(self, mock_redis):
        """Test that canonicalization works in integration."""
        # Set data with one location format
        payload = {"temperature": 15.5}
        asyncio.run(cache_set(mock_redis, "daily", "London, England, United Kingdom", "2024-01-15", payload))
        
        # Mock Redis to return cached data
        mock_redis.zscore.return_value = datetime(2024, 1, 15).timestamp()
        mock_redis.get.return_value = json.dumps(payload)
        
        # Try to get data with different location format (should find same canonical key)
        result = asyncio.run(cache_get(mock_redis, "daily", "london, england", "2024-01-15"))
        
        assert result is not None
        cached_payload, meta = result
        assert cached_payload == payload
        assert meta["served_from"]["canonical_location"] == "london_england"


class TestErrorHandling:
    """Test error handling in the caching system."""
    
    @pytest.fixture
    def error_redis(self):
        """Create a Redis client that raises exceptions."""
        mock = Mock(spec=redis.Redis)
        mock.get.side_effect = redis.RedisError("Connection failed")
        mock.setex.side_effect = redis.RedisError("Connection failed")
        mock.zadd.side_effect = redis.RedisError("Connection failed")
        return mock
    
    def test_cache_set_error_handling(self, error_redis):
        """Test error handling in cache_set."""
        cache = ImprovedCache(error_redis)
        payload = {"test": "data"}
        
        result = asyncio.run(cache.cache_set(error_redis, "daily", "London", "2024-01-15", payload))
        
        assert result is False
    
    def test_cache_get_error_handling(self, error_redis):
        """Test error handling in cache_get."""
        cache = ImprovedCache(error_redis)
        
        result = asyncio.run(cache.cache_get(error_redis, "daily", "London", "2024-01-15"))
        
        assert result is None


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = Mock(spec=redis.Redis)
        mock.get = Mock(return_value=None)
        mock.setex = Mock(return_value=True)
        mock.zadd = Mock(return_value=1)
        mock.zrangebyscore = Mock(return_value=[])
        mock.zscore = Mock(return_value=None)
        mock.expire = Mock(return_value=True)
        return mock
    
    def test_initialize_and_get_cache(self, mock_redis):
        """Test cache initialization and retrieval."""
        # Initialize the cache
        initialize_improved_cache(mock_redis)
        
        # Get the cache instance
        cache = get_improved_cache()
        
        assert cache is not None
        assert isinstance(cache, ImprovedCache)
    
    def test_get_cache_without_initialization(self):
        """Test that getting cache without initialization raises error."""
        # Reset the global cache
        import app.cache_utils
        app.cache_utils._improved_cache = None
        
        with pytest.raises(RuntimeError, match="Improved cache not initialized"):
            get_improved_cache()


class TestMetadataStructure:
    """Test that metadata structure is correct."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = Mock(spec=redis.Redis)
        mock.get = Mock(return_value=None)
        mock.setex = Mock(return_value=True)
        mock.zadd = Mock(return_value=1)
        mock.zrangebyscore = Mock(return_value=[])
        mock.zscore = Mock(return_value=None)
        mock.expire = Mock(return_value=True)
        return mock
    
    def test_metadata_structure_exact_match(self, mock_redis):
        """Test metadata structure for exact match."""
        cache = ImprovedCache(mock_redis)
        
        # Mock exact match
        mock_redis.zscore.return_value = datetime(2024, 1, 15).timestamp()
        mock_redis.get.return_value = json.dumps({"test": "data"})
        
        result = asyncio.run(cache.cache_get(mock_redis, "daily", "London", "2024-01-15"))
        
        assert result is not None
        payload, meta = result
        
        # Check metadata structure
        assert "requested" in meta
        assert "served_from" in meta
        assert "approximate" in meta
        
        assert meta["requested"]["location"] == "London"
        assert meta["requested"]["end_date"] == "2024-01-15"
        
        assert meta["served_from"]["canonical_location"] == "london"
        assert meta["served_from"]["end_date"] == "2024-01-15"
        assert meta["served_from"]["temporal_delta_days"] == 0
        
        assert meta["approximate"]["temporal"] is False
    
    def test_metadata_structure_temporal_approximation(self, mock_redis):
        """Test metadata structure for temporal approximation."""
        cache = ImprovedCache(mock_redis)
        
        # Mock temporal approximation
        target_timestamp = datetime(2024, 1, 15).timestamp()
        cached_timestamp = datetime(2024, 1, 17).timestamp()  # 2 days difference
        
        mock_redis.zrangebyscore.return_value = [
            (b"2024-01-17", cached_timestamp)
        ]
        mock_redis.get.return_value = json.dumps({"test": "data"})
        
        result = asyncio.run(cache.cache_get(mock_redis, "monthly", "London", "2024-01-15"))
        
        assert result is not None
        payload, meta = result
        
        # Check metadata structure
        assert meta["requested"]["location"] == "London"
        assert meta["requested"]["end_date"] == "2024-01-15"
        
        assert meta["served_from"]["canonical_location"] == "london"
        assert meta["served_from"]["end_date"] == "2024-01-17"
        assert meta["served_from"]["temporal_delta_days"] == 2
        
        assert meta["approximate"]["temporal"] is True


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
