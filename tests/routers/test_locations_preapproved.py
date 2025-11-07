"""
Tests for the preapproved locations endpoint.

Tests cover:
- Happy path scenarios (all, filtered, combined, limit)
- ETag and Last-Modified caching
- Redis caching behavior
- Input validation
- Rate limiting
- Error handling
"""

import json
import pytest
import redis
from datetime import datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the router and models
from routers.locations_preapproved import (
    router,
    LocationItem,
    initialize_locations_data,
    get_cache_key,
    validate_country_code,
    validate_limit,
    generate_etag,
    filter_locations
)

# Test data
SAMPLE_LOCATIONS = [
    {
        "id": "london",
        "slug": "london",
        "name": "London",
        "admin1": "England",
        "country_name": "United Kingdom",
        "country_code": "GB",
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timezone": "Europe/London",
        "tier": "global"
    },
    {
        "id": "new_york",
        "slug": "new-york",
        "name": "New York",
        "admin1": "New York",
        "country_name": "United States",
        "country_code": "US",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone": "America/New_York",
        "tier": "global"
    },
    {
        "id": "paris",
        "slug": "paris",
        "name": "Paris",
        "admin1": "Île-de-France",
        "country_name": "France",
        "country_code": "FR",
        "latitude": 48.8566,
        "longitude": 2.3522,
        "timezone": "Europe/Paris",
        "tier": "global"
    }
]

@pytest.fixture
def app():
    """Create test FastAPI app with the router."""
    app = FastAPI()
    app.include_router(router)
    return app

@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = MagicMock(spec=redis.Redis)
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    return mock_redis

@pytest.fixture
def sample_locations():
    """Sample location data for testing."""
    return [LocationItem(**loc) for loc in SAMPLE_LOCATIONS]

@pytest.fixture
def mock_locations_data(sample_locations, mock_redis):
    """Mock the global locations data."""
    with patch('routers.locations_preapproved.locations_data', sample_locations), \
         patch('routers.locations_preapproved.locations_etag', '"test-etag"'), \
         patch('routers.locations_preapproved.locations_last_modified', 'Mon, 01 Jan 2024 00:00:00 GMT'), \
         patch('routers.locations_preapproved.redis_client', mock_redis):
        yield

class TestLocationItem:
    """Test LocationItem model validation."""
    
    def test_valid_location_item(self):
        """Test valid location item creation."""
        location = LocationItem(**SAMPLE_LOCATIONS[0])
        assert location.id == "london"
        assert location.country_code == "GB"
    
    def test_invalid_country_code(self):
        """Test invalid country code validation."""
        with pytest.raises(ValueError, match="Country code must be a 2-letter ISO 3166-1 alpha-2 code"):
            LocationItem(
                id="test",
                slug="test",
                name="Test",
                admin1="Test",
                country_name="Test",
                country_code="INVALID",
                latitude=0.0,
                longitude=0.0,
                timezone="UTC",
                tier="test"
            )

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_country_code(self):
        """Test country code validation."""
        assert validate_country_code("US") == True
        assert validate_country_code("GB") == True
        assert validate_country_code("INVALID") == False
        assert validate_country_code("us") == False  # lowercase
        assert validate_country_code("USA") == False  # 3 letters
    
    def test_validate_limit(self):
        """Test limit validation."""
        assert validate_limit(1) == True
        assert validate_limit(500) == True
        assert validate_limit(0) == False
        assert validate_limit(501) == False
    
    def test_generate_etag(self):
        """Test ETag generation."""
        etag1 = generate_etag("test data")
        etag2 = generate_etag("test data")
        etag3 = generate_etag("different data")
        
        assert etag1 == etag2  # Same data should produce same ETag
        assert etag1 != etag3  # Different data should produce different ETag
        assert etag1.startswith('"') and etag1.endswith('"')  # Should be quoted
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        assert get_cache_key() == "preapproved:v1:all"
        assert get_cache_key("US") == "preapproved:v1:country:US"
        assert get_cache_key(tier="global") == "preapproved:v1:tier:global"
        assert get_cache_key("US", "global") == "preapproved:v1:country:US:tier:global"
    
    def test_filter_locations(self, sample_locations):
        """Test location filtering."""
        # No filters
        filtered = filter_locations(sample_locations)
        assert len(filtered) == 3
        
        # Country filter
        filtered = filter_locations(sample_locations, country_code="US")
        assert len(filtered) == 1
        assert filtered[0].country_code == "US"
        
        # Tier filter
        filtered = filter_locations(sample_locations, tier="global")
        assert len(filtered) == 3  # All are global tier
        
        # Combined filters
        filtered = filter_locations(sample_locations, country_code="GB", tier="global")
        assert len(filtered) == 1
        assert filtered[0].country_code == "GB"
        
        # Limit
        filtered = filter_locations(sample_locations, limit=2)
        assert len(filtered) == 2
        
        # Sort by name
        filtered = filter_locations(sample_locations)
        names = [loc.name for loc in filtered]
        assert names == sorted(names)

class TestPreapprovedLocationsEndpoint:
    """Test the main preapproved locations endpoint."""
    
    def test_get_all_locations(self, client, mock_locations_data):
        """Test getting all locations."""
        response = client.get("/v1/locations/preapproved")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["version"] == 1
        assert data["count"] == 3
        assert len(data["locations"]) == 3
        assert "generated_at" in data
        
        # Check cache headers
        assert "Cache-Control" in response.headers
        assert "ETag" in response.headers
        assert "Last-Modified" in response.headers
    
    def test_filter_by_country_code(self, client, mock_locations_data):
        """Test filtering by country code."""
        response = client.get("/v1/locations/preapproved?country_code=US")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 1
        assert data["locations"][0]["country_code"] == "US"
    
    def test_filter_by_tier(self, client, mock_locations_data):
        """Test filtering by tier."""
        response = client.get("/v1/locations/preapproved?tier=global")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 3
        for location in data["locations"]:
            assert location["tier"] == "global"
    
    def test_combined_filters(self, client, mock_locations_data):
        """Test combined filters."""
        response = client.get("/v1/locations/preapproved?country_code=GB&tier=global")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 1
        assert data["locations"][0]["country_code"] == "GB"
        assert data["locations"][0]["tier"] == "global"
    
    def test_limit_parameter(self, client, mock_locations_data):
        """Test limit parameter."""
        response = client.get("/v1/locations/preapproved?limit=2")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["count"] == 2
        assert len(data["locations"]) == 2
    
    def test_invalid_country_code(self, client, mock_locations_data):
        """Test invalid country code validation."""
        response = client.get("/v1/locations/preapproved?country_code=INVALID")
        
        assert response.status_code == 400
        assert "Invalid country code format" in response.json()["detail"]
    
    def test_invalid_limit_too_low(self, client, mock_locations_data):
        """Test invalid limit (too low)."""
        response = client.get("/v1/locations/preapproved?limit=0")
        
        assert response.status_code == 422  # Pydantic validation error
    
    def test_invalid_limit_too_high(self, client, mock_locations_data):
        """Test invalid limit (too high)."""
        response = client.get("/v1/locations/preapproved?limit=501")
        
        assert response.status_code == 422  # Pydantic validation error

class TestCaching:
    """Test caching behavior."""
    
    def test_etag_not_modified(self, client, mock_locations_data):
        """Test ETag not modified response."""
        # First request to get ETag
        response1 = client.get("/v1/locations/preapproved")
        assert response1.status_code == 200
        etag = response1.headers["ETag"]
        
        # Second request with If-None-Match
        response2 = client.get(
            "/v1/locations/preapproved",
            headers={"If-None-Match": etag}
        )
        assert response2.status_code == 304
    
    def test_last_modified_not_modified(self, client, mock_locations_data):
        """Test Last-Modified not modified response."""
        # First request to get Last-Modified
        response1 = client.get("/v1/locations/preapproved")
        assert response1.status_code == 200
        last_modified = response1.headers["Last-Modified"]
        
        # Second request with If-Modified-Since
        response2 = client.get(
            "/v1/locations/preapproved",
            headers={"If-Modified-Since": last_modified}
        )
        assert response2.status_code == 304
    
    def test_redis_cache_miss(self, client, mock_redis, sample_locations):
        """Test Redis cache miss behavior."""
        with patch('routers.locations_preapproved.locations_data', sample_locations), \
             patch('routers.locations_preapproved.locations_etag', '"test-etag"'), \
             patch('routers.locations_preapproved.locations_last_modified', 'Mon, 01 Jan 2024 00:00:00 GMT'), \
             patch('routers.locations_preapproved.redis_client', mock_redis):
            
            mock_redis.get.return_value = None  # Cache miss
            
            response = client.get("/v1/locations/preapproved")
            
            assert response.status_code == 200
            # Should have called setex to cache the result
            mock_redis.setex.assert_called_once()
    
    def test_redis_cache_hit(self, client, mock_redis, sample_locations):
        """Test Redis cache hit behavior."""
        cached_data = {
            "version": 1,
            "count": 3,
            "generated_at": datetime.now().isoformat(),
            "locations": [loc.model_dump() for loc in sample_locations]
        }
        
        with patch('routers.locations_preapproved.locations_data', sample_locations), \
             patch('routers.locations_preapproved.locations_etag', '"test-etag"'), \
             patch('routers.locations_preapproved.locations_last_modified', 'Mon, 01 Jan 2024 00:00:00 GMT'), \
             patch('routers.locations_preapproved.redis_client', mock_redis):
            
            mock_redis.get.return_value = json.dumps(cached_data, default=str)
            
            response = client.get("/v1/locations/preapproved")
            
            assert response.status_code == 200
            # Should not have called setex since we got from cache
            mock_redis.setex.assert_not_called()

class TestRateLimiting:
    """Test rate limiting behavior."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded scenario."""
        from routers.locations_preapproved import check_rate_limit
        
        # Simulate exceeding rate limit
        ip = "192.168.1.1"
        
        # Make requests up to the limit
        for _ in range(60):  # RATE_LIMIT_REQUESTS = 60
            allowed, reason = await check_rate_limit(ip)
            assert allowed == True
        
        # Next request should be rate limited
        allowed, reason = await check_rate_limit(ip)
        assert allowed == False
        assert "Rate limit exceeded" in reason
    
    def test_rate_limit_integration(self, client, mock_locations_data):
        """Test rate limiting integration with the endpoint."""
        # This is a simplified test - in a real scenario, we'd need to mock
        # the rate limiting more carefully to avoid affecting other tests
        response = client.get("/v1/locations/preapproved")
        assert response.status_code == 200

class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize_locations_data(self, mock_redis, tmp_path):
        """Test locations data initialization."""
        # Create temporary data file
        data_file = tmp_path / "preapproved_locations.json"
        data_file.write_text(json.dumps(SAMPLE_LOCATIONS))
        
        with patch('routers.locations_preapproved.os.path.join', return_value=str(data_file)):
            await initialize_locations_data(mock_redis)
        
        # Verify cache was warmed
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_locations_data_file_not_found(self, mock_redis):
        """Test initialization with missing data file."""
        with patch('routers.locations_preapproved.os.path.join', return_value="/nonexistent/file.json"):
            with pytest.raises(Exception):  # Should raise HTTPException
                await initialize_locations_data(mock_redis)
    
    @pytest.mark.asyncio
    async def test_initialize_locations_data_invalid_json(self, mock_redis, tmp_path):
        """Test initialization with invalid JSON."""
        data_file = tmp_path / "preapproved_locations.json"
        data_file.write_text("invalid json")
        
        with patch('routers.locations_preapproved.os.path.join', return_value=str(data_file)):
            with pytest.raises(Exception):  # Should raise HTTPException
                await initialize_locations_data(mock_redis)

class TestStatusEndpoint:
    """Test the status endpoint."""
    
    def test_get_status(self, client, mock_locations_data):
        """Test status endpoint."""
        response = client.get("/v1/locations/preapproved/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["locations_loaded"] == 3
        assert data["etag"] == '"test-etag"'
        assert data["last_modified"] == 'Mon, 01 Jan 2024 00:00:00 GMT'
        assert "cache_enabled" in data
        assert "rate_limit" in data

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_redis_unavailable(self, client, sample_locations):
        """Test behavior when Redis is unavailable."""
        with patch('routers.locations_preapproved.locations_data', sample_locations), \
             patch('routers.locations_preapproved.locations_etag', '"test-etag"'), \
             patch('routers.locations_preapproved.locations_last_modified', 'Mon, 01 Jan 2024 00:00:00 GMT'), \
             patch('routers.locations_preapproved.redis_client', None):
            
            # Should still work without Redis, just without caching
            response = client.get("/v1/locations/preapproved")
            assert response.status_code == 200
    
    def test_malformed_json_in_cache(self, client, mock_redis, sample_locations):
        """Test handling of malformed JSON in cache."""
        with patch('routers.locations_preapproved.locations_data', sample_locations), \
             patch('routers.locations_preapproved.locations_etag', '"test-etag"'), \
             patch('routers.locations_preapproved.locations_last_modified', 'Mon, 01 Jan 2024 00:00:00 GMT'), \
             patch('routers.locations_preapproved.redis_client', mock_redis):
            
            mock_redis.get.return_value = "invalid json"
            
            # Should fall back to generating response from data
            response = client.get("/v1/locations/preapproved")
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])
