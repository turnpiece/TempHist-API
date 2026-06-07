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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import CANONICALIZATION_RADIUS_KM
from main import app as main_app

# Import the router and models
from routers.locations import (
    EU_MEMBER_CODES,
    LocationItem,
    SelectionRequest,
    _find_preapproved_id,
    _resolve_canonical_id,
    convert_image_urls,
    filter_locations,
    generate_etag,
    get_cache_key,
    get_popular_cache_key,
    initialize_locations_data,
    resolve_country_code,
    router,
    validate_country_code,
    validate_limit,
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
        "continent": "Europe",
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timezone": "Europe/London",
        "tier": "global",
        "imageUrl": {
            "webp": "http://localhost:8000/data/locations/processed/london.webp",
            "jpeg": "http://localhost:8000/data/locations/processed/london.jpg",
        },
        "imageAlt": "London Eye and Thames river",
    },
    {
        "id": "new_york",
        "slug": "new-york",
        "name": "New York",
        "admin1": "New York",
        "country_name": "United States",
        "country_code": "US",
        "continent": "North America",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone": "America/New_York",
        "tier": "global",
        "imageUrl": {
            "webp": "http://localhost:8000/data/locations/processed/new-york.webp",
            "jpeg": "http://localhost:8000/data/locations/processed/new-york.jpg",
        },
        "imageAlt": "New York skyline",
    },
    {
        "id": "paris",
        "slug": "paris",
        "name": "Paris",
        "admin1": "Île-de-France",
        "country_name": "France",
        "country_code": "FR",
        "continent": "Europe",
        "latitude": 48.8566,
        "longitude": 2.3522,
        "timezone": "Europe/Paris",
        "tier": "global",
        "imageUrl": {
            "webp": "http://localhost:8000/data/locations/processed/paris.webp",
            "jpeg": "http://localhost:8000/data/locations/processed/paris.jpg",
        },
        "imageAlt": "Paris cityscape",
    },
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
    with (
        patch("routers.locations.locations_data", sample_locations),
        patch("routers.locations.locations_etag", '"test-etag"'),
        patch("routers.locations.locations_last_modified", "Mon, 01 Jan 2024 00:00:00 GMT"),
        patch("routers.locations.redis_client", mock_redis),
    ):
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
                tier="test",
                imageUrl={"webp": "http://localhost:8000/test.webp", "jpeg": "http://localhost:8000/test.jpg"},
                imageAlt="Test image",
            )

    def test_location_with_images(self):
        """Test location item with image URLs and alt text."""
        location = LocationItem(**SAMPLE_LOCATIONS[0])
        assert location.imageUrl.webp == "http://localhost:8000/data/locations/processed/london.webp"
        assert location.imageUrl.jpeg == "http://localhost:8000/data/locations/processed/london.jpg"
        assert location.imageAlt == "London Eye and Thames river"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_country_code(self):
        """Test country code validation."""
        assert validate_country_code("US") == (True, None)
        assert validate_country_code("GB") == (True, None)
        assert validate_country_code("EU") == (True, None)  # special grouping
        assert validate_country_code("UK") == (True, None)  # alias for GB

        valid, msg = validate_country_code("INVALID")
        assert valid is False
        assert msg is not None

        valid, msg = validate_country_code("XX")  # format ok, not a real code
        assert valid is False
        assert "Unknown country code" in msg

    def test_resolve_country_code(self):
        """Test alias and EU resolution."""
        assert resolve_country_code("GB") == "GB"
        assert resolve_country_code("UK") == "GB"
        assert resolve_country_code("EU") == EU_MEMBER_CODES
        assert isinstance(resolve_country_code("EU"), frozenset)

    def test_validate_limit(self):
        """Test limit validation."""
        assert validate_limit(1)
        assert validate_limit(500)
        assert not validate_limit(0)
        assert not validate_limit(501)

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
        assert get_cache_key() == "preapproved:v2:all"
        assert get_cache_key("US") == "preapproved:v2:country:US"
        assert get_cache_key(tier="global") == "preapproved:v2:tier:global"
        assert get_cache_key("US", "global") == "preapproved:v2:country:US:tier:global"

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

    def test_convert_image_urls(self):
        """Test image URL conversion from relative to full URLs."""
        # Test relative URLs
        relative_urls = {"webp": "/data/locations/processed/test.webp", "jpeg": "/data/locations/processed/test.jpg"}
        converted = convert_image_urls(relative_urls)
        assert converted["webp"] == "http://localhost:8000/data/locations/processed/test.webp"
        assert converted["jpeg"] == "http://localhost:8000/data/locations/processed/test.jpg"

        # Test already full URLs
        full_urls = {"webp": "https://example.com/test.webp", "jpeg": "https://example.com/test.jpg"}
        converted = convert_image_urls(full_urls)
        assert converted["webp"] == "https://example.com/test.webp"
        assert converted["jpeg"] == "https://example.com/test.jpg"


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

        # Check that image fields are present
        for location in data["locations"]:
            assert "imageUrl" in location
            assert "webp" in location["imageUrl"]
            assert "jpeg" in location["imageUrl"]
            assert "imageAlt" in location

        # Check that continent is present and correct
        continents = {loc["id"]: loc["continent"] for loc in data["locations"]}
        assert continents == {
            "london": "Europe",
            "new_york": "North America",
            "paris": "Europe",
        }

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
        response = client.get("/v1/locations/preapproved?country_code=INVALID")
        assert response.status_code == 400

    def test_unknown_country_code(self, client, mock_locations_data):
        response = client.get("/v1/locations/preapproved?country_code=XX")
        assert response.status_code == 400
        assert "Unknown country code" in response.json()["detail"]

    def test_uk_alias_accepted(self, client, mock_locations_data):
        """UK should be silently resolved to GB."""
        response = client.get("/v1/locations/preapproved?country_code=UK")
        assert response.status_code == 200

    def test_lowercase_country_code_normalised(self, client, mock_locations_data):
        response = client.get("/v1/locations/preapproved?country_code=gb")
        assert response.status_code == 200
        data = response.json()
        assert all(loc["country_code"] == "GB" for loc in data["locations"])

    def test_eu_grouping(self, client, mock_locations_data):
        """EU should return locations from any EU member state."""
        response = client.get("/v1/locations/preapproved?country_code=EU")
        assert response.status_code == 200
        data = response.json()
        # Sample data has FR (Paris); all returned codes must be EU members
        for loc in data["locations"]:
            assert loc["country_code"] in EU_MEMBER_CODES

    def test_invalid_limit_too_low(self, client, mock_locations_data):
        response = client.get("/v1/locations/preapproved?limit=0")
        assert response.status_code == 422

    def test_invalid_limit_too_high(self, client, mock_locations_data):
        response = client.get("/v1/locations/preapproved?limit=501")
        assert response.status_code == 422


class TestCaching:
    """Test caching behavior."""

    def test_etag_not_modified(self, client, mock_locations_data):
        """Test ETag not modified response."""
        # First request to get ETag
        response1 = client.get("/v1/locations/preapproved")
        assert response1.status_code == 200
        etag = response1.headers["ETag"]

        # Second request with If-None-Match
        response2 = client.get("/v1/locations/preapproved", headers={"If-None-Match": etag})
        assert response2.status_code == 304

    def test_last_modified_not_modified(self, client, mock_locations_data):
        """Test Last-Modified not modified response."""
        # First request to get Last-Modified
        response1 = client.get("/v1/locations/preapproved")
        assert response1.status_code == 200
        last_modified = response1.headers["Last-Modified"]

        # Second request with If-Modified-Since
        response2 = client.get("/v1/locations/preapproved", headers={"If-Modified-Since": last_modified})
        assert response2.status_code == 304

    def test_redis_cache_miss(self, client, mock_redis, sample_locations):
        """Test Redis cache miss behavior."""
        with (
            patch("routers.locations.locations_data", sample_locations),
            patch("routers.locations.locations_etag", '"test-etag"'),
            patch("routers.locations.locations_last_modified", "Mon, 01 Jan 2024 00:00:00 GMT"),
            patch("routers.locations.redis_client", mock_redis),
        ):
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
            "locations": [loc.model_dump() for loc in sample_locations],
        }

        with (
            patch("routers.locations.locations_data", sample_locations),
            patch("routers.locations.locations_etag", '"test-etag"'),
            patch("routers.locations.locations_last_modified", "Mon, 01 Jan 2024 00:00:00 GMT"),
            patch("routers.locations.redis_client", mock_redis),
        ):
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
        from routers.locations import check_rate_limit

        # Simulate exceeding rate limit
        ip = "192.168.1.1"

        # Make requests up to the limit
        for _ in range(60):  # RATE_LIMIT_REQUESTS = 60
            allowed, reason = await check_rate_limit(ip)
            assert allowed

        # Next request should be rate limited
        allowed, reason = await check_rate_limit(ip)
        assert not allowed
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
    async def test_initialize_locations_data(self, mock_redis):
        """Test locations data initialization stores data and warms cache."""
        sample_items = [LocationItem(**item) for item in SAMPLE_LOCATIONS]
        with patch(
            "routers.locations.load_locations_data",
            new_callable=AsyncMock,
            return_value=(sample_items, "etag-test", "Wed, 01 Jan 2025 00:00:00 GMT"),
        ):
            await initialize_locations_data(mock_redis)

        # Verify cache was warmed (preapproved:v2:all only — popular cache is not pre-populated)
        assert mock_redis.setex.call_count == 1

    @pytest.mark.asyncio
    async def test_initialize_locations_data_file_not_found(self, mock_redis):
        """Test initialization propagates HTTPException when file not found."""
        from fastapi import HTTPException

        with patch(
            "routers.locations.load_locations_data",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=500, detail="Locations data not available"),
        ):
            with pytest.raises(Exception):  # Should raise HTTPException
                await initialize_locations_data(mock_redis)

    @pytest.mark.asyncio
    async def test_initialize_locations_data_invalid_json(self, mock_redis):
        """Test initialization propagates HTTPException for invalid JSON."""
        from fastapi import HTTPException

        with patch(
            "routers.locations.load_locations_data",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=500, detail="Invalid locations data format"),
        ):
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
        assert data["last_modified"] == "Mon, 01 Jan 2024 00:00:00 GMT"
        assert "cache_enabled" in data
        assert "rate_limit" in data


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_redis_unavailable(self, client, sample_locations):
        """Test behavior when Redis is unavailable."""
        with (
            patch("routers.locations.locations_data", sample_locations),
            patch("routers.locations.locations_etag", '"test-etag"'),
            patch("routers.locations.locations_last_modified", "Mon, 01 Jan 2024 00:00:00 GMT"),
            patch("routers.locations.redis_client", None),
        ):
            # Should still work without Redis, just without caching
            response = client.get("/v1/locations/preapproved")
            assert response.status_code == 200

    def test_malformed_json_in_cache(self, client, mock_redis, sample_locations):
        """Test handling of malformed JSON in cache."""
        with (
            patch("routers.locations.locations_data", sample_locations),
            patch("routers.locations.locations_etag", '"test-etag"'),
            patch("routers.locations.locations_last_modified", "Mon, 01 Jan 2024 00:00:00 GMT"),
            patch("routers.locations.redis_client", mock_redis),
        ):
            mock_redis.get.return_value = "invalid json"

            # Should fall back to generating response from data
            response = client.get("/v1/locations/preapproved")
            assert response.status_code == 200


class TestPopularCacheKey:
    """Test popular cache key generation."""

    def test_get_popular_cache_key(self):
        assert get_popular_cache_key() == "popular:v2:all"
        assert get_popular_cache_key("US") == "popular:v2:country:US"
        assert get_popular_cache_key(tier="global") == "popular:v2:tier:global"
        assert get_popular_cache_key("US", "global") == "popular:v2:country:US:tier:global"


class TestPopularLocationsEndpoint:
    """Test the popular locations endpoint (currently falls back to preapproved data)."""

    @pytest.fixture(autouse=True)
    def no_usage_tracker(self):
        """Force the preapproved-fallback path so these tests are deterministic.

        Without this, get_usage_tracker() returns whatever live tracker the
        process-wide cache singleton holds (e.g. a real Redis connection shared
        with other test modules / a local dev server), and real selection
        signal recorded elsewhere would leak into these counts and orderings.
        """
        with patch("routers.locations.get_usage_tracker", return_value=None):
            yield

    def test_get_all_locations(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == 1
        assert data["count"] == 3
        assert len(data["locations"]) == 3
        assert "generated_at" in data
        assert "Cache-Control" in response.headers

    def test_filter_by_country(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?country_code=US")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["locations"][0]["country_code"] == "US"

    def test_filter_by_tier(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?tier=global")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3

    def test_combined_filters(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?country_code=GB&tier=global")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["locations"][0]["country_code"] == "GB"

    def test_limit(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["locations"]) == 2

    def test_invalid_country_code(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?country_code=INVALID")
        assert response.status_code == 400

    def test_unknown_country_code(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?country_code=XX")
        assert response.status_code == 400
        assert "Unknown country code" in response.json()["detail"]

    def test_uk_alias_accepted(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?country_code=UK")
        assert response.status_code == 200

    def test_lowercase_country_code_normalised(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?country_code=gb")
        assert response.status_code == 200

    def test_eu_grouping(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?country_code=EU")
        assert response.status_code == 200
        data = response.json()
        for loc in data["locations"]:
            assert loc["country_code"] in EU_MEMBER_CODES

    def test_invalid_limit_zero(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?limit=0")
        assert response.status_code == 422

    def test_invalid_limit_too_large(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular?limit=501")
        assert response.status_code == 422

    def test_returns_same_locations_as_preapproved(self, client, mock_locations_data):
        """Popular falls back to preapproved data until usage tracking is implemented."""
        popular = client.get("/v1/locations/popular").json()
        preapproved = client.get("/v1/locations/preapproved").json()

        assert popular["count"] == preapproved["count"]
        popular_ids = sorted(loc["id"] for loc in popular["locations"])
        preapproved_ids = sorted(loc["id"] for loc in preapproved["locations"])
        assert popular_ids == preapproved_ids

    def test_no_image_fields_by_default(self, client, mock_locations_data):
        """Image fields are omitted from the popular response by default."""
        response = client.get("/v1/locations/popular")
        assert response.status_code == 200
        for loc in response.json()["locations"]:
            assert "imageUrl" not in loc
            assert "imageAlt" not in loc
            assert "imageAttribution" not in loc

    def test_include_images_returns_image_fields(self, client, mock_locations_data):
        """include_images=true includes image fields."""
        response = client.get("/v1/locations/popular?include_images=true")
        assert response.status_code == 200
        for loc in response.json()["locations"]:
            assert "imageUrl" in loc
            assert "imageAlt" in loc


class TestPopularStatusEndpoint:
    """Test the popular locations status endpoint."""

    def test_get_status(self, client, mock_locations_data):
        response = client.get("/v1/locations/popular/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["locations_loaded"] == 3
        assert data["fallback"] == "preapproved"
        assert "cache_enabled" in data
        assert "rate_limit" in data


class TestSearchEndpoint:
    """Test GET /v1/locations/search — location_id enrichment."""

    def test_fallback_results_include_location_id(self, client, mock_locations_data):
        """Fallback path (no Mapbox token) always returns location_id."""
        response = client.get("/v1/locations/search?q=London")
        assert response.status_code == 200
        locs = response.json()["locations"]
        assert len(locs) > 0
        for loc in locs:
            assert "location_id" in loc
        london = next(loc for loc in locs if loc["name"] == "London")
        assert london["location_id"] == "london"

    def test_fallback_no_match_still_has_location_id(self, client, mock_locations_data):
        """All fallback results are preapproved locations so location_id is always set."""
        response = client.get("/v1/locations/search?q=Paris")
        assert response.status_code == 200
        for loc in response.json()["locations"]:
            assert loc["location_id"] is not None

    def test_find_preapproved_id_match(self, sample_locations):
        """Helper returns canonical id for exact name+country match."""
        with patch("routers.locations.locations_data", sample_locations):
            assert _find_preapproved_id("London", "GB") == "london"
            assert _find_preapproved_id("london", "GB") == "london"  # case-insensitive
            assert _find_preapproved_id("Paris", "FR") == "paris"
            assert _find_preapproved_id("New York", "US") == "new_york"

    def test_resolve_canonical_id_uses_explicit_location_id(self, sample_locations):
        with patch("routers.locations.locations_data", sample_locations):
            req = SelectionRequest(location_id="my_custom_id")
            assert _resolve_canonical_id(req) == "my_custom_id"

    def test_resolve_canonical_id_matches_preapproved(self, sample_locations):
        with patch("routers.locations.locations_data", sample_locations):
            req = SelectionRequest(name="London", country_code="GB")
            assert _resolve_canonical_id(req) == "london"

    def test_resolve_canonical_id_generates_slug(self):
        with patch("routers.locations.locations_data", []):
            req = SelectionRequest(name="Stoke-on-Trent", country_code="GB")
            assert _resolve_canonical_id(req) == "stoke_on_trent"

    def test_resolve_canonical_id_slug_no_country(self):
        req = SelectionRequest(name="Macclesfield")
        assert _resolve_canonical_id(req) == "macclesfield"

    def test_find_preapproved_id_no_match(self, sample_locations):
        """Helper returns None when no preapproved location matches."""
        with patch("routers.locations.locations_data", sample_locations):
            assert _find_preapproved_id("Shoreditch", "GB") is None
            assert _find_preapproved_id("London", "US") is None  # wrong country

    def test_mapbox_path_enriches_location_id(self, client, mock_locations_data):
        """Mapbox path attaches location_id when result matches a preapproved location."""
        mapbox_results = [
            {"name": "London", "admin1": "England", "country_name": "United Kingdom", "country_code": "GB"},
            {"name": "Shoreditch", "admin1": "England", "country_name": "United Kingdom", "country_code": "GB"},
        ]
        with (
            patch("routers.locations.MAPBOX_TOKEN", "fake-token"),
            patch("routers.locations._geocode_mapbox", new=AsyncMock(return_value=mapbox_results)),
        ):
            response = client.get("/v1/locations/search?q=London")
        assert response.status_code == 200
        locs = {loc["name"]: loc for loc in response.json()["locations"]}
        assert locs["London"]["location_id"] == "london"
        assert locs["Shoreditch"]["location_id"] is None


class TestPopularStatsEndpoint:
    """Test GET /v1/locations/popular/stats."""

    def test_returns_unavailable_when_no_tracker(self, client, mock_locations_data):
        with patch("routers.locations.get_usage_tracker", return_value=None):
            response = client.get("/v1/locations/popular/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert data["total_selections"] == 0
        assert data["using_signal"] is False
        assert data["locations"] == []

    def test_returns_ranked_locations_with_counts(self, client, mock_locations_data):
        tracker = MagicMock()
        tracker.get_popular_from_selections.return_value = [
            ("london", 50),
            ("paris", 30),
        ]
        tracker.get_total_selections.return_value = 80
        with patch("routers.locations.get_usage_tracker", return_value=tracker):
            response = client.get("/v1/locations/popular/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["total_selections"] == 80
        assert data["locations"][0]["location_id"] == "london"
        assert data["locations"][0]["count"] == 50
        assert data["locations"][1]["location_id"] == "paris"
        assert data["locations"][1]["count"] == 30

    def test_using_signal_false_with_no_selections(self, client, mock_locations_data):
        tracker = MagicMock()
        tracker.get_popular_from_selections.return_value = []
        tracker.get_total_selections.return_value = 0
        with patch("routers.locations.get_usage_tracker", return_value=tracker):
            response = client.get("/v1/locations/popular/stats")
        data = response.json()
        assert data["using_signal"] is False

    def test_using_signal_true_with_any_selections(self, client, mock_locations_data):
        tracker = MagicMock()
        tracker.get_popular_from_selections.return_value = []
        tracker.get_total_selections.return_value = 1
        with patch("routers.locations.get_usage_tracker", return_value=tracker):
            response = client.get("/v1/locations/popular/stats")
        data = response.json()
        assert data["using_signal"] is True

    def test_in_preapproved_flag(self, client, mock_locations_data):
        tracker = MagicMock()
        tracker.get_popular_from_selections.return_value = [
            ("london", 10),
            ("unknown_place", 5),
        ]
        tracker.get_total_selections.return_value = 15
        with patch("routers.locations.get_usage_tracker", return_value=tracker):
            response = client.get("/v1/locations/popular/stats")
        locs = {loc["location_id"]: loc for loc in response.json()["locations"]}
        assert locs["london"]["in_preapproved"] is True
        assert locs["unknown_place"]["in_preapproved"] is False
        assert locs["unknown_place"]["name"] is None


class TestSelectionEndpoint:
    """Test POST /v1/locations/selections endpoint."""

    @pytest.fixture
    def main_client(self):
        with TestClient(main_app) as c:
            yield c

    @pytest.fixture(autouse=True)
    def _mock_env(self):
        with patch.dict(
            "os.environ",
            {
                "CACHE_ENABLED": "true",
                "API_ACCESS_TOKEN": "test_api_token",
            },
        ):
            yield

    @pytest.fixture(autouse=True)
    def _mock_firebase(self):
        with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "testuser"}):
            yield

    @pytest.fixture
    def mock_tracker(self):
        tracker = MagicMock()
        # Defaults: no existing metadata, no nearby canonical ID — i.e. every
        # selection in these tests is treated as fresh/standalone unless a
        # test explicitly configures find_nearby_canonical_id to converge.
        tracker.get_location_metadata.return_value = None
        tracker.find_nearby_canonical_id.return_value = None
        with patch("routers.locations.get_usage_tracker", return_value=tracker):
            yield tracker

    def test_record_selection_success(self, main_client, mock_tracker):
        """204 returned and tracker.record_selection called with correct args."""
        response = main_client.post(
            "/v1/locations/selections",
            json={"location_id": "london"},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 204
        mock_tracker.record_selection.assert_called_once_with("london", "testuser")

    def test_record_selection_no_auth(self, main_client):
        """Unauthenticated request should be rejected."""
        response = main_client.post(
            "/v1/locations/selections",
            json={"location_id": "london"},
        )
        assert response.status_code == 401

    def test_record_selection_empty_location_id(self, main_client):
        """Empty location_id with no name fails Pydantic min_length validation."""
        response = main_client.post(
            "/v1/locations/selections",
            json={"location_id": ""},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422

    def test_record_selection_no_fields(self, main_client):
        """Neither location_id nor name → 422 from model_validator."""
        response = main_client.post(
            "/v1/locations/selections",
            json={},
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == 422

    def test_record_selection_with_name_only(self, main_client, mock_tracker):
        """Name-only submission generates a slug and records it."""
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Macclesfield"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.record_selection.assert_called_once_with("macclesfield", "testuser")

    def test_record_selection_name_matches_preapproved(self, main_client, mock_tracker):
        """Name + country_code matching a preapproved location uses its canonical ID."""
        with patch("routers.locations.locations_data", [LocationItem(**SAMPLE_LOCATIONS[0])]):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "London", "country_code": "GB"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.record_selection.assert_called_once_with("london", "testuser")

    def test_record_selection_name_no_preapproved_match_uses_slug(self, main_client, mock_tracker):
        """Name + country_code with no preapproved match falls back to slug and stores metadata."""
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Stoke-on-Trent", "country_code": "GB"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.record_selection.assert_called_once_with("stoke_on_trent", "testuser")
        mock_tracker.store_location_metadata.assert_called_once()
        call_args = mock_tracker.store_location_metadata.call_args
        assert call_args[0][0] == "stoke_on_trent"
        meta = call_args[0][1]
        assert meta["name"] == "Stoke-on-Trent"
        assert meta["country_code"] == "GB"

    def test_record_selection_resolves_country_code_from_country_name(self, main_client, mock_tracker):
        """country_name without country_code is reverse-resolved via pycountry fuzzy search."""
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Chennai", "country_name": "India"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.store_location_metadata.assert_called_once()
        meta = mock_tracker.store_location_metadata.call_args[0][1]
        assert meta["country_name"] == "India"
        assert meta["country_code"] == "IN"

    def test_record_selection_resolves_country_name_from_country_code(self, main_client, mock_tracker):
        """country_code without country_name is resolved to the full name via pycountry."""
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Chennai", "country_code": "IN"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        meta = mock_tracker.store_location_metadata.call_args[0][1]
        assert meta["country_name"] == "India"
        assert meta["country_code"] == "IN"

    def test_record_selection_stores_coordinates(self, main_client, mock_tracker):
        """latitude/longitude are captured in stored metadata when supplied."""
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Chennai", "country_code": "IN", "latitude": 13.0827, "longitude": 80.2707},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        meta = mock_tracker.store_location_metadata.call_args[0][1]
        assert meta["latitude"] == 13.0827
        assert meta["longitude"] == 80.2707

    def test_record_selection_converges_onto_nearby_canonical_id(self, main_client, mock_tracker):
        """A freshly-minted slug with coordinates near an existing canonical ID converges onto it.

        Prevents the same physical place from fragmenting signal across
        differently-spelled submissions, e.g. "Delhi" vs "New Delhi".
        """
        mock_tracker.find_nearby_canonical_id.return_value = "new_delhi"
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Delhi", "country_code": "IN", "latitude": 28.7041, "longitude": 77.1025},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.find_nearby_canonical_id.assert_called_once_with(28.7041, 77.1025, CANONICALIZATION_RADIUS_KM)
        mock_tracker.record_selection.assert_called_once_with("new_delhi", "testuser")
        mock_tracker.add_to_geo_index.assert_not_called()

    def test_record_selection_registers_geo_index_when_no_match(self, main_client, mock_tracker):
        """A freshly-minted slug with coordinates and no nearby match registers itself in the geo-index."""
        mock_tracker.find_nearby_canonical_id.return_value = None
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Chennai", "country_code": "IN", "latitude": 13.0827, "longitude": 80.2707},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.add_to_geo_index.assert_called_once_with("chennai", 13.0827, 80.2707)
        mock_tracker.record_selection.assert_called_once_with("chennai", "testuser")

    def test_record_selection_skips_geo_lookup_without_coordinates(self, main_client, mock_tracker):
        """No lat/lon supplied → geo-canonicalization is skipped entirely."""
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Chennai", "country_code": "IN"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.find_nearby_canonical_id.assert_not_called()
        mock_tracker.add_to_geo_index.assert_not_called()
        mock_tracker.record_selection.assert_called_once_with("chennai", "testuser")

    def test_record_selection_skips_geo_lookup_for_explicit_location_id(self, main_client, mock_tracker):
        """Explicit location_id is trusted as-is — geo-canonicalization never runs for it."""
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"location_id": "my_custom_id", "latitude": 13.0827, "longitude": 80.2707},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.find_nearby_canonical_id.assert_not_called()
        mock_tracker.add_to_geo_index.assert_not_called()
        mock_tracker.record_selection.assert_called_once_with("my_custom_id", "testuser")

    def test_record_selection_skips_geo_lookup_for_preapproved_match(self, main_client, mock_tracker):
        """Name matching a preapproved location is trusted as-is — geo-canonicalization never runs for it."""
        with patch("routers.locations.locations_data", [LocationItem(**SAMPLE_LOCATIONS[0])]):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "London", "country_code": "GB", "latitude": 51.5072, "longitude": -0.1276},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.find_nearby_canonical_id.assert_not_called()
        mock_tracker.add_to_geo_index.assert_not_called()
        mock_tracker.record_selection.assert_called_once_with("london", "testuser")

    def test_record_selection_does_not_overwrite_existing_metadata_after_convergence(self, main_client, mock_tracker):
        """Converging onto an existing canonical ID must not clobber its stored metadata."""
        mock_tracker.find_nearby_canonical_id.return_value = "new_delhi"
        mock_tracker.get_location_metadata.return_value = {"id": "new_delhi", "name": "New Delhi"}
        with patch("routers.locations.locations_data", []):
            response = main_client.post(
                "/v1/locations/selections",
                json={"name": "Delhi", "country_code": "IN", "latitude": 28.7041, "longitude": 77.1025},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204
        mock_tracker.store_location_metadata.assert_not_called()

    def test_record_selection_tracker_unavailable(self, main_client):
        """204 returned silently when tracker is None."""
        with patch("routers.locations.get_usage_tracker", return_value=None):
            response = main_client.post(
                "/v1/locations/selections",
                json={"location_id": "london"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 204

    def test_record_selection_rate_limited(self, main_client, mock_tracker):
        """429 returned when rate limit is exceeded."""
        with patch(
            "routers.locations.check_rate_limit",
            new=AsyncMock(return_value=(False, "Rate limit exceeded")),
        ):
            response = main_client.post(
                "/v1/locations/selections",
                json={"location_id": "london"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == 429


if __name__ == "__main__":
    pytest.main([__file__])
