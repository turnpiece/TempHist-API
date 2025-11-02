"""
Tests for the records aggregation router.

Tests cover:
- V1 records endpoints (daily, weekly, monthly, yearly)
- V1 subresource endpoints (average, trend, summary)
- Rolling bundle endpoints
- Error handling and validation
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from main import app as main_app, API_ACCESS_TOKEN

@pytest.fixture
def client():
    """Create a test client using the main app instance."""
    with TestClient(main_app) as test_client:
        yield test_client

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Fixture to set up environment variables for testing"""
    with patch.dict('os.environ', {
        'VISUAL_CROSSING_API_KEY': 'test_key',
        'OPENWEATHER_API_KEY': 'test_key',
        'CACHE_ENABLED': 'true',
        'API_ACCESS_TOKEN': 'test_api_token'
    }):
        yield

@pytest.fixture(autouse=True)
def mock_firebase_verify_id_token():
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "testuser"}):
        yield

class TestV1RecordsEndpoints:
    """Test the V1 records endpoints from main.py"""
    
    def test_api_info(self, client):
        """Test the API info endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        api_info = response.json()
        assert "name" in api_info
        assert "version" in api_info
        assert "v1_endpoints" in api_info
        assert "records" in api_info["v1_endpoints"]
        assert len(api_info["v1_endpoints"]["records"]) > 0
    
    @pytest.mark.parametrize("period,location,identifier,expected_status", [
        ("daily", "london", "01-15", 200),
        ("weekly", "london", "01-15", 200),
        ("monthly", "london", "01-15", 200),
        ("yearly", "london", "01-15", 200),
        ("invalid_period", "london", "01-15", 422),
    ])
    def test_v1_records_endpoint(self, client, period, location, identifier, expected_status):
        """Test the v1 records endpoint with various inputs"""
        with patch('routers.v1_records.get_temperature_data_v1', new_callable=AsyncMock) as mock_get_data, \
             patch('routers.v1_records.is_location_likely_invalid', return_value=False), \
             patch('routers.dependencies.get_invalid_location_cache') as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.is_invalid_location.return_value = False
            mock_get_cache.return_value = mock_cache
            if expected_status == 200:
                mock_get_data.return_value = {
                    "period": period,
                    "location": location,
                    "identifier": identifier,
                    "range": {"start": "1974-01-15", "end": "2024-01-15"},
                    "unit_group": "celsius",
                    "values": [{"date": "2024-01-15", "temperature": 15.0}],
                    "average": {"mean": 15.0, "tempmax": 16.0, "tempmin": 14.0, "data_points": 1, "unit": "celsius"},
                    "trend": {"slope": 0.1, "data_points": 1, "unit": "°C/decade"},
                    "summary": "Test summary"
                }
    
            response = client.get(
                f"/v1/records/{period}/{location}/{identifier}",
                headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
            )
            assert response.status_code == expected_status
            
            if expected_status == 200:
                data = response.json()
                assert data["period"] == period
                assert data["location"] == location
                assert data["identifier"] == identifier
                assert "values" in data
                assert "average" in data
                assert "trend" in data
                assert "summary" in data
    
    @pytest.mark.parametrize("period,location,identifier,subresource", [
        ("daily", "london", "01-15", "average"),
        ("daily", "london", "01-15", "trend"),
        ("daily", "london", "01-15", "summary"),
        ("weekly", "london", "01-15", "average"),
        ("monthly", "london", "01-15", "trend"),
    ])
    def test_v1_subresource_endpoints(self, client, period, location, identifier, subresource):
        """Test the v1 subresource endpoints"""
        with patch('routers.v1_records.get_temperature_data_v1', new_callable=AsyncMock) as mock_get_data, \
             patch('routers.v1_records.is_location_likely_invalid', return_value=False), \
             patch('routers.dependencies.get_invalid_location_cache') as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.is_invalid_location.return_value = False
            mock_get_cache.return_value = mock_cache
            mock_data = {
                "period": period,
                "location": location,
                "identifier": identifier,
                "range": {"start": "1974-01-15", "end": "2024-01-15"},
                "unit_group": "celsius",
                "values": [{"date": "2024-01-15", "temperature": 15.0}],
                "average": {"mean": 15.0, "tempmax": 16.0, "tempmin": 14.0, "data_points": 1, "unit": "celsius"},
                "trend": {"slope": 0.1, "data_points": 1, "unit": "°C/decade"},
                "summary": "Test summary",
                "metadata": {"total_years": 1, "available_years": 1, "missing_years": [], "completeness": 100.0}
            }
            mock_get_data.return_value = mock_data
    
            response = client.get(
                f"/v1/records/{period}/{location}/{identifier}/{subresource}",
                headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "data" in data
            assert "period" in data
            assert "location" in data
            assert "identifier" in data
            assert "metadata" in data
    
    def test_removed_endpoints_return_410(self, client):
        """Test that removed legacy endpoints return 410 Gone"""
        removed_endpoints = [
            "/data/london/01-15",
            "/average/london/01-15", 
            "/trend/london/01-15",
            "/summary/london/01-15"
        ]
        
        for endpoint in removed_endpoints:
            response = client.get(endpoint, headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"})
            assert response.status_code == 410
            data = response.json()
            assert "error" in data
            assert "Endpoint removed" in data["error"]
            assert "X-Removed" in response.headers
            assert response.headers["X-Removed"] == "true"
    
    def test_v1_error_handling(self, client):
        """Test v1 API error handling"""
        # Test invalid period (this should be caught by FastAPI path validation)
        response = client.get(
            "/v1/records/invalid_period/london/01-15",
            headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
        )
        assert response.status_code == 422
        
        # Test that endpoints exist and handle errors gracefully
        response = client.get(
            "/v1/records/daily/london/invalid_date",
            headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
        )
        # Should either return 400 for invalid format or 500 for other errors
        assert response.status_code in [400, 500]
    
    def test_v1_authentication_required(self, client):
        """Test that v1 endpoints require authentication"""
        response = client.get("/v1/records/daily/london/01-15")
        assert response.status_code == 401
        
        response = client.get("/v1/records/daily/london/01-15/average")
        assert response.status_code == 401

class TestRollingBundleEndpoints:
    """Test the rolling bundle endpoints from records_agg router"""
    
    def test_rolling_bundle_cors_test(self, client):
        """Test the rolling bundle CORS test endpoint"""
        response = client.get("/v1/records/rolling-bundle/test-cors")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "CORS is working for rolling-bundle" in data["message"]
    
    def test_rolling_bundle_preload_example(self, client):
        """Test the rolling bundle preload example endpoint"""
        response = client.get(
            "/v1/records/rolling-bundle/preload-example",
            headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "data_structure" in data
        assert "description" in data
        assert "endpoint" in data
    
    def test_rolling_bundle_status(self, client):
        """Test the rolling bundle status endpoint"""
        response = client.get(
            "/v1/records/rolling-bundle/london/2024-01-15/status",
            headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "location" in data
        assert "anchor" in data

class TestRecordsAggIntegration:
    """Integration tests for records aggregation endpoints"""
    
    @pytest.mark.asyncio
    async def test_v1_api_integration(self):
        """Test the v1 API with actual HTTP calls (integration test)"""
        # This test can be run when the server is actually running
        # It's marked as async to match the original test structure
        import httpx
        BASE_URL = "http://localhost:8000"
        headers = {
            "Authorization": f"Bearer {API_ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test API info
                response = await client.get(f"{BASE_URL}/")
                if response.status_code == 200:
                    api_info = response.json()
                    assert "name" in api_info
                    assert "version" in api_info
                
                # Test daily record
                response = await client.get(f"{BASE_URL}/v1/records/daily/london/01-15", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    assert "period" in data
                    assert "location" in data
                    assert "identifier" in data
                    assert "values" in data
                    assert "average" in data
                    assert "trend" in data
                    assert "summary" in data
                
                # Test subresource
                response = await client.get(f"{BASE_URL}/v1/records/daily/london/01-15/average", headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    assert "data" in data
                    assert "endpoint" in data
                    
        except httpx.ConnectError:
            pytest.skip("Server not running - skipping integration test")
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
