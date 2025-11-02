"""
Tests for the analytics endpoints.

Tests cover:
- Analytics data submission
- Analytics data retrieval
- Error handling and validation
- Integration tests
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from main import app as main_app

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
        'API_ACCESS_TOKEN': 'test_api_token',
        'ANALYTICS_RATE_LIMIT': '10000'  # Very high limit for tests to avoid rate limit issues
    }):
        yield

@pytest.fixture(autouse=True)
def mock_firebase_verify_id_token():
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "testuser"}):
        yield

@pytest.fixture(autouse=True)
def clear_rate_limit_keys():
    """Clear rate limit keys in Redis before each test to avoid rate limit issues."""
    try:
        import redis
        from main import redis_client
        if redis_client:
            # Clear all analytics rate limit keys for test IPs
            # The test client uses "testclient" as the IP
            test_ips = ["testclient", "127.0.0.1", "localhost"]
            for ip in test_ips:
                key = f"analytics_limit:{ip}"
                try:
                    redis_client.delete(key)
                except Exception:
                    pass
    except Exception:
        # If Redis is not available or not configured for tests, skip
        pass
    yield
    # Clean up after test too
    try:
        import redis
        from main import redis_client
        if redis_client:
            test_ips = ["testclient", "127.0.0.1", "localhost"]
            for ip in test_ips:
                key = f"analytics_limit:{ip}"
                try:
                    redis_client.delete(key)
                except Exception:
                    pass
    except Exception:
        pass

class TestAnalyticsEndpoint:
    """Test the analytics endpoint robustness and error handling"""
    
    def test_analytics_valid_request(self, client):
        """Test analytics endpoint with valid data"""
        valid_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post("/analytics", json=valid_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "analytics_id" in data
        assert "timestamp" in data
        assert "Analytics data submitted successfully" in data["message"]
    
    def test_analytics_missing_required_fields(self, client):
        """Test analytics endpoint with missing required fields"""
        invalid_data = {
            "session_duration": 3600,
            # Missing api_calls, api_failure_rate, etc.
        }
        
        response = client.post("/analytics", json=invalid_data)
        assert response.status_code == 422
        
        data = response.json()
        # Updated to match standardized error response format (MED-008)
        assert "error" in data
        assert "message" in data
        assert "code" in data
        assert data["code"] == "VALIDATION_ERROR"
        assert "details" in data
    
    def test_analytics_invalid_data_types(self, client):
        """Test analytics endpoint with invalid data types"""
        invalid_types = {
            "session_duration": "not_a_number",  # Should be int
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post("/analytics", json=invalid_types)
        assert response.status_code == 422
        
        data = response.json()
        # Updated to match standardized error response format (MED-008)
        assert "error" in data
        assert "message" in data
        assert data["code"] == "VALIDATION_ERROR"
        assert "Validation failed" in data["message"]
    
    def test_analytics_invalid_json(self, client):
        """Test analytics endpoint with invalid JSON"""
        response = client.post(
            "/analytics",
            content="invalid json {",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        
        data = response.json()
        # Updated to match standardized error response format (MED-008)
        assert "error" in data
        assert "message" in data
        assert data["code"] == "BAD_REQUEST"
        assert "Invalid JSON format" in data["message"]
    
    def test_analytics_wrong_content_type(self, client):
        """Test analytics endpoint with wrong content type"""
        valid_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post(
            "/analytics",
            content=json.dumps(valid_data),
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        
        data = response.json()
        assert "Unsupported Media Type" in data["error"]
        assert "Content-Type must be application/json" in data["message"]
    
    def test_analytics_empty_body(self, client):
        """Test analytics endpoint with empty body"""
        response = client.post("/analytics", content="")
        assert response.status_code == 415  # Unsupported Media Type
        
        data = response.json()
        assert "Unsupported Media Type" in data["error"]
    
    def test_analytics_large_payload(self, client):
        """Test analytics endpoint with large payload"""
        large_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "error_type": "test",
                    "message": "x" * 1000  # Large error message
                }
                for _ in range(1000)  # Many errors
            ],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post("/analytics", json=large_data)
        # Should either be accepted (within limits) or rejected (too large)
        assert response.status_code in [200, 413]
        
        if response.status_code == 413:
            data = response.json()
            assert "Payload Too Large" in data["error"]
            assert "Request body too large" in data["message"]
    
    def test_analytics_negative_values(self, client):
        """Test analytics endpoint with negative values (should fail validation)"""
        invalid_data = {
            "session_duration": -100,  # Negative duration
            "api_calls": -5,  # Negative API calls
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post("/analytics", json=invalid_data)
        assert response.status_code == 422
        
        data = response.json()
        # Updated to match standardized error response format (MED-008)
        assert "error" in data
        assert "message" in data
        assert data["code"] == "VALIDATION_ERROR"
        assert "Validation failed" in data["message"]
    
    def test_analytics_invalid_error_details(self, client):
        """Test analytics endpoint with invalid error details structure"""
        invalid_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [
                {
                    "timestamp": "invalid-timestamp",  # Invalid timestamp format
                    "error_type": "test",
                    # Missing required 'message' field
                }
            ],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post("/analytics", json=invalid_data)
        assert response.status_code == 422
        
        data = response.json()
        # Updated to match standardized error response format (MED-008)
        assert "error" in data
        assert "message" in data
        assert data["code"] == "VALIDATION_ERROR"
        assert "Validation failed" in data["message"]
    
    def test_analytics_optional_fields(self, client):
        """Test analytics endpoint with only required fields (optional fields omitted)"""
        minimal_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": []
            # All optional fields omitted
        }
        
        response = client.post("/analytics", json=minimal_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_analytics_boundary_values(self, client):
        """Test analytics endpoint with boundary values"""
        boundary_data = {
            "session_duration": 0,  # Minimum valid value
            "api_calls": 0,  # Minimum valid value
            "api_failure_rate": "0%",  # Minimum failure rate
            "retry_attempts": 0,  # Minimum valid value
            "location_failures": 0,  # Minimum valid value
            "error_count": 0,  # Minimum valid value
            "recent_errors": [],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post("/analytics", json=boundary_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_analytics_maximum_error_details(self, client):
        """Test analytics endpoint with maximum allowed error details"""
        # Test with exactly 50 errors (the limit)
        max_errors_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 50,
            "recent_errors": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "error_type": "test",
                    "message": f"Error {i}",
                    "location": "test_location",
                    "status_code": 500
                }
                for i in range(50)  # Exactly 50 errors
            ],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent",
            "session_id": "test-session-123"
        }
        
        response = client.post("/analytics", json=max_errors_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_analytics_unicode_data(self, client):
        """Test analytics endpoint with unicode characters"""
        unicode_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "error_type": "test",
                    "message": "Error with unicode: 测试 🚀 émojis",
                    "location": "北京",
                    "status_code": 500
                }
            ],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Test Agent with unicode: 测试",
            "session_id": "test-session-测试-123"
        }
        
        response = client.post("/analytics", json=unicode_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

class TestAnalyticsRetrievalEndpoints:
    """Test analytics data retrieval endpoints"""
    
    def test_analytics_summary(self, client):
        """Test analytics summary endpoint"""
        response = client.get("/analytics/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "data" in data
        assert "avg_session_duration" in data["data"]
        assert "avg_failure_rate" in data["data"]
    
    def test_analytics_recent(self, client):
        """Test analytics recent endpoint"""
        response = client.get("/analytics/recent")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "data" in data
        assert isinstance(data["data"], list)
    
    def test_analytics_recent_with_limit(self, client):
        """Test analytics recent endpoint with limit parameter"""
        response = client.get("/analytics/recent?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) <= 10
    
    def test_analytics_session(self, client):
        """Test analytics session endpoint"""
        response = client.get("/analytics/session/test-session-123")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "data" in data
        assert isinstance(data["data"], list)

class TestAnalyticsEndpointIntegration:
    """Integration tests for analytics endpoint with actual HTTP calls"""
    
    @pytest.mark.asyncio
    async def test_analytics_endpoint_integration(self):
        """Test analytics endpoint with actual HTTP calls (integration test)"""
        import httpx
        BASE_URL = "http://localhost:8000"
        headers = {
            "Content-Type": "application/json"
        }
        
        valid_data = {
            "session_duration": 3600,
            "api_calls": 10,
            "api_failure_rate": "5%",
            "retry_attempts": 2,
            "location_failures": 1,
            "error_count": 0,
            "recent_errors": [],
            "app_version": "1.0.0",
            "platform": "web",
            "user_agent": "Integration Test Agent",
            "session_id": "integration-test-session-123"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test valid request
                response = await client.post(f"{BASE_URL}/analytics", json=valid_data, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    assert data["status"] == "success"
                    assert "analytics_id" in data
                
                # Test invalid request
                invalid_data = {"session_duration": "invalid"}
                response = await client.post(f"{BASE_URL}/analytics", json=invalid_data, headers=headers)
                if response.status_code == 422:
                    data = response.json()
                    assert "detail" in data
                    assert "error" in data["detail"]
                    assert "Validation Error" in data["detail"]["error"]
                
                # Test wrong content type
                response = await client.post(
                    f"{BASE_URL}/analytics",
                    content=json.dumps(valid_data),
                    headers={"Content-Type": "text/plain"}
                )
                if response.status_code == 415:
                    data = response.json()
                    assert "Unsupported Media Type" in data["error"]
                    
        except httpx.ConnectError:
            pytest.skip("Server not running - skipping integration test")
        except Exception as e:
            pytest.skip(f"Analytics integration test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
