import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import (
    calculate_historical_average, 
    calculate_trend_slope, 
    CACHE_ENABLED, 
    get_temperature_series,
    app as main_app,
    LocationDiversityMonitor,
    RequestRateMonitor,
    get_client_ip
)

@pytest.fixture
def client():
    """Create a test client using the main app instance."""
    with TestClient(main_app) as test_client:
        yield test_client

# Sample temperature data for testing
SAMPLE_TEMPERATURE_DATA = {
    "data": [
        {"x": 1970, "y": 15.0},
        {"x": 1971, "y": 15.5},
        {"x": 1972, "y": 16.0},
        {"x": 1973, "y": 15.8},
        {"x": 1974, "y": 16.2},
        {"x": datetime.now().year, "y": 17.0}  # Current year
    ],
    "metadata": {
        "total_years": 6,
        "available_years": 6,
        "missing_years": [],
        "completeness": 100.0
    }
}

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Fixture to set up environment variables for testing"""
    with patch.dict('os.environ', {
        'VISUAL_CROSSING_API_KEY': 'test_key',
        'OPENWEATHER_API_KEY': 'test_key',
        'CACHE_ENABLED': 'true',
        'API_ACCESS_TOKEN': 'testing'  # Add the API token
    }):
        yield

@pytest.fixture(autouse=True)
def mock_firebase_verify_id_token():
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "testuser"}):
        yield

def test_calculate_historical_average():
    """Test the historical average calculation function"""
    avg = calculate_historical_average(SAMPLE_TEMPERATURE_DATA["data"])
    # Should only use data up to 1974 (excluding 2024)
    expected_avg = (15.0 + 15.5 + 16.0 + 15.8 + 16.2) / 5
    assert round(avg, 1) == round(expected_avg, 1)

def test_calculate_historical_average_insufficient_data():
    """Test historical average with insufficient data"""
    assert calculate_historical_average([]) == 0.0
    assert calculate_historical_average([{"x": 2024, "y": 17.0}]) == 0.0

def test_calculate_trend_slope():
    """Test the trend slope calculation"""
    slope = calculate_trend_slope(SAMPLE_TEMPERATURE_DATA["data"])
    # The slope should be positive as temperatures are generally increasing
    assert slope > 0

def test_calculate_trend_slope_insufficient_data():
    """Test trend slope with insufficient data"""
    assert calculate_trend_slope([]) == 0.0
    assert calculate_trend_slope([{"x": 2024, "y": 17.0}]) == 0.0

@pytest.mark.parametrize("location,month_day,expected_status", [
    ("London", "05-15", 200),  # Valid date
    ("London", "13-15", 400),  # Invalid month
    ("London", "05-32", 400),  # Invalid day
    ("London", "invalid", 400),  # Invalid format
])
def test_average_endpoint(client, location, month_day, expected_status):
    """Test the average endpoint with various inputs"""
    with patch('main.get_temperature_series', return_value=SAMPLE_TEMPERATURE_DATA):
        response = client.get(
            f"/average/{location}/{month_day}",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            assert "average" in data
            assert "unit" in data
            assert data["unit"] == "celsius"
            assert "data_points" in data
            assert "year_range" in data
            assert "completeness" in data
            assert "missing_years" in data

def test_average_endpoint_no_token(client):
    """Test the average endpoint without an API token"""
    response = client.get("/average/London/05-15")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header." in response.json()["detail"]

@pytest.mark.parametrize("location,month_day,expected_status", [
    ("London", "05-15", 200),  # Valid date
    ("London", "13-15", 400),  # Invalid month
    ("London", "05-32", 400),  # Invalid day
    ("London", "invalid", 400),  # Invalid format
])
def test_trend_endpoint(client, location, month_day, expected_status):
    """Test the trend endpoint with various inputs"""
    with patch('main.get_temperature_series', return_value=SAMPLE_TEMPERATURE_DATA):
        response = client.get(
            f"/trend/{location}/{month_day}",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            assert "slope" in data
            assert "units" in data
            assert data["units"] == "°C/decade"
            assert "data_points" in data
            assert "completeness" in data
            assert "missing_years" in data

def test_weather_endpoint(client):
    """Test the weather endpoint"""
    mock_weather_data = {
        "temp": 20.0,
        "tempmax": 22.0,
        "tempmin": 18.0
    }
    with patch('main.fetch_weather_batch', return_value={"2024-05-15": mock_weather_data}):
        response = client.get(
            "/weather/London/2024-05-15",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "days" in data
        assert len(data["days"]) > 0
        assert "temp" in data["days"][0]

def test_summary_endpoint(client):
    """Test the summary endpoint"""
    with patch('main.get_temperature_series', return_value=SAMPLE_TEMPERATURE_DATA):
        response = client.get(
            "/summary/London/05-15",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 0

@pytest.mark.asyncio
async def test_get_temperature_series_success():
    """Test successful temperature series retrieval"""
    mock_weather_data = {
        "days": [{
            "temp": 15.5,
            "tempmax": 16.0,
            "tempmin": 15.0
        }]
    }
    
    with patch('main.fetch_weather_batch', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"2024-06-02": mock_weather_data}
        result = await get_temperature_series("London", 6, 2)
        assert isinstance(result, dict)
        assert "data" in result
        assert "metadata" in result
        assert len(result["data"]) > 0
        assert all("x" in item and "y" in item for item in result["data"])
        assert all(isinstance(item["y"], float) for item in result["data"])

@pytest.mark.asyncio
async def test_get_temperature_series_partial_failures():
    """Test temperature series retrieval with partial failures"""
    mock_weather_data = {
        "2024-06-02": {"days": [{"temp": 15.5, "tempmax": 16.0, "tempmin": 15.0}]},
        "2023-06-02": {"days": [{"temp": 15.0, "tempmax": 15.5, "tempmin": 14.5}]}
    }
    
    with patch('main.fetch_weather_batch', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_weather_data
        result = await get_temperature_series("London", 6, 2)
        assert isinstance(result, dict)
        assert "data" in result
        assert "metadata" in result
        assert len(result["data"]) > 0
        assert all("x" in item and "y" in item for item in result["data"])
        assert "missing_years" in result["metadata"]

@pytest.mark.asyncio
async def test_average_vs_current_temperature():
    """Test that average temperature and current temperature comparison is accurate"""
    test_data = SAMPLE_TEMPERATURE_DATA["data"]
    
    # Test average calculation
    avg = calculate_historical_average(test_data)
    expected_avg = (15.0 + 15.5 + 16.0 + 15.8 + 16.2) / 5
    assert round(avg, 1) == round(expected_avg, 1)
    
    # Test summary text generation
    from main import get_summary
    # Patch the last year to be current year
    test_data = [
        {"x": 1970, "y": 15.0},
        {"x": 1971, "y": 15.5},
        {"x": 1972, "y": 16.0},
        {"x": 1973, "y": 15.8},
        {"x": 1974, "y": 16.2},
        {"x": datetime.now().year, "y": 17.0}
    ]
    summary = await get_summary(location="London", month_day="05-15", weather_data=test_data)
    current_temp = test_data[-1]["y"]  # 17.0
    temp_diff = round(current_temp - avg, 1)
    
    # Verify the summary text contains the correct temperature difference
    assert f"{temp_diff}°C warmer than average" in summary

@pytest.mark.asyncio
async def test_summary_text_accuracy():
    """Test that summary text accurately reflects temperature differences"""
    test_cases = [
        {
            "data": [
                {"x": 1970, "y": 15.0},
                {"x": 1971, "y": 15.0},
                {"x": 1972, "y": 15.0},
                {"x": 1973, "y": 15.0},
                {"x": 1974, "y": 15.0},
                {"x": datetime.now().year, "y": 17.0}  # 2.0°C warmer
            ],
            "expected_diff": 2.0
        },
        {
            "data": [
                {"x": 1970, "y": 15.0},
                {"x": 1971, "y": 15.0},
                {"x": 1972, "y": 15.0},
                {"x": 1973, "y": 15.0},
                {"x": 1974, "y": 15.0},
                {"x": datetime.now().year, "y": 13.0}  # 2.0°C cooler
            ],
            "expected_diff": -2.0
        }
    ]
    
    from main import get_summary
    for case in test_cases:
        summary = await get_summary(location="London", month_day="05-15", weather_data=case["data"])
        if case["expected_diff"] > 0:
            assert f"{case['expected_diff']}°C warmer than average" in summary
        else:
            assert f"{abs(case['expected_diff'])}°C cooler than average" in summary

# Rate Limiting Tests
class TestLocationDiversityMonitor:
    """Test the LocationDiversityMonitor class"""
    
    def test_init(self):
        """Test monitor initialization"""
        monitor = LocationDiversityMonitor(max_locations=5, window_hours=2)
        assert monitor.max_locations == 5
        assert monitor.window_hours == 2
        assert monitor.window_seconds == 7200
    
    def test_check_location_diversity_under_limit(self):
        """Test location diversity check when under limit"""
        monitor = LocationDiversityMonitor(max_locations=3, window_hours=1)
        
        # Add 2 different locations
        allowed, reason = monitor.check_location_diversity("192.168.1.1", "London")
        assert allowed is True
        assert reason == "OK"
        
        allowed, reason = monitor.check_location_diversity("192.168.1.1", "Paris")
        assert allowed is True
        assert reason == "OK"
        
        # Should still be under limit
        assert len(monitor.ip_locations["192.168.1.1"]) == 2
    
    def test_check_location_diversity_over_limit(self):
        """Test location diversity check when over limit"""
        monitor = LocationDiversityMonitor(max_locations=2, window_hours=1)
        
        # Add 3 different locations
        monitor.check_location_diversity("192.168.1.1", "London")
        monitor.check_location_diversity("192.168.1.1", "Paris")
        
        # Third location should trigger limit
        allowed, reason = monitor.check_location_diversity("192.168.1.1", "Berlin")
        assert allowed is False
        assert "Too many different locations" in reason
        assert "3 > 2" in reason
    
    def test_check_location_diversity_same_location_multiple_times(self):
        """Test that same location multiple times doesn't count as different"""
        monitor = LocationDiversityMonitor(max_locations=2, window_hours=1)
        
        # Add same location multiple times
        monitor.check_location_diversity("192.168.1.1", "London")
        monitor.check_location_diversity("192.168.1.1", "London")
        monitor.check_location_diversity("192.168.1.1", "London")
        
        # Should still be allowed (only 1 unique location)
        allowed, reason = monitor.check_location_diversity("192.168.1.1", "Paris")
        assert allowed is True
        assert reason == "OK"
    
    def test_cleanup_old_entries(self):
        """Test cleanup of old entries"""
        monitor = LocationDiversityMonitor(max_locations=10, window_hours=1)
        
        # Add some entries
        monitor.check_location_diversity("192.168.1.1", "London")
        
        # Manually set old timestamp to trigger cleanup
        old_timestamp = str(time.time() - 7200)  # 2 hours ago
        monitor.ip_locations["192.168.1.1"][old_timestamp] = {"OldLocation"}
        
        # Force cleanup by bypassing the interval check
        monitor.last_cleanup = 0  # Reset to force cleanup
        monitor._cleanup_old_entries()
        
        # Old entries should be removed
        assert old_timestamp not in monitor.ip_locations["192.168.1.1"]
    
    def test_cleanup_interval_respect(self):
        """Test that cleanup respects the interval setting"""
        monitor = LocationDiversityMonitor(max_locations=10, window_hours=1)
        
        # Add some entries
        monitor.check_location_diversity("192.168.1.1", "London")
        
        # Manually set old timestamp
        old_timestamp = str(time.time() - 7200)  # 2 hours ago
        monitor.ip_locations["192.168.1.1"][old_timestamp] = {"OldLocation"}
        
        # Try cleanup without resetting interval - should not run
        monitor._cleanup_old_entries()
        
        # Old entries should still be there (cleanup didn't run due to interval)
        assert old_timestamp in monitor.ip_locations["192.168.1.1"]
        
        # Now force cleanup by resetting interval
        monitor.last_cleanup = 0
        monitor._cleanup_old_entries()
        
        # Now old entries should be removed
        assert old_timestamp not in monitor.ip_locations["192.168.1.1"]
    
    def test_is_suspicious(self):
        """Test suspicious IP detection"""
        monitor = LocationDiversityMonitor(max_locations=2, window_hours=1)
        
        # Initially not suspicious
        assert not monitor.is_suspicious("192.168.1.1")
        
        # Trigger limit to mark as suspicious
        monitor.check_location_diversity("192.168.1.1", "London")
        monitor.check_location_diversity("192.168.1.1", "Paris")
        monitor.check_location_diversity("192.168.1.1", "Berlin")  # This will trigger limit
        
        # Should now be suspicious
        assert monitor.is_suspicious("192.168.1.1")
    
    def test_get_stats(self):
        """Test getting statistics for an IP"""
        monitor = LocationDiversityMonitor(max_locations=5, window_hours=1)
        
        # Add some data
        monitor.check_location_diversity("192.168.1.1", "London")
        monitor.check_location_diversity("192.168.1.1", "Paris")
        
        stats = monitor.get_stats("192.168.1.1")
        assert stats["unique_locations"] == 2
        assert stats["max_locations"] == 5
        assert stats["window_hours"] == 1
        assert stats["is_suspicious"] is False

class TestRequestRateMonitor:
    """Test the RequestRateMonitor class"""
    
    def test_init(self):
        """Test monitor initialization"""
        monitor = RequestRateMonitor(max_requests=50, window_hours=2)
        assert monitor.max_requests == 50
        assert monitor.window_hours == 2
        assert monitor.window_seconds == 7200
    
    def test_check_request_rate_under_limit(self):
        """Test request rate check when under limit"""
        monitor = RequestRateMonitor(max_requests=3, window_hours=1)
        
        # Add 2 requests
        allowed, reason = monitor.check_request_rate("192.168.1.1")
        assert allowed is True
        assert reason == "OK"
        
        allowed, reason = monitor.check_request_rate("192.168.1.1")
        assert allowed is True
        assert reason == "OK"
        
        # Should still be under limit
        assert len(monitor.ip_requests["192.168.1.1"]) > 0
    
    def test_check_request_rate_over_limit(self):
        """Test request rate check when over limit"""
        monitor = RequestRateMonitor(max_requests=2, window_hours=1)
        
        # Add 2 requests
        monitor.check_request_rate("192.168.1.1")
        monitor.check_request_rate("192.168.1.1")
        
        # Third request should trigger limit
        allowed, reason = monitor.check_request_rate("192.168.1.1")
        assert allowed is False
        assert "Too many requests" in reason
        assert "3 > 2" in reason
    
    def test_get_stats(self):
        """Test getting statistics for an IP"""
        monitor = RequestRateMonitor(max_requests=10, window_hours=1)
        
        # Add some requests
        monitor.check_request_rate("192.168.1.1")
        monitor.check_request_rate("192.168.1.1")
        
        stats = monitor.get_stats("192.168.1.1")
        assert stats["total_requests"] == 2
        assert stats["max_requests"] == 10
        assert stats["window_hours"] == 1
        assert stats["remaining_requests"] == 8

class TestRateLimitingIntegration:
    """Test rate limiting integration with FastAPI"""
    
    def test_rate_limit_status_endpoint(self, client):
        """Test the rate limit status endpoint"""
        response = client.get("/rate-limit-status")
        assert response.status_code == 200
        
        data = response.json()
        assert "client_ip" in data
        assert "location_monitor" in data
        assert "request_monitor" in data
        assert "rate_limits" in data
    
    def test_rate_limit_stats_endpoint(self, client):
        """Test the rate limit stats endpoint"""
        response = client.get("/rate-limit-stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_monitored_ips" in data
        assert "suspicious_ips" in data
        assert "ip_details" in data
    
    def test_rate_limiting_on_weather_endpoint(self, client):
        """Test that rate limiting is applied to weather endpoints"""
        # Mock the rate limiting to be enabled
        with patch('main.RATE_LIMIT_ENABLED', True), \
             patch('main.location_monitor') as mock_location_monitor, \
             patch('main.request_monitor') as mock_request_monitor:
            
            # Mock rate limiting responses
            mock_request_monitor.check_request_rate.return_value = (True, "OK")
            mock_location_monitor.check_location_diversity.return_value = (True, "OK")
            
            # Mock weather data
            with patch('main.get_weather_for_date', return_value={"days": [{"temp": 20.0}]}):
                response = client.get(
                    "/weather/London/2024-05-15",
                    headers={"Authorization": "Bearer test_token"}
                )
                
                # Should call rate limiting
                mock_request_monitor.check_request_rate.assert_called_once()
                mock_location_monitor.check_location_diversity.assert_called_once()
    
    def test_rate_limit_exceeded_response(self, client):
        """Test response when rate limit is exceeded"""
        # Mock the rate limiting to be enabled
        with patch('main.RATE_LIMIT_ENABLED', True), \
             patch('main.location_monitor') as mock_location_monitor, \
             patch('main.request_monitor') as mock_request_monitor:
            
            # Mock rate limiting to fail
            mock_request_monitor.check_request_rate.return_value = (False, "Too many requests (101 > 100) in 1 hour(s)")
            
            response = client.get(
                "/weather/London/2024-05-15",
                headers={"Authorization": "Bearer test_token"}
            )
            
            # Should return 429 status
            assert response.status_code == 429
            data = response.json()
            assert "Rate limit exceeded" in data["detail"]
            assert "Too many requests" in data["reason"]
            assert "Retry-After" in response.headers
    
    def test_location_diversity_limit_exceeded(self, client):
        """Test response when location diversity limit is exceeded"""
        # Mock the rate limiting to be enabled
        with patch('main.RATE_LIMIT_ENABLED', True), \
             patch('main.location_monitor') as mock_location_monitor, \
             patch('main.request_monitor') as mock_request_monitor:
            
            # Mock request rate to pass, but location diversity to fail
            mock_request_monitor.check_request_rate.return_value = (True, "OK")
            mock_location_monitor.check_location_diversity.return_value = (False, "Too many different locations (11 > 10) in 1 hour(s)")
            
            response = client.get(
                "/weather/London/2024-05-15",
                headers={"Authorization": "Bearer test_token"}
            )
            
            # Should return 429 status
            assert response.status_code == 429
            data = response.json()
            assert "Location diversity limit exceeded" in data["detail"]
            assert "Too many different locations" in data["reason"]

class TestClientIPDetection:
    """Test client IP address detection"""
    
    def test_get_client_ip_direct(self):
        """Test getting client IP from direct connection"""
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"
        
        ip = get_client_ip(mock_request)
        assert ip == "192.168.1.100"
    
    def test_get_client_ip_x_forwarded_for(self):
        """Test getting client IP from X-Forwarded-For header"""
        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "203.0.113.1, 10.0.0.1"}
        mock_request.client.host = "192.168.1.100"
        
        ip = get_client_ip(mock_request)
        assert ip == "203.0.113.1"  # Should take first IP in chain
    
    def test_get_client_ip_x_real_ip(self):
        """Test getting client IP from X-Real-IP header"""
        mock_request = MagicMock()
        mock_request.headers = {"X-Real-IP": "203.0.113.1"}
        mock_request.client.host = "192.168.1.100"
        
        ip = get_client_ip(mock_request)
        assert ip == "203.0.113.1"  # X-Real-IP takes precedence
    
    def test_get_client_ip_unknown(self):
        """Test getting client IP when no client info available"""
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client = None
        
        ip = get_client_ip(mock_request)
        assert ip == "unknown"

# Performance and Load Testing
class TestRateLimitingPerformance:
    """Test rate limiting performance under load"""
    
    def test_rapid_requests_performance(self):
        """Test performance with rapid requests"""
        monitor = RequestRateMonitor(max_requests=1000, window_hours=1)
        
        start_time = time.time()
        
        # Make 1000 rapid requests
        for i in range(1000):
            allowed, _ = monitor.check_request_rate("192.168.1.1")
            assert allowed is True
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        
        # Check that 1001st request is blocked
        allowed, _ = monitor.check_request_rate("192.168.1.1")
        assert allowed is False
    
    def test_memory_usage_cleanup(self):
        """Test that memory usage doesn't grow indefinitely"""
        monitor = LocationDiversityMonitor(max_locations=10, window_hours=1)
        
        # Add many IPs and locations
        for i in range(100):
            ip = f"192.168.1.{i}"
            for j in range(5):
                location = f"City{j}"
                monitor.check_location_diversity(ip, location)
        
        # Force cleanup by bypassing the interval check
        monitor.last_cleanup = 0  # Reset to force cleanup
        monitor._cleanup_old_entries()
        
        # Memory should be reasonable (not thousands of entries)
        total_entries = sum(len(timestamps) for timestamps in monitor.ip_locations.values())
        assert total_entries < 1000  # Reasonable limit

# Edge Cases and Error Handling
class TestRateLimitingEdgeCases:
    """Test rate limiting edge cases and error handling"""
    
    def test_empty_location_string(self):
        """Test handling of empty location strings"""
        monitor = LocationDiversityMonitor(max_locations=5, window_hours=1)
        
        # Empty string should be treated as a valid location
        allowed, reason = monitor.check_location_diversity("192.168.1.1", "")
        assert allowed is True
        assert reason == "OK"
    
    def test_special_characters_in_location(self):
        """Test handling of special characters in location names"""
        monitor = LocationDiversityMonitor(max_locations=5, window_hours=1)
        
        special_locations = ["New York", "São Paulo", "München", "北京", "México"]
        
        for location in special_locations:
            allowed, reason = monitor.check_location_diversity("192.168.1.1", location)
            assert allowed is True
            assert reason == "OK"
    
    def test_very_long_location_names(self):
        """Test handling of very long location names"""
        monitor = LocationDiversityMonitor(max_locations=5, window_hours=1)
        
        # Very long location name
        long_location = "A" * 1000
        allowed, reason = monitor.check_location_diversity("192.168.1.1", long_location)
        assert allowed is True
        assert reason == "OK"
    
    def test_concurrent_access_safety(self):
        """Test that rate limiting is thread-safe for concurrent access"""
        import threading
        
        monitor = RequestRateMonitor(max_requests=100, window_hours=1)
        results = []
        
        def make_requests(thread_id):
            for i in range(10):
                allowed, _ = monitor.check_request_rate(f"192.168.1.{thread_id}")
                results.append(allowed)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_requests, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should be allowed (50 total < 100 limit)
        assert all(results)
        assert len(results) == 50

# Manual Testing Helpers (for interactive testing)
class TestRateLimitingManual:
    """Helper tests for manual testing scenarios"""
    
    def test_basic_rate_limiting_flow(self, client):
        """Test the basic rate limiting flow manually"""
        # This test can be run manually to verify basic functionality
        response = client.get("/rate-limit-status")
        assert response.status_code == 200
        
        # Check if rate limiting is enabled
        data = response.json()
        if data.get("status") == "disabled":
            pytest.skip("Rate limiting is disabled")
        
        # Make a few requests to see rate limiting in action
        for i in range(3):
            response = client.get(
                "/average/London/05-15",
                headers={"Authorization": "Bearer test_token"}
            )
            # Should succeed initially
            assert response.status_code in [200, 429]  # Either success or rate limited
    
    def test_rate_limit_configuration(self, client):
        """Test rate limiting configuration"""
        response = client.get("/rate-limit-status")
        assert response.status_code == 200
        
        data = response.json()
        rate_limits = data.get("rate_limits", {})
        
        # Verify configuration is reasonable
        assert rate_limits.get("max_locations_per_hour", 0) > 0
        assert rate_limits.get("max_requests_per_hour", 0) > 0
        assert rate_limits.get("window_hours", 0) > 0
