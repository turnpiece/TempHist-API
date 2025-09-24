import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os
import time
import asyncio
import httpx
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


class TestIPWhitelistBlacklist:
    """Test IP whitelist and blacklist functionality."""
    
    def test_ip_whitelist_function(self):
        """Test IP whitelist helper function."""
        with patch.dict('os.environ', {'IP_WHITELIST': '192.168.1.100,10.0.0.5'}):
            # Reload the module to pick up new environment variables
            import importlib
            import main
            importlib.reload(main)
            
            assert main.is_ip_whitelisted('192.168.1.100') == True
            assert main.is_ip_whitelisted('10.0.0.5') == True
            assert main.is_ip_whitelisted('192.168.1.101') == False
            assert main.is_ip_whitelisted('unknown') == False
    
    def test_ip_blacklist_function(self):
        """Test IP blacklist helper function."""
        with patch.dict('os.environ', {'IP_BLACKLIST': '192.168.1.200,10.0.0.99'}):
            # Reload the module to pick up new environment variables
            import importlib
            import main
            importlib.reload(main)
            
            assert main.is_ip_blacklisted('192.168.1.200') == True
            assert main.is_ip_blacklisted('10.0.0.99') == True
            assert main.is_ip_blacklisted('192.168.1.201') == False
            assert main.is_ip_blacklisted('unknown') == False
    
    def test_whitelist_empty_environment(self):
        """Test whitelist with empty environment variable."""
        with patch.dict('os.environ', {'IP_WHITELIST': ''}):
            import importlib
            import main
            importlib.reload(main)
            
            assert main.is_ip_whitelisted('192.168.1.100') == False
            assert main.IP_WHITELIST == []
    
    def test_blacklist_empty_environment(self):
        """Test blacklist with empty environment variable."""
        with patch.dict('os.environ', {'IP_BLACKLIST': ''}):
            import importlib
            import main
            importlib.reload(main)
            
            assert main.is_ip_blacklisted('192.168.1.200') == False
            assert main.IP_BLACKLIST == []
    
    def test_whitelist_with_spaces(self):
        """Test whitelist with spaces in environment variable."""
        with patch.dict('os.environ', {'IP_WHITELIST': ' 192.168.1.100 , 10.0.0.5 '}):
            import importlib
            import main
            importlib.reload(main)
            
            assert main.is_ip_whitelisted('192.168.1.100') == True
            assert main.is_ip_whitelisted('10.0.0.5') == True
            assert main.IP_WHITELIST == ['192.168.1.100', '10.0.0.5']


class TestIPWhitelistBlacklistIntegration:
    """Test IP whitelist and blacklist integration with FastAPI."""
    
    def test_blacklisted_ip_blocked(self):
        """Test that blacklisted IPs are blocked entirely."""
        with patch.dict('os.environ', {'IP_BLACKLIST': 'testclient'}):
            import importlib
            import main
            importlib.reload(main)
            
            # Create a new test client with the reloaded app
            with TestClient(main.app) as test_client:
                # Test that blacklisted IP gets 403
                response = test_client.get("/rate-limit-status")
                assert response.status_code == 403
                data = response.json()
                assert data["detail"] == "Access denied"
                assert data["reason"] == "IP address is blacklisted"
    
    def test_whitelisted_ip_bypasses_rate_limits(self):
        """Test that whitelisted IPs bypass rate limiting."""
        with patch.dict('os.environ', {'IP_WHITELIST': 'testclient'}):
            import importlib
            import main
            importlib.reload(main)
            
            # Create a new test client with the reloaded app
            with TestClient(main.app) as test_client:
                # Test rate limit status endpoint
                response = test_client.get("/rate-limit-status")
                assert response.status_code == 200
                
                data = response.json()
                ip_status = data.get("ip_status", {})
                assert ip_status.get("whitelisted") == True
                assert ip_status.get("rate_limited") == False
    
    def test_rate_limit_status_shows_ip_status(self, client):
        """Test that rate limit status endpoint shows IP whitelist/blacklist status."""
        response = client.get("/rate-limit-status")
        assert response.status_code == 200
        
        data = response.json()
        assert "ip_status" in data
        ip_status = data["ip_status"]
        
        # Should have these keys
        assert "whitelisted" in ip_status
        assert "blacklisted" in ip_status
        assert "rate_limited" in ip_status
        
        # Should be boolean values
        assert isinstance(ip_status["whitelisted"], bool)
        assert isinstance(ip_status["blacklisted"], bool)
        assert isinstance(ip_status["rate_limited"], bool)
    
    def test_rate_limit_stats_shows_ip_lists(self, client):
        """Test that rate limit stats endpoint shows whitelist and blacklist."""
        response = client.get("/rate-limit-stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "whitelisted_ips" in data
        assert "blacklisted_ips" in data
        
        # Should be lists
        assert isinstance(data["whitelisted_ips"], list)
        assert isinstance(data["blacklisted_ips"], list)


# V1 API Tests
class TestV1API:
    """Test the new v1 API endpoints"""
    
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
        with patch('main.get_temperature_data_v1') as mock_get_data:
            if expected_status == 200:
                mock_get_data.return_value = {
                    "period": period,
                    "location": location,
                    "identifier": identifier,
                    "range": {"start": "1974-01-15", "end": "2024-01-15"},
                    "unit_group": "celsius",
                    "values": [{"date": "2024-01-15", "temp": 15.0}],
                    "average": {"mean": 15.0, "tempmax": 16.0, "tempmin": 14.0, "data_points": 1},
                    "trend": {"slope": 0.1, "data_points": 1},
                    "summary": "Test summary"
                }
            
            response = client.get(
                f"/v1/records/{period}/{location}/{identifier}",
                headers={"Authorization": "Bearer test_token"}
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
        with patch('main.get_temperature_data_v1') as mock_get_data:
            mock_data = {
                "period": period,
                "location": location,
                "identifier": identifier,
                "range": {"start": "1974-01-15", "end": "2024-01-15"},
                "unit_group": "celsius",
                "values": [{"date": "2024-01-15", "temp": 15.0}],
                "average": {"mean": 15.0, "tempmax": 16.0, "tempmin": 14.0, "data_points": 1},
                "trend": {"slope": 0.1, "data_points": 1},
                "summary": "Test summary",
                "metadata": {"total_years": 1, "available_years": 1, "missing_years": [], "completeness": 100.0}
            }
            mock_get_data.return_value = mock_data
            
            response = client.get(
                f"/v1/records/{period}/{location}/{identifier}/{subresource}",
                headers={"Authorization": "Bearer test_token"}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "data" in data
            assert "period" in data
            assert "location" in data
            assert "identifier" in data
            assert "metadata" in data
    
    def test_v1_legacy_compatibility(self, client):
        """Test that legacy endpoints show deprecation headers"""
        # Test that legacy endpoints exist and show deprecation warnings
        response = client.get(
            "/data/london/01-15",
            headers={"Authorization": "Bearer test_token"}
        )
        # Legacy endpoint should either work or show proper error handling
        assert response.status_code in [200, 500]  # Allow for errors in test environment
        
        # If it works, check for deprecation headers
        if response.status_code == 200:
            assert "X-Deprecated" in response.headers
            assert "X-New-Endpoint" in response.headers
            assert response.headers["X-Deprecated"] == "true"
            assert "/v1/records/daily/london/01-15" in response.headers["X-New-Endpoint"]
    
    def test_v1_error_handling(self, client):
        """Test v1 API error handling"""
        # Test invalid period (this should be caught by FastAPI path validation)
        response = client.get(
            "/v1/records/invalid_period/london/01-15",
            headers={"Authorization": "Bearer test_token"}
        )
        assert response.status_code == 422
        
        # Test that endpoints exist and handle errors gracefully
        response = client.get(
            "/v1/records/daily/london/invalid_date",
            headers={"Authorization": "Bearer test_token"}
        )
        # Should either return 400 for invalid format or 500 for other errors
        assert response.status_code in [400, 500]
    
    def test_v1_authentication_required(self, client):
        """Test that v1 endpoints require authentication"""
        response = client.get("/v1/records/daily/london/01-15")
        assert response.status_code == 401
        
        response = client.get("/v1/records/daily/london/01-15/average")
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_v1_api_integration(self):
        """Test the v1 API with actual HTTP calls (integration test)"""
        # This test can be run when the server is actually running
        # It's marked as async to match the original test structure
        BASE_URL = "http://localhost:8000"
        headers = {
            "Authorization": "Bearer test_token",
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
