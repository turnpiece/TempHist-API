import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import (
    calculate_historical_average, 
    calculate_trend_slope, 
    CACHE_ENABLED, 
    get_temperature_series,
    verify_token,
    app as main_app
)

@pytest.fixture
def test_app():
    """Create a test-specific FastAPI app without GZip middleware"""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Copy all routes from main app to test app
    for route in main_app.routes:
        app.routes.append(route)
    
    return app

@pytest.fixture
def client(test_app):
    """Create a test client using the test app"""
    with TestClient(test_app) as test_client:
        yield test_client

# Sample temperature data for testing
SAMPLE_TEMPERATURE_DATA = {
    "data": [
        {"x": 1970, "y": 15.0},
        {"x": 1971, "y": 15.5},
        {"x": 1972, "y": 16.0},
        {"x": 1973, "y": 15.8},
        {"x": 1974, "y": 16.2},
        {"x": 2024, "y": 17.0}  # Current year
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
            headers={"X-API-Token": "testing"}
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
            headers={"X-API-Token": "testing"}
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
            headers={"X-API-Token": "testing"}
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
            headers={"X-API-Token": "testing"}
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
        "temp": 15.5,
        "tempmax": 16.0,
        "tempmin": 15.0
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
        "2024-06-02": {"temp": 15.5, "tempmax": 16.0, "tempmin": 15.0},
        "2023-06-02": {"temp": 15.0, "tempmax": 15.5, "tempmin": 14.5}
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

def test_average_vs_current_temperature():
    """Test that average temperature and current temperature comparison is accurate"""
    test_data = SAMPLE_TEMPERATURE_DATA["data"]
    
    # Test average calculation
    avg = calculate_historical_average(test_data)
    expected_avg = (15.0 + 15.5 + 16.0 + 15.8 + 16.2) / 5
    assert round(avg, 1) == round(expected_avg, 1)
    
    # Test summary text generation
    from main import get_summary
    summary = get_summary(test_data, "2024-05-15")
    current_temp = test_data[-1]["y"]  # 17.0
    temp_diff = round(current_temp - avg, 1)
    
    # Verify the summary text contains the correct temperature difference
    assert f"{temp_diff}°C warmer than average" in summary

def test_summary_text_accuracy():
    """Test that summary text accurately reflects temperature differences"""
    test_cases = [
        {
            "data": [
                {"x": 1970, "y": 15.0},
                {"x": 1971, "y": 15.0},
                {"x": 1972, "y": 15.0},
                {"x": 1973, "y": 15.0},
                {"x": 1974, "y": 15.0},
                {"x": 2024, "y": 17.0}  # 2.0°C warmer
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
                {"x": 2024, "y": 13.0}  # 2.0°C cooler
            ],
            "expected_diff": -2.0
        }
    ]
    
    from main import get_summary
    for case in test_cases:
        summary = get_summary(case["data"], "2024-05-15")
        if case["expected_diff"] > 0:
            assert f"{case['expected_diff']}°C warmer than average" in summary
        else:
            assert f"{abs(case['expected_diff'])}°C cooler than average" in summary
