import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock
import json
import os
from main import app, calculate_historical_average, calculate_trend_slope, CACHE_ENABLED, get_temperature_series

client = TestClient(app)

# Sample temperature data for testing
SAMPLE_TEMPERATURE_DATA = [
    {"x": 1970, "y": 15.0},
    {"x": 1971, "y": 15.5},
    {"x": 1972, "y": 16.0},
    {"x": 1973, "y": 15.8},
    {"x": 1974, "y": 16.2},
    {"x": 2024, "y": 17.0},  # Current year
]

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Fixture to set up environment variables for testing"""
    with patch.dict('os.environ', {
        'VISUAL_CROSSING_API_KEY': 'test_key',
        'OPENWEATHER_API_KEY': 'test_key',
        'CACHE_ENABLED': 'true'
    }):
        yield

def test_calculate_historical_average():
    """Test the historical average calculation function"""
    avg = calculate_historical_average(SAMPLE_TEMPERATURE_DATA)
    # Should only use data up to 1974 (excluding 2024)
    expected_avg = (15.0 + 15.5 + 16.0 + 15.8 + 16.2) / 5
    assert round(avg, 1) == round(expected_avg, 1)

def test_calculate_historical_average_insufficient_data():
    """Test historical average with insufficient data"""
    assert calculate_historical_average([]) == 0.0
    assert calculate_historical_average([{"x": 2024, "y": 17.0}]) == 0.0

def test_calculate_trend_slope():
    """Test the trend slope calculation"""
    slope = calculate_trend_slope(SAMPLE_TEMPERATURE_DATA)
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
def test_average_endpoint(location, month_day, expected_status):
    """Test the average endpoint with various inputs"""
    with patch('main.get_temperature_series', return_value=SAMPLE_TEMPERATURE_DATA):
        response = client.get(f"/average/{location}/{month_day}")
        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            assert "average" in data
            assert "unit" in data
            assert data["unit"] == "celsius"
            assert "data_points" in data
            assert "year_range" in data

@pytest.mark.parametrize("location,month_day,expected_status", [
    ("London", "05-15", 200),  # Valid date
    ("London", "13-15", 400),  # Invalid month
    ("London", "05-32", 400),  # Invalid day
    ("London", "invalid", 400),  # Invalid format
])
def test_trend_endpoint(location, month_day, expected_status):
    """Test the trend endpoint with various inputs"""
    with patch('main.get_temperature_series', return_value=SAMPLE_TEMPERATURE_DATA):
        response = client.get(f"/trend/{location}/{month_day}")
        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            assert "slope" in data
            assert "units" in data
            assert data["units"] == "°C/year"

def test_weather_endpoint():
    """Test the weather endpoint"""
    mock_weather_data = {
        "days": [{"temp": 20.0}],
        "address": "London"
    }
    with patch('main.fetch_weather_from_api', return_value=mock_weather_data):
        response = client.get("/weather/London/2024-05-15")
        assert response.status_code == 200
        data = response.json()
        assert "days" in data
        assert len(data["days"]) > 0
        assert "temp" in data["days"][0]

def test_summary_endpoint():
    """Test the summary endpoint"""
    with patch('main.get_temperature_series', return_value=SAMPLE_TEMPERATURE_DATA):
        response = client.get("/summary/London/05-15")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 0

@pytest.fixture
def mock_weather_response():
    return {
        "days": [{"temp": 15.5}]
    }

@patch("main.fetch_weather_from_api")
def test_get_temperature_series_success(mock_fetch, mock_weather_response):
    mock_fetch.return_value = mock_weather_response

    # Test for a known date that won't raise edge cases
    result = get_temperature_series("London", 6, 2)

    # Should get 50+1 = 51 years of data if cache and mock both work
    assert isinstance(result, list)
    assert len(result) >= 10  # Allow for some skipped
    assert all("x" in item and "y" in item for item in result)
    assert all(isinstance(item["y"], float) for item in result)

@patch("main.fetch_weather_from_api")
def test_get_temperature_series_partial_failures(mock_fetch):
    def conditional_response(location, date):
        year = int(date.split("-")[0])
        if year % 2 == 0:
            return {"days": [{"temp": 10 + (year % 5)}]}
        return {"error": "No data"}
    mock_fetch.side_effect = conditional_response

    result = get_temperature_series("London", 6, 2)
    assert len(result) > 0
    assert all("x" in item and "y" in item for item in result)

def test_average_vs_current_temperature():
    """Test that average temperature and current temperature comparison is accurate"""
    # Sample data with a clear difference between current and average
    test_data = [
        {"x": 1970, "y": 15.0},
        {"x": 1971, "y": 15.5},
        {"x": 1972, "y": 16.0},
        {"x": 1973, "y": 15.8},
        {"x": 1974, "y": 16.2},
        {"x": 2024, "y": 17.3}  # Current year with higher temperature
    ]
    
    # Test average calculation
    avg = calculate_historical_average(test_data)
    expected_avg = (15.0 + 15.5 + 16.0 + 15.8 + 16.2) / 5
    assert round(avg, 1) == round(expected_avg, 1)
    
    # Test summary text generation
    from main import get_summary
    summary = get_summary(test_data, "2024-05-15")
    current_temp = test_data[-1]["y"]  # 17.3
    temp_diff = round(current_temp - avg, 1)  # Should be about 1.3°C
    
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

def test_live_site_scenario():
    """Test the specific scenario from the live site"""
    # Using the exact temperatures mentioned: current 17.3°C, average 15.1°C
    test_data = [
        {"x": 1975, "y": 15.0},
        {"x": 1976, "y": 15.0},
        {"x": 1977, "y": 15.0},
        {"x": 1978, "y": 15.0},
        {"x": 1979, "y": 15.0},
        {"x": 1980, "y": 15.0},
        {"x": 1981, "y": 15.0},
        {"x": 1982, "y": 15.0},
        {"x": 1983, "y": 15.0},
        {"x": 1984, "y": 15.0},
        {"x": 1985, "y": 15.0},
        {"x": 1986, "y": 15.0},
        {"x": 1987, "y": 15.0},
        {"x": 1988, "y": 15.0},
        {"x": 1989, "y": 15.0},
        {"x": 1990, "y": 15.0},
        {"x": 1991, "y": 15.0},
        {"x": 1992, "y": 15.0},
        {"x": 1993, "y": 15.0},
        {"x": 1994, "y": 15.0},
        {"x": 1995, "y": 15.0},
        {"x": 1996, "y": 15.0},
        {"x": 1997, "y": 15.0},
        {"x": 1998, "y": 15.0},
        {"x": 1999, "y": 15.0},
        {"x": 2000, "y": 15.0},
        {"x": 2001, "y": 15.0},
        {"x": 2002, "y": 15.0},
        {"x": 2003, "y": 15.0},
        {"x": 2004, "y": 15.0},
        {"x": 2005, "y": 15.0},
        {"x": 2006, "y": 15.0},
        {"x": 2007, "y": 15.0},
        {"x": 2008, "y": 15.0},
        {"x": 2009, "y": 15.0},
        {"x": 2010, "y": 15.0},
        {"x": 2011, "y": 15.0},
        {"x": 2012, "y": 15.0},
        {"x": 2013, "y": 15.0},
        {"x": 2014, "y": 15.0},
        {"x": 2015, "y": 15.0},
        {"x": 2016, "y": 15.0},
        {"x": 2017, "y": 15.0},
        {"x": 2018, "y": 15.0},
        {"x": 2019, "y": 15.0},
        {"x": 2020, "y": 15.0},
        {"x": 2021, "y": 15.0},
        {"x": 2022, "y": 15.0},
        {"x": 2023, "y": 15.0},
        {"x": 2024, "y": 17.3}  # Current year
    ]
    
    # Add variation to get an average of 15.1
    # We need 20% of the years to be 15.5°C to get an average of 15.1°C
    # (0.8 * 15.0 + 0.2 * 15.5 = 15.1)
    for i in range(0, len(test_data) - 1, 5):  # Every 5th year (20% of years)
        test_data[i]["y"] = 15.5
    
    # Test average calculation
    avg = calculate_historical_average(test_data)
    print(f"Calculated average: {avg}")  # Debug print
    assert round(avg, 1) == 15.1  # Should match the live site average
    
    # Test summary text generation
    from main import get_summary
    summary = get_summary(test_data, "2024-05-15")
    current_temp = test_data[-1]["y"]  # 17.3
    temp_diff = round(current_temp - avg, 1)  # Should be 2.2°C
    print(f"Temperature difference: {temp_diff}°C")  # Debug print
    
    # Verify the summary text contains the correct temperature difference
    assert f"{temp_diff}°C warmer than average" in summary
    assert "4.0°C warmer than average" not in summary  # Should not show the incorrect value
