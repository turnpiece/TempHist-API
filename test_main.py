from main import get_temperature_series, cache
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest

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

@patch("main.fetch_weather_from_api")
def test_cache_usage(mock_fetch):
    # Clear any existing cache
    cache.clear()

    year = datetime.now().year
    date_str = f"{year}-06-02"
    cache_key = f"london_{date_str}"

    # First run: simulate API returning data
    mock_fetch.return_value = {"days": [{"temp": 15.0}]}
    result1 = get_temperature_series("London", 6, 2)

    assert any(d["x"] == year and d["y"] == 15.0 for d in result1)
    assert mock_fetch.called, "Expected API to be called"

    mock_fetch.reset_mock()

    # Second run: should hit the cache, not the API
    result2 = get_temperature_series("London", 6, 2)
    assert any(d["x"] == year and d["y"] == 15.0 for d in result2)
    mock_fetch.assert_not_called()
