"""Utility functions for the application."""
from .sanitization import sanitize_url, sanitize_for_logging
from .validation import validate_location_for_ssrf, validate_date_format, clean_location_string, build_visual_crossing_url
from .weather import (
    get_year_range, create_metadata, track_missing_year,
    is_today, is_today_or_future, get_forecast_cache_duration
)
from .weather_data import (
    get_weather_for_date, get_forecast_data, fetch_weather_batch, get_temperature_series
)
from .temperature import calculate_historical_average, calculate_trend_slope, get_friendly_date, generate_summary
from .location_validation import (
    InvalidLocationCache, validate_location_response, is_location_likely_invalid
)
from .cache_headers import set_weather_cache_headers
from .redis_client import create_redis_client
from .firebase import initialize_firebase, verify_firebase_token

__all__ = [
    "sanitize_url",
    "sanitize_for_logging",
    "validate_location_for_ssrf",
    "validate_date_format",
    "clean_location_string",
    "build_visual_crossing_url",
    "get_year_range",
    "create_metadata",
    "track_missing_year",
    "calculate_historical_average",
    "calculate_trend_slope",
    "get_friendly_date",
    "generate_summary",
    "is_today",
    "is_today_or_future",
    "get_forecast_cache_duration",
    "InvalidLocationCache",
    "validate_location_response",
    "is_location_likely_invalid",
    "set_weather_cache_headers",
    "create_redis_client",
    "initialize_firebase",
    "verify_firebase_token",
    "get_weather_for_date",
    "get_forecast_data",
    "fetch_weather_batch",
    "get_temperature_series",
]
