"""Utility functions for the application."""

from .cache_headers import set_weather_cache_headers
from .firebase import initialize_firebase, verify_firebase_token
from .location_validation import InvalidLocationCache, is_location_likely_invalid, validate_location_response
from .redis_client import create_redis_client
from .sanitization import sanitize_for_logging, sanitize_url
from .temperature import calculate_historical_average, calculate_trend_slope, generate_summary, get_friendly_date
from .validation import clean_location_string, validate_date_format, validate_location_for_ssrf
from .weather import (
    create_metadata,
    get_forecast_cache_duration,
    get_year_range,
    is_today,
    is_today_or_future,
    track_missing_year,
)
from .weather_data import fetch_weather_batch, get_forecast_data, get_temperature_series, get_weather_for_date

__all__ = [
    "sanitize_url",
    "sanitize_for_logging",
    "validate_location_for_ssrf",
    "validate_date_format",
    "clean_location_string",
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
