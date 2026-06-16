"""Application configuration and environment variables."""

import logging
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

# Single source of truth for .env location (same directory as this file / main.py).
DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379").strip()

# Mapbox Geocoding API — used for location search autocomplete.
# Obtain a public token from https://account.mapbox.com/
# If unset, location search falls back to the small preapproved list (dev mode only).
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "").strip()

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_CONTROL_HEADER = "public, max-age=3600, stale-while-revalidate=86400, stale-if-error=86400"
LOCATIONS_CACHE_CONTROL_HEADER = "public, max-age=3600, s-maxage=604800"
FILTER_WEATHER_DATA = os.getenv("FILTER_WEATHER_DATA", "true").lower() == "true"

# Environment and debug settings
ENVIRONMENT = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Prevent DEBUG mode in production
if ENVIRONMENT == "production" and DEBUG:
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    _temp_logger = _logging.getLogger(__name__)
    _temp_logger.error("❌ DEBUG mode cannot be enabled in production environment")
    raise ValueError("DEBUG=true is forbidden in production. Set ENVIRONMENT=production and DEBUG=false")

# Logging configuration
LOG_VERBOSITY = os.getenv("LOG_VERBOSITY", "normal").lower()  # "minimal", "normal", "verbose"
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN")  # API access token for automated systems
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")  # Admin key for operational/monitoring endpoints
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")  # Public API URL for job callbacks

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
MAX_LOCATIONS_PER_HOUR = int(os.getenv("MAX_LOCATIONS_PER_HOUR", "10"))
MAX_REQUESTS_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "100"))
RATE_LIMIT_WINDOW_HOURS = int(os.getenv("RATE_LIMIT_WINDOW_HOURS", "1"))

# Service Token Rate Limiting Configuration
SERVICE_TOKEN_REQUESTS_PER_HOUR = int(os.getenv("SERVICE_TOKEN_REQUESTS_PER_HOUR", "5000"))
SERVICE_TOKEN_LOCATIONS_PER_HOUR = int(os.getenv("SERVICE_TOKEN_LOCATIONS_PER_HOUR", "500"))
SERVICE_TOKEN_WINDOW_HOURS = int(os.getenv("SERVICE_TOKEN_WINDOW_HOURS", "1"))

# IP address management
IP_WHITELIST = [ip.strip() for ip in os.getenv("IP_WHITELIST", "").split(",") if ip.strip()]
IP_BLACKLIST = [ip.strip() for ip in os.getenv("IP_BLACKLIST", "").split(",") if ip.strip()]

# Usage Tracking Configuration
USAGE_TRACKING_ENABLED = os.getenv("USAGE_TRACKING_ENABLED", "true").lower() == "true"
USAGE_RETENTION_DAYS = int(os.getenv("USAGE_RETENTION_DAYS", "7"))
POPULARITY_WINDOW_DAYS = int(os.getenv("POPULARITY_WINDOW_DAYS", "90"))
# Default limit when the popular endpoint is called without ?limit=.
# Set high — clients that need fewer should pass ?limit=N explicitly.
# The hard cap MAX_LIMIT=500 in the router is the real safety net.
POPULARITY_MAX_LOCATIONS = int(os.getenv("POPULARITY_MAX_LOCATIONS", "500"))
# Identity dedup radius: merges different name strings for the *same* physical
# point (e.g. "Chennai" vs "Chennai, Tamil Nadu, India") into one canonical
# identity — one DB row, one popularity bucket, one display name. Deliberately
# small so genuinely distinct nearby places keep their own identity: Lambeth,
# Croydon, Birmingham and Coventry all rank and display separately. This is NOT
# the data-sharing radius — see METRO_CACHE_SNAP_KM for that.
SAME_LOCATION_RADIUS_KM = float(os.getenv("SAME_LOCATION_RADIUS_KM", "2.0"))
# Metro-area cache snapping: the single data-sharing radius. Distinct places
# within this radius of each other share one Redis temperature record for faster
# cache hits and fewer upstream fetches. TempHist is a temperature-history app
# (climate trends/comparisons), not a hyperlocal forecast — so a place may be
# *labelled* as itself while being *served* a near neighbour's data. This is the
# main speed↔locational-accuracy tuning lever.
METRO_CACHE_SNAP_KM = int(os.getenv("METRO_CACHE_SNAP_KM", "30"))
METRO_CACHE_GRID_DEGREES = float(
    os.getenv("METRO_CACHE_GRID_DEGREES", str(round(METRO_CACHE_SNAP_KM / 100.0, 2)))
)
# HTTP timeout configuration
HTTP_TIMEOUT_DEFAULT = 60.0  # Default HTTP timeout in seconds
HTTP_TIMEOUT_SHORT = 5.0  # Short timeout for health checks
HTTP_TIMEOUT_LONG = 120.0  # Long timeout for large data requests
HTTP_TIMEOUT = HTTP_TIMEOUT_DEFAULT  # Alias for backward compatibility
MAX_CONCURRENT_REQUESTS = 2  # Reduced for cold start protection

# Weather provider selection
# Set WEATHER_PROVIDER=visual_crossing to use Visual Crossing (requires VISUAL_CROSSING_API_KEY).
# Defaults to open_meteo (free, no API key required).
WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "open_meteo").strip().lower()

# Open-Meteo API configuration
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST_PAST_DAYS = int(os.getenv("OPEN_METEO_FORECAST_PAST_DAYS", "7"))
OPEN_METEO_MONITORING_ENABLED = os.getenv("OPEN_METEO_MONITORING_ENABLED", "true").lower() == "true"
OPEN_METEO_STATS_WINDOW_SECONDS = int(os.getenv("OPEN_METEO_STATS_WINDOW_SECONDS", "300"))
OPEN_METEO_DEGRADED_FAILURE_RATE = float(os.getenv("OPEN_METEO_DEGRADED_FAILURE_RATE", "0.05"))
OPEN_METEO_UNHEALTHY_FAILURE_RATE = float(os.getenv("OPEN_METEO_UNHEALTHY_FAILURE_RATE", "0.25"))
OPEN_METEO_CONSECUTIVE_FAILURE_THRESHOLD = int(os.getenv("OPEN_METEO_CONSECUTIVE_FAILURE_THRESHOLD", "10"))

# Cache durations
SHORT_CACHE_DURATION_SECONDS = 3600  # 1 hour for today's data
LONG_CACHE_DURATION_SECONDS = 604800  # 1 week for historical data (168 hours * 3600)
FORECAST_DAY_CACHE_DURATION_SECONDS = 1800  # 30 minutes
FORECAST_NIGHT_CACHE_DURATION_SECONDS = 7200  # 2 hours

SHORT_CACHE_DURATION = timedelta(seconds=SHORT_CACHE_DURATION_SECONDS)
LONG_CACHE_DURATION = timedelta(seconds=LONG_CACHE_DURATION_SECONDS)
FORECAST_DAY_CACHE_DURATION = timedelta(seconds=FORECAST_DAY_CACHE_DURATION_SECONDS)
FORECAST_NIGHT_CACHE_DURATION = timedelta(seconds=FORECAST_NIGHT_CACHE_DURATION_SECONDS)

# Analytics rate limiting
ANALYTICS_RATE_LIMIT = int(os.getenv("ANALYTICS_RATE_LIMIT", "1000"))  # requests per hour per IP (0 = disabled)


# CORS configuration
def validate_cors_config():
    """Validate CORS configuration to prevent misconfiguration."""
    origins = os.getenv("CORS_ORIGINS", "").strip()
    regex = os.getenv("CORS_ORIGIN_REGEX", "").strip()
    env = ENVIRONMENT

    # Configure logging first if not already configured
    if not logging.getLogger().handlers:
        log_level = (
            logging.WARNING
            if LOG_VERBOSITY == "minimal"
            else (logging.DEBUG if DEBUG or LOG_VERBOSITY == "verbose" else logging.INFO)
        )
        logging.basicConfig(level=log_level)

    logger = logging.getLogger(__name__)

    if regex:
        # Test regex is valid and not too permissive
        import re

        try:
            re.compile(regex)
            # Warn if regex looks too permissive
            permissive_patterns = [".*", ".+", r".*\.*"]
            if regex in permissive_patterns or (".*" in regex and env == "production"):
                logger.error(f"❌ CORS regex very permissive: {regex}")
                if env == "production":
                    raise ValueError("Overly permissive CORS regex not allowed in production")
                else:
                    logger.warning(f"⚠️  Permissive CORS regex in {env} environment: {regex}")
        except re.error as e:
            logger.error(f"❌ Invalid CORS_ORIGIN_REGEX: {e}")
            raise ValueError(f"Invalid CORS_ORIGIN_REGEX: {e}")

    if origins == "*":
        logger.error("❌ CORS_ORIGINS set to '*' - this is insecure!")
        if env == "production":
            raise ValueError("Wildcard CORS not allowed in production")
        else:
            logger.warning("⚠️  Wildcard CORS in non-production environment")

    return origins, regex


CORS_ORIGINS, CORS_ORIGIN_REGEX = validate_cors_config()

# Service token rate limits dictionary
SERVICE_TOKEN_RATE_LIMITS = {
    "requests_per_hour": SERVICE_TOKEN_REQUESTS_PER_HOUR,
    "locations_per_hour": SERVICE_TOKEN_LOCATIONS_PER_HOUR,
    "window_hours": SERVICE_TOKEN_WINDOW_HOURS,
}
