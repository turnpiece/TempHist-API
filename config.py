"""Application configuration and environment variables."""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Environment variables - strip whitespace/newlines from API keys
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379").strip()

# Cache configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_CONTROL_HEADER = "public, max-age=3600, stale-while-revalidate=86400, stale-if-error=86400"
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
# HTTP timeout configuration
HTTP_TIMEOUT_DEFAULT = 60.0  # Default HTTP timeout in seconds
HTTP_TIMEOUT_SHORT = 5.0     # Short timeout for health checks
HTTP_TIMEOUT_LONG = 120.0    # Long timeout for large data requests
HTTP_TIMEOUT = HTTP_TIMEOUT_DEFAULT  # Alias for backward compatibility
MAX_CONCURRENT_REQUESTS = 2  # Reduced for cold start protection

# Visual Crossing API configuration
VISUAL_CROSSING_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
VISUAL_CROSSING_UNIT_GROUP = "metric"
VISUAL_CROSSING_INCLUDE_PARAMS = "days"
VISUAL_CROSSING_REMOTE_DATA = "options=useremote&forecastDataset=era5core"

# Cache durations
SHORT_CACHE_DURATION_SECONDS = 3600  # 1 hour for today's data
LONG_CACHE_DURATION_SECONDS = 604800  # 1 week for historical data (168 hours * 3600)
FORECAST_DAY_CACHE_DURATION_SECONDS = 1800  # 30 minutes
FORECAST_NIGHT_CACHE_DURATION_SECONDS = 7200  # 2 hours

# Import timedelta for compatibility
from datetime import timedelta
SHORT_CACHE_DURATION = timedelta(seconds=SHORT_CACHE_DURATION_SECONDS)
LONG_CACHE_DURATION = timedelta(seconds=LONG_CACHE_DURATION_SECONDS)
FORECAST_DAY_CACHE_DURATION = timedelta(seconds=FORECAST_DAY_CACHE_DURATION_SECONDS)
FORECAST_NIGHT_CACHE_DURATION = timedelta(seconds=FORECAST_NIGHT_CACHE_DURATION_SECONDS)

# Analytics rate limiting
ANALYTICS_RATE_LIMIT = int(os.getenv("ANALYTICS_RATE_LIMIT", "100"))  # 100 requests per hour per IP

# CORS configuration
def validate_cors_config():
    """Validate CORS configuration to prevent misconfiguration."""
    origins = os.getenv("CORS_ORIGINS", "").strip()
    regex = os.getenv("CORS_ORIGIN_REGEX", "").strip()
    env = ENVIRONMENT
    
    # Configure logging first if not already configured
    if not logging.getLogger().handlers:
        log_level = logging.WARNING if LOG_VERBOSITY == "minimal" else (logging.DEBUG if DEBUG or LOG_VERBOSITY == "verbose" else logging.INFO)
        logging.basicConfig(level=log_level)
    
    logger = logging.getLogger(__name__)
    
    # Warn about permissive configurations
    if not origins and not regex:
        logger.warning("⚠️  No CORS origins configured - API may be inaccessible to web clients")
    
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
