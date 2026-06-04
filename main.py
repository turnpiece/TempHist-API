# Standard library imports
import asyncio
import json
import logging
import mimetypes
import os
import time
from datetime import date as dt_date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Third-party imports
import firebase_admin
import httpx
import redis
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from firebase_admin import app_check, auth, credentials
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

# Load .env and populate os.environ before routers (see config.DOTENV_PATH).
import config  # noqa: F401
from cache.accessors import get_cache_warmer, get_usage_tracker, initialize_cache

# Import enhanced caching utilities
from cache.warming import CACHE_WARMING_ENABLED, scheduled_cache_warming

# Import configuration and rate limiting
from config import (
    API_ACCESS_TOKEN,
    CORS_ORIGIN_REGEX,
    CORS_ORIGINS,
    DEBUG,
    ENVIRONMENT,
    HTTP_TIMEOUT,
    IP_BLACKLIST,
    IP_WHITELIST,
    LOG_VERBOSITY,
    MAX_LOCATIONS_PER_HOUR,
    MAX_REQUESTS_PER_HOUR,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_WINDOW_HOURS,
    SERVICE_TOKEN_RATE_LIMITS,
)
from exceptions import register_exception_handlers
from rate_limiting import LocationDiversityMonitor, RequestRateMonitor, ServiceTokenRateLimiter
from routers.analytics import router as analytics_router
from routers.cache import router as cache_router
from routers.dependencies import initialize_dependencies
from routers.health import router as health_router
from routers.jobs import router as jobs_router
from routers.legacy import router as legacy_router
from routers.locations import initialize_locations_data

# Import routers
from routers.locations import router as locations_router
from routers.og_image import router as og_image_router
from routers.root import router as root_router
from routers.shares import router as shares_router
from routers.stats import router as stats_router
from routers.v1_records import router as v1_records_router
from routers.weather import router as weather_router
from utils.ip_utils import get_client_ip, is_ip_blacklisted, is_ip_whitelisted

if not CORS_ORIGINS and not CORS_ORIGIN_REGEX:
    logging.getLogger(__name__).warning("⚠️  No CORS origins configured - API may be inaccessible to web clients")


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379").strip()

# Debug: Log the REDIS_URL value to diagnose connection issues (sanitized)
import logging as _logging
from urllib.parse import urlparse, urlunparse


def sanitize_url(url: str) -> str:
    """Remove credentials and API keys from URL for logging to prevent sensitive data exposure."""
    try:
        from urllib.parse import parse_qs, urlencode

        parsed = urlparse(url)

        # Redact password from netloc
        if parsed.password:
            username = parsed.username or ""
            netloc = f"{username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            parsed = parsed._replace(netloc=netloc)

        # Redact API keys from query parameters
        if parsed.query:
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            # Redact sensitive parameters
            sensitive_params = ["key", "api_key", "apikey", "token", "password", "secret"]
            for param in sensitive_params:
                if param in query_params:
                    query_params[param] = ["[REDACTED]"]
            # Reconstruct query string
            sanitized_query = urlencode(query_params, doseq=True)
            parsed = parsed._replace(query=sanitized_query)

        return urlunparse(parsed)
    except (ValueError, TypeError) as e:
        # If parsing fails, return a safe placeholder
        logger.debug(f"Failed to parse URL for sanitization: {e}")
        return "[REDACTED_URL]"


def sanitize_for_logging(data: str, max_length: int = 100) -> str:
    """Sanitize user input data before logging to prevent log injection and sensitive data exposure (MED-005).

    Args:
        data: The input string to sanitize
        max_length: Maximum length to keep (default 100 chars)

    Returns:
        Sanitized string safe for logging
    """
    if not data or not isinstance(data, str):
        return str(data)[:max_length] if data else ""

    # Truncate to max length
    if len(data) > max_length:
        data = data[:max_length] + "..."

    # Remove or replace control characters that could be used for log injection
    # Replace newlines, tabs, and other control chars with spaces
    import re

    data = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", data)

    # Remove potential sensitive patterns
    # Redact bearer tokens
    data = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", data, flags=re.IGNORECASE)
    # Redact API keys in URLs
    data = re.sub(r"key=[^&\s]+", "key=[REDACTED]", data, flags=re.IGNORECASE)
    data = re.sub(r"api_key=[^&\s]+", "api_key=[REDACTED]", data, flags=re.IGNORECASE)
    # Redact tokens
    data = re.sub(r"token=[^&\s]+", "token=[REDACTED]", data, flags=re.IGNORECASE)

    # Clean up multiple spaces
    data = re.sub(r"\s+", " ", data).strip()

    return data


_temp_logger = _logging.getLogger(__name__)
_temp_logger.info(f"🔍 DEBUG: REDIS_URL environment variable = {sanitize_url(REDIS_URL)}")

# LOW-008: Validate debug mode with environment check
# CACHE_ENABLED, ENVIRONMENT, and DEBUG are imported from config

# Prevent DEBUG mode in production
if ENVIRONMENT == "production" and DEBUG:
    # Use basic logging since logger not yet initialized
    _logging.basicConfig(level=_logging.INFO)
    _temp_logger = _logging.getLogger(__name__)
    _temp_logger.error("❌ DEBUG mode cannot be enabled in production environment")
    raise ValueError("DEBUG=true is forbidden in production. Set ENVIRONMENT=production and DEBUG=false")

# Configuration variables are imported from config
# LOG_VERBOSITY, API_ACCESS_TOKEN, CACHE_CONTROL_HEADER, FILTER_WEATHER_DATA,
# RATE_LIMIT_ENABLED, MAX_LOCATIONS_PER_HOUR, MAX_REQUESTS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS,
# IP_WHITELIST, IP_BLACKLIST


# Analytics Models
class ErrorDetail(BaseModel):
    """Individual error detail."""

    timestamp: str = Field(..., description="Error timestamp in ISO format")
    error_type: str = Field(..., description="Type of error (network, api, validation, etc.)")
    message: str = Field(..., description="Error message")
    location: Optional[str] = Field(None, description="Location where error occurred")
    endpoint: Optional[str] = Field(None, description="API endpoint that failed")
    status_code: Optional[int] = Field(None, description="HTTP status code if applicable")


class AnalyticsData(BaseModel):
    """Analytics data from client applications."""

    session_duration: int = Field(..., ge=0, description="Session duration in seconds")
    api_calls: int = Field(..., ge=0, description="Total number of API calls made")
    api_failure_rate: str = Field(..., description="API failure rate as percentage (e.g., '0%', '15%')")
    retry_attempts: int = Field(..., ge=0, description="Number of retry attempts made")
    location_failures: int = Field(..., ge=0, description="Number of location-related failures")
    error_count: int = Field(..., ge=0, description="Total number of errors encountered")
    recent_errors: List[ErrorDetail] = Field(default_factory=list, description="Recent error details")
    app_version: Optional[str] = Field(None, description="Client application version")
    platform: Optional[str] = Field(None, description="Platform (web, mobile, desktop)")
    user_agent: Optional[str] = Field(None, description="User agent string")
    session_id: Optional[str] = Field(None, description="Unique session identifier")


class AnalyticsResponse(BaseModel):
    """Response for analytics submission."""

    status: str = Field(..., description="Submission status")
    message: str = Field(..., description="Response message")
    analytics_id: str = Field(..., description="Unique analytics record ID")
    timestamp: str = Field(..., description="Submission timestamp")


# Configure logging based on verbosity setting
if LOG_VERBOSITY == "minimal":
    log_level = logging.WARNING  # Only warnings and errors
elif LOG_VERBOSITY == "verbose" or DEBUG:
    log_level = logging.DEBUG
else:
    log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("temphist.log") if DEBUG else logging.NullHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Reduce verbosity of noisy third-party loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Firebase App Check enforcement mode
# off     — skip verification entirely (safe default until frontend ships App Check)
# monitor — verify token when present; log failures but never block (use while rolling out)
# enforce — require a valid App Check token on every protected request
APP_CHECK_ENFORCEMENT = os.getenv("APP_CHECK_ENFORCEMENT", "off").lower().strip()
if APP_CHECK_ENFORCEMENT not in ("off", "monitor", "enforce"):
    logging.getLogger(__name__).warning(
        f"Unknown APP_CHECK_ENFORCEMENT value {APP_CHECK_ENFORCEMENT!r}; defaulting to 'off'"
    )
    APP_CHECK_ENFORCEMENT = "off"

# Usage Tracking Configuration
USAGE_TRACKING_ENABLED = os.getenv("USAGE_TRACKING_ENABLED", "true").lower() == "true"
USAGE_RETENTION_DAYS = int(os.getenv("USAGE_RETENTION_DAYS", "7"))
# Initialize rate limiting monitors
if RATE_LIMIT_ENABLED:
    location_monitor = LocationDiversityMonitor(MAX_LOCATIONS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS)
    request_monitor = RequestRateMonitor(MAX_REQUESTS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS)
else:
    location_monitor = None
    request_monitor = None

# Initialize service token rate limiter (Redis-based, always enabled for security)
# Note: redis_client is initialized later, so we'll create this in the lifespan handler
service_token_rate_limiter: Optional[ServiceTokenRateLimiter] = None
# Optional async Redis client placeholder (used when an async client is configured elsewhere)
async_redis_client = None

if RATE_LIMIT_ENABLED:
    if DEBUG:
        logger.info(
            f"🛡️  RATE LIMITING INITIALIZED: {MAX_LOCATIONS_PER_HOUR} locations/hour, {MAX_REQUESTS_PER_HOUR} requests/hour, {RATE_LIMIT_WINDOW_HOURS}h window"
        )
        if IP_WHITELIST:
            logger.debug(f"⭐ WHITELISTED IPS: {', '.join(IP_WHITELIST)}")
        if IP_BLACKLIST:
            logger.debug(f"🚫 BLACKLISTED IPS: {', '.join(IP_BLACKLIST)}")
    else:
        logger.debug(
            f"Rate limiting enabled: max {MAX_LOCATIONS_PER_HOUR} locations, max {MAX_REQUESTS_PER_HOUR} requests per {RATE_LIMIT_WINDOW_HOURS} hour(s)"
        )
        if IP_WHITELIST:
            logger.debug(f"Whitelisted IPs: {len(IP_WHITELIST)} configured")
        if IP_BLACKLIST:
            logger.debug(f"Blacklisted IPs: {len(IP_BLACKLIST)} configured")
else:
    location_monitor = None
    request_monitor = None
    if DEBUG:
        logger.info("⚠️  RATE LIMITING DISABLED")
    else:
        logger.info("Rate limiting disabled")


def clean_location_string(location: str) -> str:
    """Clean location string by removing non-printable ASCII characters."""
    import re

    # Remove any non-printable ASCII characters (keep only printable ASCII + common Unicode)
    # This removes control characters, zero-width spaces, etc.
    cleaned = "".join(char for char in location if char.isprintable() or char in [" ", ","])
    # Remove any multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def validate_location_for_ssrf(location: str) -> str:
    """Validate location string to prevent SSRF attacks.

    Args:
        location: The location string to validate

    Returns:
        Validated location string

    Raises:
        ValueError: If location is invalid or potentially dangerous
    """

    if not location or not isinstance(location, str):
        raise ValueError("Location must be a non-empty string")

    # Length validation
    if len(location) > 200:
        raise ValueError(f"Location string too long (max 200 characters, got {len(location)})")

    # Check for null bytes
    if "\x00" in location:
        raise ValueError("Location contains null bytes")

    # Check for control characters
    if any(ord(c) < 32 and c not in ["\t", "\n", "\r"] for c in location):
        raise ValueError("Location contains control characters")

    # Allow printable characters (letters, numbers, spaces, common punctuation)
    # This includes Unicode letters (accents, non-Latin scripts) which are common in location names
    # Block only control characters and specific dangerous patterns
    if not all(c.isprintable() or c in ["\t", "\n", "\r"] for c in location):
        raise ValueError("Location contains non-printable or control characters")

    # Prevent path traversal attempts
    dangerous_patterns = ["..", "/", "\\", "//"]
    for pattern in dangerous_patterns:
        if pattern in location:
            raise ValueError(f"Location contains dangerous pattern: {pattern}")

    # Prevent SSRF patterns - block URLs, IP addresses, and special schemes
    location_lower = location.lower()
    ssrf_patterns = [
        "://",  # URL scheme
        "@",  # URL auth separator
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "169.254",  # Link-local
        "10.",  # Private IP range start
        "172.16",  # Private IP range start
        "172.17",
        "172.18",
        "172.19",
        "172.20",
        "172.21",
        "172.22",
        "172.23",
        "172.24",
        "172.25",
        "172.26",
        "172.27",
        "172.28",
        "172.29",
        "172.30",
        "172.31",
        "192.168",  # Private IP range start
        "[::1]",  # IPv6 localhost
        "[fc00:",  # IPv6 private range
        "[fe80:",  # IPv6 link-local
    ]

    for pattern in ssrf_patterns:
        if pattern in location_lower:
            raise ValueError(f"Location contains potentially dangerous SSRF pattern: {pattern}")

    # Additional validation: block URL-encoded dangerous characters
    # This prevents encoding-based bypasses
    if "%" in location:
        # Check for encoded versions of dangerous patterns
        encoded_patterns = [
            "%2f",  # /
            "%5c",  # \
            "%2e",  # .
            "%40",  # @
            "%3a",  # :
        ]
        for enc_pattern in encoded_patterns:
            if enc_pattern in location_lower:
                raise ValueError(f"Location contains encoded dangerous character: {enc_pattern}")

    return location.strip()


def validate_date_format(date: str) -> str:
    """Validate date format to prevent injection attacks.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        Validated date string

    Raises:
        ValueError: If date format is invalid
    """
    import re

    if not date or not isinstance(date, str):
        raise ValueError("Date must be a non-empty string")

    # Strict date format validation
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise ValueError(f"Invalid date format. Must be YYYY-MM-DD, got: {date}")

    # Validate date values are reasonable
    try:
        year, month, day = map(int, date.split("-"))
        # Check year range (reasonable bounds)
        if year < 1800 or year > 2100:
            raise ValueError(f"Year out of range: {year}")
        if month < 1 or month > 12:
            raise ValueError(f"Month out of range: {month}")
        if day < 1 or day > 31:
            raise ValueError(f"Day out of range: {day}")

        # Try to create date to validate it's a real date
        from datetime import datetime

        datetime(year, month, day)
    except ValueError as e:
        if "out of range" in str(e):
            raise
        raise ValueError(f"Invalid date: {date}")

    return date


# Cache durations
SHORT_CACHE_DURATION = timedelta(hours=1)  # For today's data
LONG_CACHE_DURATION = timedelta(hours=168)  # 1 week for historical data
FORECAST_DAY_CACHE_DURATION = timedelta(minutes=30)  # Short cache during active hours (Midnight - 6 PM)
FORECAST_NIGHT_CACHE_DURATION = timedelta(hours=2)  # Longer cache during stable hours (6 PM - Midnight)


# Smart cache headers based on data age
def set_weather_cache_headers(response: Response, *, req_date: dt_date, key_parts: str):
    """Set appropriate cache headers based on the age of the weather data."""
    import hashlib
    from datetime import timezone

    # Strong long-term cache for any day before (UTC) today - 2 days
    today_utc = datetime.now(timezone.utc).date()
    if req_date <= today_utc - timedelta(days=2):
        # Historical data - very unlikely to change
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, s-maxage=31536000, immutable, stale-if-error=604800"
        )
    else:
        # Safer policy for the last ~48h (recent data might be revised)
        response.headers["Cache-Control"] = "public, max-age=21600, stale-while-revalidate=86400, stale-if-error=86400"

    # Deterministic weak ETag: location|date|unit_group|schema_version
    etag = hashlib.sha256(key_parts.encode("utf-8")).hexdigest()[:16]
    response.headers["ETag"] = f'W/"{etag}"'

    # Use the requested calendar day as Last-Modified (UTC midnight)
    response.headers["Last-Modified"] = f"{req_date.isoformat()}T00:00:00Z"


class AnalyticsStorage:
    """Store and manage client analytics data."""

    def __init__(self):
        self.analytics_prefix = "analytics_"
        self.retention_seconds = 7 * 24 * 3600  # 7 days retention
        self.max_errors_per_session = 50  # Limit errors per session

    def store_analytics(self, analytics_data: AnalyticsData, client_ip: str) -> str:
        """Store analytics data and return unique ID."""
        analytics_id = f"analytics_{int(time.time() * 1000)}_{hash(client_ip) % 10000}"

        # Prepare data for storage
        analytics_record = {
            "id": analytics_id,
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip,
            "session_duration": analytics_data.session_duration,
            "api_calls": analytics_data.api_calls,
            "api_failure_rate": analytics_data.api_failure_rate,
            "retry_attempts": analytics_data.retry_attempts,
            "location_failures": analytics_data.location_failures,
            "error_count": analytics_data.error_count,
            "recent_errors": [
                error.model_dump() for error in analytics_data.recent_errors[: self.max_errors_per_session]
            ],
            "app_version": analytics_data.app_version,
            "platform": analytics_data.platform,
            "user_agent": analytics_data.user_agent,
            "session_id": analytics_data.session_id,
        }

        try:
            # Store in Redis with expiration
            redis_client.setex(
                f"{self.analytics_prefix}{analytics_id}", self.retention_seconds, json.dumps(analytics_record)
            )

            # Add to analytics index for easy retrieval
            redis_client.lpush("analytics_index", analytics_id)
            redis_client.expire("analytics_index", self.retention_seconds)

            # Update analytics summary stats
            self._update_analytics_summary(analytics_record)

            if DEBUG:
                logger.debug(
                    f"📊 ANALYTICS STORED: {analytics_id} | Errors: {analytics_data.error_count} | Duration: {analytics_data.session_duration}s"
                )

            return analytics_id

        except (redis.RedisError, redis.ConnectionError) as redis_error:
            logger.error(f"Redis error storing analytics data: {redis_error}")
            raise HTTPException(status_code=503, detail="Storage service unavailable")
        except (json.JSONEncodeError, TypeError) as encode_error:
            logger.error(f"Invalid analytics data: {encode_error}")
            raise HTTPException(status_code=400, detail="Invalid analytics data format")

    def _update_analytics_summary(self, analytics_record: dict):
        """Update analytics summary statistics."""
        try:
            # Get current summary
            summary_key = "analytics_summary"
            summary = redis_client.get(summary_key)
            if summary:
                summary_data = json.loads(summary)
            else:
                summary_data = {
                    "total_sessions": 0,
                    "total_api_calls": 0,
                    "total_errors": 0,
                    "avg_session_duration": 0,
                    "avg_failure_rate": 0,
                    "platforms": {},
                    "error_types": {},
                    "last_updated": datetime.now().isoformat(),
                }

            # Update counters
            summary_data["total_sessions"] += 1
            summary_data["total_api_calls"] += analytics_record["api_calls"]
            summary_data["total_errors"] += analytics_record["error_count"]

            # Update averages
            total_sessions = summary_data["total_sessions"]
            summary_data["avg_session_duration"] = (
                summary_data["avg_session_duration"] * (total_sessions - 1) + analytics_record["session_duration"]
            ) / total_sessions

            # Update platform stats
            platform = analytics_record.get("platform", "unknown")
            summary_data["platforms"][platform] = summary_data["platforms"].get(platform, 0) + 1

            # Update error type stats
            for error in analytics_record["recent_errors"]:
                error_type = error.get("error_type", "unknown")
                summary_data["error_types"][error_type] = summary_data["error_types"].get(error_type, 0) + 1

            # Store updated summary
            redis_client.setex(summary_key, self.retention_seconds, json.dumps(summary_data))

        except (redis.RedisError, redis.ConnectionError) as redis_error:
            logger.error(f"Redis error updating analytics summary: {redis_error}")
        except (json.JSONEncodeError, json.JSONDecodeError) as json_error:
            logger.error(f"JSON error updating analytics summary: {json_error}")
        except (KeyError, ValueError) as data_error:
            logger.error(f"Data error updating analytics summary: {data_error}")

    def get_analytics_summary(self) -> dict:
        """Get analytics summary statistics."""
        try:
            summary_key = "analytics_summary"
            summary = redis_client.get(summary_key)
            if summary:
                return json.loads(summary)
            else:
                return {
                    "total_sessions": 0,
                    "total_api_calls": 0,
                    "total_errors": 0,
                    "avg_session_duration": 0,
                    "avg_failure_rate": 0,
                    "platforms": {},
                    "error_types": {},
                    "last_updated": datetime.now().isoformat(),
                }
        except (redis.RedisError, redis.ConnectionError) as redis_error:
            logger.error(f"Redis error getting analytics summary: {redis_error}")
            return {"error": "Storage service unavailable"}
        except (json.JSONDecodeError, ValueError) as decode_error:
            logger.error(f"Data error getting analytics summary: {decode_error}")
            return {"error": "Failed to parse analytics summary"}

    def get_recent_analytics(self, limit: int = 100) -> List[dict]:
        """Get recent analytics records."""
        try:
            # Get recent analytics IDs
            analytics_ids = redis_client.lrange("analytics_index", 0, limit - 1)
            analytics_records = []

            for analytics_id in analytics_ids:
                analytics_id = analytics_id.decode("utf-8")
                record = redis_client.get(f"{self.analytics_prefix}{analytics_id}")
                if record:
                    analytics_records.append(json.loads(record))

            return analytics_records

        except Exception as e:
            logger.error(f"Failed to get recent analytics: {e}")
            return []

    def get_analytics_by_session(self, session_id: str) -> List[dict]:
        """Get analytics records for a specific session."""
        try:
            # This would require a more sophisticated indexing system
            # For now, we'll search through recent analytics
            recent_analytics = self.get_recent_analytics(1000)  # Get more records
            session_analytics = [record for record in recent_analytics if record.get("session_id") == session_id]
            return session_analytics

        except (redis.RedisError, redis.ConnectionError) as redis_error:
            logger.error(f"Redis error getting analytics by session: {redis_error}")
            return []
        except (json.JSONDecodeError, KeyError, ValueError) as data_error:
            logger.error(f"Data error getting analytics by session: {data_error}")
            return []


# Lifespan event handler for startup and shutdown
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global service_token_rate_limiter

    # Startup
    # Initialize service token rate limiter (Redis-based, always enabled for security)
    try:
        service_token_rate_limiter = ServiceTokenRateLimiter(redis_client)
        if DEBUG:
            logger.info(
                f"🛡️  SERVICE TOKEN RATE LIMITING: {SERVICE_TOKEN_RATE_LIMITS['requests_per_hour']} requests/hour, {SERVICE_TOKEN_RATE_LIMITS['locations_per_hour']} locations/hour"
            )
    except (redis.RedisError, redis.ConnectionError) as redis_error:
        logger.error(f"❌ SERVICE TOKEN RATE LIMITER: Redis connection failed - {redis_error}")
        # Create None limiter if Redis fails - will fail open in middleware
        service_token_rate_limiter = None
    except ImportError as import_error:
        logger.error(f"❌ SERVICE TOKEN RATE LIMITER: Import failed - {import_error}")
        service_token_rate_limiter = None

    # Initialize router dependencies (must happen after service_token_rate_limiter is created)
    try:
        from analytics_storage import AnalyticsStorage
        from utils.location_validation import InvalidLocationCache

        # Create instances needed for dependencies
        invalid_location_cache = InvalidLocationCache(redis_client)
        analytics_storage = AnalyticsStorage(redis_client)

        # Initialize dependencies
        initialize_dependencies(
            redis_client=redis_client,
            invalid_location_cache=invalid_location_cache,
            service_token_rate_limiter=service_token_rate_limiter,
            location_monitor=location_monitor,
            request_monitor=request_monitor,
            analytics_storage=analytics_storage,
        )

        if DEBUG:
            logger.info("✅ ROUTER DEPENDENCIES: Initialized successfully")
    except (redis.RedisError, redis.ConnectionError) as redis_error:
        logger.error(f"❌ ROUTER DEPENDENCIES: Redis connection failed - {redis_error}")
    except ImportError as import_error:
        logger.error(f"❌ ROUTER DEPENDENCIES: Import failed - {import_error}")
    except (ValueError, TypeError) as config_error:
        logger.error(f"❌ ROUTER DEPENDENCIES: Configuration error - {config_error}")

    # Initialize cache system first
    try:
        initialize_cache(redis_client)
        if DEBUG:
            logger.info("✅ CACHE SYSTEM: Initialized successfully")
    except (redis.RedisError, redis.ConnectionError) as redis_error:
        logger.error(f"❌ CACHE SYSTEM: Redis connection failed - {redis_error}")
    except ImportError as import_error:
        logger.error(f"❌ CACHE SYSTEM: Import failed - {import_error}")

    # Initialize locations data (carousel / search)
    try:
        await initialize_locations_data(redis_client)
        if DEBUG:
            logger.info("✅ LOCATIONS: Data loaded and cache warmed")
    except (redis.RedisError, redis.ConnectionError) as redis_error:
        logger.error(f"❌ LOCATIONS: Redis connection failed - {redis_error}")
    except (FileNotFoundError, json.JSONDecodeError) as file_error:
        logger.error(f"❌ LOCATIONS: File error - {file_error}")
    except (IOError, PermissionError) as io_error:
        logger.error(f"❌ LOCATIONS: I/O error - {io_error}")

    if CACHE_WARMING_ENABLED and get_cache_warmer():
        # Wait a moment for the server to fully start
        await asyncio.sleep(2)

        if DEBUG:
            logger.info("🚀 STARTUP CACHE WARMING: Creating initial warming job")

        # Create initial warming job instead of running directly
        from cache.accessors import get_job_manager

        job_manager = get_job_manager()
        if job_manager:
            try:
                job_id = job_manager.create_job(
                    "cache_warming",
                    {
                        "type": "all",
                        "locations": [],
                        "triggered_by": "startup",
                        "triggered_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                logger.info(f"✅ Startup cache warming job created: {job_id}")
            except Exception as job_error:
                logger.warning(f"⚠️  Startup cache warming job skipped: {job_error}")
        else:
            logger.warning("⚠️  Job manager not available, skipping startup warming")

        # Start scheduled warming background task (job-based)
        asyncio.create_task(scheduled_cache_warming(get_cache_warmer()))
        if DEBUG:
            logger.info("⏰ SCHEDULED CACHE WARMING: Background task started")

    yield  # Application runs here

    # Shutdown
    if DEBUG:
        logger.info("🛑 APPLICATION SHUTDOWN: Cleaning up resources")

    # Clean up async Redis client (global already declared at function start)
    if async_redis_client:
        try:
            await async_redis_client.aclose()
            if DEBUG:
                logger.info("✅ Async Redis client closed successfully")
        except Exception as e:
            logger.error(f"⚠️  Error closing async Redis client: {e}")

    # Clean up HTTP client sessions
    try:
        from utils.weather_provider import close_client_session

        await close_client_session()
        if DEBUG:
            logger.info("✅ HTTP client sessions closed successfully")
    except Exception as e:
        logger.error(f"⚠️  Error closing HTTP client sessions: {e}")


app = FastAPI(lifespan=lifespan)

# Register exception handlers from exceptions module
register_exception_handlers(app)

# Include all routers
app.include_router(root_router)
app.include_router(health_router)
app.include_router(weather_router)
app.include_router(v1_records_router)
app.include_router(locations_router)
app.include_router(cache_router)
app.include_router(jobs_router)
app.include_router(legacy_router)
app.include_router(stats_router)
app.include_router(analytics_router)
app.include_router(shares_router)
app.include_router(og_image_router)

# Initialize mimetypes at module level to ensure proper configuration across workers
mimetypes.init()
# Add common image types explicitly to ensure they're recognized
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/gif", ".gif")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/svg+xml", ".svg")


# Custom StaticFiles class to ensure correct Content-Type headers for images
class ImageStaticFiles(StaticFiles):
    """Custom StaticFiles that ensures correct Content-Type headers for image files."""

    async def get_response(self, path: str, scope):
        """Override to ensure proper content-type for images."""
        response = await super().get_response(path, scope)

        # Ensure proper content-type for image files
        if isinstance(response, FileResponse):
            file_path = str(response.path)
            content_type, _ = mimetypes.guess_type(file_path)

            if content_type:
                # Force the content type in both places to ensure it takes effect
                response.headers["Content-Type"] = content_type
                response.media_type = content_type
                # Also set it in raw_headers for Starlette to ensure it's not overridden
                response.raw_headers = [(k, v) for k, v in response.raw_headers if k.lower() != b"content-type"]
                response.raw_headers.append((b"content-type", content_type.encode()))

                if DEBUG or LOG_VERBOSITY == "verbose":
                    logger.debug(f"📷 Serving {path} with Content-Type: {content_type}")

        return response


# Mount static files for location images
# Must be done AFTER including routers so router paths take precedence
data_dir = Path(__file__).resolve().parent / "data"
if data_dir.exists():
    app.mount("/data", ImageStaticFiles(directory=str(data_dir)), name="data")
    logger.info(f"📁 Mounted static files from {data_dir}")
else:
    logger.warning(f"⚠️  Data directory not found at {data_dir}")


# Initialize Redis with security validation
def create_redis_client(url: str):
    """Create Redis client with security validation."""
    import ssl
    from urllib.parse import urlparse

    parsed = urlparse(url)
    env = ENVIRONMENT  # Use ENVIRONMENT imported from config

    # Enforce password in production
    if env == "production" and not parsed.password:
        logger.error("❌ Redis password required in production")
        raise ValueError("Redis password required in production environment")

    # Enforce SSL in production
    # Note: redis.from_url handles SSL automatically for rediss:// URLs
    if parsed.scheme != "rediss" and env == "production":
        logger.warning(
            "⚠️  Redis not using SSL (rediss://) in production! Consider using rediss:// for encrypted connections."
        )
        # In production, warn but don't fail (some providers handle SSL at network level)

    # Create Redis client with SSL if needed
    # Note: from_url handles SSL automatically when using rediss:// scheme
    client = redis.from_url(url, decode_responses=True)

    # Test connection
    try:
        client.ping()
        if DEBUG:
            logger.info("✅ Redis connection validated successfully")
    except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as redis_error:
        logger.error(f"❌ Redis connection failed: {redis_error}")
        raise
    except redis.AuthenticationError as auth_error:
        logger.error(f"❌ Redis authentication failed - check password: {auth_error}")
        raise

    return client


redis_client = create_redis_client(REDIS_URL)

# Note: initialize_cache() is called in the lifespan handler (line 614)
# to ensure proper initialization order during app startup

# Background worker is now handled by a separate service (worker_service.py)
# This provides better isolation, scaling, and eliminates event loop conflicts
logger.debug("ℹ️  Background worker runs as separate service - no in-process worker needed")

# Initialize analytics storage
analytics_storage = AnalyticsStorage()
if DEBUG:
    logger.debug("📊 ANALYTICS STORAGE INITIALIZED: 7 days retention, 50 errors per session limit")

# HTTP client configuration
# LOW-004: Extract magic numbers to named constants
# HTTP Request Timeouts (imported from config)


# HTTP client for external API calls
async def get_http_client():
    """Get configured httpx client."""
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT)


# check Firebase credentials
try:
    # Try to load from environment variable first (for Railway/production)
    firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT") or os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if firebase_creds_json:
        import json

        firebase_creds = json.loads(firebase_creds_json)
        cred = credentials.Certificate(firebase_creds)
    else:
        # Fall back to file (for local development)
        if os.path.exists("firebase-service-account.json"):
            cred = credentials.Certificate("firebase-service-account.json")
        else:
            logger.warning("⚠️ No Firebase credentials found - Firebase features will be disabled")
            cred = None

    if cred:
        firebase_admin.initialize_app(cred)
        logger.info("✅ Firebase initialized successfully")
except ValueError:
    # Firebase app already initialized, skip
    logger.info("Firebase app already initialized")
except FileNotFoundError as file_error:
    logger.error(f"❌ Firebase credentials file not found: {file_error}")
    # Continue without Firebase - the app can still work without it
except json.JSONDecodeError as json_error:
    logger.error(f"❌ Invalid Firebase credentials JSON: {json_error}")
    # Continue without Firebase - the app can still work without it
except (IOError, PermissionError) as io_error:
    logger.error(f"❌ Cannot read Firebase credentials: {io_error}")
    # Continue without Firebase - the app can still work without it
except Exception as e:
    logger.error(f"❌ Unexpected error initializing Firebase: {e}")
    # Continue without Firebase - the app can still work without it


def verify_firebase_token(request: Request):
    """Verify Firebase authentication token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    id_token = auth_header.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token  # Can use user ID, etc.
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid token")


# Middleware to fix Content-Type for static image files
# This runs first in the middleware stack (last in response processing)
@app.middleware("http")
async def fix_image_content_type_middleware(request: Request, call_next):
    """Ensure correct Content-Type headers for image files served from /data."""
    response = await call_next(request)

    # Check if this is a request for an image file from /data
    if request.url.path.startswith("/data/"):
        path_lower = request.url.path.lower()

        # Map file extensions to content types
        content_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }

        # Check if the path ends with an image extension
        for ext, content_type in content_type_map.items():
            if path_lower.endswith(ext):
                # Force the correct content type
                response.headers["Content-Type"] = content_type
                if DEBUG or LOG_VERBOSITY == "verbose":
                    logger.debug(f"📷 Fixed Content-Type for {request.url.path} to {content_type}")
                break

    return response


# LOW-003: Request ID middleware for distributed tracing
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request ID for distributed tracing (LOW-003)."""
    import uuid

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


@app.middleware("http")
async def performance_timing_middleware(request: Request, call_next):
    """Add X-Response-Time header and log slow requests for performance monitoring."""
    start_time = time.perf_counter()

    # Process request
    response = await call_next(request)

    # Calculate total time in milliseconds
    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Add header with response time in milliseconds
    response.headers["X-Response-Time"] = f"{total_time_ms:.2f}ms"

    # Log slow requests (>1000ms)
    if total_time_ms > 1000:
        client_ip = get_client_ip(request)
        logger.warning(
            f"⚠️  SLOW REQUEST: {request.method} {request.url.path} took {total_time_ms:.0f}ms | IP={client_ip}"
        )

    return response


@app.middleware("http")
async def x_cache_header_middleware(request: Request, call_next):
    """Derive a simple X-Cache: HIT/MISS header from X-Cache-Status for client analytics."""
    response = await call_next(request)
    status = response.headers.get("X-Cache-Status", "")
    if status in ("HIT", "APPROX", "STALE"):
        response.headers["X-Cache"] = "HIT"
    elif status in ("MISS", "PARTIAL"):
        response.headers["X-Cache"] = "MISS"
    return response


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Log all requests when DEBUG is enabled or verbosity is verbose."""
    if DEBUG or LOG_VERBOSITY == "verbose":
        start_time = time.time()
        client_ip = get_client_ip(request)

        # Log request details (only for non-public paths to reduce noise)
        if request.url.path not in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/rate-limit-status"]:
            user_agent = request.headers.get("user-agent", "Unknown")
            logger.debug(
                f"🌐 REQUEST: {request.method} {request.url.path} | IP: {client_ip} | User-Agent: {sanitize_for_logging(user_agent, max_length=150)}"
            )

        # Process request
        response = await call_next(request)

        # Log response details (only for non-public paths to reduce noise)
        process_time = time.time() - start_time
        if request.url.path not in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/rate-limit-status"]:
            logger.debug(
                f"✅ RESPONSE: {response.status_code} | {request.method} {request.url.path} | {process_time:.3f}s | IP: {client_ip}"
            )

        return response
    else:
        # Skip logging in production
        return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses to prevent various attacks."""
    # Enforce HTTPS in production (MED-001)
    env = ENVIRONMENT  # Use ENVIRONMENT imported from config
    if env == "production" and request.url.scheme != "https":
        return JSONResponse(
            status_code=400,
            content={"error": "HTTPS required in production"},
            headers={"Location": f"https://{request.url.netloc}{request.url.path}"},
        )

    response = await call_next(request)

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Prevent MIME sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # XSS protection (legacy browsers)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Content Security Policy
    # Note: Adjust based on your frontend requirements
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' https://weather.visualcrossing.com; "
        "frame-ancestors 'none';"
    )
    response.headers["Content-Security-Policy"] = csp_policy

    # HSTS (only if using HTTPS)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Permissions policy (disable unnecessary browser features)
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    return response


@app.middleware("http")
async def request_size_middleware(request: Request, call_next):
    """Middleware to enforce request size limits and validate content types."""
    client_ip = get_client_ip(request)

    # Only apply to POST/PUT/PATCH requests with bodies
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        content_length = request.headers.get("content-length", "unknown")

        # Check content type for JSON endpoints
        if request.url.path.startswith("/analytics") and not content_type.startswith("application/json"):
            logger.warning(f"⚠️  INVALID CONTENT-TYPE: {content_type} | IP={client_ip} | Path={request.url.path}")
            return JSONResponse(
                status_code=415,
                content={
                    "error": "Unsupported Media Type",
                    "message": "Content-Type must be application/json for this endpoint",
                    "path": request.url.path,
                },
            )

        # Check content length
        max_size = 1024 * 1024  # 1MB default limit
        if content_length != "unknown":
            try:
                content_length_int = int(content_length)
                if content_length_int > max_size:
                    logger.warning(
                        f"⚠️  REQUEST TOO LARGE: {content_length_int} bytes | IP={client_ip} | Path={request.url.path}"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Payload Too Large",
                            "message": f"Request body too large. Maximum size is {max_size} bytes",
                            "path": request.url.path,
                        },
                    )
            except ValueError:
                logger.warning(
                    f"⚠️  INVALID CONTENT-LENGTH: {content_length} | IP={client_ip} | Path={request.url.path}"
                )

    response = await call_next(request)
    return response


@app.middleware("http")
async def verify_token_middleware(request: Request, call_next):
    """Middleware to verify Firebase tokens and apply rate limiting for protected routes."""
    if DEBUG:
        logger.debug(f"[DEBUG] Middleware: Processing {request.method} request to {request.url.path}")

    # Allow OPTIONS requests for CORS preflight
    if request.method == "OPTIONS":
        if DEBUG:
            logger.debug("[DEBUG] Middleware: OPTIONS request, allowing through")
        return await call_next(request)

    # Get client IP for security checks
    client_ip = get_client_ip(request)
    if DEBUG:
        logger.debug(f"[DEBUG] Middleware: Client IP: {client_ip}")

    # Check if IP is blacklisted (block entirely, except for health checks and analytics)
    if (
        is_ip_blacklisted(client_ip)
        and request.url.path not in ["/health"]
        and not request.url.path.startswith("/analytics")
    ):
        if DEBUG:
            logger.warning(f"🚫 BLACKLISTED IP BLOCKED: {client_ip} | {request.method} {request.url.path}")
        return JSONResponse(status_code=403, content={"detail": "Access denied", "reason": "IP address is blacklisted"})

    # Public paths that don't require a token or rate limiting
    # Note: Stats endpoints removed - they require authentication (HIGH-012)
    public_paths = [
        "/",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/test-cors",
        "/test-redis",
        "/rate-limit-status",
        "/analytics",
        "/health",
        "/health/detailed",
        "/v1/jobs/diagnostics/worker-status",
    ]
    # GET /v1/shares (feed) is public; POST /v1/shares still requires Firebase auth
    if request.url.path == "/v1/shares" and request.method == "GET":
        return await call_next(request)
    if request.url.path in public_paths or any(
        request.url.path.startswith(p) for p in ["/static", "/analytics", "/data", "/v1/shares/", "/v1/og/"]
    ):
        if DEBUG:
            logger.debug("[DEBUG] Middleware: Public path, allowing through")
        return await call_next(request)

    # Check if this is a service job using API_ACCESS_TOKEN (bypass rate limiting)
    auth_header = request.headers.get("Authorization")
    is_service_job = False
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
            is_service_job = True
            if DEBUG:
                logger.info(
                    f"🔧 SERVICE JOB DETECTED: {client_ip} | {request.method} {request.url.path} | Rate limiting bypassed"
                )

    # Define endpoints that actually query Visual Crossing API (cost money)
    vc_api_paths = ["/weather/", "/forecast/", "/v1/records/"]
    is_vc_api_endpoint = any(request.url.path.startswith(path) for path in vc_api_paths)

    # Apply rate limiting only to Visual Crossing API endpoints
    # Skip rate limiting for: whitelisted IPs, service jobs (API_ACCESS_TOKEN)
    if (
        RATE_LIMIT_ENABLED
        and location_monitor
        and request_monitor
        and not is_ip_whitelisted(client_ip)
        and not is_service_job
    ):
        if is_vc_api_endpoint:
            # Check request rate first
            rate_allowed, rate_reason = request_monitor.check_request_rate(client_ip)
            if not rate_allowed:
                if DEBUG:
                    logger.warning(
                        f"🚫 RATE LIMIT EXCEEDED: {client_ip} | {request.method} {request.url.path} | {rate_reason}"
                    )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "reason": rate_reason,
                        "retry_after": RATE_LIMIT_WINDOW_HOURS * 3600,
                    },
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW_HOURS * 3600)},
                )
            elif DEBUG:
                logger.debug(f"✅ RATE LIMIT CHECK: {client_ip} | {request.method} {request.url.path} | Rate: OK")

            # Check location diversity for Visual Crossing API endpoints
            if location_monitor:
                # Extract location from path
                path_parts = request.url.path.split("/")
                if len(path_parts) >= 3:
                    location = path_parts[2]  # e.g., "/weather/london/2024-01-15" -> "london"
                    location_allowed, location_reason = location_monitor.check_location_diversity(client_ip, location)
                    if not location_allowed:
                        if DEBUG:
                            logger.warning(
                                f"🌍 LOCATION DIVERSITY LIMIT: {client_ip} | {request.method} {request.url.path} | {sanitize_for_logging(location_reason)}"
                            )
                        return JSONResponse(
                            status_code=429,
                            content={
                                "detail": "Location diversity limit exceeded",
                                "reason": location_reason,
                                "retry_after": RATE_LIMIT_WINDOW_HOURS * 3600,
                            },
                            headers={"Retry-After": str(RATE_LIMIT_WINDOW_HOURS * 3600)},
                        )
                    elif DEBUG:
                        logger.debug(
                            f"✅ LOCATION DIVERSITY CHECK: {client_ip} | {request.method} {request.url.path} | Location: {sanitize_for_logging(location)} | OK"
                        )
        elif DEBUG:
            logger.debug(
                f"ℹ️  NON-VC ENDPOINT: {client_ip} | {request.method} {request.url.path} | Rate limiting skipped"
            )
    elif is_ip_whitelisted(client_ip) and DEBUG:
        logger.info(f"⭐ WHITELISTED IP: {client_ip} | {request.method} {request.url.path} | Rate limiting bypassed")
    elif is_service_job and not DEBUG:
        # Log service job bypass in non-DEBUG mode (already logged in DEBUG mode above)
        pass

    # Track usage for Visual Crossing API endpoints
    if USAGE_TRACKING_ENABLED and get_usage_tracker():
        vc_api_paths = ["/weather/", "/forecast/", "/v1/records/"]
        if any(request.url.path.startswith(path) for path in vc_api_paths):
            path_parts = request.url.path.split("/")
            if len(path_parts) >= 3:
                location = path_parts[2]
                endpoint = path_parts[1] if len(path_parts) > 1 else "unknown"
                get_usage_tracker().track_location_request(location, endpoint)

    if DEBUG:
        logger.debug("[DEBUG] Middleware: Protected path, checking Firebase token...")
    # All other paths require a Firebase token
    # Note: auth_header was already extracted earlier for service job detection
    if not auth_header or not auth_header.startswith("Bearer "):
        if DEBUG:
            logger.debug("[DEBUG] Middleware: No valid Authorization header")
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header."})

    id_token = auth_header.split(" ")[1]

    # Production token bypass for automated systems (cron jobs, etc.)
    if is_service_job:
        # Already verified this is API_ACCESS_TOKEN during rate limiting check
        if DEBUG:
            logger.debug("[DEBUG] Middleware: Using production token bypass")
        request.state.user = {"uid": "admin", "system": True, "source": "production_token"}
    else:
        logger.info("[DEBUG] Middleware: Verifying Firebase token...")
        try:
            decoded_token = auth.verify_id_token(id_token)
            provider = decoded_token.get("firebase", {}).get("sign_in_provider", "unknown")
            uid = decoded_token.get("uid", "unknown")
            logger.info(f"Auth: uid={uid} provider={provider} ip={client_ip} path={request.url.path}")
            request.state.user = decoded_token

            # Firebase App Check verification
            if APP_CHECK_ENFORCEMENT != "off":
                app_check_token = request.headers.get("X-Firebase-AppCheck")
                app_check_valid = False
                app_check_error = None
                if app_check_token:
                    try:
                        app_check.verify_token(app_check_token)
                        app_check_valid = True
                    except Exception as ace:
                        app_check_error = str(ace)
                if app_check_valid:
                    if DEBUG:
                        logger.debug(f"App Check: valid token | uid={uid}")
                elif APP_CHECK_ENFORCEMENT == "enforce":
                    reason = app_check_error or "missing X-Firebase-AppCheck header"
                    logger.warning(
                        f"App Check: BLOCKED uid={uid} ip={client_ip} path={request.url.path} reason={reason}"
                    )
                    return JSONResponse(status_code=403, content={"detail": "App Check verification failed"})
                else:  # monitor
                    reason = app_check_error or "missing X-Firebase-AppCheck header"
                    logger.warning(
                        f"App Check: MONITOR uid={uid} ip={client_ip} path={request.url.path} reason={reason}"
                    )

        except Exception as e:
            logger.error(f"[DEBUG] Middleware: Firebase token verification failed: {e}")
            logger.error(f"[DEBUG] Middleware: Error type: {type(e).__name__}")
            logger.error(f"[DEBUG] Middleware: Error message: {str(e)}")
            # Log detailed error server-side only
            logger.error(f"Firebase token verification failed: {e}", exc_info=True)

            # Return generic error message to client (don't expose internal details)
            if DEBUG:
                # In debug mode, provide more details
                return JSONResponse(status_code=403, content={"detail": f"Invalid Firebase token: {str(e)}"})
            else:
                # In production, use generic error message
                return JSONResponse(status_code=403, content={"detail": "Authentication failed"})

    logger.info("[DEBUG] Middleware: Token verified, calling next handler...")
    response = await call_next(request)
    logger.info("[DEBUG] Middleware: Response received, returning")
    return response


@app.middleware("http")
async def health_check_cors_middleware(request: Request, call_next):
    """Custom middleware to handle CORS for health check requests from Render."""
    # Check if this is a health check request
    if request.url.path == "/health":
        # Add CORS headers for health check requests
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "authorization, content-type, accept, x-requested-with"
        response.headers["Access-Control-Max-Age"] = "600"
        return response

    # For all other requests, proceed normally
    return await call_next(request)


# Configure CORS
def get_cors_origins():
    """Parse CORS origins from environment variable or use defaults."""
    if CORS_ORIGINS:
        # Split by comma and strip whitespace
        origins = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
        return origins
    else:
        # Default origins for development
        default_origins = [
            "http://localhost:3000",  # Local development
            "http://localhost:5173",  # Vite default port
            "https://temphist-develop.up.railway.app",  # development site on Railway
            "https://temphist-api-staging.up.railway.app",  # staging site on Railway
        ]
        return default_origins


def get_cors_origin_regex():
    """Parse CORS origin regex from environment variable or use default."""
    if CORS_ORIGIN_REGEX:
        return CORS_ORIGIN_REGEX
    else:
        # Default regex for temphist.com and all its subdomains
        default_regex = r"^https://(.*\.)?temphist\.com$"
        return default_regex


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_origin_regex=get_cors_origin_regex(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "authorization",
        "content-type",
        "accept",
        "x-requested-with",
        "x-firebase-appcheck",
    ],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Note: Endpoints have been moved to routers:
# - /, /test-cors, /test-cors-rolling -> routers/root.py
# - /health, /health/detailed -> routers/health.py
# - /weather/*, /forecast/* -> routers/weather.py
# - /v1/records/* -> routers/v1_records.py
# - /cache-* -> routers/cache.py
# - /v1/jobs/*, /debug/jobs -> routers/jobs.py
# - /data/*, /average/*, /trend/*, /summary/*, /protected-endpoint -> routers/legacy.py
# - /rate-limit-*, /usage-stats/* -> routers/stats.py
# - /analytics/* -> routers/analytics.py

# Note: All endpoints have been moved to router modules:
# - /weather/*, /forecast/* -> routers/weather.py
# - /v1/records/* -> routers/v1_records.py
# - /health/* -> routers/health.py
# - /cache/* -> routers/cache.py
# - /v1/jobs/* -> routers/jobs.py
# - /data/*, /average/*, /trend/*, /summary/*, /protected-endpoint -> routers/legacy.py
# - /rate-limit-*, /usage-stats/* -> routers/stats.py
# - /analytics/* -> routers/analytics.py


@app.post("/admin/clear-job-queue")
async def admin_clear_job_queue(admin_key: str = Header(None, alias="X-Admin-Key")):
    """
    Admin endpoint to clear the job queue.
    Requires X-Admin-Key header matching ADMIN_API_KEY environment variable.
    """
    expected_key = os.getenv("ADMIN_API_KEY")

    if not expected_key:
        raise HTTPException(status_code=503, detail="Admin API not configured (ADMIN_API_KEY not set)")

    if admin_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    try:
        from admin_clear_queue import clear_job_queue

        result = clear_job_queue()

        return {
            "status": result["status"],
            "message": result["message"],
            "jobs_cleared": result.get("jobs_cleared", 0),
            "remaining_jobs": result.get("remaining_jobs", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error clearing job queue: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing job queue: {str(e)}")


# For local testing
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
