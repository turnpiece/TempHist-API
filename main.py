# Standard library imports
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, date as dt_date, timezone
from typing import List, Dict, Optional, Set, Union, Literal
from collections import defaultdict
import time

# Third-party imports
import firebase_admin
import redis
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from firebase_admin import auth, credentials
from pydantic import BaseModel, Field

# Load environment variables before importing routers
load_dotenv()

# Import routers
from routers.records_agg import router as records_agg_router, daily_cache, cleanup_http_sessions
from routers.locations_preapproved import router as locations_preapproved_router, initialize_locations_data
from routers.weather import router as weather_router
from routers.v1_records import router as v1_records_router
from routers.cache import router as cache_router
from routers.jobs import router as jobs_router
from routers.legacy import router as legacy_router
from routers.health import router as health_router
from routers.stats import router as stats_router
from routers.analytics import router as analytics_router
from routers.root import router as root_router
from routers.dependencies import initialize_dependencies
from exceptions import register_exception_handlers

# Import enhanced caching utilities
from cache_utils import (
    # Cache utility functions
    get_cache_value, set_cache_value, generate_cache_key,
    # Global instances
    get_cache_stats, get_usage_tracker, get_cache_warmer,
    # Cache configuration
    CACHE_WARMING_ENABLED,
    # Cache warming
    scheduled_cache_warming,
    # Cache initialization
    initialize_cache
)
from version import __version__

# Environment variables - strip whitespace/newlines from API keys
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
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
            sensitive_params = ['key', 'api_key', 'apikey', 'token', 'password', 'secret']
            for param in sensitive_params:
                if param in query_params:
                    query_params[param] = ['[REDACTED]']
            # Reconstruct query string
            sanitized_query = urlencode(query_params, doseq=True)
            parsed = parsed._replace(query=sanitized_query)
        
        return urlunparse(parsed)
    except Exception:
        # If parsing fails, return a safe placeholder
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
    data = re.sub(r'[\r\n\t\x00-\x1f\x7f-\x9f]', ' ', data)
    
    # Remove potential sensitive patterns
    # Redact bearer tokens
    data = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', data, flags=re.IGNORECASE)
    # Redact API keys in URLs
    data = re.sub(r'key=[^&\s]+', 'key=[REDACTED]', data, flags=re.IGNORECASE)
    data = re.sub(r'api_key=[^&\s]+', 'api_key=[REDACTED]', data, flags=re.IGNORECASE)
    # Redact tokens
    data = re.sub(r'token=[^&\s]+', 'token=[REDACTED]', data, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    data = re.sub(r'\s+', ' ', data).strip()
    
    return data

_temp_logger = _logging.getLogger(__name__)
_temp_logger.info(f"🔍 DEBUG: REDIS_URL environment variable = {sanitize_url(REDIS_URL)}")

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
# LOW-008: Validate debug mode with environment check
ENVIRONMENT = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Prevent DEBUG mode in production
if ENVIRONMENT == "production" and DEBUG:
    # Use basic logging since logger not yet initialized
    _logging.basicConfig(level=_logging.INFO)
    _temp_logger = _logging.getLogger(__name__)
    _temp_logger.error("❌ DEBUG mode cannot be enabled in production environment")
    raise ValueError("DEBUG=true is forbidden in production. Set ENVIRONMENT=production and DEBUG=false")
# Logging verbosity control - set to "minimal" to reduce Railway logging limits
LOG_VERBOSITY = os.getenv("LOG_VERBOSITY", "normal").lower()  # "minimal", "normal", "verbose"
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN")  # API access token for automated systems
CACHE_CONTROL_HEADER = "public, max-age=3600, stale-while-revalidate=86400, stale-if-error=86400"
FILTER_WEATHER_DATA = os.getenv("FILTER_WEATHER_DATA", "true").lower() == "true"

# CORS configuration from environment variables
def validate_cors_config():
    """Validate CORS configuration to prevent misconfiguration."""
    origins = os.getenv("CORS_ORIGINS", "").strip()
    regex = os.getenv("CORS_ORIGIN_REGEX", "").strip()
    env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
    
    # Use _temp_logger since logger is not yet defined at module initialization time
    # This function is called at module level before logger is initialized
    _log = _logging.getLogger(__name__)
    
    # Warn about permissive configurations
    if not origins and not regex:
        _log.warning("⚠️  No CORS origins configured - API may be inaccessible to web clients")
    
    if regex:
        # Test regex is valid and not too permissive
        import re
        try:
            pattern = re.compile(regex)
            # Warn if regex looks too permissive
            permissive_patterns = [".*", ".+", r".*\.*"]
            if regex in permissive_patterns or (".*" in regex and env == "production"):
                _log.error(f"❌ CORS regex very permissive: {regex}")
                if env == "production":
                    raise ValueError("Overly permissive CORS regex not allowed in production")
                else:
                    _log.warning(f"⚠️  Permissive CORS regex in {env} environment: {regex}")
        except re.error as e:
            _log.error(f"❌ Invalid CORS_ORIGIN_REGEX: {e}")
            raise ValueError(f"Invalid CORS_ORIGIN_REGEX: {e}")
    
    if origins == "*":
        _log.error("❌ CORS_ORIGINS set to '*' - this is insecure!")
        if env == "production":
            raise ValueError("Wildcard CORS not allowed in production")
        else:
            _log.warning("⚠️  Wildcard CORS in non-production environment")
    
    return origins, regex

CORS_ORIGINS, CORS_ORIGIN_REGEX = validate_cors_config()

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
MAX_LOCATIONS_PER_HOUR = int(os.getenv("MAX_LOCATIONS_PER_HOUR", "10"))
MAX_REQUESTS_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "100"))
RATE_LIMIT_WINDOW_HOURS = int(os.getenv("RATE_LIMIT_WINDOW_HOURS", "1"))

# IP address management
IP_WHITELIST = os.getenv("IP_WHITELIST", "").split(",") if os.getenv("IP_WHITELIST") else []
IP_BLACKLIST = os.getenv("IP_BLACKLIST", "").split(",") if os.getenv("IP_BLACKLIST") else []

# Clean up empty strings from lists
IP_WHITELIST = [ip.strip() for ip in IP_WHITELIST if ip.strip()]
IP_BLACKLIST = [ip.strip() for ip in IP_BLACKLIST if ip.strip()]

# Pydantic Models for v1 API
class TemperatureValue(BaseModel):
    """Individual temperature data point."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    year: int = Field(..., description="Year")
    temperature: float = Field(..., description="Temperature value")

class DateRange(BaseModel):
    """Date range for the data."""
    start: str = Field(..., description="Start date in YYYY-MM-DD format")
    end: str = Field(..., description="End date in YYYY-MM-DD format")
    years: int = Field(..., description="Number of years in range")

class AverageData(BaseModel):
    """Average temperature statistics."""
    mean: float = Field(..., description="Mean temperature")
    unit: str = Field("celsius", description="Temperature unit (celsius or fahrenheit)")
    data_points: int = Field(..., description="Number of data points used")

class TrendData(BaseModel):
    """Temperature trend analysis."""
    slope: float = Field(..., description="Temperature change per decade")
    unit: str = Field("°C/decade", description="Trend unit (changes based on temperature unit)")
    data_points: int = Field(..., description="Number of data points used")
    r_squared: Optional[float] = Field(None, description="R-squared value for trend fit")

class UpdatedResponse(BaseModel):
    """Response model for updated timestamp endpoint."""
    period: str = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Date identifier")
    updated: Optional[str] = Field(None, description="ISO timestamp when data was last updated, null if not cached")
    cached: bool = Field(..., description="Whether the data is currently cached")
    cache_key: str = Field(..., description="Cache key used for this endpoint")

class RecordResponse(BaseModel):
    """Main record response for v1 API."""
    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Date identifier (MM-DD for daily, YYYY-MM for monthly, etc.)")
    range: DateRange = Field(..., description="Date range covered")
    unit_group: str = Field("celsius", description="Temperature unit used")
    values: List[TemperatureValue] = Field(..., description="Temperature data points")
    average: AverageData = Field(..., description="Average temperature statistics")
    trend: TrendData = Field(..., description="Temperature trend analysis")
    summary: str = Field(..., description="Human-readable summary")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    updated: Optional[str] = Field(None, description="ISO timestamp when data was last updated (if cached)")

class SubResourceResponse(BaseModel):
    """Response for subresource endpoints."""
    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Date identifier")
    data: Union[AverageData, TrendData, str] = Field(..., description="Subresource data")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('temphist.log') if DEBUG else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce verbosity of noisy third-party loggers
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Service Token Rate Limiting Configuration
# High limits for legitimate cache warming, but protection against abuse
# Configurable via environment variables with sensible defaults
SERVICE_TOKEN_REQUESTS_PER_HOUR = int(os.getenv("SERVICE_TOKEN_REQUESTS_PER_HOUR", "5000"))
SERVICE_TOKEN_LOCATIONS_PER_HOUR = int(os.getenv("SERVICE_TOKEN_LOCATIONS_PER_HOUR", "500"))
SERVICE_TOKEN_WINDOW_HOURS = int(os.getenv("SERVICE_TOKEN_WINDOW_HOURS", "1"))

# Validate limits are reasonable
if SERVICE_TOKEN_REQUESTS_PER_HOUR < 100:
    logger.warning(f"SERVICE_TOKEN_REQUESTS_PER_HOUR is very low ({SERVICE_TOKEN_REQUESTS_PER_HOUR}). Consider increasing for cache warming.")
elif SERVICE_TOKEN_REQUESTS_PER_HOUR > 100000:
    logger.warning(f"SERVICE_TOKEN_REQUESTS_PER_HOUR is very high ({SERVICE_TOKEN_REQUESTS_PER_HOUR}). This may allow excessive costs.")

if SERVICE_TOKEN_LOCATIONS_PER_HOUR < 10:
    logger.warning(f"SERVICE_TOKEN_LOCATIONS_PER_HOUR is very low ({SERVICE_TOKEN_LOCATIONS_PER_HOUR}). Consider increasing for cache warming.")
elif SERVICE_TOKEN_LOCATIONS_PER_HOUR > 10000:
    logger.warning(f"SERVICE_TOKEN_LOCATIONS_PER_HOUR is very high ({SERVICE_TOKEN_LOCATIONS_PER_HOUR}). This may allow excessive costs.")

if SERVICE_TOKEN_WINDOW_HOURS < 1:
    logger.warning(f"SERVICE_TOKEN_WINDOW_HOURS is less than 1 ({SERVICE_TOKEN_WINDOW_HOURS}). Using minimum of 1 hour.")
    SERVICE_TOKEN_WINDOW_HOURS = 1
elif SERVICE_TOKEN_WINDOW_HOURS > 24:
    logger.warning(f"SERVICE_TOKEN_WINDOW_HOURS is very high ({SERVICE_TOKEN_WINDOW_HOURS}). Consider using 1-24 hours.")

SERVICE_TOKEN_RATE_LIMITS = {
    "requests_per_hour": SERVICE_TOKEN_REQUESTS_PER_HOUR,
    "locations_per_hour": SERVICE_TOKEN_LOCATIONS_PER_HOUR,
    "window_hours": SERVICE_TOKEN_WINDOW_HOURS,
}

class ServiceTokenRateLimiter:
    """Redis-based rate limiter for service tokens to prevent abuse while allowing legitimate cache warming.
    
    Uses Redis for distributed rate limiting across multiple worker instances.
    """
    
    def __init__(self, redis_client: redis.Redis, 
                 requests_per_hour: int = SERVICE_TOKEN_RATE_LIMITS["requests_per_hour"],
                 locations_per_hour: int = SERVICE_TOKEN_RATE_LIMITS["locations_per_hour"],
                 window_hours: int = SERVICE_TOKEN_RATE_LIMITS["window_hours"]):
        self.redis = redis_client
        self.requests_per_hour = requests_per_hour
        self.locations_per_hour = locations_per_hour
        self.window_seconds = window_hours * 3600
        self.requests_key_prefix = "service_rate:requests:"
        self.locations_key_prefix = "service_rate:locations:"
    
    def check_request_rate(self, client_ip: str) -> tuple[bool, str]:
        """Check if service token request rate is within limits.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        try:
            key = f"{self.requests_key_prefix}{client_ip}"
            now = time.time()
            window_start = now - self.window_seconds
            
            # Use Redis sorted set for sliding window
            # Remove old entries
            self.redis.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window
            count = self.redis.zcard(key)
            
            if count >= self.requests_per_hour:
                return False, f"Service token rate limit exceeded: {count}/{self.requests_per_hour} requests per hour"
            
            # Add current request with current timestamp
            self.redis.zadd(key, {str(now): now})
            self.redis.expire(key, self.window_seconds)
            
            return True, "OK"
        except Exception as e:
            logger.error(f"Error checking service token request rate: {e}")
            # Fail open - allow request if Redis fails (to prevent DoS from Redis issues)
            return True, "OK"
    
    def check_location_diversity(self, client_ip: str, location: str) -> tuple[bool, str]:
        """Check if service token location diversity is within limits.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        try:
            key = f"{self.locations_key_prefix}{client_ip}"
            now = time.time()
            window_start = now - self.window_seconds
            
            # Use Redis sorted set for sliding window
            # Remove old entries
            self.redis.zremrangebyscore(key, 0, window_start)
            
            # Check if location already in window
            location_exists = self.redis.zscore(key, location) is not None
            
            if not location_exists:
                # Count unique locations in window
                count = self.redis.zcard(key)
                
                if count >= self.locations_per_hour:
                    return False, f"Service token location limit exceeded: {count}/{self.locations_per_hour} unique locations per hour"
            
            # Add/update location with current timestamp
            self.redis.zadd(key, {location: now})
            self.redis.expire(key, self.window_seconds)
            
            return True, "OK"
        except Exception as e:
            logger.error(f"Error checking service token location diversity: {e}")
            # Fail open - allow request if Redis fails
            return True, "OK"
    
    def get_stats(self, client_ip: str) -> Dict:
        """Get service token rate limiting stats."""
        try:
            requests_key = f"{self.requests_key_prefix}{client_ip}"
            locations_key = f"{self.locations_key_prefix}{client_ip}"
            now = time.time()
            window_start = now - self.window_seconds
            
            # Count requests
            self.redis.zremrangebyscore(requests_key, 0, window_start)
            request_count = self.redis.zcard(requests_key)
            
            # Count locations
            self.redis.zremrangebyscore(locations_key, 0, window_start)
            location_count = self.redis.zcard(locations_key)
            
            return {
                "requests_count": request_count,
                "requests_limit": self.requests_per_hour,
                "locations_count": location_count,
                "locations_limit": self.locations_per_hour,
                "window_hours": self.window_seconds / 3600,
                "remaining_requests": max(0, self.requests_per_hour - request_count),
                "remaining_locations": max(0, self.locations_per_hour - location_count)
            }
        except Exception as e:
            logger.error(f"Error getting service token stats: {e}")
            return {
                "error": str(e),
                "requests_count": 0,
                "requests_limit": self.requests_per_hour,
                "locations_count": 0,
                "locations_limit": self.locations_per_hour
            }

# Rate Limiting Classes
class LocationDiversityMonitor:
    """Monitor and limit location diversity per IP address to prevent API abuse."""
    
    def __init__(self, max_locations: int = 10, window_hours: int = 1):
        self.max_locations = max_locations
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600
        
        # Track unique locations per IP over time windows
        self.ip_locations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.suspicious_ips: Set[str] = set()
        self.last_cleanup = time.time()
        # LOW-002: Use background task for cleanup instead of on every request
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self._cleanup_task = None  # Background cleanup task
        
    def _cleanup_old_entries(self):
        """Clean up old entries to prevent memory bloat."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        window_start = current_time - self.window_seconds
        
        for ip_addr in list(self.ip_locations.keys()):
            for timestamp in list(self.ip_locations[ip_addr].keys()):
                if float(timestamp) < window_start:
                    del self.ip_locations[ip_addr][timestamp]
            
            # Remove IP if no timestamps remain
            if not self.ip_locations[ip_addr]:
                del self.ip_locations[ip_addr]
        
        self.last_cleanup = current_time
    
    def check_location_diversity(self, ip: str, location: str) -> tuple[bool, str]:
        """Check if IP is requesting too many different locations.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Add current request
        timestamp_str = str(current_time)
        self.ip_locations[ip][timestamp_str].add(location)
        
        # Count unique locations in time window
        unique_locations = set()
        for timestamp, locations in self.ip_locations[ip].items():
            if float(timestamp) >= window_start:
                unique_locations.update(locations)
        
        # Check if suspicious
        if len(unique_locations) > self.max_locations:
            self.suspicious_ips.add(ip)
            return False, f"Too many different locations ({len(unique_locations)} > {self.max_locations}) in {self.window_hours} hour(s)"
        
        return True, "OK"
    
    def is_suspicious(self, ip: str) -> bool:
        """Check if an IP has been flagged as suspicious."""
        return ip in self.suspicious_ips
    
    def get_stats(self, ip: str) -> Dict:
        """Get rate limiting stats for an IP address."""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        unique_locations = set()
        total_requests = 0
        
        for timestamp, locations in self.ip_locations[ip].items():
            if float(timestamp) >= window_start:
                unique_locations.update(locations)
                total_requests += len(locations)
        
        return {
            "unique_locations": len(unique_locations),
            "total_requests": total_requests,
            "max_locations": self.max_locations,
            "window_hours": self.window_hours,
            "is_suspicious": self.is_suspicious(ip)
        }

class RequestRateMonitor:
    """Monitor and limit total request rate per IP address."""
    
    def __init__(self, max_requests: int = 100, window_hours: int = 1):
        self.max_requests = max_requests
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600
        
        # Track request counts per IP over time windows
        self.ip_requests: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_cleanup = time.time()
        # LOW-002: Use background task for cleanup instead of on every request
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self._cleanup_task = None  # Background cleanup task
        
    def _cleanup_old_entries(self):
        """Clean up old entries to prevent memory bloat."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        window_start = current_time - self.window_seconds
        
        for ip_addr in list(self.ip_requests.keys()):
            for timestamp in list(self.ip_requests[ip_addr].keys()):
                if float(timestamp) < window_start:
                    del self.ip_requests[ip_addr][timestamp]
            
            # Remove IP if no timestamps remain
            if not self.ip_requests[ip_addr]:
                del self.ip_requests[ip_addr]
        
        self.last_cleanup = current_time
    
    def check_request_rate(self, ip: str) -> tuple[bool, str]:
        """Check if IP is making too many requests.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Add current request
        timestamp_str = str(int(current_time / 60) * 60)  # Round to minute for better grouping
        self.ip_requests[ip][timestamp_str] += 1
        
        # Count total requests in time window
        total_requests = sum(
            count for timestamp, count in self.ip_requests[ip].items()
            if float(timestamp) >= window_start
        )
        
        # Check if rate limit exceeded
        if total_requests > self.max_requests:
            return False, f"Too many requests ({total_requests} > {self.max_requests}) in {self.window_hours} hour(s)"
        
        return True, "OK"
    
    def get_stats(self, ip: str) -> Dict:
        """Get rate limiting stats for an IP address."""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        total_requests = sum(
            count for timestamp, count in self.ip_requests[ip].items()
            if float(timestamp) >= window_start
        )
        
        return {
            "total_requests": total_requests,
            "max_requests": self.max_requests,
            "window_hours": self.window_hours,
            "remaining_requests": max(0, self.max_requests - total_requests)
        }

# Usage Tracking Configuration
USAGE_TRACKING_ENABLED = os.getenv("USAGE_TRACKING_ENABLED", "true").lower() == "true"
USAGE_RETENTION_DAYS = int(os.getenv("USAGE_RETENTION_DAYS", "7"))
DEFAULT_POPULAR_LOCATIONS = os.getenv("CACHE_WARMING_POPULAR_LOCATIONS", "london,new_york,paris,tokyo,sydney,berlin,madrid,rome,amsterdam,dublin").split(",")
DEFAULT_POPULAR_LOCATIONS = [loc.strip().lower() for loc in DEFAULT_POPULAR_LOCATIONS if loc.strip()]


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

if RATE_LIMIT_ENABLED:
    if DEBUG:
        logger.info(f"🛡️  RATE LIMITING INITIALIZED: {MAX_LOCATIONS_PER_HOUR} locations/hour, {MAX_REQUESTS_PER_HOUR} requests/hour, {RATE_LIMIT_WINDOW_HOURS}h window")
        if IP_WHITELIST:
            logger.debug(f"⭐ WHITELISTED IPS: {', '.join(IP_WHITELIST)}")
        if IP_BLACKLIST:
            logger.debug(f"🚫 BLACKLISTED IPS: {', '.join(IP_BLACKLIST)}")
    else:
        logger.debug(f"Rate limiting enabled: max {MAX_LOCATIONS_PER_HOUR} locations, max {MAX_REQUESTS_PER_HOUR} requests per {RATE_LIMIT_WINDOW_HOURS} hour(s)")
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

# API Configuration
VISUAL_CROSSING_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
VISUAL_CROSSING_UNIT_GROUP = "metric"  # Visual Crossing API still uses "metric"/"us"
VISUAL_CROSSING_INCLUDE_PARAMS = "days"
VISUAL_CROSSING_REMOTE_DATA = "options=useremote&forecastDataset=era5core"

def clean_location_string(location: str) -> str:
    """Clean location string by removing non-printable ASCII characters."""
    import re
    # Remove any non-printable ASCII characters (keep only printable ASCII + common Unicode)
    # This removes control characters, zero-width spaces, etc.
    cleaned = ''.join(char for char in location if char.isprintable() or char in [' ', ','])
    # Remove any multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
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
    import re
    
    if not location or not isinstance(location, str):
        raise ValueError("Location must be a non-empty string")
    
    # Length validation
    if len(location) > 200:
        raise ValueError(f"Location string too long (max 200 characters, got {len(location)})")
    
    # Check for null bytes
    if '\x00' in location:
        raise ValueError("Location contains null bytes")
    
    # Check for control characters
    if any(ord(c) < 32 and c not in ['\t', '\n', '\r'] for c in location):
        raise ValueError("Location contains control characters")
    
    # Allow printable characters (letters, numbers, spaces, common punctuation)
    # This includes Unicode letters (accents, non-Latin scripts) which are common in location names
    # Block only control characters and specific dangerous patterns
    if not all(c.isprintable() or c in ['\t', '\n', '\r'] for c in location):
        raise ValueError("Location contains non-printable or control characters")
    
    # Prevent path traversal attempts
    dangerous_patterns = ['..', '/', '\\', '//']
    for pattern in dangerous_patterns:
        if pattern in location:
            raise ValueError(f"Location contains dangerous pattern: {pattern}")
    
    # Prevent SSRF patterns - block URLs, IP addresses, and special schemes
    location_lower = location.lower()
    ssrf_patterns = [
        '://',           # URL scheme
        '@',             # URL auth separator
        'localhost',
        '127.0.0.1',
        '0.0.0.0',
        '169.254',       # Link-local
        '10.',           # Private IP range start
        '172.16',        # Private IP range start
        '172.17',
        '172.18',
        '172.19',
        '172.20',
        '172.21',
        '172.22',
        '172.23',
        '172.24',
        '172.25',
        '172.26',
        '172.27',
        '172.28',
        '172.29',
        '172.30',
        '172.31',
        '192.168',       # Private IP range start
        '[::1]',         # IPv6 localhost
        '[fc00:',        # IPv6 private range
        '[fe80:',        # IPv6 link-local
    ]
    
    for pattern in ssrf_patterns:
        if pattern in location_lower:
            raise ValueError(f"Location contains potentially dangerous SSRF pattern: {pattern}")
    
    # Additional validation: block URL-encoded dangerous characters
    # This prevents encoding-based bypasses
    if '%' in location:
        # Check for encoded versions of dangerous patterns
        encoded_patterns = [
            '%2f',   # /
            '%5c',   # \
            '%2e',   # .
            '%40',   # @
            '%3a',   # :
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
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        raise ValueError(f"Invalid date format. Must be YYYY-MM-DD, got: {date}")
    
    # Validate date values are reasonable
    try:
        year, month, day = map(int, date.split('-'))
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

def build_visual_crossing_url(location: str, date: str, remote: bool = True) -> str:
    """Build Visual Crossing API URL with consistent parameters and SSRF protection.
    
    Args:
        location: The location to get weather data for (will be validated)
        date: The date in YYYY-MM-DD format (will be validated)
        remote: Whether to include remote data parameters (default: True)
        
    Returns:
        URL string for Visual Crossing API
        
    Raises:
        ValueError: If location or date validation fails
    """
    from urllib.parse import quote
    
    # Validate inputs to prevent SSRF and injection attacks
    try:
        validated_location = validate_location_for_ssrf(location)
        validated_date = validate_date_format(date)
    except ValueError as e:
        # Log the error but don't expose the exact validation failure to prevent information disclosure
        logger.error(f"Location or date validation failed: {str(e)}")
        raise ValueError("Invalid location or date format") from e
    
    # Clean and URL-encode the validated location
    cleaned_location = clean_location_string(validated_location)
    encoded_location = quote(cleaned_location, safe='')
    
    # Final validation: ensure encoded location doesn't reintroduce dangerous patterns
    encoded_lower = encoded_location.lower()
    dangerous_encoded = ['%2f', '%5c', '%40', '%3a%3a%2f', 'localhost', '127.0.0.1']
    for pattern in dangerous_encoded:
        if pattern in encoded_lower:
            logger.error(f"Encoded location contains dangerous pattern after encoding: {pattern}")
            raise ValueError("Invalid location format")
    
    base_params = f"unitGroup={VISUAL_CROSSING_UNIT_GROUP}&include={VISUAL_CROSSING_INCLUDE_PARAMS}&key={API_KEY}"
    if remote:
        return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{validated_date}?{base_params}&{VISUAL_CROSSING_REMOTE_DATA}"
    else:
        return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{validated_date}?{base_params}"

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
        response.headers["Cache-Control"] = (
            "public, max-age=21600, stale-while-revalidate=86400, stale-if-error=86400"
        )

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
            "recent_errors": [error.model_dump() for error in analytics_data.recent_errors[:self.max_errors_per_session]],
            "app_version": analytics_data.app_version,
            "platform": analytics_data.platform,
            "user_agent": analytics_data.user_agent,
            "session_id": analytics_data.session_id
        }
        
        try:
            # Store in Redis with expiration
            redis_client.setex(
                f"{self.analytics_prefix}{analytics_id}",
                self.retention_seconds,
                json.dumps(analytics_record)
            )
            
            # Add to analytics index for easy retrieval
            redis_client.lpush("analytics_index", analytics_id)
            redis_client.expire("analytics_index", self.retention_seconds)
            
            # Update analytics summary stats
            self._update_analytics_summary(analytics_record)
            
            if DEBUG:
                logger.debug(f"📊 ANALYTICS STORED: {analytics_id} | Errors: {analytics_data.error_count} | Duration: {analytics_data.session_duration}s")
            
            return analytics_id
            
        except Exception as e:
            logger.error(f"Failed to store analytics data: {e}")
            raise HTTPException(status_code=500, detail="Failed to store analytics data")
    
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
                    "last_updated": datetime.now().isoformat()
                }
            
            # Update counters
            summary_data["total_sessions"] += 1
            summary_data["total_api_calls"] += analytics_record["api_calls"]
            summary_data["total_errors"] += analytics_record["error_count"]
            
            # Update averages
            total_sessions = summary_data["total_sessions"]
            summary_data["avg_session_duration"] = (
                (summary_data["avg_session_duration"] * (total_sessions - 1) + analytics_record["session_duration"]) / total_sessions
            )
            
            # Update platform stats
            platform = analytics_record.get("platform", "unknown")
            summary_data["platforms"][platform] = summary_data["platforms"].get(platform, 0) + 1
            
            # Update error type stats
            for error in analytics_record["recent_errors"]:
                error_type = error.get("error_type", "unknown")
                summary_data["error_types"][error_type] = summary_data["error_types"].get(error_type, 0) + 1
            
            # Store updated summary
            redis_client.setex(summary_key, self.retention_seconds, json.dumps(summary_data))
            
        except Exception as e:
            logger.error(f"Failed to update analytics summary: {e}")
    
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
                    "last_updated": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {"error": "Failed to retrieve analytics summary"}
    
    def get_recent_analytics(self, limit: int = 100) -> List[dict]:
        """Get recent analytics records."""
        try:
            # Get recent analytics IDs
            analytics_ids = redis_client.lrange("analytics_index", 0, limit - 1)
            analytics_records = []
            
            for analytics_id in analytics_ids:
                analytics_id = analytics_id.decode('utf-8')
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
            session_analytics = [
                record for record in recent_analytics 
                if record.get("session_id") == session_id
            ]
            return session_analytics
            
        except Exception as e:
            logger.error(f"Failed to get analytics by session: {e}")
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
            logger.info(f"🛡️  SERVICE TOKEN RATE LIMITING: {SERVICE_TOKEN_RATE_LIMITS['requests_per_hour']} requests/hour, {SERVICE_TOKEN_RATE_LIMITS['locations_per_hour']} locations/hour")
    except Exception as e:
        logger.error(f"❌ SERVICE TOKEN RATE LIMITER: Failed to initialize - {e}")
        # Create None limiter if Redis fails - will fail open in middleware
        service_token_rate_limiter = None
    
    # Initialize router dependencies (must happen after service_token_rate_limiter is created)
    try:
        from utils.location_validation import InvalidLocationCache
        from analytics_storage import AnalyticsStorage
        
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
            analytics_storage=analytics_storage
        )
        
        if DEBUG:
            logger.info("✅ ROUTER DEPENDENCIES: Initialized successfully")
    except Exception as e:
        logger.error(f"❌ ROUTER DEPENDENCIES: Failed to initialize - {e}")
    
    # Initialize cache system first
    try:
        initialize_cache(redis_client)
        if DEBUG:
            logger.info("✅ CACHE SYSTEM: Initialized successfully")
    except Exception as e:
        logger.error(f"❌ CACHE SYSTEM: Failed to initialize - {e}")
    
    # Initialize preapproved locations data
    try:
        await initialize_locations_data(redis_client)
        if DEBUG:
            logger.info("✅ PREAPPROVED LOCATIONS: Data loaded and cache warmed")
    except Exception as e:
        logger.error(f"❌ PREAPPROVED LOCATIONS: Failed to initialize - {e}")
    
    if CACHE_WARMING_ENABLED and get_cache_warmer():
        # Wait a moment for the server to fully start
        await asyncio.sleep(2)
        
        if DEBUG:
            logger.info("🚀 STARTUP CACHE WARMING: Creating initial warming job")
        
        # Create initial warming job instead of running directly
        from cache_utils import get_job_manager
        job_manager = get_job_manager()
        if job_manager:
            job_id = job_manager.create_job("cache_warming", {
                "type": "all",
                "locations": [],
                "triggered_by": "startup",
                "triggered_at": datetime.now(timezone.utc).isoformat()
            })
            logger.info(f"✅ Startup cache warming job created: {job_id}")
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
    
    # Clean up HTTP client sessions
    try:
        await cleanup_http_sessions()
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
app.include_router(records_agg_router)  # Must come before v1_records_router
app.include_router(v1_records_router)
app.include_router(locations_preapproved_router)
app.include_router(cache_router)
app.include_router(jobs_router)
app.include_router(legacy_router)
app.include_router(stats_router)
app.include_router(analytics_router)

# Initialize Redis with security validation
def create_redis_client(url: str):
    """Create Redis client with security validation."""
    import ssl
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
    
    # Enforce password in production
    if env == "production" and not parsed.password:
        logger.error("❌ Redis password required in production")
        raise ValueError("Redis password required in production environment")
    
    # Enforce SSL in production
    ssl_context = None
    if parsed.scheme == "rediss":
        ssl_context = ssl.create_default_context()
    elif env == "production":
        logger.warning("⚠️  Redis not using SSL (rediss://) in production! Consider using rediss:// for encrypted connections.")
        # In production, warn but don't fail (some providers handle SSL at network level)
    
    # Create Redis client with SSL if needed
    # Note: from_url handles SSL automatically when using rediss:// scheme
    client = redis.from_url(
        url,
        decode_responses=True
    )
    
    # Test connection
    try:
        client.ping()
        if DEBUG:
            logger.info("✅ Redis connection validated successfully")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise
    
    return client

redis_client = create_redis_client(REDIS_URL)

# Note: initialize_cache() is called in the lifespan handler (line 614)
# to ensure proper initialization order during app startup

# Background worker is now handled by a separate service (worker_service.py)
# This provides better isolation, scaling, and eliminates event loop conflicts
logger.debug("ℹ️  Background worker runs as separate service - no in-process worker needed")

# Wire up Redis cache for rolling bundle
daily_cache.redis = redis_client


# Initialize analytics storage
analytics_storage = AnalyticsStorage()
if DEBUG:
    logger.debug("📊 ANALYTICS STORAGE INITIALIZED: 7 days retention, 50 errors per session limit")

# HTTP client configuration
# LOW-004: Extract magic numbers to named constants
# HTTP Request Timeouts
HTTP_TIMEOUT_DEFAULT = 60.0  # Default HTTP timeout in seconds
HTTP_TIMEOUT_SHORT = 5.0     # Short timeout for health checks
HTTP_TIMEOUT_LONG = 120.0    # Long timeout for large data requests

HTTP_TIMEOUT = HTTP_TIMEOUT_DEFAULT  # Alias for backward compatibility
MAX_CONCURRENT_REQUESTS = 2  # Reduced for cold start protection - prevents stampeding Visual Crossing API

# Simple global semaphore - no longer need complex event loop handling with separate services
visual_crossing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

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
    pass
except Exception as e:
    logger.error(f"❌ Error initializing Firebase: {e}")
    # Continue without Firebase - the app can still work without it

def get_client_ip(request: Request) -> str:
    """Get the client IP address from the request."""
    # Check for forwarded headers first (for proxy/load balancer scenarios)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"

def is_ip_whitelisted(ip: str) -> bool:
    """Check if an IP address is whitelisted (exempt from rate limiting)."""
    return ip in IP_WHITELIST

def is_ip_blacklisted(ip: str) -> bool:
    """Check if an IP address is blacklisted (blocked entirely)."""
    return ip in IP_BLACKLIST

def verify_firebase_token(request: Request):
    """Verify Firebase authentication token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    id_token = auth_header.split(" ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token  # Can use user ID, etc.
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid token")

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
async def log_requests_middleware(request: Request, call_next):
    """Log all requests when DEBUG is enabled or verbosity is verbose."""
    if DEBUG or LOG_VERBOSITY == "verbose":
        start_time = time.time()
        client_ip = get_client_ip(request)
        
        # Log request details (only for non-public paths to reduce noise)
        if not request.url.path in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/rate-limit-status"]:
            user_agent = request.headers.get('user-agent', 'Unknown')
            logger.debug(f"🌐 REQUEST: {request.method} {request.url.path} | IP: {client_ip} | User-Agent: {sanitize_for_logging(user_agent, max_length=150)}")
        
        # Process request
        response = await call_next(request)
        
        # Log response details (only for non-public paths to reduce noise)
        process_time = time.time() - start_time
        if not request.url.path in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/rate-limit-status"]:
            logger.debug(f"✅ RESPONSE: {response.status_code} | {request.method} {request.url.path} | {process_time:.3f}s | IP: {client_ip}")
        
        return response
    else:
        # Skip logging in production
        return await call_next(request)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses to prevent various attacks."""
    # Enforce HTTPS in production (MED-001)
    env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
    if env == "production" and request.url.scheme != "https":
        return JSONResponse(
            status_code=400,
            content={"error": "HTTPS required in production"},
            headers={"Location": f"https://{request.url.netloc}{request.url.path}"}
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
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' https://weather.visualcrossing.com; "
        "frame-ancestors 'none';"
    )
    response.headers["Content-Security-Policy"] = csp_policy
    
    # HSTS (only if using HTTPS)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Permissions policy (disable unnecessary browser features)
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=()"
    )
    
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
                    "path": request.url.path
                }
            )
        
        # Check content length
        max_size = 1024 * 1024  # 1MB default limit
        if content_length != "unknown":
            try:
                content_length_int = int(content_length)
                if content_length_int > max_size:
                    logger.warning(f"⚠️  REQUEST TOO LARGE: {content_length_int} bytes | IP={client_ip} | Path={request.url.path}")
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Payload Too Large",
                            "message": f"Request body too large. Maximum size is {max_size} bytes",
                            "path": request.url.path
                        }
                    )
            except ValueError:
                logger.warning(f"⚠️  INVALID CONTENT-LENGTH: {content_length} | IP={client_ip} | Path={request.url.path}")
    
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
            logger.debug(f"[DEBUG] Middleware: OPTIONS request, allowing through")
        return await call_next(request)

    # Get client IP for security checks
    client_ip = get_client_ip(request)
    if DEBUG:
        logger.debug(f"[DEBUG] Middleware: Client IP: {client_ip}")

    # Check if IP is blacklisted (block entirely, except for health checks and analytics)
    if is_ip_blacklisted(client_ip) and request.url.path not in ["/health"] and not request.url.path.startswith("/analytics"):
        if DEBUG:
            logger.warning(f"🚫 BLACKLISTED IP BLOCKED: {client_ip} | {request.method} {request.url.path}")
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Access denied",
                "reason": "IP address is blacklisted"
            }
        )

    # Public paths that don't require a token or rate limiting
    # Note: Stats endpoints removed - they require authentication (HIGH-012)
    public_paths = ["/", "/docs", "/openapi.json", "/redoc", "/test-cors", "/test-redis", "/rate-limit-status", "/analytics", "/health", "/health/detailed", "/v1/jobs/diagnostics/worker-status"]
    if request.url.path in public_paths or any(request.url.path.startswith(p) for p in ["/static", "/analytics"]):
        if DEBUG:
            logger.debug(f"[DEBUG] Middleware: Public path, allowing through")
        return await call_next(request)

    # Check if this is a service job using API_ACCESS_TOKEN (bypass rate limiting)
    auth_header = request.headers.get("Authorization")
    is_service_job = False
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
            is_service_job = True
            if DEBUG:
                logger.info(f"🔧 SERVICE JOB DETECTED: {client_ip} | {request.method} {request.url.path} | Rate limiting bypassed")

    # Define endpoints that actually query Visual Crossing API (cost money)
    vc_api_paths = ["/weather/", "/forecast/", "/v1/records/"]
    is_vc_api_endpoint = any(request.url.path.startswith(path) for path in vc_api_paths)
    
    # Apply rate limiting only to Visual Crossing API endpoints
    # Skip rate limiting for: whitelisted IPs, service jobs (API_ACCESS_TOKEN)
    if RATE_LIMIT_ENABLED and location_monitor and request_monitor and not is_ip_whitelisted(client_ip) and not is_service_job:
        if is_vc_api_endpoint:
            # Check request rate first
            rate_allowed, rate_reason = request_monitor.check_request_rate(client_ip)
            if not rate_allowed:
                if DEBUG:
                    logger.warning(f"🚫 RATE LIMIT EXCEEDED: {client_ip} | {request.method} {request.url.path} | {rate_reason}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "reason": rate_reason,
                        "retry_after": RATE_LIMIT_WINDOW_HOURS * 3600
                    },
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW_HOURS * 3600)}
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
                            logger.warning(f"🌍 LOCATION DIVERSITY LIMIT: {client_ip} | {request.method} {request.url.path} | {sanitize_for_logging(location_reason)}")
                        return JSONResponse(
                            status_code=429,
                            content={
                                "detail": "Location diversity limit exceeded",
                                "reason": location_reason,
                                "retry_after": RATE_LIMIT_WINDOW_HOURS * 3600
                            },
                            headers={"Retry-After": str(RATE_LIMIT_WINDOW_HOURS * 3600)}
                        )
                    elif DEBUG:
                        logger.debug(f"✅ LOCATION DIVERSITY CHECK: {client_ip} | {request.method} {request.url.path} | Location: {sanitize_for_logging(location)} | OK")
        elif DEBUG:
            logger.debug(f"ℹ️  NON-VC ENDPOINT: {client_ip} | {request.method} {request.url.path} | Rate limiting skipped")
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
        logger.debug(f"[DEBUG] Middleware: Protected path, checking Firebase token...")
    # All other paths require a Firebase token
    # Note: auth_header was already extracted earlier for service job detection
    if not auth_header or not auth_header.startswith("Bearer "):
        if DEBUG:
            logger.debug(f"[DEBUG] Middleware: No valid Authorization header")
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header."}
        )

    id_token = auth_header.split(" ")[1]
    
    # Production token bypass for automated systems (cron jobs, etc.)
    if is_service_job:
        # Already verified this is API_ACCESS_TOKEN during rate limiting check
        if DEBUG:
            logger.debug(f"[DEBUG] Middleware: Using production token bypass")
        request.state.user = {"uid": "admin", "system": True, "source": "production_token"}
    else:
        logger.info(f"[DEBUG] Middleware: Verifying Firebase token...")
        try:
            decoded_token = auth.verify_id_token(id_token)
            logger.info(f"[DEBUG] Middleware: Firebase token verified successfully")
            # Optionally, attach user info to request.state
            request.state.user = decoded_token
        except Exception as e:
            logger.error(f"[DEBUG] Middleware: Firebase token verification failed: {e}")
            logger.error(f"[DEBUG] Middleware: Error type: {type(e).__name__}")
            logger.error(f"[DEBUG] Middleware: Error message: {str(e)}")
            # Log detailed error server-side only
            logger.error(f"Firebase token verification failed: {e}", exc_info=True)
            
            # Return generic error message to client (don't expose internal details)
            if DEBUG:
                # In debug mode, provide more details
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"Invalid Firebase token: {str(e)}"}
                )
            else:
                # In production, use generic error message
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Authentication failed"}
                )

    logger.info(f"[DEBUG] Middleware: Token verified, calling next handler...")
    response = await call_next(request)
    logger.info(f"[DEBUG] Middleware: Response received, returning")
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
            "https://temphist-api-staging.up.railway.app"  # staging site on Railway
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

def calculate_historical_average(data: List[Dict[str, float]]) -> float:
    """
    Calculate the average temperature using only historical data (excluding current year).
    Returns the average temperature rounded to 1 decimal place.
    """
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not data or len(data) < 2:
        return 0.0
    # Filter out None values
    historical_data = [p for p in data[:-1] if p.get('y') is not None]
    if not historical_data:
        return 0.0
    avg_temp = sum(p['y'] for p in historical_data) / len(historical_data)
    return round(avg_temp, 1)

def get_friendly_date(date: datetime) -> str:
    """Get a friendly date string with ordinal suffix."""
    day = date.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {date.strftime('%B')}"

def generate_summary(data: List[Dict[str, float]], date: datetime, period: str = "daily") -> str:
    """Generate a summary text for temperature data.
    
    Note: Time-sensitive summaries (e.g., "the past week/month/year") should have
    very short cache durations (minutes) as they become invalid quickly.
    """
    # Filter out data points with None temperature
    data = [d for d in data if d.get('y') is not None]
    if not data or len(data) < 2:
        return "Not enough data to generate summary."

    # Check if we have data for the expected year (from the date parameter)
    expected_year = date.year
    latest = data[-1]
    
    # Verify the latest data point is actually for the expected year
    if latest.get('x') != expected_year:
        # Current year data is missing - don't generate a misleading summary
        return f"Temperature data for {date.year} is not yet available."
    
    if latest.get('y') is None:
        return "No valid temperature data for the latest year."

    avg_temp = calculate_historical_average(data)
    diff = latest['y'] - avg_temp
    rounded_diff = round(diff, 1)

    friendly_date = get_friendly_date(date)
    warm_summary = ''
    cold_summary = ''
    temperature = f"{latest['y']}°C."

    # Determine tense based on date and period
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    target_date = date.date()
    
    
    # For non-daily periods, determine if the period includes today or ended recently
    if period == "daily":
        if target_date == today:
            # Today - use present tense
            tense_context = "is"
            tense_context_alt = "is"
            tense_warm_cold = "is"
        elif target_date == yesterday:
            # Yesterday - use past tense but keep the actual date
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"
            # Don't override friendly_date - keep the actual date like "October 25th"
        else:
            # Past date - use past tense
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"
    else:
        # For weekly, monthly, yearly periods
        if period == "weekly":
            # Check if the week ending on target_date includes today
            week_start = target_date - timedelta(days=6)
            period_includes_today = week_start <= today <= target_date
            period_ended_recently = target_date == yesterday or target_date == today - timedelta(days=2)
        elif period == "monthly":
            # Check if the month containing target_date includes today
            month_start = target_date.replace(day=1)
            next_month = (month_start + timedelta(days=32)).replace(day=1)
            month_end = next_month - timedelta(days=1)
            period_includes_today = month_start <= today <= month_end
            # Check if the month ended recently (same month as yesterday or day before yesterday)
            period_ended_recently = (target_date.month == yesterday.month and target_date.year == yesterday.year) or \
                                  (target_date.month == (yesterday - timedelta(days=1)).month and target_date.year == (yesterday - timedelta(days=1)).year)
        elif period == "yearly":
            # Check if the target year is the current year
            period_includes_today = target_date.year == today.year
            period_ended_recently = target_date.year == yesterday.year
        else:
            # Default for other periods
            period_includes_today = target_date == today
            period_ended_recently = target_date == yesterday
        
        if period_includes_today:
            # Period includes today - use present perfect for consistency
            tense_context = "has been"
            tense_context_alt = "has been"
            tense_warm_cold = "has been"
        elif period_ended_recently:
            # Period ended recently - use past perfect for consistency
            tense_context = "had been"
            tense_context_alt = "had been"
            tense_warm_cold = "had been"
        else:
            # Period is in the past - use past tense
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"
        

    previous = [p for p in data[:-1] if p.get('y') is not None]
    is_warmest = all(latest['y'] >= p['y'] for p in previous)
    is_coldest = all(latest['y'] <= p['y'] for p in previous)

    # Check against last year first for consistency
    last_year_temp = next((p['y'] for p in reversed(previous) if p['x'] == latest['x'] - 1), None)
    
    # Generate mutually exclusive summaries to avoid contradictions
    if is_warmest:
        warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} on record."
    elif is_coldest:
        cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} on record."
    elif last_year_temp is not None:
        # Compare against last year first
        if latest['y'] > last_year_temp:
            # Warmer than last year - find last warmer year
            last_warmer = next((p['x'] for p in reversed(previous) if p['y'] > latest['y']), None)
            if last_warmer:
                years_since = int(latest['x'] - last_warmer)
                if years_since == 2:
                    warm_summary = f"It's warmer than last year but not as warm as {last_warmer}."
                elif years_since <= 10:
                    warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} since {last_warmer}."
                else:
                    warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} in {years_since} years."
            else:
                warm_summary = f"It's warmer than last year."
        elif latest['y'] < last_year_temp:
            # Colder than last year - find last colder year
            last_colder = next((p['x'] for p in reversed(previous) if p['y'] < latest['y']), None)
            if last_colder:
                years_since = int(latest['x'] - last_colder)
                if years_since == 2:
                    cold_summary = f"It's colder than last year but not as cold as {last_colder}."
                elif years_since <= 10:
                    cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} since {last_colder}."
                else:
                    cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} in {years_since} years."
            else:
                cold_summary = f"It's colder than last year."
        # If equal to last year, no warm/cold summary is generated

    # Generate period-appropriate language with correct tense and context
    if period == "daily":
        if target_date == today:
            period_context = "today"
            period_context_alt = "this date"
        elif target_date == yesterday:
            period_context = "yesterday"
            period_context_alt = "yesterday"
        else:
            period_context = "that day"
            period_context_alt = "that date"
    elif period == "weekly":
        if tense_context == "has been":
            period_context = "this week"
            period_context_alt = "this week"
        elif tense_context == "had been":
            period_context = "the past week"
            period_context_alt = "the past week"
        else:
            # For distant past weeks, use more specific language
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the week ending {day}{suffix} {target_date.strftime('%B')}"
            period_context_alt = period_context
    elif period == "monthly":
        if tense_context == "has been":
            period_context = "this month"
            period_context_alt = "this month"
        elif tense_context == "had been":
            period_context = "the past month"
            period_context_alt = "the past month"
        else:
            # For distant past months, use more specific language
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the month ending {day}{suffix} {target_date.strftime('%B')}"
            period_context_alt = period_context
    elif period == "yearly":
        if tense_context == "has been":
            period_context = "this year"
            period_context_alt = "this year"
        elif tense_context == "had been":
            period_context = "the past year"
            period_context_alt = "the past year"
        else:
            # For distant past years, use more specific language
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the year ending {day}{suffix} {target_date.strftime('%B %Y')}"
            period_context_alt = period_context
    else:
        if tense_context == "has been":
            period_context = "this period"
            period_context_alt = "this period"
        elif tense_context == "had been":
            period_context = "the past period"
            period_context_alt = "the past period"
        else:
            period_context = "that period"
            period_context_alt = "that period"

    if abs(diff) < 0.05:
        # Special case for yearly summaries to sound more natural
        if period == "yearly":
            avg_summary = f"It {tense_context_alt} an average year."
        else:
            avg_summary = f"It {tense_context_alt} about average for {period_context_alt}."
    elif diff > 0:
        # For weekly/monthly/yearly periods, use "It was" to avoid repetition with "has been"
        if period in ["weekly", "monthly", "yearly"]:
            avg_summary = "However, " if cold_summary else ""
            avg_summary += f"{'it' if cold_summary else 'It'} was {rounded_diff}°C warmer than average."
        else:
            # For daily periods, use the period context
            avg_summary = "However, " if cold_summary else ""
            # Don't capitalise the period context when it follows "However, "
            if cold_summary:  # Use the same condition as above
                # Force lowercase for period context when following "However, "
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {rounded_diff}°C warmer than average."
            else:
                period_capitalised = period_context.capitalize()
                avg_summary += f"{period_capitalised} {tense_context} {rounded_diff}°C warmer than average."
    else:
        # For weekly/monthly/yearly periods, use "It was" to avoid repetition with "has been"
        if period in ["weekly", "monthly", "yearly"]:
            avg_summary = "However, " if warm_summary else ""
            avg_summary += f"{'it' if warm_summary else 'It'} was {abs(rounded_diff)}°C cooler than average."
        else:
            # For daily periods, use the period context
            avg_summary = "However, " if warm_summary else ""
            # Don't capitalise the period context when it follows "However, "
            if warm_summary:  # Use the same condition as above
                # Force lowercase for period context when following "However, "
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {abs(rounded_diff)}°C cooler than average."
            else:
                period_capitalised = period_context.capitalize()
                avg_summary += f"{period_capitalised} {tense_context} {abs(rounded_diff)}°C cooler than average."

    return " ".join(filter(None, [temperature, warm_summary, cold_summary, avg_summary]))

async def get_summary(location: str, month_day: str, weather_data: Optional[List[Dict]] = None) -> str:

    try:
        if weather_data is None:
            month, day = map(int, month_day.split("-"))
            weather_data = await get_temperature_series(location, month, day)

        # If weather_data is a dict, extract the list
        if isinstance(weather_data, dict) and "data" in weather_data:
            weather_data = weather_data["data"]
        # Filter out None values
        weather_data = [d for d in weather_data if d.get('y') is not None]

        if not weather_data or not isinstance(weather_data, list) or len(weather_data) == 0:
            return "No weather data available."

        # Use the year from the latest data point or fallback to current year
        latest_year = max(d.get("x") for d in weather_data if d.get("x"))
        current_year = datetime.now().year
        if latest_year != current_year:
            return "Failed to get today's temperature data."

        date = datetime.strptime(f"{latest_year}-{month_day}", "%Y-%m-%d")
        return generate_summary(weather_data, date, "daily")

    except Exception as e:
        logger.error(f"Error in get_summary: {e}")
        return "Error generating summary."

def calculate_trend_slope(data: List[Dict[str, float]]) -> float:
    """Calculate temperature trend slope using linear regression.
    
    This function handles missing years correctly by using the actual years
    in the linear regression calculation. The slope represents the rate of
    temperature change per year, which is then converted to per decade.
    
    Args:
        data: List of dictionaries with 'x' (year) and 'y' (temperature) keys
        
    Returns:
        Slope in °C/decade, rounded to 2 decimal places
    """
    # Filter out None values
    data = [d for d in data if d.get('y') is not None]
    n = len(data)
    if n < 2:
        return 0.0

    # Sort by year to ensure proper ordering
    data = sorted(data, key=lambda d: d['x'])
    
    # Use actual years for the calculation - this is mathematically correct
    # Linear regression works with any x-values, not just consecutive integers
    sum_x = sum(p['x'] for p in data)
    sum_y = sum(p['y'] for p in data)
    sum_xy = sum(p['x'] * p['y'] for p in data)
    sum_xx = sum(p['x'] ** 2 for p in data)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x ** 2
    
    if denominator == 0:
        return 0.0
    
    # Calculate slope in °C per year
    slope_per_year = numerator / denominator
    
    # Convert to °C per decade
    slope_per_decade = slope_per_year * 10.0
    
    return round(slope_per_decade, 2)

# Removed duplicate endpoints - now in routers:
# - /forecast/{location} -> routers/weather.py
# - /health, /health/detailed -> routers/health.py


# /test-redis endpoint moved to routers/root.py

# Removed duplicate endpoints - now in routers/stats.py:
# - /rate-limit-status -> routers/stats.py
# - /rate-limit-stats -> routers/stats.py
# - /usage-stats -> routers/stats.py
# - /usage-stats/{location} -> routers/stats.py

# Removed duplicate cache endpoints - now in routers/cache.py
# All /cache-* endpoints moved to routers/cache.py

# Helper to check if a given year, month, day is today
def is_today(year: int, month: int, day: int) -> bool:
    """Check if the given date is today."""
    today = dt_date.today()
    return year == today.year and month == today.month and day == today.day

def is_today_or_future(year: int, month: int, day: int) -> bool:
    """Check if the given date is today or in the future."""
    today = dt_date.today()
    date = dt_date(year, month, day)
    return date >= today

def get_forecast_cache_duration() -> timedelta:
    """Get appropriate cache duration for forecast data based on time of day.
    
    Returns:
        timedelta: Short duration during active forecast hours (30 min), longer when stable (2 hours)
    """
    current_hour = datetime.now().hour
    
    # Stable hours (6 PM to Midnight) - forecast is more stable
    if current_hour >= 18:
        return FORECAST_NIGHT_CACHE_DURATION
    else:
        # Active hours (Midnight to 6 PM) - forecast can change frequently
        return FORECAST_DAY_CACHE_DURATION


    
async def get_trend(location: str, month_day: str, weather_data: Optional[List[Dict]] = None) -> dict:
    """Calculate temperature trend for a location and date."""
    if weather_data is None:
        month, day = map(int, month_day.split("-"))
        weather_data = await get_temperature_series(location, month, day)

    if isinstance(weather_data, dict) and "data" in weather_data:
        weather_data = weather_data["data"]
    trend_input = [{"x": d["x"], "y": d["y"]} for d in weather_data if d.get("y") is not None]
    slope = calculate_trend_slope(trend_input)
    return {
        "slope": slope,
        "units": "°C/decade"
    }

def validate_month_day(month_day: str):
    """Validate month-day format and return month, day integers."""
    try:
        month, day = map(int, month_day.split("-"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid month-day format. Use MM-DD.")
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    if not (1 <= day <= 31):
        raise HTTPException(status_code=400, detail="Day must be between 1 and 31")
    if month in [4, 6, 9, 11] and day > 30:
        raise HTTPException(status_code=400, detail=f"Month {month} has only 30 days")
    if month == 2 and day > 29:
        raise HTTPException(status_code=400, detail="February has only 29 days")
    return month, day


async def is_valid_location(location: str) -> bool:
    """Check if a location is valid by testing the API (async version)."""
    import httpx
    today = datetime.now().strftime("%Y-%m-%d")
    url = build_visual_crossing_url(location, today, remote=False)
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url)
            if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
                return True
    except Exception:
        # If request fails, assume location is invalid
        pass
    return False

class InvalidLocationCache:
    """Cache for invalid locations to avoid repeated API calls."""
    
    def __init__(self, redis_client, ttl_hours: int = 24):
        self.redis_client = redis_client
        self.ttl_seconds = ttl_hours * 3600
        self.invalid_key_prefix = "invalid_location:"
    
    def is_invalid_location(self, location: str) -> bool:
        """Check if a location is known to be invalid."""
        if not self.redis_client:
            return False
        try:
            key = f"{self.invalid_key_prefix}{location.lower()}"
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking invalid location cache: {e}")
            return False
    
    def mark_location_invalid(self, location: str, reason: str = "no_data"):
        """Mark a location as invalid with a reason."""
        if not self.redis_client:
            return
        try:
            key = f"{self.invalid_key_prefix}{location.lower()}"
            data = {
                "location": location,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.redis_client.setex(key, self.ttl_seconds, json.dumps(data))
            logger.info(f"Marked location as invalid: {location} (reason: {reason})")
        except Exception as e:
            logger.error(f"Error marking location as invalid: {e}")
    
    def get_invalid_location_info(self, location: str) -> Optional[dict]:
        """Get information about why a location was marked as invalid."""
        if not self.redis_client:
            return None
        try:
            key = f"{self.invalid_key_prefix}{location.lower()}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting invalid location info: {e}")
        return None

# Initialize invalid location cache
invalid_location_cache = InvalidLocationCache(redis_client)

def validate_location_response(data: dict, location: str) -> tuple[bool, str]:
    """
    Validate that the response contains meaningful data.
    Returns (is_valid, error_message).
    """
    # Check if values array is empty
    if not data.get("values") or len(data["values"]) == 0:
        return False, f"No temperature data found for location '{location}'. The location may be invalid or not supported by the weather service."
    
    # Check if all temperature values are None or 0
    values = data.get("values", [])
    valid_temps = [v for v in values if v.get("temperature") is not None and v.get("temperature") != 0]
    if not valid_temps:
        return False, f"No valid temperature data found for location '{location}'. The location may be invalid or not supported by the weather service."
    
    # Check if data points count is 0
    if data.get("average", {}).get("data_points", 0) == 0:
        return False, f"No data points available for location '{location}'. The location may be invalid or not supported by the weather service."
    
    return True, ""

def is_location_likely_invalid(location: str) -> bool:
    """
    Check if a location string looks obviously invalid.
    This is a quick check before making API calls.
    Enhanced with security checks for SSRF prevention.
    """
    if not location or not isinstance(location, str):
        return True
    
    # Check for common invalid patterns
    invalid_patterns = [
        "[object Object]",
        "undefined",
        "null",
        "NaN",
        "object Object",
        "Object",
        "[object",
        "object]"
    ]
    
    location_lower = location.lower().strip()
    if any(pattern.lower() in location_lower for pattern in invalid_patterns):
        return True
    
    # Security checks for SSRF prevention
    # Length check
    if len(location) > 200:
        return True
    
    # Check for dangerous patterns that could indicate SSRF attempts
    dangerous_patterns = [
        '://',      # URL scheme
        '@',        # URL auth
        'localhost',
        '127.0.0.1',
        '0.0.0.0',
        '..',       # Path traversal
        '/',        # Path separator
        '\\',       # Windows path separator
        '//',       # Protocol-relative
    ]
    
    for pattern in dangerous_patterns:
        if pattern in location_lower:
            return True
    
    # Check for private IP ranges
    private_ip_patterns = ['10.', '172.16', '172.17', '172.18', '172.19', '172.20',
                          '172.21', '172.22', '172.23', '172.24', '172.25', '172.26',
                          '172.27', '172.28', '172.29', '172.30', '172.31', '192.168']
    
    for pattern in private_ip_patterns:
        if pattern in location_lower:
            return True
    
    return False

def get_year_range(current_year: int, years_back: int = 50) -> List[int]:
    """Get a list of years for historical data analysis."""
    return list(range(current_year - years_back, current_year + 1))

def create_metadata(total_years: int, available_years: int, missing_years: List[Dict], 
                   additional_metadata: Dict = None) -> Dict:
    """Create standardized metadata for temperature data responses."""
    metadata = {
        "total_years": total_years,
        "available_years": available_years,
        "missing_years": missing_years,
        "completeness": round(available_years / total_years * 100, 1) if total_years > 0 else 0.0
    }
    if additional_metadata:
        metadata.update(additional_metadata)
    return metadata

def track_missing_year(missing_years: List[Dict], year: int, reason: str):
    """Add a missing year entry to the missing_years list."""
    missing_years.append({"year": year, "reason": reason})

async def get_temperature_series(location: str, month: int, day: int) -> Dict:
    """Get temperature series data for a location and date over multiple years."""
    # Check for cached series first
    series_cache_key = generate_cache_key("series", location, f"{month:02d}_{day:02d}")
    if CACHE_ENABLED:
        cached_series = get_cache_value(series_cache_key, redis_client, "series", location, get_cache_stats())
        if cached_series:
            logger.debug(f"Cache hit: {series_cache_key}")
            try:
                return json.loads(cached_series)
            except Exception as e:
                logger.error(f"Error decoding cached series for {series_cache_key}: {e}")
    # Note: Removed is_valid_location() check - it makes unnecessary API calls.
    # The Visual Crossing API will naturally return an error if the location is invalid.
    logger.debug(f"get_temperature_series for {location} on {day}/{month}")
    today = datetime.now()
    current_year = today.year
    years = get_year_range(current_year)

    data = []
    missing_years = []
    uncached_years = []
    uncached_date_strs = []
    year_to_date_str = {}
    for year in years:
        logger.debug(f"get_temperature_series year: {year}")
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = generate_cache_key("weather", location, date_str)
        year_to_date_str[year] = date_str
        # If this is today's date, use the forecast data
        if year == current_year and month == today.month and day == today.day:
            logger.debug("get_temperature_series forecast for today")
            try:
                forecast_data = await get_forecast_data(location, datetime(year, month, day).date())
                logger.debug(f"Got forecast data {forecast_data}")
                if "error" in forecast_data:
                    logger.warning(f"Forecast error: {forecast_data['error']}")
                    track_missing_year(missing_years, year, "forecast_error")
                else:
                    data.append({"x": year, "y": forecast_data["average_temperature"]})
                continue
            except Exception as e:
                logger.error(f"Error fetching forecast: {str(e)}")
                track_missing_year(missing_years, year, "forecast_failed")
                continue
        # Use historical data for all other dates
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED TEMPERATURE: {cache_key} | Location: {location} | Year: {year}")
                weather = json.loads(cached_data)
                try:
                    temp = weather["days"][0]["temp"]
                    if temp is not None:
                        data.append({"x": year, "y": temp})
                        # Cache the result if caching is enabled
                        if CACHE_ENABLED:
                            cache_key = generate_cache_key("weather", location, date_str)
                            cache_duration = SHORT_CACHE_DURATION if is_today(year, month, day) else LONG_CACHE_DURATION
                            set_cache_value(cache_key, cache_duration, json.dumps(weather), redis_client)
                    else:
                        logger.debug(f"Temperature is None for {year}, marking as missing.")
                        track_missing_year(missing_years, year, "no_temperature_data")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error processing cached data for {date_str}: {str(e)}")
                    track_missing_year(missing_years, year, "data_processing_error")
                continue
            else:
                uncached_years.append(year)
                uncached_date_strs.append(date_str)
        else:
            uncached_years.append(year)
            uncached_date_strs.append(date_str)

    # Batch fetch for all uncached years
    if uncached_date_strs:
        batch_results = await fetch_weather_batch(location, uncached_date_strs, redis_client)
        for year, date_str in zip(uncached_years, uncached_date_strs):
            weather = batch_results.get(date_str)
            #logger.debug(f"get_temperature_series weather: {weather}")
            if weather and "error" not in weather:
                try:
                    temp = weather["days"][0]["temp"]
                    logger.debug(f"Got {year} temperature = {temp}")
                    if temp is not None:
                        data.append({"x": year, "y": temp})
                        # Cache the result if caching is enabled
                        if CACHE_ENABLED:
                            cache_key = generate_cache_key("weather", location, date_str)
                            cache_duration = SHORT_CACHE_DURATION if is_today(year, month, day) else LONG_CACHE_DURATION
                            set_cache_value(cache_key, cache_duration, json.dumps(weather), redis_client)
                    else:
                        logger.debug(f"Temperature is None for {year}, marking as missing.")
                        track_missing_year(missing_years, year, "no_temperature_data")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error processing batch data for {date_str}: {str(e)}")
                    track_missing_year(missing_years, year, "data_processing_error")
            else:
                track_missing_year(missing_years, year, weather.get("error", "api_error") if weather else "api_error")

    # Print summary of collected data
    logger.debug("Data summary:")
    logger.debug(f"Total data points: {len(data)}")
    if data:
        temps = [d['y'] for d in data]
        if temps:  # Add check to ensure temps list is not empty
            logger.debug(f"Temperature range: {min(temps):.1f}°C to {max(temps):.1f}°C")
            logger.debug(f"Average temperature: {sum(temps)/len(temps):.1f}°C")

    data_list = sorted(data, key=lambda d: d['x'])

    if data_list:
        latest_year = data_list[-1]['x']
        if isinstance(latest_year, str):
            try:
                latest_year = int(latest_year)
            except ValueError:
                latest_year = None
        if latest_year is not None and latest_year != current_year:
            if not any(entry.get("year") == current_year for entry in missing_years):
                track_missing_year(missing_years, current_year, "no_data_current_year")
    elif current_year not in [entry.get("year") for entry in missing_years]:
        track_missing_year(missing_years, current_year, "no_data_current_year")

    if not data_list:
        logger.warning(f"No valid temperature data found for {location} on {month}-{day}")
        return {
            "data": [],
            "metadata": create_metadata(len(years), 0, [{"year": y, "reason": "no_data"} for y in years]),
            "error": f"Invalid location: {location}"
        }

    # Cache the entire series for a short duration
    if CACHE_ENABLED:
        set_cache_value(series_cache_key, SHORT_CACHE_DURATION, json.dumps({
            "data": data_list,
            "metadata": create_metadata(len(years), len(data), missing_years)
        }), redis_client)

    return {
        "data": data_list,
        "metadata": create_metadata(len(years), len(data), missing_years)
    }

async def get_forecast_data(location: str, date) -> Dict:
    """Get forecast data for a location and date using httpx."""
    # Convert date to string format if it's a date object
    if hasattr(date, 'strftime'):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)
    
    # Debug logging to see what location we're processing
    logger.debug(f"🌐 Fetching forecast for location: '{location}' (repr: {repr(location)}), date: {date_str}")
    
    try:
        url = build_visual_crossing_url(location, date_str, remote=False)
        logger.debug(f"🔗 Built URL (sanitized): {sanitize_url(url)}")
    except Exception as url_error:
        logger.error(f"❌ Error building URL for location '{location}': {url_error}")
        raise
    
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.get(url, headers={"Accept-Encoding": "gzip"})
    if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
        data = response.json()
        if data.get('days') and len(data['days']) > 0:
            day_data = data['days'][0]
            temp = day_data.get('temp')
            if temp is None:
                return {"error": "No temperature data in forecast response"}
            return {
                "location": location,
                "date": date,
                "average_temperature": round(temp, 1),
                "unit": "celsius"
            }
        else:
            return {"error": "No days data in forecast response"}
    return {"error": response.text, "status": response.status_code}

async def fetch_weather_batch(location: str, date_strs: list, redis_client: redis.Redis, max_concurrent: int = None) -> dict:
    """
    Fetch weather data for multiple dates in parallel using httpx with concurrency control.
    Returns a dict mapping date_str to weather data.
    Now uses caching for each date and global concurrency semaphore.
    """
    from utils.weather_data import get_weather_for_date
    
    logger.debug(f"fetch_weather_batch for {location}")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Visual Crossing API key not configured")
    
    # Use global semaphore if max_concurrent not specified
    if max_concurrent is None:
        semaphore = visual_crossing_semaphore
    else:
        semaphore = asyncio.Semaphore(max_concurrent)
    
    results = {}
    
    async def fetch_one(date_str):
        async with semaphore:
            return date_str, await get_weather_for_date(location, date_str, redis_client)
    
    tasks = [fetch_one(date_str) for date_str in date_strs]
    for fut in asyncio.as_completed(tasks):
        date_str, result = await fut
        results[date_str] = result
    return results


# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
