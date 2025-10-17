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
import aiohttp
import firebase_admin
import redis
import requests
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, Path, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from firebase_admin import auth, credentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Load environment variables before importing routers
load_dotenv()

# Import the new router
from routers.records_agg import router as records_agg_router, daily_cache, cleanup_http_sessions
from routers.locations_preapproved import router as locations_preapproved_router, initialize_locations_data

# Import enhanced caching utilities
from cache_utils import (
    initialize_cache, get_cache as get_enhanced_cache, get_job_manager,
    CACHE_CONTROL_PUBLIC, JobStatus,
    # Cache management classes
    LocationUsageTracker, CacheWarmer, CacheStats, CacheInvalidator,
    # Cache utility functions
    get_cache_value, set_cache_value, get_weather_cache_key, generate_cache_key,
    # Cache configuration
    CACHE_WARMING_ENABLED, CACHE_WARMING_INTERVAL_HOURS, CACHE_WARMING_DAYS_BACK,
    CACHE_WARMING_CONCURRENT_REQUESTS, CACHE_WARMING_MAX_LOCATIONS,
    CACHE_STATS_ENABLED, CACHE_STATS_RETENTION_HOURS, CACHE_HEALTH_THRESHOLD,
    CACHE_INVALIDATION_ENABLED, CACHE_INVALIDATION_DRY_RUN, CACHE_INVALIDATION_BATCH_SIZE,
    USAGE_TRACKING_ENABLED, USAGE_RETENTION_DAYS, DEFAULT_POPULAR_LOCATIONS,
    # Global instances
    get_usage_tracker, get_cache_warmer, get_cache_stats, get_cache_invalidator,
    scheduled_cache_warming
)
from version import __version__

# Environment variables - strip whitespace/newlines from API keys
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379").strip()

# Debug: Log the REDIS_URL value to diagnose connection issues
import logging as _logging
_temp_logger = _logging.getLogger(__name__)
_temp_logger.info(f"🔍 DEBUG: REDIS_URL environment variable = {REDIS_URL}")

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
# Logging verbosity control - set to "minimal" to reduce Railway logging limits
LOG_VERBOSITY = os.getenv("LOG_VERBOSITY", "normal").lower()  # "minimal", "normal", "verbose"
TEST_TOKEN = os.getenv("TEST_TOKEN")
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN")  # API access token for automated systems
CACHE_CONTROL_HEADER = "public, max-age=3600, stale-while-revalidate=86400, stale-if-error=86400"
FILTER_WEATHER_DATA = os.getenv("FILTER_WEATHER_DATA", "true").lower() == "true"

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
        self.cleanup_interval = 300  # Clean up every 5 minutes
        
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
        self.cleanup_interval = 300  # Clean up every 5 minutes
        
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

def build_visual_crossing_url(location: str, date: str, remote: bool = True) -> str:
    """Build Visual Crossing API URL with consistent parameters.
    
    Args:
        location: The location to get weather data for
        date: The date in YYYY-MM-DD format
        remote: Whether to include remote data parameters (default: True)
    """
    from urllib.parse import quote
    # Clean and URL-encode the location to handle special characters
    cleaned_location = clean_location_string(location)
    encoded_location = quote(cleaned_location, safe='')
    base_params = f"unitGroup={VISUAL_CROSSING_UNIT_GROUP}&include={VISUAL_CROSSING_INCLUDE_PARAMS}&key={API_KEY}"
    if remote:
        return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{date}?{base_params}&{VISUAL_CROSSING_REMOTE_DATA}"
    else:
        return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{date}?{base_params}"

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
    # Startup
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

# Global exception handlers for better error handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed error messages."""
    client_ip = get_client_ip(request)
    
    # Extract detailed error information
    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error['loc'])
        error_details.append({
            "field": field,
            "message": error['msg'],
            "type": error['type'],
            "input": error.get('input', 'N/A')
        })
    
    logger.error(f"❌ VALIDATION ERROR: {exc.errors()} | IP={client_ip} | Path={request.url.path}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request data validation failed",
            "details": error_details,
            "path": request.url.path,
            "method": request.method
        }
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic model validation errors."""
    client_ip = get_client_ip(request)
    
    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error['loc'])
        error_details.append({
            "field": field,
            "message": error['msg'],
            "type": error['type'],
            "input": error.get('input', 'N/A')
        })
    
    logger.error(f"❌ PYDANTIC VALIDATION ERROR: {exc.errors()} | IP={client_ip} | Path={request.url.path}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Model Validation Error",
            "message": "Data model validation failed",
            "details": error_details,
            "path": request.url.path,
            "method": request.method
        }
    )

# Include the routers
app.include_router(records_agg_router)
app.include_router(locations_preapproved_router)

# Initialize Redis with decode_responses for consistent string handling
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

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
HTTP_TIMEOUT = 60.0  # Increased from 30s to 60s for better reliability
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

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Log all requests when DEBUG is enabled or verbosity is verbose."""
    if DEBUG or LOG_VERBOSITY == "verbose":
        start_time = time.time()
        client_ip = get_client_ip(request)
        
        # Log request details (only for non-public paths to reduce noise)
        if not request.url.path in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/rate-limit-status"]:
            logger.debug(f"🌐 REQUEST: {request.method} {request.url.path} | IP: {client_ip} | User-Agent: {request.headers.get('user-agent', 'Unknown')}")
        
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
    public_paths = ["/", "/docs", "/openapi.json", "/redoc", "/test-cors", "/test-redis", "/rate-limit-status", "/rate-limit-stats", "/analytics", "/health", "/health/detailed", "/v1/records/rolling-bundle/test-cors", "/v1/jobs/diagnostics/worker-status"]
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
    
    # Apply rate limiting only to Visual Crossing API endpoints
    # Skip rate limiting for: whitelisted IPs, service jobs (API_ACCESS_TOKEN)
    if RATE_LIMIT_ENABLED and location_monitor and request_monitor and not is_ip_whitelisted(client_ip) and not is_service_job:
        # Check if this endpoint queries Visual Crossing API
        is_vc_api_endpoint = any(request.url.path.startswith(path) for path in vc_api_paths)
        
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
                            logger.warning(f"🌍 LOCATION DIVERSITY LIMIT: {client_ip} | {request.method} {request.url.path} | {location_reason}")
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
                        logger.debug(f"✅ LOCATION DIVERSITY CHECK: {client_ip} | {request.method} {request.url.path} | Location: {location} | OK")
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
    
    # Special bypass for testing
    if id_token == TEST_TOKEN:
        if DEBUG:
            logger.debug(f"[DEBUG] Middleware: Using test token bypass")
        request.state.user = {"uid": "testuser"}
    # Production token bypass for automated systems (cron jobs, etc.)
    elif is_service_job:
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
            return JSONResponse(
                status_code=403,
                content={"detail": f"Invalid Firebase token: {str(e)}"}
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite default port
        "https://temphist.com",  # Main domain
        "https://www.temphist.com",  # www subdomain
        "https://dev.temphist.com",  # development site
        "https://temphist-develop.up.railway.app",  # development site on Railway
        "https://temphist-staging.up.railway.app"  # staging site on Railway
    ],
    allow_origin_regex=r"^https://.*\.onrender\.com$",  # Allow any Render subdomain
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

@app.api_route("/test-cors", methods=["GET", "OPTIONS"])
async def test_cors():
    """Test endpoint for CORS"""
    return {"message": "CORS is working"}

@app.api_route("/test-cors-rolling", methods=["GET", "OPTIONS"])
async def test_cors_rolling():
    """Test endpoint for CORS with rolling-bundle path"""
    return {"message": "CORS is working for rolling-bundle", "path": "/test-cors-rolling"}

@app.api_route("/", methods=["GET", "OPTIONS"])
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "Temperature History API",
        "version": __version__,
        "description": "Temperature history API with v1 unified endpoints and legacy support",
        "v1_endpoints": {
            "records": [
                "/v1/records/{period}/{location}/{identifier}",
                "/v1/records/{period}/{location}/{identifier}/average",
                "/v1/records/{period}/{location}/{identifier}/trend",
                "/v1/records/{period}/{location}/{identifier}/summary",
                "/v1/records/{period}/{location}/{identifier}/updated"
            ],
            "rolling_bundle": [
                "/v1/records/rolling-bundle/{location}/{anchor}"
            ],
            "periods": ["daily", "weekly", "monthly", "yearly"],
            "examples": [
                "/v1/records/daily/london/01-15",
                "/v1/records/weekly/london/01-15",
                "/v1/records/monthly/london/01-15",
                "/v1/records/yearly/london/01-15",
                "/v1/records/daily/london/01-15/updated",
                "/v1/records/rolling-bundle/london/2024-01-15"
            ]
        },
        "removed_endpoints": {
            "status": "removed",
            "endpoints": [
                "/data/{location}/{month_day}",
                "/average/{location}/{month_day}",
                "/trend/{location}/{month_day}",
                "/summary/{location}/{month_day}"
            ],
            "note": "These endpoints have been removed. Please use v1 endpoints instead.",
            "migration": "Use /v1/records/daily/{location}/{month_day} and subresources"
        },
        "other_endpoints": [
            "/weather/{location}/{date}",
            "/forecast/{location}",
            "/rate-limit-status",
            "/rate-limit-stats",
            "/usage-stats",
            "/usage-stats/{location}",
            "/cache-warm",
            "/cache-warm/status",
            "/cache-warm/locations",
            "/cache-warm/startup",
            "/cache-warm/schedule",
            "/cache-stats",
            "/cache-stats/health",
            "/cache-stats/endpoints",
            "/cache-stats/locations",
            "/cache-stats/hourly",
            "/cache-stats/reset",
            "/cache/info",
            "/cache/invalidate/key/{cache_key}",
            "/cache/invalidate/pattern",
            "/cache/invalidate/endpoint/{endpoint}",
            "/cache/invalidate/location/{location}",
            "/cache/invalidate/date/{date}",
            "/cache/invalidate/forecast",
            "/cache/invalidate/today",
            "/cache/invalidate/expired",
            "/cache/clear"
        ],
        "analytics_endpoints": [
            "/analytics",
            "/analytics/summary", 
            "/analytics/recent",
            "/analytics/session/{session_id}"
        ]
    }

# Shared async function for fetching and caching weather data for a single date

async def get_weather_for_date(location: str, date_str: str) -> dict:
    """Fetch and cache weather data for a specific date.
    
    Implements fallback logic for remote data:
    1. First tries without remote data parameters
    2. If no temperature data and year >= 2005, retries with remote data parameters
    3. Never uses remote data for today's data
    """
    logger.debug(f"get_weather_for_date for {location} on {date_str}")
    cache_key = get_weather_cache_key(location, date_str)
    if CACHE_ENABLED:
        cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
        if cached_data:
            if DEBUG:
                logger.debug(f"✅ SERVING CACHED WEATHER: {cache_key} | Location: {location} | Date: {date_str}")
            try:
                return json.loads(cached_data)
            except Exception as e:
                logger.error(f"Error decoding cached data for {cache_key}: {e}")

    logger.debug(f"Cache miss: {cache_key} — fetching from API")
    
    # Parse the date to determine the year
    try:
        year, month, day = map(int, date_str.split("-")[:3])
        is_today_date = is_today(year, month, day)
    except Exception:
        year = 2000  # Default fallback
        is_today_date = False
    
    # First attempt: without remote data parameters
    url = build_visual_crossing_url(location, date_str, remote=False)
    logger.debug(f"First attempt (no remote data): {url}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)  # Increased from 30s to 60s for better reliability
        logger.info(f"[DEBUG] Creating aiohttp session with 60-second timeout")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(f"[DEBUG] Session created successfully, making GET request to: {url}")
            async with session.get(url, headers={"Accept-Encoding": "gzip"}) as resp:
                logger.info(f"[DEBUG] Response received: status={resp.status}, content-type={resp.headers.get('Content-Type')}")
                if resp.status == 200 and 'application/json' in resp.headers.get('Content-Type', ''):
                    logger.info(f"[DEBUG] Parsing JSON response...")
                    data = await resp.json()
                    logger.info(f"[DEBUG] JSON parsed successfully, checking for 'days' data")
                    days = data.get('days')
                    if days is not None and len(days) > 0:
                        # Check if we got valid temperature data
                        day_data = days[0]
                        temp = day_data.get('temp')
                        
                        if temp is not None:
                            # Success! Cache and return the data
                            if FILTER_WEATHER_DATA:
                                # Filter to only essential temperature data
                                filtered_days = []
                                for day_data in days:
                                    filtered_day = {
                                        'datetime': day_data.get('datetime'),
                                        'temp': day_data.get('temp'),
                                        'tempmin': day_data.get('tempmin'),
                                        'tempmax': day_data.get('tempmax')
                                    }
                                    filtered_days.append(filtered_day)
                                to_cache = {"days": filtered_days}
                            else:
                                # Return full data if filtering is disabled
                                to_cache = {"days": days}

                            if DEBUG:
                                logger.debug(f"🌤️ FRESH API RESPONSE: {location} | {date_str}")

                            if CACHE_ENABLED:
                                cache_duration = SHORT_CACHE_DURATION if is_today_date else LONG_CACHE_DURATION
                                set_cache_value(cache_key, cache_duration, json.dumps(to_cache), redis_client)
                            return to_cache
                        else:
                            logger.debug(f"No temperature data in first attempt for {date_str}")
                    else:
                        logger.debug(f"No 'days' data in first attempt for {date_str}")
                else:
                    logger.debug(f"First attempt failed with status {resp.status}")
                
                # If we reach here, the first attempt didn't provide valid temperature data
                # Check if we should try with remote data parameters
                if not is_today_date and year >= 2005:
                    logger.info(f"[DEBUG] First attempt failed, trying remote fallback for {date_str} (year {year})")
                    url_with_remote = build_visual_crossing_url(location, date_str, remote=True)
                    logger.info(f"[DEBUG] Remote fallback URL: {url_with_remote}")
                    
                    logger.info(f"[DEBUG] Making remote fallback request...")
                    async with session.get(url_with_remote, headers={"Accept-Encoding": "gzip"}) as remote_resp:
                        logger.info(f"[DEBUG] Remote response received: status={remote_resp.status}, content-type={remote_resp.headers.get('Content-Type')}")
                        if remote_resp.status == 200 and 'application/json' in remote_resp.headers.get('Content-Type', ''):
                            logger.info(f"[DEBUG] Parsing remote JSON response...")
                            remote_data = await remote_resp.json()
                            logger.info(f"[DEBUG] Remote JSON parsed successfully, checking for 'days' data")
                            remote_days = remote_data.get('days')
                            if remote_days is not None and len(remote_days) > 0:
                                remote_day_data = remote_days[0]
                                remote_temp = remote_day_data.get('temp')
                                
                                if remote_temp is not None:
                                    # Success with remote data! Cache and return
                                    logger.debug(f"Remote data fallback successful for {date_str}")
                                    to_cache = {"days": remote_days}
                                    if DEBUG:
                                        logger.debug(f"🌤️ REMOTE FALLBACK RESPONSE: {location} | {date_str}")
                                    if CACHE_ENABLED:
                                        set_cache_value(cache_key, LONG_CACHE_DURATION, json.dumps(to_cache), redis_client)
                                    return to_cache
                                else:
                                    logger.debug(f"No temperature data in remote fallback for {date_str}")
                            else:
                                logger.debug(f"No 'days' data in remote fallback for {date_str}")
                        else:
                            logger.debug(f"Remote fallback failed with status {remote_resp.status}")
                
                # If we reach here, neither attempt provided valid temperature data
                logger.info(f"[DEBUG] Both attempts failed, returning error response")
                return {"error": "No temperature data available", "status": resp.status}
                
    except Exception as e:
        logger.error(f"[DEBUG] Exception occurred in get_weather_for_date for {date_str}: {str(e)}")
        logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return {"error": str(e)}

@app.get("/weather/{location}/{date}")
def get_weather(location: str, date: str, response: Response = None):
    """Get weather data for a specific location and date."""
    logger.info(f"[DEBUG] Weather endpoint called with location={location}, date={date}")
    
    # Parse the date for cache headers
    try:
        req_date = dt_date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    cache_key = generate_cache_key("weather", location, date)
    logger.info(f"[DEBUG] Cache key: {cache_key}")
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        if DEBUG:
            logger.info(f"🔍 CACHE CHECK: {cache_key}")
        cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
        if cached_data:
            if DEBUG:
                logger.info(f"✅ SERVING CACHED DATA: {cache_key} | Location: {location} | Date: {date}")
            result = json.loads(cached_data)
            # Set smart cache headers for cached data too
            set_weather_cache_headers(response, req_date=req_date, key_parts=f"{location}|{date}|metric|v1")
            return result
        if DEBUG:
            logger.info(f"❌ CACHE MISS: {cache_key} — fetching from API")
    else:
        if DEBUG:
            logger.info(f"⚠️  CACHING DISABLED: fetching from API")
    
    logger.info(f"[DEBUG] About to call get_weather_for_date...")
    # Use the shared async function for consistency
    # Since this is a sync endpoint, run the async function in the event loop
    result = asyncio.run(get_weather_for_date(location, date))
    
    # Log the final response being returned to the client
    if DEBUG:
        logger.debug(f"🎯 FINAL RESPONSE TO CLIENT: {location} | {date}")
        logger.debug(f"📄 FINAL JSON RESPONSE: {json.dumps(result, indent=2)}")
    
    # Set smart cache headers based on data age
    set_weather_cache_headers(response, req_date=req_date, key_parts=f"{location}|{date}|metric|v1")
    
    # Only cache successful results if caching is enabled and not already cached
    if CACHE_ENABLED and "error" not in result:
        try:
            year, month, day = map(int, date.split("-")[:3])
            cache_duration = SHORT_CACHE_DURATION if is_today_or_future(year, month, day) else LONG_CACHE_DURATION
        except Exception:
            cache_duration = LONG_CACHE_DURATION
        set_cache_value(cache_key, cache_duration, json.dumps(result), redis_client)
    return result

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
            # Yesterday - use past tense with "yesterday"
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"
            friendly_date = "yesterday"
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
            # Don't capitalize the period context when it follows "However, "
            if cold_summary:  # Use the same condition as above
                # Force lowercase for period context when following "However, "
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {rounded_diff}°C warmer than average."
            else:
                period_capitalized = period_context.capitalize()
                avg_summary += f"{period_capitalized} {tense_context} {rounded_diff}°C warmer than average."
    else:
        # For weekly/monthly/yearly periods, use "It was" to avoid repetition with "has been"
        if period in ["weekly", "monthly", "yearly"]:
            avg_summary = "However, " if warm_summary else ""
            avg_summary += f"{'it' if warm_summary else 'It'} was {abs(rounded_diff)}°C cooler than average."
        else:
            # For daily periods, use the period context
            avg_summary = "However, " if warm_summary else ""
            # Don't capitalize the period context when it follows "However, "
            if warm_summary:  # Use the same condition as above
                # Force lowercase for period context when following "However, "
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {abs(rounded_diff)}°C cooler than average."
            else:
                period_capitalized = period_context.capitalize()
                avg_summary += f"{period_capitalized} {tense_context} {abs(rounded_diff)}°C cooler than average."

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

@app.get("/forecast/{location}")
async def get_forecast(location: str):
    """Get weather forecast for a location with time-based caching."""
    try:
        # Create cache key for forecast
        cache_key = generate_cache_key("forecast", location)
        
        # Check cache first if caching is enabled
        if CACHE_ENABLED:
            cached_forecast = get_cache_value(cache_key, redis_client, "forecast", location, get_cache_stats())
            if cached_forecast:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED FORECAST: {cache_key} | Location: {location}")
                return JSONResponse(content=json.loads(cached_forecast), headers={"Cache-Control": "public, max-age=1800"})
            elif DEBUG:
                logger.debug(f"❌ FORECAST CACHE MISS: {cache_key} — fetching fresh forecast")
        
        # Fetch fresh forecast data
        today = datetime.now().date()
        result = await get_forecast_data(location, today)
        
        # Cache the result if caching is enabled and no error
        if CACHE_ENABLED and "error" not in result:
            cache_duration = get_forecast_cache_duration()
            # Convert date object to string for JSON serialization
            if "date" in result and hasattr(result["date"], 'strftime'):
                result["date"] = result["date"].strftime("%Y-%m-%d")
            set_cache_value(cache_key, cache_duration, json.dumps(result), redis_client)
            if DEBUG:
                current_hour = datetime.now().hour
                time_period = "stable" if current_hour >= 18 else "active"
                logger.debug(f"💾 CACHED FORECAST: {cache_key} | Duration: {cache_duration} | Time: {time_period}")
        
        # Convert date object to string for JSON response
        if "date" in result and hasattr(result["date"], 'strftime'):
            result["date"] = result["date"].strftime("%Y-%m-%d")
        
        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=1800"})
    
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}")
        return JSONResponse(
            content={"error": f"Forecast error: {str(e)}"},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Render load balancers."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check endpoint for debugging and monitoring."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {}
    }
    
    overall_healthy = True
    
    # Check Redis connection
    try:
        set_cache_value("health_check", timedelta(minutes=1), "test_value", redis_client)
        test_value = get_cache_value("health_check", redis_client, "health", "test", get_cache_stats())
        
        # Handle both string and bytes responses (depending on Redis client configuration)
        if test_value:
            if isinstance(test_value, bytes):
                test_value_str = test_value.decode('utf-8')
            else:
                test_value_str = str(test_value)
            
            if test_value_str == "test_value":
                health_status["services"]["redis"] = {
                    "status": "healthy",
                    "message": "Connection successful"
                }
            else:
                health_status["services"]["redis"] = {
                    "status": "unhealthy",
                    "message": f"Cache test failed - expected 'test_value', got '{test_value_str}'"
                }
                overall_healthy = False
        else:
            health_status["services"]["redis"] = {
                "status": "unhealthy",
                "message": "Cache test failed - no value returned"
            }
            overall_healthy = False
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "unhealthy",
            "message": f"Connection error: {str(e)}"
        }
        overall_healthy = False
    
    # Check API keys
    api_keys_healthy = True
    if not API_KEY:
        health_status["services"]["visual_crossing_api"] = {
            "status": "unhealthy",
            "message": "API key not configured"
        }
        api_keys_healthy = False
    else:
        health_status["services"]["visual_crossing_api"] = {
            "status": "healthy",
            "message": "API key configured"
        }
    
    if not OPENWEATHER_API_KEY:
        health_status["services"]["openweather_api"] = {
            "status": "unhealthy",
            "message": "API key not configured"
        }
        api_keys_healthy = False
    else:
        health_status["services"]["openweather_api"] = {
            "status": "healthy",
            "message": "API key configured"
        }
    
    if not api_keys_healthy:
        overall_healthy = False
    
    # Check cache statistics if available
    try:
        cache_stats_instance = get_cache_stats()
        if cache_stats_instance and hasattr(cache_stats_instance, 'get_cache_health'):
            cache_health = cache_stats_instance.get_cache_health()
            health_status["services"]["cache"] = cache_health
            # Only consider cache "unhealthy" status as a failure, not "degraded"
            if cache_health.get("status") == "unhealthy":
                overall_healthy = False
    except Exception as e:
        health_status["services"]["cache"] = {
            "status": "unknown",
            "message": f"Cache stats unavailable: {str(e)}"
        }
    
    # Set overall status
    health_status["status"] = "healthy" if overall_healthy else "unhealthy"
    
    # Return appropriate HTTP status code
    status_code = 200 if overall_healthy else 503
    
    return JSONResponse(
        content=health_status,
        status_code=status_code,
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/test-redis")
async def test_redis():
    """Test Redis connection."""
    try:
        # Try to set a test value
        set_cache_value("test_key", timedelta(minutes=5), "test_value", redis_client)
        # Try to get the test value
        test_value = get_cache_value("test_key", redis_client, "test", "test", get_cache_stats())
        if test_value and test_value.decode('utf-8') == "test_value":
            return JSONResponse(content={"status": "success", "message": "Redis connection is working"}, headers={"Cache-Control": CACHE_CONTROL_HEADER})
        else:
            return JSONResponse(content={"status": "error", "message": "Redis connection test failed"}, headers={"Cache-Control": CACHE_CONTROL_HEADER})
    except Exception as e:
        return JSONResponse(
    content={
        "status": "error",
        "message": f"Redis connection error: {str(e)}"
    },
    headers={"Cache-Control": CACHE_CONTROL_HEADER}
)

@app.get("/rate-limit-status")
async def get_rate_limit_status(request: Request):
    """Get rate limiting status for the current client IP."""
    if not RATE_LIMIT_ENABLED:
        return {"status": "disabled", "message": "Rate limiting is not enabled"}
    
    client_ip = get_client_ip(request)
    
    # Check if this is a service job using API_ACCESS_TOKEN
    auth_header = request.headers.get("Authorization")
    is_service_job = False
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
            is_service_job = True
    
    # Check IP status
    is_whitelisted = is_ip_whitelisted(client_ip)
    is_blacklisted = is_ip_blacklisted(client_ip)
    
    # Skip stats if whitelisted or service job
    location_stats = location_monitor.get_stats(client_ip) if location_monitor and not is_whitelisted and not is_service_job else {}
    request_stats = request_monitor.get_stats(client_ip) if request_monitor and not is_whitelisted and not is_service_job else {}
    
    return {
        "client_ip": client_ip,
        "ip_status": {
            "whitelisted": is_whitelisted,
            "blacklisted": is_blacklisted,
            "service_job": is_service_job,
            "rate_limited": not is_whitelisted and not is_blacklisted and not is_service_job
        },
        "location_monitor": location_stats,
        "request_monitor": request_stats,
        "rate_limits": {
            "max_locations_per_hour": MAX_LOCATIONS_PER_HOUR,
            "max_requests_per_hour": MAX_REQUESTS_PER_HOUR,
            "window_hours": RATE_LIMIT_WINDOW_HOURS
        }
    }

@app.get("/rate-limit-stats")
async def get_rate_limit_stats():
    """Get overall rate limiting statistics (admin endpoint)."""
    if not RATE_LIMIT_ENABLED:
        return {"status": "disabled", "message": "Rate limiting is not enabled"}
    
    # Get stats for all monitored IPs
    all_stats = {}
    
    if location_monitor:
        for ip in location_monitor.ip_locations.keys():
            all_stats[ip] = {
                "location_stats": location_monitor.get_stats(ip),
                "request_stats": request_monitor.get_stats(ip) if request_monitor else {},
                "suspicious": location_monitor.is_suspicious(ip),
                "whitelisted": is_ip_whitelisted(ip),
                "blacklisted": is_ip_blacklisted(ip)
            }
    
    return {
        "total_monitored_ips": len(all_stats),
        "suspicious_ips": list(location_monitor.suspicious_ips) if location_monitor else [],
        "whitelisted_ips": IP_WHITELIST,
        "blacklisted_ips": IP_BLACKLIST,
        "ip_details": all_stats
    }

@app.get("/usage-stats")
async def get_usage_stats():
    """Get usage tracking statistics."""
    if not USAGE_TRACKING_ENABLED or not get_usage_tracker():
        return {"status": "disabled", "message": "Usage tracking is not enabled"}
    
    return {
        "enabled": USAGE_TRACKING_ENABLED,
        "retention_days": USAGE_RETENTION_DAYS,
        "popular_locations_24h": get_usage_tracker().get_popular_locations(limit=10, hours=24),
        "popular_locations_7d": get_usage_tracker().get_popular_locations(limit=10, hours=168),
        "all_location_stats": get_usage_tracker().get_all_location_stats()
    }

@app.get("/usage-stats/{location}")
async def get_location_usage_stats(location: str):
    """Get usage statistics for a specific location."""
    if not USAGE_TRACKING_ENABLED or not get_usage_tracker():
        return {"status": "disabled", "message": "Usage tracking is not enabled"}
    
    stats = get_usage_tracker().get_location_stats(location)
    if not stats:
        return {"status": "not_found", "message": f"No usage data found for location: {location}"}
    
    return stats

@app.post("/cache-warm")
async def trigger_cache_warming():
    """Trigger manual cache warming for all popular locations (legacy endpoint - now uses job system)."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    # Import job manager
    from cache_utils import get_job_manager
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    # Create cache warming job
    job_id = job_manager.create_job("cache_warming", {
        "type": "all",
        "locations": [],
        "triggered_by": "legacy_api",
        "triggered_at": datetime.now(timezone.utc).isoformat()
    })
    
    return {
        "status": "job_created",
        "message": "Cache warming job created successfully (legacy endpoint)",
        "job_id": job_id,
        "job_status_url": f"/cache-warm/job/{job_id}",
        "locations_to_warm": get_cache_warmer().get_locations_to_warm()
    }

@app.get("/cache-warm/status")
async def get_cache_warming_status():
    """Get cache warming status and statistics."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    return get_cache_warmer().get_warming_stats()

@app.get("/cache-warm/locations")
async def get_locations_to_warm():
    """Get list of locations that would be warmed."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    return {
        "locations": get_cache_warmer().get_locations_to_warm(),
        "dates": get_cache_warmer().get_dates_to_warm(),
        "month_days": get_cache_warmer().get_month_days_to_warm()
    }

@app.post("/cache-warm/startup")
async def trigger_startup_warming():
    """Trigger cache warming on startup (useful for deployment) - now uses job system."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    # Import job manager
    from cache_utils import get_job_manager
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    # Create cache warming job
    job_id = job_manager.create_job("cache_warming", {
        "type": "all",
        "locations": [],
        "triggered_by": "startup_api",
        "triggered_at": datetime.now(timezone.utc).isoformat()
    })
    
    return {
        "status": "job_created",
        "message": "Startup cache warming job created successfully",
        "job_id": job_id,
        "job_status_url": f"/cache-warm/job/{job_id}",
        "locations_to_warm": get_cache_warmer().get_locations_to_warm()
    }

@app.get("/cache-warm/schedule")
async def get_warming_schedule():
    """Get information about the warming schedule."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    return {
        "enabled": CACHE_WARMING_ENABLED,
        "interval_hours": CACHE_WARMING_INTERVAL_HOURS,
        "next_warming_in_hours": CACHE_WARMING_INTERVAL_HOURS,  # Simplified - would need more complex logic for exact timing
        "last_warming": get_cache_warmer().last_warming_time.isoformat() if get_cache_warmer().last_warming_time else None,
        "warming_in_progress": get_cache_warmer().warming_in_progress
    }

@app.post("/cache-warm/job")
async def trigger_cache_warming_job(
    warming_type: str = "all",
    locations: List[str] = None
):
    """Trigger cache warming as a background job."""
    if not CACHE_WARMING_ENABLED or not get_cache_warmer():
        return {"status": "disabled", "message": "Cache warming is not enabled"}
    
    # Import job manager
    from cache_utils import get_job_manager
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    # Validate warming type
    if warming_type not in ["all", "popular", "specific"]:
        return {"status": "error", "message": "Invalid warming type. Must be 'all', 'popular', or 'specific'"}
    
    # Validate locations for specific warming
    if warming_type == "specific" and not locations:
        return {"status": "error", "message": "Locations must be provided for specific warming"}
    
    # Create cache warming job
    job_id = job_manager.create_job("cache_warming", {
        "type": warming_type,
        "locations": locations or [],
        "triggered_by": "api",
        "triggered_at": datetime.now(timezone.utc).isoformat()
    })
    
    return {
        "status": "job_created",
        "message": "Cache warming job created successfully",
        "job_id": job_id,
        "warming_type": warming_type,
        "locations": locations or [],
        "job_status_url": f"/jobs/{job_id}"
    }

@app.get("/cache-warm/job/{job_id}")
async def get_cache_warming_job_status(job_id: str):
    """Get status of a cache warming job."""
    from cache_utils import get_job_manager
    job_manager = get_job_manager()
    
    if not job_manager:
        return {"status": "error", "message": "Job manager not available"}
    
    job_status = job_manager.get_job_status(job_id)
    if not job_status:
        return {"status": "not_found", "message": "Job not found"}
    
    return job_status

@app.get("/cache-stats")
async def get_cache_statistics():
    """Get comprehensive cache statistics and performance metrics."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return cache_stats_instance.get_comprehensive_stats()

@app.get("/cache-stats/health")
async def get_cache_health():
    """Get cache health assessment and alerts."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return cache_stats_instance.get_cache_health()

@app.get("/cache-stats/endpoints")
async def get_cache_endpoint_stats():
    """Get cache statistics broken down by endpoint."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return {
        "by_endpoint": cache_stats_instance.get_endpoint_stats(),
        "overall": {
            "total_requests": cache_stats_instance.stats["total_requests"],
            "hit_rate": cache_stats_instance.get_hit_rate(),
            "error_rate": cache_stats_instance.get_error_rate()
        }
    }

@app.get("/cache-stats/locations")
async def get_cache_location_stats():
    """Get cache statistics broken down by location."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return {
        "by_location": cache_stats_instance.get_location_stats(),
        "overall": {
            "total_requests": cache_stats_instance.stats["total_requests"],
            "hit_rate": cache_stats_instance.get_hit_rate(),
            "error_rate": cache_stats_instance.get_error_rate()
        }
    }

@app.get("/cache-stats/hourly")
async def get_cache_hourly_stats(hours: int = 24):
    """Get hourly cache statistics for the last N hours."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    return {
        "hourly_data": cache_stats_instance.get_hourly_stats(hours),
        "requested_hours": hours
    }

@app.post("/cache-stats/reset")
async def reset_cache_statistics():
    """Reset all cache statistics (admin endpoint)."""
    cache_stats_instance = get_cache_stats()
    if not CACHE_STATS_ENABLED or not cache_stats_instance:
        return {"status": "disabled", "message": "Cache statistics are not enabled"}
    
    cache_stats_instance.reset_stats()
    return {
        "status": "success",
        "message": "Cache statistics have been reset",
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/cache/invalidate/key/{cache_key:path}")
async def invalidate_cache_key(cache_key: str, dry_run: bool = False):
    """Invalidate a specific cache key."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_key(cache_key, dry_run)
    return result

@app.delete("/cache/invalidate/pattern")
async def invalidate_by_pattern(pattern: str, dry_run: bool = False):
    """Invalidate cache keys matching a pattern."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_pattern(pattern, dry_run)
    return result

@app.delete("/cache/invalidate/endpoint/{endpoint}")
async def invalidate_by_endpoint(endpoint: str, location: str = None, dry_run: bool = False):
    """Invalidate cache keys for a specific endpoint and optionally location."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_endpoint(endpoint, location, dry_run)
    return result

@app.delete("/cache/invalidate/location/{location}")
async def invalidate_by_location(location: str, dry_run: bool = False):
    """Invalidate all cache keys for a specific location."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_location(location, dry_run)
    return result

@app.delete("/cache/invalidate/date/{date}")
async def invalidate_by_date(date: str, dry_run: bool = False):
    """Invalidate cache keys for a specific date."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_by_date(date, dry_run)
    return result

@app.delete("/cache/invalidate/forecast")
async def invalidate_forecast_data(dry_run: bool = False):
    """Invalidate all forecast data."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_forecast_data(dry_run)
    return result

@app.delete("/cache/invalidate/today")
async def invalidate_today_data(dry_run: bool = False):
    """Invalidate all data for today."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_today_data(dry_run)
    return result

@app.delete("/cache/invalidate/expired")
async def invalidate_expired_keys(dry_run: bool = False):
    """Invalidate keys that have expired."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().invalidate_expired_keys(dry_run)
    return result

@app.get("/cache/info")
async def get_cache_info():
    """Get information about current cache state."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    return get_cache_invalidator().get_cache_info()

@app.delete("/cache/clear")
async def clear_all_cache(dry_run: bool = False):
    """Clear all cache data (use with caution!)."""
    if not CACHE_INVALIDATION_ENABLED or not get_cache_invalidator():
        return {"status": "disabled", "message": "Cache invalidation is not enabled"}
    
    result = get_cache_invalidator().clear_all_cache(dry_run)
    return result

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


def is_valid_location(location: str) -> bool:
    """Check if a location is valid by testing the API."""
    today = datetime.now().strftime("%Y-%m-%d")
    url = build_visual_crossing_url(location, today, remote=False)
    response = requests.get(url)
    if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
        return True
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
    return any(pattern.lower() in location_lower for pattern in invalid_patterns)

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
        batch_results = await fetch_weather_batch(location, uncached_date_strs)
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
        logger.debug(f"🔗 Built URL (first 200 chars): {url[:200]}...")
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

async def fetch_weather_batch(location: str, date_strs: list, max_concurrent: int = None) -> dict:
    """
    Fetch weather data for multiple dates in parallel using httpx with concurrency control.
    Returns a dict mapping date_str to weather data.
    Now uses caching for each date and global concurrency semaphore.
    """
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
            return date_str, await get_weather_for_date(location, date_str)
    
    tasks = [fetch_one(date_str) for date_str in date_strs]
    for fut in asyncio.as_completed(tasks):
        date_str, result = await fut
        results[date_str] = result
    return results

# ============================================================================
# V1 API ENDPOINTS
# ============================================================================

def parse_identifier(period: str, identifier: str) -> tuple:
    """Parse identifier based on period type. All periods use MM-DD format representing the end date."""
    # All periods use MM-DD format representing the end date of the period
    try:
        month, day = map(int, identifier.split("-"))
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        return month, day, period
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Identifier must be in MM-DD format: {str(e)}")

async def _fetch_yearly_summary(location: str, start_year: int, end_year: int, unit_group: str = "metric"):
    """Fetch yearly summary data from Visual Crossing historysummary endpoint."""
    from constants import VC_BASE_URL
    url = f"{VC_BASE_URL}/weatherdata/historysummary"
    params = {
        "aggregateHours": 24,
        "minYear": start_year,
        "maxYear": end_year,
        "chronoUnit": "years",
        "breakBy": "years",
        "dailySummaries": "false",
        "contentType": "json",
        "unitGroup": unit_group,
        "locations": location,
        "key": API_KEY,
    }
    
    async with visual_crossing_semaphore:
        http = await get_http_client()
        async with http:
            r = await http.get(url, params=params, headers={"Accept-Encoding": "gzip"})
    r.raise_for_status()
    data = r.json()
    
    # Parse the response to extract yearly temperature data
    yearly_data = []
    if 'locations' in data and location in data['locations']:
        location_data = data['locations'][location]
        if 'values' in location_data:
            for value in location_data['values']:
                year = value.get('year')
                temp = value.get('temp')
                if year and temp is not None:
                    yearly_data.append((year, temp))
    
    return yearly_data

async def get_temperature_data_v1(location: str, period: str, identifier: str, unit_group: str = "celsius") -> Dict:
    """Get temperature data for v1 API with unified logic using historysummary for weekly/monthly/yearly."""
    # Parse identifier based on period (all use MM-DD format representing end date)
    month, day, period_type = parse_identifier(period, identifier)
    
    # Calculate date range based on period and end date
    from datetime import datetime, timedelta
    
    # Use 50 years of data ending at current year (consistent with other endpoints)
    current_year = datetime.now().year
    start_year = current_year - 50  # 50 years back + current year = 51 years total
    years = get_year_range(current_year)
    end_date = datetime(current_year, month, day)
    
    if period == "daily":
        # Single day across all years
        start_date = end_date
        date_range_days = 1
    elif period == "weekly":
        # 7 days ending on the specified date
        start_date = end_date - timedelta(days=6)
        date_range_days = 7
    elif period == "monthly":
        # 30 days ending on the specified date
        start_date = end_date - timedelta(days=29)
        date_range_days = 30
    elif period == "yearly":
        # 365 days ending on the specified date
        start_date = end_date - timedelta(days=364)
        date_range_days = 365
    else:
        raise HTTPException(status_code=400, detail="Invalid period. Must be daily, weekly, monthly, or yearly")
    
    # Get temperature data for the date range across all years
    values = []
    all_temps = []
    missing_years = []
    
    if period == "daily":
        # For daily, get data for just the specific day across all years
        weather_data = await get_temperature_series(location, month, day)
        if weather_data and 'data' in weather_data:
            # Extract missing years from the series metadata
            if 'metadata' in weather_data and 'missing_years' in weather_data['metadata']:
                missing_years.extend(weather_data['metadata']['missing_years'])
            
            for data_point in weather_data['data']:
                year = int(data_point['x'])
                temp = data_point['y']
                if temp is not None:
                    all_temps.append(temp)
                    values.append(TemperatureValue(
                        date=f"{year}-{month:02d}-{day:02d}",
                        year=year,
                        temperature=temp
                    ))
    
    elif period in ["weekly", "monthly"]:
        # Use historysummary endpoint for weekly/monthly data (much more efficient)
        try:
            from routers.records_agg import fetch_historysummary, _historysummary_values, _row_year, _row_mean_temp
            
            # Determine chrono_unit based on period
            chrono_unit = "weeks" if period == "weekly" else "months"
            
            if DEBUG:
                logger.debug(f"Fetching {period} summary for {location} from {start_year} to {current_year}")
            
            payload = await fetch_historysummary(
                location, 
                start_year, 
                current_year, 
                chrono_unit=chrono_unit, 
                break_by="years"
            )
            
            rows = _historysummary_values(payload)
            if DEBUG:
                logger.debug(f"Got {len(rows)} {period} data points")
            
            # For weekly, we need to match by ISO week number
            if period == "weekly":
                desired_week = datetime(current_year, month, day).isocalendar().week
                for r in rows:
                    y = _row_year(r)
                    t = _row_mean_temp(r)
                    if y is not None and t is not None:
                        # Check if this row corresponds to the desired week
                        try:
                            # Try to extract week info from the row
                            period_str = r.get('period') or r.get('datetimeStr') or r.get('start')
                            if period_str:
                                row_date = datetime.strptime(period_str[:10], "%Y-%m-%d")
                                row_week = row_date.isocalendar().week
                                if row_week == desired_week:
                                    all_temps.append(t)
                                    values.append(TemperatureValue(
                                        date=f"{y}-{month:02d}-{day:02d}",
                                        year=y,
                                        temperature=round(t, 1),
                                    ))
                        except Exception as e:
                            if DEBUG:
                                logger.debug(f"Error processing weekly row: {e}")
                            continue
            
            # For monthly, we need to match by month
            elif period == "monthly":
                for r in rows:
                    y = _row_year(r)
                    t = _row_mean_temp(r)
                    if y is not None and t is not None:
                        # Check if this row corresponds to the desired month
                        try:
                            # Try to extract month info from the row
                            period_str = r.get('period') or r.get('datetimeStr') or r.get('start')
                            if period_str:
                                row_month = int(period_str[5:7])
                                if row_month == month:
                                    all_temps.append(t)
                                    values.append(TemperatureValue(
                                        date=f"{y}-{month:02d}-{day:02d}",
                                        year=y,
                                        temperature=round(t, 1),
                                    ))
                        except Exception as e:
                            if DEBUG:
                                logger.debug(f"Error processing monthly row: {e}")
                            continue
            
            if not values:
                if DEBUG:
                    logger.debug(f"No {period} data found, falling back to sampling")
                raise Exception(f"No {period} data found")
                
        except Exception as e:
            # Only log on first occurrence to reduce log noise (historysummary often fails, fallback is expected)
            if DEBUG and "historysummary" not in str(e).lower():
                logger.debug(f"Error fetching {period} summary: {e}, falling back to sampling")
            # Fallback to sampling approach if historysummary fails
            sample_days = min(date_range_days, 7)  # Sample up to 7 days for efficiency
            step = max(1, date_range_days // sample_days)
            
            for year in range(start_year, current_year + 1):
                year_temps = []
                for day_offset in range(0, date_range_days, step):
                    current_date = start_date.replace(year=year) + timedelta(days=day_offset)
                    
                    try:
                        weather_data = await get_temperature_series(location, current_date.month, current_date.day)
                        if weather_data and 'data' in weather_data:
                            # Extract missing years from the series metadata
                            if 'metadata' in weather_data and 'missing_years' in weather_data['metadata']:
                                for missing_year_info in weather_data['metadata']['missing_years']:
                                    if missing_year_info['year'] == year:
                                        track_missing_year(missing_years, year, f"{missing_year_info['reason']}_sampling")
                                        break
                            
                            for data_point in weather_data['data']:
                                if int(data_point['x']) == year and data_point['y'] is not None:
                                    year_temps.append(data_point['y'])
                                    break
                    except Exception as e:
                        if DEBUG:
                            logger.debug(f"Error getting data for {current_date}: {e}")
                        track_missing_year(missing_years, year, "sampling_error")
                        continue
                
                # Calculate average for the year (only add one value per year)
                if year_temps:
                    avg_temp = sum(year_temps) / len(year_temps)
                    all_temps.append(avg_temp)  # Add to all_temps only once per year
                    values.append(TemperatureValue(
                        date=end_date.replace(year=year).strftime("%Y-%m-%d"),
                        year=year,
                        temperature=round(avg_temp, 1),
                    ))
                else:
                    track_missing_year(missing_years, year, "no_data_sampling")
    
    elif period == "yearly":
        # For yearly, use the Visual Crossing historysummary endpoint for efficiency
        try:
            if DEBUG:
                logger.debug(f"Fetching yearly summary for {location} from {start_year} to {current_year}")
            yearly_data = await _fetch_yearly_summary(location, start_year, current_year)
            if DEBUG:
                logger.debug(f"Got {len(yearly_data)} yearly data points")
            
            if yearly_data:
                for year, temp in yearly_data:
                    all_temps.append(temp)
                    values.append(TemperatureValue(
                        date=f"{year}-{month:02d}-{day:02d}",
                        year=year,
                        temperature=round(temp, 1),
                    ))
            else:
                if DEBUG:
                    logger.debug("No yearly data returned, falling back to sampling")
                raise Exception("No yearly data returned")
        except Exception as e:
            # Historysummary endpoint often fails (400 errors), fallback is normal - don't log every occurrence
            pass
            # Fallback to simple sampling approach if historysummary fails
            # Just sample a few representative days for each year
            sample_dates = [
                (1, 15), (4, 15), (7, 15), (10, 15)  # Mid-month samples for each season
            ]
            
            for year in range(start_year, current_year + 1):
                year_values = []
                for sample_month, sample_day in sample_dates:
                    try:
                        weather_data = await get_temperature_series(location, sample_month, sample_day)
                        if weather_data and 'data' in weather_data:
                            # Extract missing years from the series metadata
                            if 'metadata' in weather_data and 'missing_years' in weather_data['metadata']:
                                for missing_year_info in weather_data['metadata']['missing_years']:
                                    if missing_year_info['year'] == year:
                                        track_missing_year(missing_years, year, f"{missing_year_info['reason']}_yearly_sampling")
                                        break
                            
                            for data_point in weather_data['data']:
                                if int(data_point['x']) == year and data_point['y'] is not None:
                                    temp = data_point['y']
                                    year_values.append(temp)
                                    all_temps.append(temp)
                                    break
                    except Exception as e:
                        if DEBUG:
                            logger.debug(f"Error getting data for {year}-{sample_month:02d}-{sample_day:02d}: {e}")
                        track_missing_year(missing_years, year, "yearly_sampling_error")
                        continue
                
                # Calculate average for the year
                if year_values:
                    avg_temp = sum(year_values) / len(year_values)
                    values.append(TemperatureValue(
                        date=f"{year}-{month:02d}-{day:02d}",
                        year=year,
                        temperature=round(avg_temp, 1),
                    ))
                else:
                    track_missing_year(missing_years, year, "no_data_yearly_sampling")
    
    # Calculate date range
    if values:
        start_year = min(v.year for v in values)
        end_year = max(v.year for v in values)
        range_data = DateRange(
            start=f"{start_year}-{month:02d}-{day:02d}",
            end=f"{end_year}-{month:02d}-{day:02d}",
            years=end_year - start_year + 1
        )
    else:
        range_data = DateRange(start="", end="", years=0)
    
    # Calculate average
    if all_temps:
        avg_data = AverageData(
            mean=round(sum(all_temps) / len(all_temps), 1),
            unit=unit_group,
            data_points=len(all_temps)
        )
    else:
        avg_data = AverageData(mean=0.0, unit=unit_group, data_points=0)
    
    # Calculate trend
    if len(values) >= 2:
        trend_input = [{"x": v.year, "y": v.temperature} for v in values]
        slope = calculate_trend_slope(trend_input)
        trend_data = TrendData(
            slope=slope,
            unit="°C/decade" if unit_group == "celsius" else "°F/decade",
            data_points=len(values),
            r_squared=None
        )
    else:
        trend_data = TrendData(slope=0.0, unit="°C/decade" if unit_group == "celsius" else "°F/decade", data_points=len(values))
    
    # Generate summary using existing logic
    from datetime import datetime
    end_date = datetime(current_year, month, day)
    
    # Create friendly date based on period
    if period == "daily":
        friendly_date = get_friendly_date(end_date)
    elif period == "weekly":
        friendly_date = f"week ending {get_friendly_date(end_date)}"
    elif period == "monthly":
        friendly_date = f"month ending {get_friendly_date(end_date)}"
    elif period == "yearly":
        friendly_date = f"year ending {get_friendly_date(end_date)}"
    else:
        friendly_date = get_friendly_date(end_date)
    
    # Convert values to the format expected by generate_summary
    summary_data = []
    for value in values:
        summary_data.append({
            'x': value.year,
            'y': value.temperature
        })
    
    # Generate summary text
    summary_text = generate_summary(summary_data, end_date, period)
    
    # Replace the friendly date in the summary with our period-specific version
    summary_text = summary_text.replace(get_friendly_date(end_date), friendly_date)
    
    # Create comprehensive metadata
    additional_metadata = {
        "period_days": date_range_days, 
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    
    return {
        "period": period,
        "location": location,
        "identifier": identifier,
        "range": range_data.model_dump(),
        "unit_group": unit_group,
        "values": [v.model_dump() for v in values],
        "average": avg_data.model_dump(),
        "trend": trend_data.model_dump(),
        "summary": summary_text,
        "metadata": create_metadata(len(years), len(values), missing_years, additional_metadata)
    }

@app.get("/v1/records/{period}/{location}/{identifier}", response_model=RecordResponse)
async def get_record(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name"),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None
):
    """Get temperature record data for a specific period, location, and identifier."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Create cache key for v1 endpoint with comprehensive format
        from cache_utils import normalize_location_for_cache
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:values,average,trend,summary"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 RECORD: {cache_key}")
                data = json.loads(cached_data)
                
                # Validate cached data
                is_valid, error_msg = validate_location_response(data, location)
                if not is_valid:
                    # Mark location as invalid and return error
                    invalid_location_cache.mark_location_invalid(location, "no_data_cached")
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # Add updated timestamp for cached data
                from cache_utils import get_cache_updated_timestamp
                updated_timestamp = await get_cache_updated_timestamp(cache_key, redis_client)
                if updated_timestamp:
                    data["updated"] = updated_timestamp.isoformat()
                
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1")
                return json_response
        
        # Get data
        data = await get_temperature_data_v1(location, period, identifier)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Add updated timestamp for newly computed data
        current_time = datetime.now(timezone.utc)
        data["updated"] = current_time.isoformat()
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION  # Use long cache for historical data
            set_cache_value(cache_key, cache_duration, json.dumps(data), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=data)
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/records/{period}/{location}/{identifier}/average", response_model=SubResourceResponse)
async def get_record_average(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name"),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None
):
    """Get average temperature data for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Create cache key for subresource
        from cache_utils import normalize_location_for_cache
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:average"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records_average", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 AVERAGE: {cache_key}")
                data = json.loads(cached_data)
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|average|metric|v1")
                return json_response
        
        # Get full record data
        record_data = await get_temperature_data_v1(location, period, identifier)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(record_data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Extract average data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["average"],
            metadata=record_data["metadata"]
        )
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION
            set_cache_value(cache_key, cache_duration, json.dumps(response_data.model_dump()), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|average|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records average endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/records/{period}/{location}/{identifier}/trend", response_model=SubResourceResponse)
async def get_record_trend(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name"),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None
):
    """Get temperature trend data for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Create cache key for subresource
        from cache_utils import normalize_location_for_cache
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:trend"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records_trend", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 TREND: {cache_key}")
                data = json.loads(cached_data)
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|trend|metric|v1")
                return json_response
        
        # Get full record data
        record_data = await get_temperature_data_v1(location, period, identifier)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(record_data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Extract trend data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["trend"],
            metadata=record_data["metadata"]
        )
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION
            set_cache_value(cache_key, cache_duration, json.dumps(response_data.model_dump()), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|trend|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records trend endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/records/{period}/{location}/{identifier}/summary", response_model=SubResourceResponse)
async def get_record_summary(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name"),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None
):
    """Get temperature summary text for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Create cache key for subresource
        from cache_utils import normalize_location_for_cache
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:summary"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records_summary", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 SUMMARY: {cache_key}")
                data = json.loads(cached_data)
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|summary|metric|v1")
                return json_response
        
        # Get full record data
        record_data = await get_temperature_data_v1(location, period, identifier)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(record_data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Extract summary data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["summary"],
            metadata=record_data["metadata"]
        )
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION
            set_cache_value(cache_key, cache_duration, json.dumps(response_data.model_dump()), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|summary|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records summary endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/records/{period}/{location}/{identifier}/updated", response_model=UpdatedResponse)
async def get_record_updated(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name"),
    identifier: str = Path(..., description="Date identifier")
):
    """
    Get the last updated timestamp for a specific record endpoint.
    
    Returns when the data was last updated (cached) or null if it's never been queried.
    This endpoint is designed for web apps that want to check if they need to refetch data.
    """
    try:
        # Create the same cache key that would be used by the main endpoint
        from cache_utils import normalize_location_for_cache
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:values,average,trend,summary"
        
        # Import the cache timestamp function
        from cache_utils import get_cache_updated_timestamp
        
        # Get the updated timestamp from cache
        updated_timestamp = await get_cache_updated_timestamp(cache_key, redis_client)
        
        # Determine if data is cached
        is_cached = updated_timestamp is not None
        
        # Format timestamp as ISO string if available
        updated_iso = updated_timestamp.isoformat() if updated_timestamp else None
        
        return UpdatedResponse(
            period=period,
            location=location,
            identifier=identifier,
            updated=updated_iso,
            cached=is_cached,
            cache_key=cache_key
        )
    
    except Exception as e:
        logger.error(f"Error in v1 records updated endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# REMOVED ENDPOINTS (410 Gone)
# ============================================================================

@app.get("/data/{location}/{month_day}")
async def removed_data_endpoint():
    """Legacy data endpoint has been removed. Use /v1/records/daily/{location}/{month_day} instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}",
            "Cache-Control": "no-cache"
        }
    )

@app.get("/average/{location}/{month_day}")
async def removed_average_endpoint():
    """Legacy average endpoint has been removed. Use /v1/records/daily/{location}/{month_day}/average instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}/average",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}/average",
            "Cache-Control": "no-cache"
        }
    )

@app.get("/trend/{location}/{month_day}")
async def removed_trend_endpoint():
    """Legacy trend endpoint has been removed. Use /v1/records/daily/{location}/{month_day}/trend instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}/trend",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}/trend",
            "Cache-Control": "no-cache"
        }
    )

@app.get("/summary/{location}/{month_day}")
async def removed_summary_endpoint():
    """Legacy summary endpoint has been removed. Use /v1/records/daily/{location}/{month_day}/summary instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}/summary",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}/summary",
            "Cache-Control": "no-cache"
        }
    )

@app.get("/protected-endpoint")
def protected_route(user=Depends(verify_firebase_token)):
    """Protected endpoint that requires Firebase authentication."""
    return {"message": "You are authenticated!", "user": user}

# Analytics Endpoints
@app.post("/analytics", response_model=AnalyticsResponse)
async def submit_analytics(request: Request):
    """Submit client analytics data for monitoring and error tracking."""
    client_ip = get_client_ip(request)
    
    try:
        # Log request details for debugging
        content_type = request.headers.get("content-type", "")
        content_length = request.headers.get("content-length", "unknown")
        
        logger.info(f"📊 ANALYTICS REQUEST: IP={client_ip} | Content-Type={content_type} | Length={content_length}")
        
        # Validate content type
        if not content_type.startswith("application/json"):
            logger.warning(f"⚠️  ANALYTICS INVALID CONTENT-TYPE: {content_type} | IP={client_ip}")
            raise HTTPException(
                status_code=415, 
                detail="Content-Type must be application/json"
            )
        
        # Check content length (limit to 1MB for analytics data)
        max_content_length = 1024 * 1024  # 1MB
        try:
            content_length_int = int(content_length) if content_length != "unknown" else None
            if content_length_int and content_length_int > max_content_length:
                logger.warning(f"⚠️  ANALYTICS REQUEST TOO LARGE: {content_length_int} bytes | IP={client_ip}")
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Maximum size is {max_content_length} bytes"
                )
        except ValueError:
            # If we can't parse content-length, continue but log it
            logger.warning(f"⚠️  ANALYTICS INVALID CONTENT-LENGTH: {content_length} | IP={client_ip}")
        
        # Read and parse request body
        try:
            body = await request.body()
            if not body:
                logger.warning(f"⚠️  ANALYTICS EMPTY BODY: IP={client_ip}")
                raise HTTPException(
                    status_code=400,
                    detail="Request body cannot be empty"
                )
            
            # Check actual body size
            if len(body) > max_content_length:
                logger.warning(f"⚠️  ANALYTICS BODY TOO LARGE: {len(body)} bytes | IP={client_ip}")
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Maximum size is {max_content_length} bytes"
                )
            
            # Log request body for debugging (truncated for security)
            body_str = body.decode('utf-8')
            body_preview = body_str[:500] + "..." if len(body_str) > 500 else body_str
            logger.info(f"📊 ANALYTICS BODY PREVIEW: {body_preview}")
            
            # Parse JSON
            try:
                json_data = json.loads(body_str)
            except json.JSONDecodeError as e:
                logger.error(f"❌ ANALYTICS JSON PARSE ERROR: {e} | IP={client_ip} | Body: {body_preview}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON format: {str(e)}"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ ANALYTICS BODY READ ERROR: {e} | IP={client_ip}")
            raise HTTPException(
                status_code=400,
                detail="Failed to read request body"
            )
        
        # Validate data using Pydantic model
        try:
            analytics_data = AnalyticsData(**json_data)
        except Exception as e:
            logger.error(f"❌ ANALYTICS VALIDATION ERROR: {e} | IP={client_ip} | Data: {json_data}")
            
            # Provide detailed validation error information
            if hasattr(e, 'errors'):
                error_details = []
                for error in e.errors():
                    field = " -> ".join(str(loc) for loc in error['loc'])
                    error_details.append(f"{field}: {error['msg']}")
                error_message = f"Validation failed: {'; '.join(error_details)}"
            else:
                error_message = f"Validation failed: {str(e)}"
            
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Validation Error",
                    "message": error_message,
                    "details": str(e) if hasattr(e, 'errors') else None
                }
            )
        
        # Store analytics data
        try:
            analytics_id = analytics_storage.store_analytics(analytics_data, client_ip)
            logger.info(f"📊 ANALYTICS STORED: {analytics_id} | IP={client_ip}")
        except Exception as e:
            logger.error(f"❌ ANALYTICS STORAGE ERROR: {e} | IP={client_ip}")
            raise HTTPException(
                status_code=500,
                detail="Failed to store analytics data"
            )
        
        return AnalyticsResponse(
            status="success",
            message="Analytics data submitted successfully",
            analytics_id=analytics_id,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper status codes)
        raise
    except Exception as e:
        logger.error(f"❌ ANALYTICS UNEXPECTED ERROR: {e} | IP={client_ip}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while processing analytics data"
        )

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary statistics."""
    try:
        summary = analytics_storage.get_analytics_summary()
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics summary")

@app.get("/analytics/recent")
async def get_recent_analytics(limit: int = Query(100, ge=1, le=1000)):
    """Get recent analytics records."""
    try:
        analytics_records = analytics_storage.get_recent_analytics(limit)
        return {
            "status": "success",
            "data": analytics_records,
            "count": len(analytics_records),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recent analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent analytics")

@app.get("/analytics/session/{session_id}")
async def get_analytics_by_session(session_id: str):
    """Get analytics records for a specific session."""
    try:
        session_analytics = analytics_storage.get_analytics_by_session(session_id)
        return {
            "status": "success",
            "data": session_analytics,
            "count": len(session_analytics),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting session analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session analytics")

@app.get("/debug/jobs")
async def debug_jobs_endpoint():
    """Debug endpoint to check job queue and job data in Redis."""
    try:
        debug_info = {
            "queue_length": 0,
            "jobs_in_queue": [],
            "job_data_status": {},
            "redis_connection": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test Redis connection
        try:
            redis_client.ping()
            debug_info["redis_connection"] = "OK"
        except Exception as e:
            debug_info["redis_connection"] = f"FAILED: {e}"
            return debug_info
        
        # Check job queue
        queue_key = "job_queue"
        queue_length = redis_client.llen(queue_key)
        debug_info["queue_length"] = queue_length
        
        if queue_length > 0:
            jobs_in_queue = []
            for i in range(min(queue_length, 10)):
                job_id = redis_client.lindex(queue_key, i)
                if job_id:
                    # Convert bytes to string if needed
                    if isinstance(job_id, bytes):
                        job_id = job_id.decode('utf-8')
                    
                    jobs_in_queue.append(job_id)
                    
                    # Check if job data exists
                    job_key = f"job:{job_id}"
                    job_data = redis_client.get(job_key)
                    
                    if job_data:
                        try:
                            job = json.loads(job_data)
                            status = job.get("status", "unknown")
                            created = job.get("created_at", "unknown")
                            debug_info["job_data_status"][job_id] = {
                                "exists": True,
                                "status": status,
                                "created_at": created
                            }
                        except:
                            debug_info["job_data_status"][job_id] = {
                                "exists": True,
                                "error": "invalid JSON"
                            }
                    else:
                        debug_info["job_data_status"][job_id] = {
                            "exists": False
                        }
            
            debug_info["jobs_in_queue"] = jobs_in_queue
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error in debug jobs endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

# Async Job Endpoints for Heavy Operations
@app.post("/v1/records/{period}/{location}/{identifier}/async")
async def create_record_job(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name"),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None
):
    """Create an async job to compute heavy record data."""
    try:
        logger.info(f"Creating async job: period={period}, location={location}, identifier={identifier}")
        job_manager = get_job_manager()
        logger.info(f"Job manager retrieved successfully")
        
        # Create job
        job_id = job_manager.create_job("record_computation", {
            "period": period,
            "location": location,
            "identifier": identifier
        })
        logger.info(f"Job created successfully: {job_id}")
        
        # Return 202 Accepted with job info
        response.status_code = 202
        response.headers["Retry-After"] = "3"
        
        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Job created successfully",
            "retry_after": 3,
            "status_url": f"/v1/jobs/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating record job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an async job."""
    try:
        job_manager = get_job_manager()
        job_status = job_manager.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job status")

@app.post("/v1/records/rolling-bundle/{location}/{anchor}/async")
async def create_rolling_bundle_job(
    location: str = Path(..., description="Location name"),
    anchor: str = Path(..., description="Anchor date"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius"),
    month_mode: Literal["calendar", "rolling1m", "rolling30d"] = Query("rolling1m"),
    days_back: int = Query(7, ge=0, le=10),
    include: str = Query(None),
    exclude: str = Query(None),
    response: Response = None
):
    """Create an async job to compute rolling bundle data."""
    try:
        job_manager = get_job_manager()
        
        # Create job
        job_id = job_manager.create_job("rolling_bundle", {
            "location": location,
            "anchor": anchor,
            "unit_group": unit_group,
            "month_mode": month_mode,
            "days_back": days_back,
            "include": include,
            "exclude": exclude
        })
        
        # Return 202 Accepted with job info
        response.status_code = 202
        response.headers["Retry-After"] = "5"
        
        return {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "message": "Rolling bundle job created successfully",
            "retry_after": 5,
            "status_url": f"/v1/jobs/{job_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating rolling bundle job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create job")

@app.get("/v1/jobs/diagnostics/worker-status")
async def get_worker_diagnostics():
    """Get diagnostic information about the background worker and job queue."""
    try:
        # Check worker heartbeat
        heartbeat = redis_client.get("worker:heartbeat")
        worker_alive = heartbeat is not None
        
        # Get queue status
        queue_length = redis_client.llen("job_queue")
        
        # Get jobs from the queue (without KEYS command)
        # We'll examine jobs in the queue since we can't scan all keys
        jobs_by_status = {
            "pending": 0,
            "processing": 0,
            "ready": 0,
            "error": 0
        }
        
        stuck_jobs = []
        jobs_examined = []
        
        # Get all job IDs from the queue
        if queue_length > 0:
            # Get up to 100 jobs from the queue
            for i in range(min(queue_length, 100)):
                job_id = redis_client.lindex("job_queue", i)
                if job_id:
                    if isinstance(job_id, bytes):
                        job_id = job_id.decode('utf-8')
                    jobs_examined.append(job_id)
        
        # Examine each job in the queue
        for job_id in jobs_examined:
            try:
                job_key = f"job:{job_id}"
                job_data = redis_client.get(job_key)
                if job_data:
                    if isinstance(job_data, bytes):
                        job_data = job_data.decode('utf-8')
                    job = json.loads(job_data)
                    status = job.get("status", "unknown")
                    
                    if status in jobs_by_status:
                        jobs_by_status[status] += 1
                    
                    # Check for stuck jobs (older than 5 minutes in pending/processing)
                    created = job.get("created_at")
                    if created and status in ["pending", "processing"]:
                        try:
                            from datetime import datetime, timezone
                            created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                            if age > 300:  # 5 minutes
                                stuck_jobs.append({
                                    "job_id": job.get("id"),
                                    "status": status,
                                    "age_seconds": int(age),
                                    "type": job.get("type"),
                                    "params": job.get("params", {})
                                })
                        except:
                            pass
            except Exception as job_error:
                logger.warning(f"Error examining job {job_id}: {job_error}")
        
        # Parse heartbeat time if available
        heartbeat_age = None
        if heartbeat:
            try:
                if isinstance(heartbeat, bytes):
                    heartbeat = heartbeat.decode('utf-8')
                from datetime import datetime, timezone
                heartbeat_dt = datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))
                heartbeat_age = (datetime.now(timezone.utc) - heartbeat_dt).total_seconds()
            except:
                pass
        
        return {
            "worker": {
                "alive": worker_alive,
                "heartbeat": heartbeat,
                "heartbeat_age_seconds": heartbeat_age,
                "status": "healthy" if worker_alive and (heartbeat_age is None or heartbeat_age < 60) else "unhealthy"
            },
            "queue": {
                "length": queue_length,
                "jobs_examined": len(jobs_examined)
            },
            "jobs": {
                "by_status": jobs_by_status,
                "stuck_count": len(stuck_jobs),
                "stuck_jobs": stuck_jobs[:10]  # Show first 10 stuck jobs
            },
            "recommendations": get_diagnostics_recommendations(
                worker_alive, 
                heartbeat_age, 
                queue_length, 
                jobs_by_status,
                len(stuck_jobs)
            ),
            "note": "Only examining jobs in the queue (Redis KEYS command not available)"
        }
    except Exception as e:
        logger.error(f"Error getting worker diagnostics: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get diagnostics: {str(e)}")

def get_diagnostics_recommendations(worker_alive, heartbeat_age, queue_length, jobs_by_status, stuck_count):
    """Generate diagnostic recommendations based on worker and job status."""
    recommendations = []
    
    if not worker_alive:
        recommendations.append({
            "severity": "critical",
            "issue": "Background worker is not running",
            "actions": [
                "Check server logs for worker startup errors",
                "Restart the API server",
                "Verify Redis connection is available"
            ]
        })
    elif heartbeat_age and heartbeat_age > 60:
        recommendations.append({
            "severity": "warning",
            "issue": f"Worker heartbeat is stale ({int(heartbeat_age)}s old)",
            "actions": [
                "Worker may be stuck or crashed",
                "Check server logs for errors",
                "Consider restarting the API server"
            ]
        })
    
    if jobs_by_status.get("pending", 0) > 0 and worker_alive:
        recommendations.append({
            "severity": "info",
            "issue": f"{jobs_by_status['pending']} jobs in pending state",
            "actions": [
                "Jobs are waiting to be processed",
                "Worker should process these shortly",
                "If they remain pending for >1 minute, check worker logs"
            ]
        })
    
    if stuck_count > 0:
        recommendations.append({
            "severity": "warning",
            "issue": f"{stuck_count} jobs stuck for >5 minutes",
            "actions": [
                "Check server logs for processing errors",
                "These jobs may need to be manually cleared",
                "Use diagnose_jobs.py --clear-stuck to clear them"
            ]
        })
    
    if jobs_by_status.get("error", 0) > 0:
        recommendations.append({
            "severity": "warning",
            "issue": f"{jobs_by_status['error']} jobs failed with errors",
            "actions": [
                "Check individual job status for error details",
                "Common causes: API errors, timeouts, invalid parameters"
            ]
        })
    
    if not recommendations:
        recommendations.append({
            "severity": "success",
            "issue": "System is healthy",
            "actions": ["No action needed"]
        })
    
    return recommendations

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
