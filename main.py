# Standard library imports
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, date as dt_date
from typing import List, Dict, Optional, Set
from collections import defaultdict
import time

# Third-party imports
import aiohttp
import firebase_admin
import redis
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from firebase_admin import auth, credentials

load_dotenv()

# Environment variables
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CACHE_CONTROL_HEADER = "public, max-age=14400, immutable"
FILTER_WEATHER_DATA = os.getenv("FILTER_WEATHER_DATA", "true").lower() == "true"

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
MAX_LOCATIONS_PER_HOUR = int(os.getenv("MAX_LOCATIONS_PER_HOUR", "10"))
MAX_REQUESTS_PER_HOUR = int(os.getenv("MAX_REQUESTS_PER_HOUR", "100"))
RATE_LIMIT_WINDOW_HOURS = int(os.getenv("RATE_LIMIT_WINDOW_HOURS", "1"))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('temphist.log') if DEBUG else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Initialize rate limiting monitors
if RATE_LIMIT_ENABLED:
    location_monitor = LocationDiversityMonitor(MAX_LOCATIONS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS)
    request_monitor = RequestRateMonitor(MAX_REQUESTS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS)
    if DEBUG:
        logger.info(f"🛡️  RATE LIMITING INITIALIZED: {MAX_LOCATIONS_PER_HOUR} locations/hour, {MAX_REQUESTS_PER_HOUR} requests/hour, {RATE_LIMIT_WINDOW_HOURS}h window")
    else:
        logger.info(f"Rate limiting enabled: max {MAX_LOCATIONS_PER_HOUR} locations, max {MAX_REQUESTS_PER_HOUR} requests per {RATE_LIMIT_WINDOW_HOURS} hour(s)")
else:
    location_monitor = None
    request_monitor = None
    if DEBUG:
        logger.info("⚠️  RATE LIMITING DISABLED")
    else:
        logger.info("Rate limiting disabled")

# API Configuration
VISUAL_CROSSING_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
VISUAL_CROSSING_UNIT_GROUP = "metric"
VISUAL_CROSSING_INCLUDE_PARAMS = "days"
VISUAL_CROSSING_REMOTE_DATA = "options=useremote&forecastDataset=era5core"

def build_visual_crossing_url(location: str, date: str, remote: bool = True) -> str:
    """Build Visual Crossing API URL with consistent parameters.
    
    Args:
        location: The location to get weather data for
        date: The date in YYYY-MM-DD format
        remote: Whether to include remote data parameters (default: True)
    """
    base_params = f"unitGroup={VISUAL_CROSSING_UNIT_GROUP}&include={VISUAL_CROSSING_INCLUDE_PARAMS}&key={API_KEY}"
    if remote:
        return f"{VISUAL_CROSSING_BASE_URL}/{location}/{date}?{base_params}&{VISUAL_CROSSING_REMOTE_DATA}"
    else:
        return f"{VISUAL_CROSSING_BASE_URL}/{location}/{date}?{base_params}"

# Cache durations
SHORT_CACHE_DURATION = timedelta(hours=1)  # For today's data
LONG_CACHE_DURATION = timedelta(hours=168)  # 1 week for historical data

# Remove the old debug_print function - we'll use logger.debug() instead

app = FastAPI()
redis_client = redis.from_url(REDIS_URL)

# check Firebase credentials
try:
    cred = credentials.Certificate("firebase-service-account.json")  # Download from Firebase Console
    firebase_admin.initialize_app(cred)
except ValueError:
    # Firebase app already initialized, skip
    pass

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
    """Log all requests when DEBUG is enabled."""
    if DEBUG:
        start_time = time.time()
        client_ip = get_client_ip(request)
        
        # Log request details
        logger.info(f"🌐 REQUEST: {request.method} {request.url.path} | IP: {client_ip} | User-Agent: {request.headers.get('user-agent', 'Unknown')}")
        
        # Process request
        response = await call_next(request)
        
        # Log response details
        process_time = time.time() - start_time
        logger.info(f"✅ RESPONSE: {response.status_code} | {request.method} {request.url.path} | {process_time:.3f}s | IP: {client_ip}")
        
        return response
    else:
        # Skip logging in production
        return await call_next(request)

@app.middleware("http")
async def verify_token_middleware(request: Request, call_next):
    """Middleware to verify Firebase tokens and apply rate limiting for protected routes."""
    logger.info(f"[DEBUG] Middleware: Processing {request.method} request to {request.url.path}")
    
    # Allow OPTIONS requests for CORS preflight
    if request.method == "OPTIONS":
        logger.info(f"[DEBUG] Middleware: OPTIONS request, allowing through")
        return await call_next(request)

    # Get client IP for rate limiting
    client_ip = get_client_ip(request)
    logger.info(f"[DEBUG] Middleware: Client IP: {client_ip}")

    # Apply rate limiting if enabled
    if RATE_LIMIT_ENABLED and location_monitor and request_monitor:
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
        
        # For weather-related endpoints, check location diversity
        weather_paths = ["/weather/", "/data/", "/summary/", "/trend/", "/average/", "/forecast/"]
        if any(request.url.path.startswith(path) for path in weather_paths):
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

    # Public paths that don't require a token
    public_paths = ["/", "/docs", "/openapi.json", "/redoc", "/test-cors", "/test-redis", "/rate-limit-status", "/rate-limit-stats"]
    if request.url.path in public_paths or any(request.url.path.startswith(p) for p in ["/static"]):
        logger.info(f"[DEBUG] Middleware: Public path, allowing through")
        return await call_next(request)

    logger.info(f"[DEBUG] Middleware: Protected path, checking Firebase token...")
    # All other paths require a Firebase token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.info(f"[DEBUG] Middleware: No valid Authorization header")
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header."}
        )

    id_token = auth_header.split(" ")[1]
    
    # Special bypass for testing
    if id_token == "test_token":
        logger.info(f"[DEBUG] Middleware: Using test token bypass")
        request.state.user = {"uid": "testuser"}
    else:
        logger.info(f"[DEBUG] Middleware: Verifying Firebase token...")
        try:
            decoded_token = auth.verify_id_token(id_token)
            logger.info(f"[DEBUG] Middleware: Firebase token verified successfully")
            # Optionally, attach user info to request.state
            request.state.user = decoded_token
        except Exception as e:
            logger.error(f"[DEBUG] Middleware: Firebase token verification failed: {e}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid Firebase token."}
            )

    logger.info(f"[DEBUG] Middleware: Token verified, calling next handler...")
    response = await call_next(request)
    logger.info(f"[DEBUG] Middleware: Response received, returning")
    return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite default port
        "https://temphist.onrender.com",  # Render frontend
        "https://*.onrender.com",  # Any Render subdomain
        "https://temphist.com",  # Main domain
        "https://www.temphist.com",  # www subdomain
        "https://dev.temphist.com",  # development site
    ],
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

@app.api_route("/", methods=["GET", "OPTIONS"])
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "Temperature History API",
        "version": "1.0.0",
        "endpoints": [
            "/data/{location}/{month_day}",
            "/average/{location}/{month_day}",
            "/trend/{location}/{month_day}",
            "/weather/{location}/{date}",
            "/summary/{location}/{month_day}",
            "/forecast/{location}",
            "/rate-limit-status",
            "/rate-limit-stats"
        ]
    }

def fetch_weather_from_api(location: str, date: str):
    """Fetch weather data from Visual Crossing API with fallback logic.
    
    Implements fallback logic for remote data:
    1. First tries without remote data parameters
    2. If no temperature data and year >= 2005, retries with remote data parameters
    3. Never uses remote data for today's data
    """
    # Parse the date to determine the year
    try:
        year, month, day = map(int, date.split("-")[:3])
        is_today_date = is_today(year, month, day)
    except Exception:
        year = 2000  # Default fallback
        is_today_date = False
    
    # First attempt: without remote data parameters
    url = build_visual_crossing_url(location, date, remote=False)
    if DEBUG:
        logger.debug(f"🌤️  API CALL: {location} | {date} | Remote: False")
        logger.debug(f"🔗 URL: {url}")
    
    response = requests.get(url)
    if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
        data = response.json()
        days = data.get('days')
        if days is not None and len(days) > 0:
            # Check if we got valid temperature data
            day_data = days[0]
            temp = day_data.get('temp')
            
            if temp is not None:
                # Success! Return the data
                logger.debug(f"fetch_weather_from_api successful without remote data for {date}")
                return data
            else:
                logger.debug(f"No temperature data in first attempt for {date}")
        else:
            logger.debug(f"No 'days' data in first attempt for {date}")
    else:
        logger.debug(f"First attempt failed with status {response.status_code}")
    
    # If we reach here, the first attempt didn't provide valid temperature data
    # Check if we should try with remote data parameters
    if not is_today_date and year >= 2005:
        if DEBUG:
            logger.debug(f"🔄 API FALLBACK: {location} | {date} | Year: {year}")
        url_with_remote = build_visual_crossing_url(location, date, remote=True)
        
        remote_response = requests.get(url_with_remote)
        if remote_response.status_code == 200 and 'application/json' in remote_response.headers.get('Content-Type', ''):
            remote_data = remote_response.json()
            remote_days = remote_data.get('days')
            if remote_days is not None and len(remote_days) > 0:
                remote_day_data = remote_days[0]
                remote_temp = remote_day_data.get('temp')
                
                if remote_temp is not None:
                    # Success with remote data!
                    logger.debug(f"Remote data fallback successful for {date}")
                    return remote_data
                else:
                    logger.debug(f"No temperature data in remote fallback for {date}")
            else:
                logger.debug(f"No 'days' data in remote fallback for {date}")
        else:
            logger.debug(f"Remote fallback failed with status {remote_response.status_code}")
    
    # If we reach here, neither attempt provided valid temperature data
    return {"error": "No temperature data available", "status": response.status_code}

# get a value from the cache
def get_cache(cache_key):
    """Get a value from the cache."""
    if DEBUG:
        logger.debug(f"🔍 CACHE GET: {cache_key}")
    result = redis_client.get(cache_key)
    if DEBUG and result:
        logger.debug(f"✅ CACHE HIT: {cache_key}")
    elif DEBUG:
        logger.debug(f"❌ CACHE MISS: {cache_key}")
    return result

# set a value in the cache
def set_cache(cache_key, lifetime, value):
    """Set a value in the cache with specified lifetime."""
    if DEBUG:
        logger.debug(f"💾 CACHE SET: {cache_key} | TTL: {lifetime}")
    redis_client.setex(cache_key, lifetime, value)

# Shared async function for fetching and caching weather data for a single date
def get_weather_cache_key(location: str, date_str: str) -> str:
    """Generate cache key for weather data."""
    return f"{location.lower()}_{date_str}"

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
        cached_data = get_cache(cache_key)
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
        timeout = aiohttp.ClientTimeout(total=30)
        logger.info(f"[DEBUG] Creating aiohttp session with 30-second timeout")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(f"[DEBUG] Session created successfully, making GET request to: {url}")
            async with session.get(url) as resp:
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
                                set_cache(cache_key, cache_duration, json.dumps(to_cache))
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
                    async with session.get(url_with_remote) as remote_resp:
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
                                        set_cache(cache_key, LONG_CACHE_DURATION, json.dumps(to_cache))
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
def get_weather(location: str, date: str):
    """Get weather data for a specific location and date."""
    logger.info(f"[DEBUG] Weather endpoint called with location={location}, date={date}")
    
    cache_key = f"{location.lower()}_{date}"
    logger.info(f"[DEBUG] Cache key: {cache_key}")
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        if DEBUG:
            logger.info(f"🔍 CACHE CHECK: {cache_key}")
        cached_data = get_cache(cache_key)
        if cached_data:
            if DEBUG:
                logger.info(f"✅ SERVING CACHED DATA: {cache_key} | Location: {location} | Date: {date}")
            return json.loads(cached_data)
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
    
    # Only cache successful results if caching is enabled and not already cached
    if CACHE_ENABLED and "error" not in result:
        try:
            year, month, day = map(int, date.split("-")[:3])
            cache_duration = SHORT_CACHE_DURATION if is_today_or_future(year, month, day) else LONG_CACHE_DURATION
        except Exception:
            cache_duration = LONG_CACHE_DURATION
        set_cache(cache_key, cache_duration, json.dumps(result))
    return result

# get a text summary
@app.get("/summary/{location}/{month_day}")
async def summary(location: str, month_day: str, request: Request):
    """Get a text summary of temperature data for a location and date."""
    try:
        month, day = map(int, month_day.split("-"))
        today = datetime.now()
        current_year = today.year
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month-day format. Use MM-DD.")

    data = await get_temperature_series(location, month, day)
    logger.debug(f"Summary data for {location} on {month_day}:")
    logger.debug(f"Number of data points: {len(data['data'])}")
    if data['data']:
        logger.debug(f"Date range: {data['data'][0]['x']} to {data['data'][-1]['x']}")
    else:
        logger.debug("No data available for date range")
    current_temp = data['data'][-1]['y']
    if current_temp is not None:
        logger.debug(f"Current temperature: {current_temp}°C")
    else:
        logger.debug("Current temperature: None")
    
    if len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    data_list = sorted(data['data'], key=lambda d: d['x'])
    avg_temp = calculate_historical_average(data_list)
    logger.debug(f"Calculated average: {avg_temp}°C")
    if data_list[-1]['y'] is not None:
        logger.debug(f"Temperature difference: {round(data_list[-1]['y'] - avg_temp, 1)}°C")
    else:
        logger.debug("Temperature difference: Cannot calculate (current temperature is None)")

    summary_text = await get_summary(location, month_day, data)
    return JSONResponse(content={"summary": summary_text}, headers={"Cache-Control": CACHE_CONTROL_HEADER})

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

async def get_summary(location: str, month_day: str, weather_data: Optional[List[Dict]] = None) -> str:
    def get_friendly_date(date: datetime) -> str:
        day = date.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{day}{suffix} {date.strftime('%B')}"

    def generate_summary(data: List[Dict[str, float]], date: datetime) -> str:
        # Filter out data points with None temperature
        data = [d for d in data if d.get('y') is not None]
        if not data or len(data) < 2:
            return "Not enough data to generate summary."

        latest = data[-1]
        if latest.get('y') is None:
            return "No valid temperature data for the latest year."

        avg_temp = calculate_historical_average(data)
        diff = latest['y'] - avg_temp
        rounded_diff = round(diff, 1)

        friendly_date = get_friendly_date(date)
        warm_summary = ''
        cold_summary = ''
        temperature = f"{latest['y']}°C."

        previous = [p for p in data[:-1] if p.get('y') is not None]
        is_warmest = all(latest['y'] >= p['y'] for p in previous)
        is_coldest = all(latest['y'] <= p['y'] for p in previous)

        # Check against last year first for consistency
        last_year_temp = next((p['y'] for p in reversed(previous) if p['x'] == latest['x'] - 1), None)
        
        # Generate mutually exclusive summaries to avoid contradictions
        if is_warmest:
            warm_summary = f"This is the warmest {friendly_date} on record."
        elif is_coldest:
            cold_summary = f"This is the coldest {friendly_date} on record."
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
                        warm_summary = f"This is the warmest {friendly_date} since {last_warmer}."
                    else:
                        warm_summary = f"This is the warmest {friendly_date} in {years_since} years."
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
                        cold_summary = f"This is the coldest {friendly_date} since {last_colder}."
                    else:
                        cold_summary = f"This is the coldest {friendly_date} in {years_since} years."
                else:
                    cold_summary = f"It's colder than last year."
            # If equal to last year, no warm/cold summary is generated

        if abs(diff) < 0.05:
            avg_summary = "It is about average for this date."
        elif diff > 0:
            avg_summary = "However, it is " if cold_summary else "It is "
            avg_summary += f"{rounded_diff}°C warmer than average today."
        else:
            avg_summary = "However, it is " if warm_summary else "It is "
            avg_summary += f"{abs(rounded_diff)}°C cooler than average today."

        return " ".join(filter(None, [temperature, warm_summary, cold_summary, avg_summary]))

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
        return generate_summary(weather_data, date)

    except Exception as e:
        logger.error(f"Error in get_summary: {e}")
        return "Error generating summary."

# get the warming/cooling trend
@app.get("/trend/{location}/{month_day}")
async def trend(location: str, month_day: str):
    """Get the temperature trend (warming/cooling) for a location and date."""
    try:
        month, day = map(int, month_day.split("-"))
        # Validate month and day
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        # Additional validation for specific months
        if month in [4, 6, 9, 11] and day > 30:
            raise ValueError(f"Month {month} has only 30 days")
        if month == 2 and day > 29:
            raise ValueError("February has only 29 days")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create cache key for the trend
    cache_key = f"trend_{location.lower()}_{month:02d}_{day:02d}"
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = get_cache(cache_key)
        if cached_data:
            logger.debug(f"Cache hit: {cache_key}")
            return JSONResponse(content=json.loads(cached_data), headers={"Cache-Control": CACHE_CONTROL_HEADER})
        logger.debug(f"Cache miss: {cache_key} — calculating trend")

    data = await get_temperature_series(location, month, day)

    if len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    slope = calculate_trend_slope(data['data'])
    is_complete = data['metadata']['completeness'] == 100
    
    result = {
        "slope": slope, 
        "units": "°C/decade",
        "data_points": len(data['data']),
        "completeness": data['metadata']['completeness'],
        "missing_years": data['metadata']['missing_years']
    }
    
    headers = {"Cache-Control": CACHE_CONTROL_HEADER if is_complete else "no-store"}

    # Cache the result if caching is enabled and data is complete
    if CACHE_ENABLED and is_complete:
        set_cache(cache_key, SHORT_CACHE_DURATION, json.dumps(result))
    
    return JSONResponse(content=result, headers=headers)

# get the average temperature
@app.api_route("/average/{location}/{month_day}", methods=["GET", "OPTIONS"])
async def average(location: str, month_day: str):
    """Get the historical average temperature for a location and date."""
    try:
        month, day = map(int, month_day.split("-"))
        # Validate month and day
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        # Additional validation for specific months
        if month in [4, 6, 9, 11] and day > 30:
            raise ValueError(f"Month {month} has only 30 days")
        if month == 2 and day > 29:
            raise ValueError("February has only 29 days")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create cache key for the average
    cache_key = f"average_{location.lower()}_{month:02d}_{day:02d}"
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = get_cache(cache_key)
        if cached_data:
            logger.debug(f"Cache hit: {cache_key}")
            return JSONResponse(content=json.loads(cached_data), headers={"Cache-Control": CACHE_CONTROL_HEADER})
        logger.debug(f"Cache miss: {cache_key} — calculating average")

    data = await get_temperature_series(location, month, day)

    if len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    is_complete = data['metadata']['completeness'] == 100

    # Calculate average temperature using the new function
    data_list = sorted(data['data'], key=lambda d: d['x'])
    avg_temp = calculate_historical_average(data_list)
    
    result = {
        "average": avg_temp,
        "unit": "celsius",
        "data_points": len(data['data']),
        "year_range": {
            "start": data_list[0]['x'],
            "end": data_list[-1]['x']
        },
        "missing_years": data['metadata']['missing_years'],
        "completeness": data['metadata']['completeness']
    }

    headers = {"Cache-Control": CACHE_CONTROL_HEADER if is_complete else "no-store"}

    # Cache the result if caching is enabled and data is complete
    if CACHE_ENABLED and is_complete:
        set_cache(cache_key, SHORT_CACHE_DURATION, json.dumps(result))
    
    return JSONResponse(content=result, headers=headers)

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
    """Get weather forecast for a location."""
    today = datetime.now().date()
    return get_forecast_data(location, today)

@app.get("/test-redis")
async def test_redis():
    """Test Redis connection."""
    try:
        # Try to set a test value
        set_cache("test_key", timedelta(minutes=5), "test_value")
        # Try to get the test value
        test_value = get_cache("test_key")
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
    
    location_stats = location_monitor.get_stats(client_ip) if location_monitor else {}
    request_stats = request_monitor.get_stats(client_ip) if request_monitor else {}
    
    return {
        "client_ip": client_ip,
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
                "suspicious": location_monitor.is_suspicious(ip)
            }
    
    return {
        "total_monitored_ips": len(all_stats),
        "suspicious_ips": list(location_monitor.suspicious_ips) if location_monitor else [],
        "ip_details": all_stats
    }

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

@app.get("/data/{location}/{month_day}")
async def get_all_data(location: str, month_day: str):
    """Get all temperature data, summary, trend, and average for a location and date."""
    logger.debug(f"get_all_data for {location} on {month_day}")
    try:
        month, day = map(int, month_day.split("-"))
        weather_data = await get_temperature_series(location, month, day)
        if DEBUG:
            logger.debug(f"📊 WEATHER DATA SERIES: {location} | {month_day}")
        if not weather_data['data']:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "No valid temperature data found for this location and date."
                },
                headers={"Cache-Control": "public, max-age=60"},
                status_code=404
            )
        is_complete = weather_data['metadata']['completeness'] == 100
        headers = {"Cache-Control": CACHE_CONTROL_HEADER if is_complete else "no-store"}
        
        summary_data = None
        average_data = None

        # Determine if the latest data point is today
        latest_year = None
        if weather_data['data']:
            latest_year = weather_data['data'][-1]['x']
        cache_is_today = False
        if latest_year is not None:
            cache_is_today = is_today(latest_year, month, day)
        # Set cache durations using global constants
        cache_duration = SHORT_CACHE_DURATION if cache_is_today else LONG_CACHE_DURATION

        if is_complete and CACHE_ENABLED:
            # Check cache for summary
            summary_cache_key = f"summary_{location.lower()}_{month:02d}_{day:02d}"
            cached_summary = get_cache(summary_cache_key)
            if cached_summary:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED SUMMARY: {summary_cache_key}")
                summary_data = json.loads(cached_summary)
            else:
                if DEBUG:
                    logger.debug(f"🔄 GENERATING FRESH SUMMARY: {location} | {month_day}")
                summary_data = await get_summary(location, month_day, weather_data)
                set_cache(summary_cache_key, cache_duration, json.dumps(summary_data))

            # Check cache for average
            average_cache_key = f"average_{location.lower()}_{month:02d}_{day:02d}"
            cached_average = get_cache(average_cache_key)
            if cached_average:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED AVERAGE: {average_cache_key}")
                average_data = json.loads(cached_average)
            else:
                if DEBUG:
                    logger.debug(f"🔄 GENERATING FRESH AVERAGE: {location} | {month_day}")
                average_data = get_average_dict(weather_data)
                set_cache(average_cache_key, cache_duration, json.dumps(average_data))
        else:
            # If data is incomplete or cache is disabled, compute without caching
            if DEBUG:
                logger.debug(f"⚠️  CACHE DISABLED OR INCOMPLETE DATA: computing fresh summary and average")
            summary_data = await get_summary(location, month_day, weather_data)
            average_data = get_average_dict(weather_data)
        
        if DEBUG:
            logger.debug(f"🔄 GENERATING TREND DATA: {location} | {month_day}")
        trend_data = await get_trend(location, month_day, weather_data)

        # Log the final aggregated response
        final_response = {
            "weather": weather_data,
            "summary": summary_data,
            "trend": trend_data,
            "average": average_data
        }
        
        if DEBUG:
            logger.debug(f"🎯 FINAL /DATA ENDPOINT RESPONSE: {location} | {month_day}")
            logger.debug(f"📄 COMPLETE RESPONSE JSON: {json.dumps(final_response, indent=2)}")

        return JSONResponse(
            content=final_response,
            headers=headers
        )
    except HTTPException as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e.detail)
            },
            headers={"Cache-Control": "public, max-age=60"},
            status_code=e.status_code
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            },
            headers={"Cache-Control": "public, max-age=60"},
            status_code=500
        )
    
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

def get_average_dict(weather_data):
    """Create average temperature dictionary from weather data."""
    data_list = sorted([d for d in weather_data['data'] if d.get('y') is not None], key=lambda d: d['x'])
    if not data_list:
        return {
            "average": None,
            "unit": "celsius",
            "data_points": 0,
            "year_range": None,
            "missing_years": weather_data['metadata']['missing_years'],
            "completeness": weather_data['metadata']['completeness']
        }
    avg_temp = calculate_historical_average(data_list)
    return {
        "average": avg_temp,
        "unit": "celsius",
        "data_points": len(data_list),
        "year_range": {
            "start": data_list[0]['x'],
            "end": data_list[-1]['x']
        },
        "missing_years": weather_data['metadata']['missing_years'],
        "completeness": weather_data['metadata']['completeness']
    }

def is_valid_location(location: str) -> bool:
    """Check if a location is valid by testing the API."""
    today = datetime.now().strftime("%Y-%m-%d")
    url = build_visual_crossing_url(location, today, remote=False)
    response = requests.get(url)
    if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
        return True
    return False

async def get_temperature_series(location: str, month: int, day: int) -> Dict:
    """Get temperature series data for a location and date over multiple years."""
    # Check for cached series first
    series_cache_key = f"series_{location.lower()}_{month:02d}_{day:02d}"
    if CACHE_ENABLED:
        cached_series = get_cache(series_cache_key)
        if cached_series:
            logger.debug(f"Cache hit: {series_cache_key}")
            try:
                return json.loads(cached_series)
            except Exception as e:
                logger.error(f"Error decoding cached series for {series_cache_key}: {e}")
    if not is_valid_location(location):
        logger.warning(f"Invalid location: {location}")
        raise HTTPException(status_code=400, detail=f"Invalid location: {location}")
    logger.debug(f"get_temperature_series for {location} on {day}/{month}")
    today = datetime.now()
    current_year = today.year
    years = list(range(current_year - 50, current_year + 1))

    data = []
    missing_years = []
    uncached_years = []
    uncached_date_strs = []
    year_to_date_str = {}
    for year in years:
        logger.debug(f"get_temperature_series year: {year}")
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = f"{location.lower()}_{date_str}"
        year_to_date_str[year] = date_str
        # If this is today's date, use the forecast data
        if year == current_year and month == today.month and day == today.day:
            logger.debug("get_temperature_series forecast for today")
            try:
                forecast_data = get_forecast_data(location, datetime(year, month, day).date())
                logger.debug(f"Got forecast data {forecast_data}")
                if "error" in forecast_data:
                    logger.warning(f"Forecast error: {forecast_data['error']}")
                    missing_years.append({"year": year, "reason": "forecast_error"})
                else:
                    data.append({"x": year, "y": forecast_data["average_temperature"]})
                continue
            except Exception as e:
                logger.error(f"Error fetching forecast: {str(e)}")
                missing_years.append({"year": year, "reason": "forecast_failed"})
                continue
        # Use historical data for all other dates
        if CACHE_ENABLED:
            cached_data = get_cache(cache_key)
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
                            cache_key = f"{location.lower()}_{date_str}"
                            cache_duration = SHORT_CACHE_DURATION if is_today(year, month, day) else LONG_CACHE_DURATION
                            set_cache(cache_key, cache_duration, json.dumps(weather))
                    else:
                        logger.debug(f"Temperature is None for {year}, marking as missing.")
                        missing_years.append({"year": year, "reason": "no_temperature_data"})
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error processing cached data for {date_str}: {str(e)}")
                    missing_years.append({"year": year, "reason": "data_processing_error"})
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
                            cache_key = f"{location.lower()}_{date_str}"
                            cache_duration = SHORT_CACHE_DURATION if is_today(year, month, day) else LONG_CACHE_DURATION
                            set_cache(cache_key, cache_duration, json.dumps(weather))
                    else:
                        logger.debug(f"Temperature is None for {year}, marking as missing.")
                        missing_years.append({"year": year, "reason": "no_temperature_data"})
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error processing batch data for {date_str}: {str(e)}")
                    missing_years.append({"year": year, "reason": "data_processing_error"})
            else:
                missing_years.append({"year": year, "reason": weather.get("error", "api_error") if weather else "api_error"})

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
            "metadata": {
                "total_years": len(years),
                "available_years": 0,
                "missing_years": years,
                "completeness": 0.0
            },
            "error": f"Invalid location: {location}"
        }

    # Cache the entire series for a short duration
    if CACHE_ENABLED:
        set_cache(series_cache_key, SHORT_CACHE_DURATION, json.dumps({
            "data": data_list,
            "metadata": {
                "total_years": len(years),
                "available_years": len(data),
                "missing_years": missing_years,
                "completeness": round(len(data) / len(years) * 100, 1) if years else 0
            }
        }))

    return {
        "data": data_list,
        "metadata": {
            "total_years": len(years),
            "available_years": len(data),
            "missing_years": missing_years,
            "completeness": round(len(data) / len(years) * 100, 1) if years else 0
        }
    }

def get_forecast_data(location: str, date) -> Dict:
    """Get forecast data for a location and date."""
    # Convert date to string format if it's a date object
    if hasattr(date, 'strftime'):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)
    
    url = build_visual_crossing_url(location, date_str, remote=False)
    response = requests.get(url)
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

async def fetch_weather_batch(location: str, date_strs: list, max_concurrent: int = 1) -> dict:
    """
    Fetch weather data for multiple dates in parallel using aiohttp, with limited concurrency.
    Returns a dict mapping date_str to weather data.
    Now uses caching for each date.
    """
    logger.debug(f"fetch_weather_batch for {location}")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Visual Crossing API key not configured")
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)
    async def fetch_one(date_str):
        async with semaphore:
            return date_str, await get_weather_for_date(location, date_str)
    tasks = [fetch_one(date_str) for date_str in date_strs]
    for fut in asyncio.as_completed(tasks):
        date_str, result = await fut
        results[date_str] = result
    return results

@app.get("/protected-endpoint")
def protected_route(user=Depends(verify_firebase_token)):
    """Protected endpoint that requires Firebase authentication."""
    return {"message": "You are authenticated!", "user": user}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
