# Standard library imports
import asyncio
import json
import logging
import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

# Third-party imports
import redis
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from firebase_admin import app_check, auth
from starlette.responses import FileResponse, Response

# Load .env and populate os.environ before routers (see config.DOTENV_PATH).
import config  # noqa: F401
from cache.accessors import get_cache_warmer, initialize_cache

# Import enhanced caching utilities
from cache.warming import CACHE_WARMING_ENABLED, scheduled_cache_warming

# Import configuration and rate limiting
from config import (
    ADMIN_API_KEY,
    API_ACCESS_TOKEN,
    CORS_ORIGIN_REGEX,
    CORS_ORIGINS,
    DEBUG,
    ENVIRONMENT,
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
from middleware import (
    add_security_headers,
    health_check_cors_middleware,
    log_requests_middleware,
    request_id_middleware,
    request_size_middleware,
)
from middleware.cors import get_cors_origin_regex, get_cors_origins
from rate_limiting import LocationDiversityMonitor, RequestRateMonitor, ServiceTokenRateLimiter
from routers._responses import error_responses
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
from utils.admin_auth import admin_key_is_valid, is_admin_path, verify_admin_key
from utils.firebase import initialize_firebase
from utils.ip_utils import get_client_ip, is_ip_blacklisted, is_ip_whitelisted
from utils.path_parsing import extract_location_from_path
from utils.redis_client import create_redis_client
from utils.sanitization import sanitize_for_logging, sanitize_url

if not CORS_ORIGINS and not CORS_ORIGIN_REGEX:
    logging.getLogger(__name__).warning("⚠️  No CORS origins configured - API may be inaccessible to web clients")


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379").strip()

_temp_logger = logging.getLogger(__name__)
_temp_logger.info(f"🔍 DEBUG: REDIS_URL environment variable = {sanitize_url(REDIS_URL)}")

# LOW-008: Validate debug mode with environment check
# CACHE_ENABLED, ENVIRONMENT, and DEBUG are imported from config

# Prevent DEBUG mode in production
if ENVIRONMENT == "production" and DEBUG:
    # Use basic logging since logger not yet initialized
    logging.basicConfig(level=logging.INFO)
    _temp_logger = logging.getLogger(__name__)
    _temp_logger.error("❌ DEBUG mode cannot be enabled in production environment")
    raise ValueError("DEBUG=true is forbidden in production. Set ENVIRONMENT=production and DEBUG=false")

# Configuration variables are imported from config
# LOG_VERBOSITY, API_ACCESS_TOKEN, CACHE_CONTROL_HEADER, FILTER_WEATHER_DATA,
# RATE_LIMIT_ENABLED, MAX_LOCATIONS_PER_HOUR, MAX_REQUESTS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS,
# IP_WHITELIST, IP_BLACKLIST


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


# Lifespan event handler for startup and shutdown
from contextlib import asynccontextmanager  # noqa: E402


def _init_service_token_rate_limiter() -> Optional[ServiceTokenRateLimiter]:
    """Build the Redis-backed service-token rate limiter, or return None on failure."""
    try:
        limiter = ServiceTokenRateLimiter(redis_client)
        if DEBUG:
            logger.info(
                "🛡️  SERVICE TOKEN RATE LIMITING: %s requests/hour, %s locations/hour",
                SERVICE_TOKEN_RATE_LIMITS["requests_per_hour"],
                SERVICE_TOKEN_RATE_LIMITS["locations_per_hour"],
            )
        return limiter
    except (redis.RedisError, redis.ConnectionError):
        logger.exception("❌ SERVICE TOKEN RATE LIMITER: Redis connection failed")
    except ImportError:
        logger.exception("❌ SERVICE TOKEN RATE LIMITER: Import failed")
    return None


def _init_router_dependencies(limiter: Optional[ServiceTokenRateLimiter]) -> None:
    """Wire shared router dependencies (cache, monitors, analytics)."""
    try:
        from analytics_storage import AnalyticsStorage
        from utils.location_validation import InvalidLocationCache

        initialize_dependencies(
            redis_client=redis_client,
            invalid_location_cache=InvalidLocationCache(redis_client),
            service_token_rate_limiter=limiter,
            location_monitor=location_monitor,
            request_monitor=request_monitor,
            analytics_storage=AnalyticsStorage(redis_client),
        )
        if DEBUG:
            logger.info("✅ ROUTER DEPENDENCIES: Initialized successfully")
    except (redis.RedisError, redis.ConnectionError):
        logger.exception("❌ ROUTER DEPENDENCIES: Redis connection failed")
    except ImportError:
        logger.exception("❌ ROUTER DEPENDENCIES: Import failed")
    except (ValueError, TypeError):
        logger.exception("❌ ROUTER DEPENDENCIES: Configuration error")


def _init_cache_system() -> None:
    try:
        initialize_cache(redis_client)
        if DEBUG:
            logger.info("✅ CACHE SYSTEM: Initialized successfully")
    except (redis.RedisError, redis.ConnectionError):
        logger.exception("❌ CACHE SYSTEM: Redis connection failed")
    except ImportError:
        logger.exception("❌ CACHE SYSTEM: Import failed")


async def _init_locations() -> None:
    try:
        await initialize_locations_data(redis_client)
        if DEBUG:
            logger.info("✅ LOCATIONS: Data loaded and cache warmed")
    except (redis.RedisError, redis.ConnectionError):
        logger.exception("❌ LOCATIONS: Redis connection failed")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.exception("❌ LOCATIONS: File error")
    except (IOError, PermissionError):
        logger.exception("❌ LOCATIONS: I/O error")


async def _maybe_start_cache_warming() -> None:
    """Enqueue the startup warming job and launch the scheduled-warming background task."""
    if not (CACHE_WARMING_ENABLED and get_cache_warmer()):
        return

    # Wait a moment for the server to fully start
    await asyncio.sleep(2)

    if DEBUG:
        logger.info("🚀 STARTUP CACHE WARMING: Creating initial warming job")

    from cache.accessors import get_job_manager

    job_manager = get_job_manager()
    if not job_manager:
        logger.warning("⚠️  Job manager not available, skipping startup warming")
        return

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
        logger.info("✅ Startup cache warming job created: %s", job_id)
    except Exception as job_error:
        logger.warning("⚠️  Startup cache warming job skipped: %s", job_error)

    asyncio.create_task(scheduled_cache_warming(get_cache_warmer()))
    if DEBUG:
        logger.info("⏰ SCHEDULED CACHE WARMING: Background task started")


async def _shutdown_clients() -> None:
    if DEBUG:
        logger.info("🛑 APPLICATION SHUTDOWN: Cleaning up resources")

    if async_redis_client:
        try:
            await async_redis_client.aclose()
            if DEBUG:
                logger.info("✅ Async Redis client closed successfully")
        except Exception:
            logger.exception("⚠️  Error closing async Redis client")

    try:
        from utils.weather_provider import close_client_session

        await close_client_session()
        if DEBUG:
            logger.info("✅ HTTP client sessions closed successfully")
    except Exception:
        logger.exception("⚠️  Error closing HTTP client sessions")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global service_token_rate_limiter

    service_token_rate_limiter = _init_service_token_rate_limiter()
    _init_router_dependencies(service_token_rate_limiter)
    _init_cache_system()
    await _init_locations()
    await _maybe_start_cache_warming()

    yield

    await _shutdown_clients()


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


redis_client = create_redis_client(REDIS_URL)

# Note: initialize_cache() is called in the lifespan handler (line 614)
# to ensure proper initialization order during app startup

# Background worker is now handled by a separate service (worker_service.py)
# This provides better isolation, scaling, and eliminates event loop conflicts
logger.debug("ℹ️  Background worker runs as separate service - no in-process worker needed")

initialize_firebase()


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


app.middleware("http")(request_id_middleware)


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


app.middleware("http")(log_requests_middleware)
app.middleware("http")(add_security_headers)
app.middleware("http")(request_size_middleware)


def _check_ip_blacklist(request: Request, client_ip: str) -> Optional[JSONResponse]:
    """Block blacklisted IPs except for health and analytics endpoints."""
    if not is_ip_blacklisted(client_ip):
        return None
    if request.url.path in ("/health",) or request.url.path.startswith("/analytics"):
        return None
    if DEBUG:
        logger.warning(f"🚫 BLACKLISTED IP BLOCKED: {client_ip} | {request.method} {request.url.path}")
    return JSONResponse(status_code=403, content={"detail": "Access denied", "reason": "IP address is blacklisted"})


_PUBLIC_PATHS = frozenset([
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
])

_PUBLIC_PREFIXES = ("/static", "/analytics", "/data", "/v1/shares/", "/v1/og/")


def _is_public_path(request: Request) -> bool:
    """Return True for paths that require no auth or rate limiting."""
    path = request.url.path
    if path == "/v1/shares" and request.method == "GET":
        return True
    return path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES)


async def _check_admin_auth(request: Request, call_next) -> Optional[Response]:
    """Validate X-Admin-Key for admin paths.

    Returns a Response (error or successful call_next result) for admin paths,
    or None if the path is not an admin path.
    """
    if not is_admin_path(request.url.path, request.method):
        return None
    admin_key = request.headers.get("X-Admin-Key")
    if not ADMIN_API_KEY:
        return JSONResponse(
            status_code=503,
            content={"detail": "Admin API not configured (ADMIN_API_KEY not set)"},
        )
    if not admin_key_is_valid(admin_key):
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid admin key"})
    request.state.user = {"uid": "admin", "system": True, "source": "admin_key"}
    if DEBUG:
        logger.debug("[DEBUG] Middleware: Admin path, X-Admin-Key verified")
    return await call_next(request)


def _detect_service_job(auth_header: Optional[str], request: Request, client_ip: str) -> bool:
    """Return True if the request uses the API_ACCESS_TOKEN service bypass."""
    if not (auth_header and auth_header.startswith("Bearer ")):
        return False
    token = auth_header.split(" ")[1]
    if not (API_ACCESS_TOKEN and token == API_ACCESS_TOKEN):
        return False
    if DEBUG:
        logger.info(f"🔧 SERVICE JOB DETECTED: {client_ip} | {request.method} {request.url.path} | Rate limiting bypassed")
    return True


def _check_rate_limits(
    request: Request, client_ip: str, is_service_job: bool
) -> Optional[JSONResponse]:
    """Apply request-rate and location-diversity limits for VC API endpoints."""
    rate_limiting_active = (
        RATE_LIMIT_ENABLED
        and location_monitor
        and request_monitor
        and not is_ip_whitelisted(client_ip)
        and not is_service_job
    )
    if not rate_limiting_active:
        if is_ip_whitelisted(client_ip) and DEBUG:
            logger.info(f"⭐ WHITELISTED IP: {client_ip} | {request.method} {request.url.path} | Rate limiting bypassed")
        return None

    vc_api_paths = ("/weather/", "/forecast/", "/v1/records/")
    if not any(request.url.path.startswith(p) for p in vc_api_paths):
        if DEBUG:
            logger.debug(f"ℹ️  NON-VC ENDPOINT: {client_ip} | {request.method} {request.url.path} | Rate limiting skipped")
        return None

    rate_allowed, rate_reason = request_monitor.check_request_rate(client_ip)
    if not rate_allowed:
        if DEBUG:
            logger.warning(f"🚫 RATE LIMIT EXCEEDED: {client_ip} | {request.method} {request.url.path} | {rate_reason}")
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "reason": rate_reason,
                "retry_after": RATE_LIMIT_WINDOW_HOURS * 3600,
            },
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_HOURS * 3600)},
        )
    if DEBUG:
        logger.debug(f"✅ RATE LIMIT CHECK: {client_ip} | {request.method} {request.url.path} | Rate: OK")

    location, _endpoint = extract_location_from_path(request.url.path)
    if location:
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
        if DEBUG:
            logger.debug(
                f"✅ LOCATION DIVERSITY CHECK: {client_ip} | {request.method} {request.url.path} | Location: {sanitize_for_logging(location)} | OK"
            )

    return None


def _check_app_check(request: Request, uid: str, client_ip: str) -> Optional[JSONResponse]:
    """Verify the Firebase App Check token when enforcement is active."""
    if APP_CHECK_ENFORCEMENT == "off":
        return None
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
        return None
    reason = app_check_error or "missing X-Firebase-AppCheck header"
    if APP_CHECK_ENFORCEMENT == "enforce":
        logger.warning(f"App Check: BLOCKED uid={uid} ip={client_ip} path={request.url.path} reason={reason}")
        return JSONResponse(status_code=403, content={"detail": "App Check verification failed"})
    logger.warning(f"App Check: MONITOR uid={uid} ip={client_ip} path={request.url.path} reason={reason}")
    return None


async def _verify_firebase_auth(
    request: Request, auth_header: Optional[str], client_ip: str, is_service_job: bool
) -> Optional[JSONResponse]:
    """Verify the Bearer token and set request.state.user. Returns a JSONResponse on auth failure."""
    if DEBUG:
        logger.debug("[DEBUG] Middleware: Protected path, checking Firebase token...")
    if not auth_header or not auth_header.startswith("Bearer "):
        if DEBUG:
            logger.debug("[DEBUG] Middleware: No valid Authorization header")
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header."})

    id_token = auth_header.split(" ")[1]

    if is_service_job:
        if DEBUG:
            logger.debug("[DEBUG] Middleware: Using production token bypass")
        request.state.user = {"uid": "admin", "system": True, "source": "production_token"}
        return None

    logger.info("[DEBUG] Middleware: Verifying Firebase token...")
    try:
        decoded_token = auth.verify_id_token(id_token)
        provider = decoded_token.get("firebase", {}).get("sign_in_provider", "unknown")
        uid = decoded_token.get("uid", "unknown")
        logger.info(f"Auth: uid={uid} provider={provider} ip={client_ip} path={request.url.path}")
        request.state.user = decoded_token
        return _check_app_check(request, uid, client_ip)
    except Exception as e:
        logger.exception("Firebase token verification failed (type=%s)", type(e).__name__)
        if DEBUG:
            return JSONResponse(status_code=403, content={"detail": f"Invalid Firebase token: {str(e)}"})
        return JSONResponse(status_code=403, content={"detail": "Authentication failed"})


@app.middleware("http")
async def verify_token_middleware(request: Request, call_next):
    """Middleware to verify Firebase tokens and apply rate limiting for protected routes."""
    if DEBUG:
        logger.debug(f"[DEBUG] Middleware: Processing {request.method} request to {request.url.path}")

    if request.method == "OPTIONS":
        if DEBUG:
            logger.debug("[DEBUG] Middleware: OPTIONS request, allowing through")
        return await call_next(request)

    client_ip = get_client_ip(request)
    if DEBUG:
        logger.debug(f"[DEBUG] Middleware: Client IP: {client_ip}")

    block = _check_ip_blacklist(request, client_ip)
    if block:
        return block

    if _is_public_path(request):
        if DEBUG:
            logger.debug("[DEBUG] Middleware: Public path, allowing through")
        return await call_next(request)

    admin_response = await _check_admin_auth(request, call_next)
    if admin_response is not None:
        return admin_response

    auth_header = request.headers.get("Authorization")
    is_service_job = _detect_service_job(auth_header, request, client_ip)

    block = _check_rate_limits(request, client_ip, is_service_job)
    if block:
        return block

    block = await _verify_firebase_auth(request, auth_header, client_ip, is_service_job)
    if block:
        return block

    logger.info("[DEBUG] Middleware: Token verified, calling next handler...")
    response = await call_next(request)
    logger.info("[DEBUG] Middleware: Response received, returning")
    return response


app.middleware("http")(health_check_cors_middleware)


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(CORS_ORIGINS),
    allow_origin_regex=get_cors_origin_regex(CORS_ORIGIN_REGEX),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "authorization",
        "content-type",
        "accept",
        "x-requested-with",
        "x-firebase-appcheck",
        "x-admin-key",
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


@app.post("/admin/clear-job-queue", responses=error_responses(401, 500))
async def admin_clear_job_queue(_admin: Annotated[bool, Depends(verify_admin_key)]):
    """
    Admin endpoint to clear the job queue.
    Requires X-Admin-Key header matching ADMIN_API_KEY environment variable.
    """
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
        logger.exception("Error clearing job queue")
        raise HTTPException(status_code=500, detail=f"Error clearing job queue: {str(e)}")


# For local testing
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("LOCAL_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port)
