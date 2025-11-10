"""Authentication and rate limiting middleware."""
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from firebase_admin import auth
from config import (
    DEBUG, API_ACCESS_TOKEN, RATE_LIMIT_ENABLED, RATE_LIMIT_WINDOW_HOURS,
    USAGE_TRACKING_ENABLED
)
from utils.ip_utils import get_client_ip, is_ip_whitelisted, is_ip_blacklisted
from utils.sanitization import sanitize_for_logging
from cache_utils import get_usage_tracker

logger = logging.getLogger(__name__)


def create_verify_token_middleware(
    location_monitor=None,
    request_monitor=None,
    service_token_rate_limiter=None
):
    """Create verify_token_middleware with dependencies."""
    
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
        public_paths = ["/", "/docs", "/openapi.json", "/redoc", "/test-cors", "/test-redis", 
                        "/rate-limit-status", "/analytics", "/health", "/health/detailed", 
                        "/v1/jobs/diagnostics/worker-status"]
        if request.url.path in public_paths or any(request.url.path.startswith(p) for p in ["/static", "/analytics"]):
            if DEBUG:
                logger.debug(f"[DEBUG] Middleware: Public path, allowing through")
            return await call_next(request)

        # Check if this is a service job using API_ACCESS_TOKEN
        auth_header = request.headers.get("Authorization")
        is_service_job = False
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
                is_service_job = True
                if DEBUG:
                    logger.info(f"🔧 SERVICE JOB DETECTED: {client_ip} | {request.method} {request.url.path} | Rate limiting bypassed")
                
                # Apply service token rate limiting if available
                if service_token_rate_limiter:
                    # Check service token request rate
                    rate_allowed, rate_reason = service_token_rate_limiter.check_request_rate(client_ip)
                    if not rate_allowed:
                        if DEBUG:
                            logger.warning(f"🚫 SERVICE TOKEN RATE LIMIT EXCEEDED: {client_ip} | {rate_reason}")
                        return JSONResponse(
                            status_code=429,
                            content={
                                "detail": "Service token rate limit exceeded",
                                "reason": rate_reason
                            }
                        )
                    
                    # Check location diversity for service tokens
                    vc_api_paths = ["/weather/", "/forecast/", "/v1/records/"]
                    if any(request.url.path.startswith(path) for path in vc_api_paths):
                        path_parts = request.url.path.split("/")
                        if len(path_parts) >= 3:
                            location = path_parts[2]
                            location_allowed, location_reason = service_token_rate_limiter.check_location_diversity(client_ip, location)
                            if not location_allowed:
                                if DEBUG:
                                    logger.warning(f"🚫 SERVICE TOKEN LOCATION LIMIT: {client_ip} | {location_reason}")
                                return JSONResponse(
                                    status_code=429,
                                    content={
                                        "detail": "Service token location limit exceeded",
                                        "reason": location_reason
                                    }
                                )

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
        
        # Track usage for Visual Crossing API endpoints
        if USAGE_TRACKING_ENABLED and get_usage_tracker():
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
    
    return verify_token_middleware


# For backward compatibility, export a factory that accepts dependencies
verify_token_middleware = None  # Will be initialized in main.py with dependencies
