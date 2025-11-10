"""Security headers middleware."""
import os
from fastapi import Request
from fastapi.responses import JSONResponse
from config import ENVIRONMENT


async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses to prevent various attacks."""
    # Enforce HTTPS in production (MED-001)
    env = ENVIRONMENT
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
