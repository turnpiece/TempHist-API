"""CORS middleware for health checks."""
from fastapi import Request


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


def get_cors_origins(cors_origins_env: str = None):
    """Parse CORS origins from environment variable or use defaults."""
    if cors_origins_env:
        # Split by comma and strip whitespace
        origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
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


def get_cors_origin_regex(cors_origin_regex_env: str = None):
    """Parse CORS origin regex from environment variable or use default."""
    if cors_origin_regex_env:
        return cors_origin_regex_env
    else:
        # Default regex for temphist.com and all its subdomains
        default_regex = r"^https://(.*\.)?temphist\.com$"
        return default_regex
