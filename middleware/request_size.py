"""Request size validation middleware."""
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from utils.ip_utils import get_client_ip

logger = logging.getLogger(__name__)


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
