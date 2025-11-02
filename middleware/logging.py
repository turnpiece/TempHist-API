"""Request logging middleware."""
import time
import logging
from fastapi import Request
from config import DEBUG, LOG_VERBOSITY
from utils.sanitization import sanitize_for_logging
from utils.ip_utils import get_client_ip

logger = logging.getLogger(__name__)


async def log_requests_middleware(request: Request, call_next):
    """Log all requests when DEBUG is enabled or verbosity is verbose."""
    if DEBUG or LOG_VERBOSITY == "verbose":
        start_time = time.time()
        client_ip = get_client_ip(request)
        
        # Log request details (only for non-public paths to reduce noise)
        if request.url.path not in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/rate-limit-status"]:
            user_agent = request.headers.get('user-agent', 'Unknown')
            logger.debug(f"🌐 REQUEST: {request.method} {request.url.path} | IP: {client_ip} | User-Agent: {sanitize_for_logging(user_agent, max_length=150)}")
        
        # Process request
        response = await call_next(request)
        
        # Log response details (only for non-public paths to reduce noise)
        process_time = time.time() - start_time
        if request.url.path not in ["/", "/docs", "/openapi.json", "/redoc", "/health", "/rate-limit-status"]:
            logger.debug(f"✅ RESPONSE: {response.status_code} | {request.method} {request.url.path} | {process_time:.3f}s | IP: {client_ip}")
        
        return response
    else:
        # Skip logging in production
        return await call_next(request)
