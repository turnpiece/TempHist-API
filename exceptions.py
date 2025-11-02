"""Exception handlers for standardized error responses."""
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException
from models import ErrorResponse
from config import DEBUG

logger = logging.getLogger(__name__)

# Error code mappings
ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_SERVER_ERROR",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT"
}


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


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors with standardized error format (MED-008)."""
        client_ip = get_client_ip(request)
        request_id = getattr(request.state, 'request_id', None)
        
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
        
        logger.error(f"❌ VALIDATION ERROR: {exc.errors()} | IP={client_ip} | Path={request.url.path} | Request-ID={request_id}")
        
        error_response = ErrorResponse(
            error="VALIDATION_ERROR",
            message="Request data validation failed",
            code="VALIDATION_ERROR",
            details=error_details,
            path=request.url.path,
            method=request.method,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.model_dump()
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic model validation errors with standardized format (MED-008)."""
        client_ip = get_client_ip(request)
        request_id = getattr(request.state, 'request_id', None)
        
        error_details = []
        for error in exc.errors():
            field = " -> ".join(str(loc) for loc in error['loc'])
            error_details.append({
                "field": field,
                "message": error['msg'],
                "type": error['type'],
                "input": error.get('input', 'N/A')
            })
        
        logger.error(f"❌ PYDANTIC VALIDATION ERROR: {exc.errors()} | IP={client_ip} | Path={request.url.path} | Request-ID={request_id}")
        
        error_response = ErrorResponse(
            error="MODEL_VALIDATION_ERROR",
            message="Data model validation failed",
            code="MODEL_VALIDATION_ERROR",
            details=error_details,
            path=request.url.path,
            method=request.method,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=422,
            content=error_response.model_dump()
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTPException with standardized error format (MED-008)."""
        request_id = getattr(request.state, 'request_id', None)
        
        # Handle different detail formats
        if isinstance(exc.detail, dict):
            # Already in structured format, use it
            detail_data = exc.detail
            error_message = detail_data.get("message", detail_data.get("detail", "An error occurred"))
            error_code = detail_data.get("code", ERROR_CODES.get(exc.status_code, "UNKNOWN_ERROR"))
            error_details = detail_data.get("details")
        elif isinstance(exc.detail, str):
            # Simple string detail, convert to standardized format
            error_message = exc.detail
            error_code = ERROR_CODES.get(exc.status_code, "UNKNOWN_ERROR")
            error_details = None
        else:
            error_message = str(exc.detail) if exc.detail else "An error occurred"
            error_code = ERROR_CODES.get(exc.status_code, "UNKNOWN_ERROR")
            error_details = None
        
        error_response = ErrorResponse(
            error=error_code,
            message=error_message,
            code=error_code,
            details=error_details,
            path=request.url.path,
            method=request.method,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unhandled exceptions with standardized error format (MED-008)."""
        request_id = getattr(request.state, 'request_id', None)
        
        # Log full error details server-side
        logger.error(f"❌ UNHANDLED EXCEPTION: {type(exc).__name__}: {str(exc)} | Path={request.url.path} | Request-ID={request_id}", exc_info=True)
        
        # Return generic error to client (don't expose internal details)
        error_response = ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred" if not DEBUG else str(exc),
            code="INTERNAL_SERVER_ERROR",
            details={"type": type(exc).__name__} if DEBUG else None,
            path=request.url.path,
            method=request.method,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )
