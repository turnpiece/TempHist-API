"""Request ID middleware for distributed tracing."""
import uuid
from fastapi import Request


async def request_id_middleware(request: Request, call_next):
    """Add request ID for distributed tracing (LOW-003)."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response
