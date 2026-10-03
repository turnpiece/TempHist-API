"""Shared FastAPI ``responses=`` documentation for error status codes.

Each route decorator that can raise an ``HTTPException`` should pull the
relevant entries from ``ERROR_RESPONSES`` so the OpenAPI schema reflects
the codes a client may see. The ``ErrorResponse`` model is the same
shape that ``exceptions.register_exception_handlers`` returns at runtime.
"""

from typing import Dict, Iterable

from models import ErrorResponse, RateLimitErrorResponse

# Success shapes are documented the same way, via ``responses={200: {"model": ...}}`` rather than
# ``response_model=``. A ``responses`` model only describes the schema; ``response_model`` would also validate and
# filter what the handler returns, which can turn an undocumented-but-working payload into a 500.

ERROR_RESPONSES: Dict[int, Dict] = {
    304: {"description": "Not Modified"},
    400: {"model": ErrorResponse, "description": "Bad Request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    404: {"model": ErrorResponse, "description": "Not Found"},
    429: {"model": ErrorResponse, "description": "Too Many Requests"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"},
    503: {"model": ErrorResponse, "description": "Service Unavailable"},
}


RETRY_AFTER_HEADER: Dict[str, Dict] = {
    "Retry-After": {
        "description": "Seconds to wait before retrying the request.",
        "schema": {"type": "integer"},
    }
}

# 429 as produced by the request-rate / location-diversity limiter in main.py, which applies to /weather,
# /forecast and /v1/records. It answers directly from the middleware, so the body is RateLimitErrorResponse rather
# than ErrorResponse, and it always sends Retry-After.
RATE_LIMITED: Dict[int, Dict] = {
    429: {
        "model": RateLimitErrorResponse,
        "description": "Too Many Requests: the request-rate or location-diversity limit was exceeded",
        "headers": RETRY_AFTER_HEADER,
    }
}


def error_responses(*codes: int) -> Dict[int, Dict]:
    """Return a subset of ``ERROR_RESPONSES`` keyed by the requested codes."""
    return {code: ERROR_RESPONSES[code] for code in codes if code in ERROR_RESPONSES}


def merge_error_responses(codes: Iterable[int], extra: Dict[int, Dict]) -> Dict[int, Dict]:
    """Combine ``error_responses(codes)`` with any extra per-route overrides."""
    merged = error_responses(*codes)
    merged.update(extra)
    return merged
