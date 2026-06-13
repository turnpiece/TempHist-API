"""Shared FastAPI ``responses=`` documentation for error status codes.

Each route decorator that can raise an ``HTTPException`` should pull the
relevant entries from ``ERROR_RESPONSES`` so the OpenAPI schema reflects
the codes a client may see. The ``ErrorResponse`` model is the same
shape that ``exceptions.register_exception_handlers`` returns at runtime.
"""

from typing import Dict, Iterable

from models import ErrorResponse

ERROR_RESPONSES: Dict[int, Dict] = {
    304: {"description": "Not Modified"},
    400: {"model": ErrorResponse, "description": "Bad Request"},
    401: {"model": ErrorResponse, "description": "Unauthorized"},
    404: {"model": ErrorResponse, "description": "Not Found"},
    429: {"model": ErrorResponse, "description": "Too Many Requests"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"},
    503: {"model": ErrorResponse, "description": "Service Unavailable"},
}


def error_responses(*codes: int) -> Dict[int, Dict]:
    """Return a subset of ``ERROR_RESPONSES`` keyed by the requested codes."""
    return {code: ERROR_RESPONSES[code] for code in codes if code in ERROR_RESPONSES}


def merge_error_responses(codes: Iterable[int], extra: Dict[int, Dict]) -> Dict[int, Dict]:
    """Combine ``error_responses(codes)`` with any extra per-route overrides."""
    merged = error_responses(*codes)
    merged.update(extra)
    return merged
