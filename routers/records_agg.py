# routers/records_agg.py
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Literal
import os, aiohttp, asyncio, re
from datetime import datetime, date
from constants import VC_BASE_URL
from utils.visual_crossing_timeline import close_client_session as close_timeline_client
from urllib.parse import quote

router = APIRouter()
UNIT_GROUP_DEFAULT = os.getenv("UNIT_GROUP", "celsius")

def _safe_parse_date(s: str) -> date:
    """Parse and validate date string with SSRF protection."""
    if not s or not isinstance(s, str):
        raise HTTPException(400, "Date must be a non-empty string")
    
    # Validate date format strictly
    import re
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        raise HTTPException(400, "Date must be in YYYY-MM-DD format")
    
    try:
        parsed_date = datetime.strptime(s, "%Y-%m-%d").date()
        # Validate reasonable date range
        if parsed_date.year < 1800 or parsed_date.year > 2100:
            raise HTTPException(400, "Year out of valid range")
        return parsed_date
    except ValueError as e:
        raise HTTPException(400, f"Invalid date format: {s}")

def _build_endpoint_links(request: Request, location: str = None, anchor: str = None, unit_group: str = "celsius") -> Dict[str, str]:
    """Build links to individual period endpoints.
    
    Args:
        request: FastAPI Request object
        location: Optional location name (if provided, will create concrete links)
        anchor: Optional anchor date in YYYY-MM-DD format (if provided, will create concrete links)
        unit_group: Temperature unit group (default: "celsius")
    
    Returns:
        Dictionary with template or concrete links to period endpoints
    """
    base_url = str(request.base_url).rstrip('/')
    
    # If location and anchor are provided, create concrete links
    if location and anchor:
        try:
            # Parse anchor date to get MM-DD format
            anchor_date = _safe_parse_date(anchor)
            mmdd = anchor_date.strftime("%m-%d")
            encoded_location = quote(location, safe='')
            
            return {
                "daily": f"{base_url}/v1/records/daily/{encoded_location}/{mmdd}?unit_group={unit_group}",
                "weekly": f"{base_url}/v1/records/weekly/{encoded_location}/{mmdd}?unit_group={unit_group}",
                "monthly": f"{base_url}/v1/records/monthly/{encoded_location}/{mmdd}?unit_group={unit_group}",
                "yearly": f"{base_url}/v1/records/yearly/{encoded_location}/{mmdd}?unit_group={unit_group}"
            }
        except Exception:
            # Fallback if date parsing fails
            encoded_location = quote(location, safe='')
            return {
                "daily": f"{base_url}/v1/records/daily/{encoded_location}/01-15?unit_group={unit_group}",
                "weekly": f"{base_url}/v1/records/weekly/{encoded_location}/01-15?unit_group={unit_group}",
                "monthly": f"{base_url}/v1/records/monthly/{encoded_location}/01-15?unit_group={unit_group}",
                "yearly": f"{base_url}/v1/records/yearly/{encoded_location}/01-15?unit_group={unit_group}"
            }
    
    # Return template links if location/anchor not provided
    return {
        "daily": f"{base_url}/v1/records/daily/{{location}}/{{mmdd}}",
        "weekly": f"{base_url}/v1/records/weekly/{{location}}/{{mmdd}}",
        "monthly": f"{base_url}/v1/records/monthly/{{location}}/{{mmdd}}",
        "yearly": f"{base_url}/v1/records/yearly/{{location}}/{{mmdd}}"
    }

def _rolling_bundle_gone_response(request: Request, message: str, location: str = None, anchor: str = None, unit_group: str = "celsius") -> JSONResponse:
    """Return a standardized 410 Gone response for removed rolling-bundle endpoints.
    
    Args:
        request: FastAPI Request object
        message: Specific message about which endpoint was removed
        location: Optional location name (for concrete links)
        anchor: Optional anchor date in YYYY-MM-DD format (for concrete links)
        unit_group: Temperature unit group (default: "celsius")
    
    Returns:
        JSONResponse with 410 status and links to individual endpoints
    """
    links = _build_endpoint_links(request, location, anchor, unit_group)
    return JSONResponse(
        status_code=410,
        content={
            "error": "GONE",
            "message": message,
            "code": "GONE",
            "details": "This endpoint is no longer available. Please use the individual period endpoints instead.",
            "links": links
        }
    )

@router.api_route("/v1/records/rolling-bundle/test-cors", methods=["GET", "OPTIONS"])
async def test_rolling_bundle_cors(request: Request):
    """Test CORS for rolling-bundle endpoint"""
    return _rolling_bundle_gone_response(
        request,
        "The rolling-bundle endpoint has been removed"
    )

@router.get("/v1/records/rolling-bundle/{location}/{anchor}/preload")
async def rolling_bundle_preload(
    request: Request,
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"] = Query(UNIT_GROUP_DEFAULT),
):
    """This endpoint has been removed. Use individual period endpoints instead."""
    return _rolling_bundle_gone_response(
        request,
        "The rolling-bundle preload endpoint has been removed",
        location,
        anchor,
        unit_group
    )

# Cache for daily data (used by other endpoints)
class KVCache:
    def __init__(self, redis=None, ttl_seconds: int = 24 * 60 * 60):
        self.redis = redis
        self.ttl = ttl_seconds
    
    async def get(self, key: str) -> Optional[bytes]:
        if not self.redis:
            return None
        try:
            return await self.redis.get(key)
        except Exception:
            return None
    
    async def set(self, key: str, value: bytes):
        if self.redis:
            try:
                await self.redis.set(key, value, ex=self.ttl)
            except Exception:
                pass  # Silently fail if Redis is not available

daily_cache = KVCache()

# HTTP client cleanup (kept for compatibility with main.py)
_http: Optional[aiohttp.ClientSession] = None

async def _close_rolling_http_client():
    """Close the rolling bundle HTTP client (kept for cleanup compatibility)."""
    global _http
    if _http is not None and not _http.closed:
        await _http.close()
        _http = None

async def cleanup_http_sessions():
    """Clean up all HTTP client sessions."""
    await _close_rolling_http_client()
    await close_timeline_client()

# Helper functions
def _safe_parse_date(s: str) -> date:
    """Parse and validate date string with SSRF protection."""
    if not s or not isinstance(s, str):
        raise HTTPException(400, "Date must be a non-empty string")
    
    # Validate date format strictly
    import re
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        raise HTTPException(400, "Date must be in YYYY-MM-DD format")
    
    try:
        parsed_date = datetime.strptime(s, "%Y-%m-%d").date()
        # Validate reasonable date range
        if parsed_date.year < 1800 or parsed_date.year > 2100:
            raise HTTPException(400, "Year out of valid range")
        return parsed_date
    except ValueError as e:
        raise HTTPException(400, f"Invalid date format: {s}")


@router.get("/v1/records/rolling-bundle/{location}/{anchor}/status")
async def rolling_bundle_status(
    request: Request,
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"] = Query(UNIT_GROUP_DEFAULT),
):
    """This endpoint has been removed. Use individual period endpoints instead."""
    return _rolling_bundle_gone_response(
        request,
        "The rolling-bundle status endpoint has been removed",
        location,
        anchor,
        unit_group
    )

@router.get("/v1/records/rolling-bundle/preload-example")
async def preload_example(request: Request):
    """This endpoint has been removed."""
    return _rolling_bundle_gone_response(
        request,
        "The rolling-bundle preload-example endpoint has been removed"
    )