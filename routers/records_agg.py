# routers/records_agg.py
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Literal
import os, aiohttp, asyncio
from datetime import datetime, date
from constants import VC_BASE_URL
from urllib.parse import quote

router = APIRouter()
# Strip whitespace/newlines from API key to prevent authentication issues
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
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

_client: Optional[aiohttp.ClientSession] = None
_sem = asyncio.Semaphore(2)

async def _client_session() -> aiohttp.ClientSession:
    global _client
    if _client is None or _client.closed:
        # Increased timeout for large date ranges, with connection and read timeouts
        _client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=120,  # 2 minutes total timeout for large requests
                connect=30,  # 30 seconds to establish connection
                sock_read=90  # 90 seconds to read response data
            )
        )
    return _client

async def _close_client_session():
    """Close the global client session."""
    global _client
    if _client is not None and not _client.closed:
        await _client.close()
        _client = None

def _years_range() -> tuple[int, int]:
    y = datetime.now().year
    return (y - 50, y)

def _convert_unit_group_for_vc(unit_group: str) -> str:
    """Convert API unit_group format (celsius/fahrenheit) to Visual Crossing format (metric/us).
    
    Args:
        unit_group: Unit group in API format ('celsius' or 'fahrenheit')
        
    Returns:
        Visual Crossing unit group ('metric' for celsius, 'us' for fahrenheit)
    """
    if unit_group.lower() == "celsius":
        return "metric"
    elif unit_group.lower() == "fahrenheit":
        return "us"
    elif unit_group.lower() in ("metric", "us", "uk"):
        # Already in VC format, return as-is
        return unit_group.lower()
    else:
        # Default to metric if unknown
        return "metric"

async def fetch_historysummary(
    location: str,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    chrono_unit: str = "years",
    break_by: str = "years",
    unit_group: str = UNIT_GROUP_DEFAULT,
    daily_summaries: bool = False,
    params_extra: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    # Validate location to prevent SSRF attacks
    from utils.validation import validate_location_for_ssrf
    try:
        location = validate_location_for_ssrf(location)
    except ValueError as e:
        raise ValueError(f"Invalid location: {str(e)}") from e
    
    if API_KEY is None:
        raise RuntimeError("VISUAL_CROSSING_API_KEY is not configured")
    if min_year is None or max_year is None:
        min_year, max_year = _years_range()
    params = {
        "aggregateHours": "24",
        "minYear": str(min_year),
        "maxYear": str(max_year),
        "chronoUnit": chrono_unit,        # weeks | months | years
        "breakBy": break_by,              # years | self | none
        "dailySummaries": "true" if daily_summaries else "false",
        "contentType": "json",
        "unitGroup": _convert_unit_group_for_vc(unit_group),
        "locations": location,
        "maxStations": 8,
        "maxDistance": 120000,
        "key": API_KEY,
    }
    if params_extra:
        params.update(params_extra)
    url = f"{VC_BASE_URL}/weatherdata/historysummary"
    sess = await _client_session()
    async with _sem:
        async with sess.get(url, params=params, headers={"Accept-Encoding": "gzip"}) as resp:
            if resp.status >= 400:
                text = await resp.text()
                # Historysummary endpoint often returns 400s - don't raise HTTPException, let caller handle fallback
                # Log detailed error but return generic error to prevent information disclosure
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Visual Crossing API error: status={resp.status}, response={text[:200]}")
                # Return generic error - don't expose API response details
                raise ValueError(f"External API error: {resp.status}")
            return await resp.json()

def _historysummary_values(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    locs = payload.get("locations") or {}
    if not locs:
        return []
    first = next(iter(locs.values()))
    vals = first.get("values")
    if isinstance(vals, list):
        return vals
    data = first.get("data")
    if isinstance(data, dict) and isinstance(data.get("values"), list):
        return data["values"]
    return []

def _row_year(row: Dict[str, Any]) -> Optional[int]:
    for k in ("year", "years"):
        v = row.get(k)
        if isinstance(v, int): return v
        if isinstance(v, str) and v.isdigit(): return int(v)
    for k in ("datetime", "datetimeStr", "period", "startTime", "start"):
        v = row.get(k)
        if isinstance(v, str) and v[:4].isdigit(): return int(v[:4])
    return None

def _row_month(row: Dict[str, Any]) -> Optional[int]:
    for k in ("month", "months", "monthnum", "monthNum"):
        v = row.get(k)
        if isinstance(v, int): return v
        if isinstance(v, str) and v.isdigit(): return int(v)
    for k in ("datetime", "datetimeStr", "period", "start", "startTime"):
        v = row.get(k)
        if isinstance(v, str) and len(v) >= 7:
            try: return int(v[5:7])
            except: pass
    return None

def _row_week_start(row: Dict[str, Any]) -> Optional[str]:
    for k in ("period", "start", "startTime", "datetime", "datetimeStr"):
        v = row.get(k)
        if isinstance(v, str) and len(v) >= 10 and v[4] == "-" and v[7] == "-":
            return v[:10]
    return None

def _row_mean_temp(row: Dict[str, Any]) -> Optional[float]:
    for k in ("temp", "tempavg", "avgtemp", "averageTemp"):
        v = row.get(k)
        if isinstance(v, (int, float)): return float(v)
        if isinstance(v, str):
            try: return float(v)
            except: pass
    return None

@router.get("/v1/records/monthly/{location}/{ym}/series")
async def monthly_series(location: str, ym: str, unit_group: str = UNIT_GROUP_DEFAULT):
    # ym is YYYY-MM (same month across all years)
    try:
        month = datetime.strptime(ym, "%Y-%m").month
    except ValueError:
        raise HTTPException(status_code=400, detail="Identifier must be YYYY-MM")
    min_year, max_year = _years_range()
    payload = await fetch_historysummary(location, min_year, max_year, chrono_unit="months", break_by="years", unit_group=unit_group)
    rows = _historysummary_values(payload)
    items = []
    for r in rows:
        if _row_month(r) == month:
            y = _row_year(r); t = _row_mean_temp(r)
            if y is not None and t is not None:
                items.append({"year": y, "temp": round(t, 2)})
    items.sort(key=lambda x: x["year"])
    return {"period": "monthly", "location": location, "identifier": ym, "unit_group": unit_group, "values": items, "count": len(items)}

@router.get("/v1/records/weekly/{location}/{week_start}/series")
async def weekly_series(location: str, week_start: str, unit_group: str = UNIT_GROUP_DEFAULT):
    """
    week_start = MM-DD or YYYY-MM-DD (anchor). We match the *ISO week number* of that anchor
    across all years using historysummary weeks (one call).
    """
    try:
        mmdd = week_start if len(week_start) == 5 else datetime.strptime(week_start, "%Y-%m-%d").strftime("%m-%d")
    except Exception:
        raise HTTPException(status_code=400, detail="Identifier must be MM-DD or YYYY-MM-DD")
    min_year, max_year = _years_range()
    payload = await fetch_historysummary(location, min_year, max_year, chrono_unit="weeks", break_by="years", unit_group=unit_group)
    rows = _historysummary_values(payload)
    desired_week = {y: datetime.strptime(f"{y}-{mmdd}", "%Y-%m-%d").isocalendar().week for y in range(min_year, max_year + 1)}
    items = []
    for r in rows:
        y = _row_year(r); start = _row_week_start(r); t = _row_mean_temp(r)
        if y is None or t is None or not start: continue
        try:
            wn = datetime.strptime(start, "%Y-%m-%d").isocalendar().week
        except Exception:
            continue
        if wn == desired_week.get(y):
            items.append({"year": y, "temp": round(t, 2)})
    items.sort(key=lambda x: x["year"])
    return {"period": "weekly", "location": location, "identifier": mmdd, "unit_group": unit_group, "values": items, "count": len(items), "note": "historysummary uses week bins; we align via ISO week number"}

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
    await _close_client_session()
    await _close_rolling_http_client()

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