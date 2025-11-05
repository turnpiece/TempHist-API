# routers/records_agg.py
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Tuple, Literal, Set
import os, aiohttp, asyncio, json
import redis
from datetime import datetime, date, timedelta
from pydantic import BaseModel
from constants import VC_BASE_URL
from dateutil.relativedelta import relativedelta
from routers.dependencies import get_redis_client
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

# ============================================================================
# ROLLING BUNDLE ENDPOINT
# ============================================================================

# Models for rolling bundle
class YearTemp(BaseModel):
    year: int
    temperature: float

class RollingSeries(BaseModel):
    values: List[YearTemp]
    count: int

class RollingBundleResponse(BaseModel):
    period: str = "rolling"
    location: str
    anchor: date
    unit_group: Literal["celsius", "fahrenheit"]
    # Daily data (anchor day and previous days)
    day: Optional[Dict] = None  # Complete daily endpoint response
    previous_days: List[Dict] = []  # List of previous days' complete responses
    # Period data (complete endpoint responses)
    daily: Optional[Dict] = None  # Complete daily endpoint response (same as day)
    weekly: Optional[Dict] = None  # Complete weekly endpoint response
    monthly: Optional[Dict] = None  # Complete monthly endpoint response
    yearly: Optional[Dict] = None  # Complete yearly endpoint response
    # Bundle metadata
    metadata: Optional[Dict] = None
    notes: Optional[str] = None

# Configuration
YEARS_BACK = 50  # 50-year history
ROLLING_BUNDLE_YEARS_BACK = 10  # 10-year history for rolling bundle (more efficient)
VC_MAX_CONCURRENCY = 2
DEFAULT_DAYS_BACK = 0  # Default number of previous days to include (0 = only anchor day)

# Allowed sections for include/exclude parameters
ALLOWED_SECTIONS = {"day", "week", "month", "year"}

# Cache for daily data
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

# HTTP client and semaphore for rolling bundle
_http: Optional[aiohttp.ClientSession] = None
_rolling_sem = asyncio.Semaphore(VC_MAX_CONCURRENCY)

async def _get_rolling_http_client() -> aiohttp.ClientSession:
    """Get or create the rolling bundle HTTP client."""
    global _http
    if _http is None or _http.closed:
        # Increased timeout for rolling bundle requests
        _http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=120,  # 2 minutes total timeout
                connect=30,  # 30 seconds to establish connection
                sock_read=90  # 90 seconds to read response data
            )
        )
    return _http

async def _close_rolling_http_client():
    """Close the rolling bundle HTTP client."""
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

def _years_range(now: date) -> Tuple[int, int]:
    end = now.year
    start = end - YEARS_BACK
    return start, end

def _leap_clamp(md: Tuple[int, int], year: int) -> Tuple[int, int]:
    m, d = md
    # clamp Feb 29 -> Feb 28 on non-leap years
    if m == 2 and d == 29:
        try:
            date(year, 2, 29)
        except ValueError:
            return (2, 28)
    return (m, d)

def month_window(anchor: date, mode: str) -> Tuple[date, date]:
    """
    Returns (start, end) inclusive.
    - calendar: first day of anchor.month to its last day
    - rolling1m: anchor minus one calendar month (clipped) + 1 day .. anchor
    - rolling30d: last 30 days ending at anchor (inclusive)
    """
    if mode == "calendar":
        start = anchor.replace(day=1)
        end = (start + relativedelta(months=+1)) - timedelta(days=1)
        return start, end
    if mode == "rolling1m":
        end = anchor
        start = (anchor - relativedelta(months=+1)) + timedelta(days=1)
        return start, end
    if mode == "rolling30d":
        end = anchor
        start = anchor - timedelta(days=29)
        return start, end
    raise ValueError("mode must be one of: calendar, rolling1m, rolling30d")

def _rolling_7d_window(anchor: date) -> Tuple[date, date]:
    end = anchor
    start = anchor - timedelta(days=6)
    return start, end

def _rolling_365d_window(anchor: date) -> Tuple[date, date]:
    end = anchor
    start = anchor - timedelta(days=364)
    return start, end

async def _fetch_all_days(location: str, start: date, end: date, unit_group: str) -> Dict[str, float]:
    """
    Fetch daily temps for [start..end] once using VC Timeline.
    Returns dict { 'YYYY-MM-DD': temp }.
    """
    # Validate location to prevent SSRF attacks
    from utils.validation import validate_location_for_ssrf
    try:
        location = validate_location_for_ssrf(location)
    except ValueError as e:
        raise HTTPException(400, f"Invalid location: {str(e)}")
    
    if not API_KEY:
        raise HTTPException(500, "Missing VISUAL_CROSSING_API_KEY")
    
    # URL-encode location for use in path
    from urllib.parse import quote
    encoded_location = quote(location, safe='')
    url = f"{VC_BASE_URL}/timeline/{encoded_location}/{start.isoformat()}/{end.isoformat()}"
    params = {
        "unitGroup": _convert_unit_group_for_vc(unit_group),
        "include": "days",
        "elements": "datetime,temp",  # keep payload small
        "contentType": "json",
            "key": API_KEY,
    }
    
    async with _rolling_sem:
        http_client = await _get_rolling_http_client()
        async with http_client.get(url, params=params, headers={"Accept-Encoding": "gzip"}) as r:
            if r.status >= 400:
                text = await r.text()
                raise HTTPException(r.status, f"Visual Crossing error: {text[:200]}")
            data = await r.json()
    
    # Extract and store timezone from Visual Crossing response
    try:
        from cache_utils import store_location_timezone
        from config import CACHE_ENABLED
        timezone_str = data.get('timezone')
        if CACHE_ENABLED and timezone_str:
            # Try to get Redis client from global cache
            try:
                from cache_utils import get_cache
                cache = get_cache()
                if cache and cache.redis:
                    store_location_timezone(location, timezone_str, cache.redis)
            except Exception:
                pass  # Silently fail if cache not available
    except Exception:
        pass  # Silently fail if imports fail
    
    days = data.get("days") or []
    out: Dict[str, float] = {}
    for d in days:
        dt = d.get("datetime")
        t = d.get("temp")
        if isinstance(dt, str) and isinstance(t, (int, float)):
            out[dt] = float(t)
    
    if not out:
        raise HTTPException(404, "No daily temperatures returned")
    return out

async def _get_daily_map_for_days(location: str, unit_group: str, anchor_d: date) -> Dict[str, float]:
    """
    Fetch daily data only for the specific days needed (anchor and 3 previous days).
    This is much more efficient than fetching years of data.
    """
    # Only fetch the 4 days we need: anchor, anchor-1, anchor-2, anchor-3
    start = anchor_d - timedelta(days=3)
    end = anchor_d
    
    cache_key = f"dailies:{location}:{unit_group}:{start.isoformat()}:{end.isoformat()}"
    cached = await daily_cache.get(cache_key)
    if cached:
        return {k: float(v) for k, v in json.loads(cached).items()}

    daily_map = await _fetch_all_days(location, start, end, unit_group)

    await daily_cache.set(cache_key, json.dumps(daily_map).encode())
    return daily_map

def _avg_window_for_year(
    daily_map: Dict[str, float],
    year: int,
    start: date,
    end: date,
) -> Optional[float]:
    """
    Compute average temp for [start..end] in the given year, where start/end are *month/day* on anchor year.
    We retarget them into `year` (and cross into year-1 if the start underflows).
    """
    # Convert anchor-year window into this `year`
    delta_days = (end - start).days
    # Anchor end in this year (clamp 29 Feb)
    m_end, d_end = _leap_clamp((end.month, end.day), year)
    end_y = date(year, m_end, d_end)
    start_y = end_y - timedelta(days=delta_days)

    # Iterate [start_y .. end_y]
    total = 0.0
    n = 0
    cur = start_y
    while cur <= end_y:
        key = cur.isoformat()
        val = daily_map.get(key)
        if val is not None:
            total += val
            n += 1
        cur += timedelta(days=1)
    if n == 0:
        return None
    return round(total / n, 2)

def _series_for_window(
    daily_map: Dict[str, float],
    start_year: int,
    end_year: int,
    start_anchor: date,  # in anchor year
    end_anchor: date,
) -> List[YearTemp]:
    out: List[YearTemp] = []
    for y in range(start_year, end_year + 1):
        avg = _avg_window_for_year(daily_map, y, start_anchor, end_anchor)
        if avg is not None:
            out.append(YearTemp(year=y, temperature=avg))
    return out


@router.get("/v1/records/rolling-bundle/{location}/{anchor}")
async def rolling_bundle(
    request: Request,
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"] = Query(UNIT_GROUP_DEFAULT),
):
    """This endpoint has been removed. Use individual period endpoints instead."""
    return _rolling_bundle_gone_response(
        request,
        "The rolling-bundle endpoint has been removed",
        location,
        anchor,
        unit_group
    )

async def _rolling_bundle_impl(
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"],
    month_mode: Literal["calendar", "rolling1m", "rolling30d"],
    days_back: int,
    include: str | None,
    exclude: str | None,
    redis_client: redis.Redis,
):
    anchor_d = _safe_parse_date(anchor)
    
    # Parse CSV parameters and determine which sections to include
    def _parse_csv(s: str | None, allowed: Set[str], max_length: int = 100, max_items: int = 20) -> Set[str]:
        """Parse and validate CSV parameter with security checks."""
        if not s:
            return set()
        
        # Length validation to prevent DoS
        if len(s) > max_length:
            raise HTTPException(
                status_code=400,
                detail=f"CSV parameter too long (max {max_length} chars)"
            )
        
        parsed = {p.strip() for p in s.split(",") if p.strip()}
        
        # Limit number of items to prevent DoS
        if len(parsed) > max_items:
            raise HTTPException(
                status_code=400,
                detail=f"Too many items in CSV (max {max_items})"
            )
        
        # Validate against whitelist of allowed values
        invalid = parsed - allowed
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid values: {', '.join(sorted(invalid))}. Allowed: {', '.join(sorted(allowed))}"
            )
        
        return parsed

    inc = _parse_csv(include, ALLOWED_SECTIONS) if include else set()
    exc = _parse_csv(exclude, ALLOWED_SECTIONS) if exclude else set()

    if inc:
        wanted = inc
    else:
        # Default to all sections for full functionality
        # Users can exclude sections if needed
        wanted = ALLOWED_SECTIONS - exc
    
    # Get the anchor day's complete daily response
    async def get_daily_response(anchor_date: date) -> Dict:
        """Get complete daily endpoint response for a given date."""
        try:
            mmdd = anchor_date.strftime("%m-%d")
            from routers.v1_records import get_temperature_data_v1
            return await get_temperature_data_v1(location, "daily", mmdd, unit_group, redis_client)
        except Exception as e:
            return {"error": str(e), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}

    # Get daily responses concurrently (anchor day + previous days)
    daily_tasks = []
    if "day" in wanted:
        daily_tasks.append(("anchor", get_daily_response(anchor_d)))
    
    if days_back > 0:
        for i in range(1, days_back + 1):
            prev_date = anchor_d - timedelta(days=i)
            daily_tasks.append((f"day_{i}", get_daily_response(prev_date)))
    
    # Execute all daily requests concurrently
    daily_results = {}
    if daily_tasks:
        results = await asyncio.gather(*[task[1] for task in daily_tasks], return_exceptions=True)
        for i, (day_name, result) in enumerate(zip([task[0] for task in daily_tasks], results)):
            if isinstance(result, Exception):
                daily_results[day_name] = {"error": str(result), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}
            else:
                daily_results[day_name] = result
    
    day_response = daily_results.get("anchor")
    previous_days = [daily_results[f"day_{i}"] for i in range(1, days_back + 1) if f"day_{i}" in daily_results]

    # Get complete responses for weekly, monthly, and yearly
    async def get_period_response(period: str, anchor_date: date) -> Optional[Dict]:
        """Get complete endpoint response for a given period."""
        try:
            if period in ["weekly", "monthly", "yearly"]:
                mmdd = anchor_date.strftime("%m-%d")
                from routers.v1_records import get_temperature_data_v1
                return await get_temperature_data_v1(location, period, mmdd, unit_group, redis_client)
            return None
        except Exception as e:
            return {"error": str(e), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}

    # Get period responses concurrently (only if requested)
    period_tasks = []
    if "week" in wanted:
        period_tasks.append(("weekly", get_period_response("weekly", anchor_d)))
    if "month" in wanted:
        period_tasks.append(("monthly", get_period_response("monthly", anchor_d)))
    if "year" in wanted:
        period_tasks.append(("yearly", get_period_response("yearly", anchor_d)))
    
    # Execute all period requests concurrently
    period_results = {}
    if period_tasks:
        results = await asyncio.gather(*[task[1] for task in period_tasks], return_exceptions=True)
        for i, (period_name, result) in enumerate(zip([task[0] for task in period_tasks], results)):
            if isinstance(result, Exception):
                period_results[period_name] = {"error": str(result), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}
            else:
                period_results[period_name] = result
    
    weekly_response = period_results.get("weekly")
    monthly_response = period_results.get("monthly")
    yearly_response = period_results.get("yearly")

    # Generate appropriate notes
    if month_mode == "calendar":
        notes = "Month uses full calendar month (1st to last day of anchor month)."
    elif month_mode == "rolling1m":
        notes = "Month uses calendar-aware 1-month window ending on anchor (EOM-clipped)."
    else:  # rolling30d
        notes = "Month uses fixed 30-day rolling window ending on anchor (consistent with /v1/records/monthly)."
    
    notes += f" Includes {days_back} previous days. Weekly/monthly/yearly data uses historysummary endpoints for 50-year coverage. All API calls are optimized to run concurrently for better performance."

    # Build response with only requested sections
    response_data = {
        "location": location,
        "anchor": anchor_d,
        "unit_group": unit_group,
        "metadata": {
            "anchor_date": anchor_d.isoformat(),
            "month_mode": month_mode,
            "days_back": days_back,
            "included_sections": list(wanted),
            "data_sources": {
                "daily": "Timeline API",
                "weekly": "historysummary API",
                "monthly": "historysummary API", 
                "yearly": "historysummary API"
            }
        },
        "notes": notes,
    }
    
    # Only include sections that were requested
    if "day" in wanted and day_response is not None:
        response_data["day"] = day_response
        response_data["daily"] = day_response  # Same as day for convenience
    
    # Always include previous_days if we have any (controlled by days_back parameter)
    if previous_days:
        response_data["previous_days"] = previous_days
    
    if "week" in wanted and weekly_response is not None:
        response_data["weekly"] = weekly_response
    
    if "month" in wanted and monthly_response is not None:
        response_data["monthly"] = monthly_response
    
    if "year" in wanted and yearly_response is not None:
        response_data["yearly"] = yearly_response

    return RollingBundleResponse(**response_data)

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