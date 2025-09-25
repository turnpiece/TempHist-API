# routers/records_agg.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional, Tuple, Literal, Set
import os, aiohttp, asyncio, json
from datetime import datetime, date, timedelta
from pydantic import BaseModel
from constants import VC_BASE_URL
from dateutil.relativedelta import relativedelta

router = APIRouter()
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
UNIT_GROUP_DEFAULT = os.getenv("UNIT_GROUP", "metric")

_client: Optional[aiohttp.ClientSession] = None
_sem = asyncio.Semaphore(2)

async def _client_session() -> aiohttp.ClientSession:
    global _client
    if _client is None or _client.closed:
        _client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    return _client

def _years_range() -> tuple[int, int]:
    y = datetime.now().year
    return (y - 50, y)

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
        "unitGroup": unit_group,
        "locations": location,
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
                raise HTTPException(status_code=502, detail=f"VC historysummary {resp.status}: {text[:180]}")
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
    unit_group: Literal["metric", "us"]
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
        _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
    return _http

# Helper functions
def _safe_parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "anchor must be YYYY-MM-DD")

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
    if not API_KEY:
        raise HTTPException(500, "Missing VISUAL_CROSSING_API_KEY")
    
    url = f"{VC_BASE_URL}/timeline/{location}/{start.isoformat()}/{end.isoformat()}"
    params = {
        "unitGroup": unit_group,
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


@router.get("/v1/records/rolling-bundle/{location}/{anchor}", response_model=RollingBundleResponse)
async def rolling_bundle(
    location: str,
    anchor: str,
    unit_group: Literal["metric", "us"] = Query(UNIT_GROUP_DEFAULT),
    month_mode: Literal["calendar", "rolling1m", "rolling30d"] = Query("rolling1m"),
    days_back: int = Query(DEFAULT_DAYS_BACK, ge=0, le=10, description="Number of previous days to include (0-10)"),
    include: str | None = Query(None, description="CSV of sections to include"),
    exclude: str | None = Query(None, description="CSV of sections to exclude (ignored if include is present)"),
):
    """
    Returns cross-year series for:
      - day (anchor day) and previous days (configurable via days_back parameter)
      - weekly data (complete weekly endpoint response)
      - monthly data (complete monthly endpoint response)
      - yearly data (complete yearly endpoint response)
    
    Each period enclosure contains the complete JSON response from its respective endpoint,
    including values, average, trend, summary, and metadata.
    
    Query params:
    - include: CSV of sections to include. If present, exclude is ignored.
    - exclude: CSV of sections to exclude (valid: day,week,month,year).
    - days_back: Number of previous days to include (0-10, controlled separately from include/exclude).
    Examples:
    - ?include=week,month,year
    - ?exclude=day
    - ?days_back=3 (includes 3 previous days regardless of include/exclude)
    """
    anchor_d = _safe_parse_date(anchor)
    
    # Parse CSV parameters and determine which sections to include
    def _parse_csv(s: str | None) -> Set[str]:
        return {p.strip() for p in s.split(",")} if s else set()

    inc = (_parse_csv(include) & ALLOWED_SECTIONS) if include else set()
    exc = (_parse_csv(exclude) & ALLOWED_SECTIONS) if exclude else set()

    if inc:
        wanted = inc
    else:
        wanted = ALLOWED_SECTIONS - exc
    
    # Get the anchor day's complete daily response
    async def get_daily_response(anchor_date: date) -> Dict:
        """Get complete daily endpoint response for a given date."""
        try:
            mmdd = anchor_date.strftime("%m-%d")
            from main import get_temperature_data_v1
            return await get_temperature_data_v1(location, "daily", mmdd)
        except Exception as e:
            return {"error": str(e), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}

    # Get the anchor day's response (only if day is wanted)
    day_response = None
    if "day" in wanted:
        day_response = await get_daily_response(anchor_d)
    
    # Get previous days' responses (controlled by days_back parameter)
    previous_days = []
    if days_back > 0:
        for i in range(1, days_back + 1):
            prev_date = anchor_d - timedelta(days=i)
            prev_response = await get_daily_response(prev_date)
            previous_days.append(prev_response)

    # Get complete responses for weekly, monthly, and yearly
    async def get_period_response(period: str, anchor_date: date) -> Optional[Dict]:
        """Get complete endpoint response for a given period."""
        try:
            if period == "weekly":
                mmdd = anchor_date.strftime("%m-%d")
                from main import get_temperature_data_v1
                return await get_temperature_data_v1(location, "weekly", mmdd)
            elif period == "monthly":
                mmdd = anchor_date.strftime("%m-%d")
                from main import get_temperature_data_v1
                return await get_temperature_data_v1(location, "monthly", mmdd)
            elif period == "yearly":
                # Use December 31st as the identifier for yearly data
                from main import get_temperature_data_v1
                return await get_temperature_data_v1(location, "yearly", "12-31")
            return None
        except Exception as e:
            return {"error": str(e), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}

    # Get period responses (only if requested)
    weekly_response = None
    if "week" in wanted:
        weekly_response = await get_period_response("weekly", anchor_d)
    
    monthly_response = None
    if "month" in wanted:
        monthly_response = await get_period_response("monthly", anchor_d)
    
    yearly_response = None
    if "year" in wanted:
        yearly_response = await get_period_response("yearly", anchor_d)

    # Generate appropriate notes
    if month_mode == "calendar":
        notes = "Month uses full calendar month (1st to last day of anchor month)."
    elif month_mode == "rolling1m":
        notes = "Month uses calendar-aware 1-month window ending on anchor (EOM-clipped)."
    else:  # rolling30d
        notes = "Month uses fixed 30-day rolling window ending on anchor (consistent with /v1/records/monthly)."
    
    notes += f" Includes {days_back} previous days. Weekly/monthly/yearly data uses historysummary endpoints for 50-year coverage."

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