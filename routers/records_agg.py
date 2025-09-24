# routers/records_agg.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional, Tuple, Literal
import os, aiohttp, asyncio, json
from datetime import datetime, date, timedelta
from pydantic import BaseModel
from constants import VC_BASE_URL
from dateutil.relativedelta import relativedelta

router = APIRouter()
VC_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
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
    if VC_KEY is None:
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
        "key": VC_KEY,
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
    temp: float

class RollingSeries(BaseModel):
    values: List[YearTemp]
    count: int

class RollingBundleResponse(BaseModel):
    period: str = "rolling"
    location: str
    anchor: date
    unit_group: Literal["metric", "us"]
    day: RollingSeries
    day_minus_1: RollingSeries
    day_minus_2: RollingSeries
    day_minus_3: RollingSeries
    week: RollingSeries
    month: RollingSeries
    year: RollingSeries
    notes: Optional[str] = None

# Configuration
YEARS_BACK = 50  # 50-year history
VC_MAX_CONCURRENCY = 2

# Cache for daily data
class KVCache:
    def __init__(self, redis=None, ttl_seconds: int = 24 * 60 * 60):
        self.redis = redis
        self.ttl = ttl_seconds
    
    async def get(self, key: str) -> Optional[bytes]:
        if not self.redis:
            return None
        return await self.redis.get(key)
    
    async def set(self, key: str, value: bytes):
        if self.redis:
            await self.redis.set(key, value, ex=self.ttl)

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
    if not VC_KEY:
        raise HTTPException(500, "Missing VISUAL_CROSSING_API_KEY")
    
    url = f"{VC_BASE_URL}/timeline/{location}/{start.isoformat()}/{end.isoformat()}"
    params = {
        "unitGroup": unit_group,
        "include": "days",
        "elements": "datetime,temp",  # keep payload small
        "contentType": "json",
        "key": VC_KEY,
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

async def _get_daily_map(location: str, unit_group: str, now: date) -> Dict[str, float]:
    """
    Load from cache or fetch and cache. Add a 31-day pad before the earliest year
    so windows crossing New Year still work for that first year.
    """
    start_year, end_year = _years_range(now)
    pad_start = date(start_year - 1, 12, 1)
    end = date(end_year, 12, 31)

    cache_key = f"dailies:{location}:{unit_group}:{pad_start.isoformat()}:{end.isoformat()}"
    cached = await daily_cache.get(cache_key)
    if cached:
        return {k: float(v) for k, v in json.loads(cached).items()}

    daily_map = await _fetch_all_days(location, pad_start, end, unit_group)

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
            out.append(YearTemp(year=y, temp=avg))
    return out

@router.get("/v1/records/rolling-bundle/{location}/{anchor}", response_model=RollingBundleResponse)
async def rolling_bundle(
    location: str,
    anchor: str,
    unit_group: Literal["metric", "us"] = Query(UNIT_GROUP_DEFAULT),
    month_mode: Literal["calendar", "rolling1m", "rolling30d"] = Query("rolling1m"),
):
    """
    Returns cross-year series for:
      - day (anchor day), day-1, day-2, day-3
      - week ending anchor (7 days)
      - month ending anchor (calendar, rolling1m, or rolling30d)
      - year ending anchor (365d)
    All computed from a single cached daily series per location.
    """
    anchor_d = _safe_parse_date(anchor)
    now = anchor_d  # for range selection
    start_year, end_year = _years_range(now)

    daily_map = await _get_daily_map(location, unit_group, now)

    # Day windows (offsets 0..3)
    def anchor_for_offset(off: int) -> date:
        return anchor_d - timedelta(days=off)

    # Day-of-year retarget window equals (start=end=that day)
    def day_series(off: int) -> RollingSeries:
        a = anchor_for_offset(off)
        # define window as [a..a]
        vals = _series_for_window(daily_map, start_year, end_year, a, a)
        return RollingSeries(values=vals, count=len(vals))

    day = day_series(0)
    day_minus_1 = day_series(1)
    day_minus_2 = day_series(2)
    day_minus_3 = day_series(3)

    # Week window (7d)
    w_start, w_end = _rolling_7d_window(anchor_d)
    week_vals = _series_for_window(daily_map, start_year, end_year, w_start, w_end)

    # Month window
    m_start, m_end = month_window(anchor_d, month_mode)
    if month_mode == "calendar":
        notes = "Month uses full calendar month (1st to last day of anchor month)."
    elif month_mode == "rolling1m":
        notes = "Month uses calendar-aware 1-month window ending on anchor (EOM-clipped)."
    else:  # rolling30d
        notes = "Month uses fixed 30-day rolling window ending on anchor (consistent with /v1/records/monthly)."
    month_vals = _series_for_window(daily_map, start_year, end_year, m_start, m_end)

    # Year window (365d)
    y_start, y_end = _rolling_365d_window(anchor_d)
    year_vals = _series_for_window(daily_map, start_year, end_year, y_start, y_end)

    return RollingBundleResponse(
        location=location,
        anchor=anchor_d,
        unit_group=unit_group,
        day=RollingSeries(values=day.values, count=day.count),
        day_minus_1=RollingSeries(values=day_minus_1.values, count=day_minus_1.count),
        day_minus_2=RollingSeries(values=day_minus_2.values, count=day_minus_2.count),
        day_minus_3=RollingSeries(values=day_minus_3.values, count=day_minus_3.count),
        week=RollingSeries(values=week_vals, count=len(week_vals)),
        month=RollingSeries(values=month_vals, count=len(month_vals)),
        year=RollingSeries(values=year_vals, count=len(year_vals)),
        notes=notes,
    )