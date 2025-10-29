# routers/records_agg.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional, Tuple, Literal, Set, Iterable
import os, aiohttp, asyncio, json, logging
from datetime import datetime, date, timedelta
from pydantic import BaseModel
from constants import VC_BASE_URL
from dateutil.relativedelta import relativedelta

# Import improved caching utilities
from app.cache_utils import cache_get, cache_set, canonicalize_location

logger = logging.getLogger(__name__)

router = APIRouter()
# Strip whitespace/newlines from API key to prevent authentication issues
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
UNIT_GROUP_DEFAULT = os.getenv("UNIT_GROUP", "celsius")

# Log API key status (without exposing the actual key)
if API_KEY:
    logger.info(f"VISUAL_CROSSING_API_KEY loaded: {API_KEY[:10]}... (length: {len(API_KEY)})")
else:
    logger.error("❌ VISUAL_CROSSING_API_KEY is NOT loaded or is empty!")

def _vc_unit_group(u: str) -> str:
    """Map our unit groups to Visual Crossing's expected values."""
    u = (u or "").lower()
    if u in ("c", "celsius", "metric", "si"):
        return "metric"
    if u in ("f", "fahrenheit", "us"):
        return "us"
    # Default sensibly
    return "metric"

@router.api_route("/v1/records/rolling-bundle/test-cors", methods=["GET", "OPTIONS"])
async def test_rolling_bundle_cors():
    """Test CORS for rolling-bundle endpoint"""
    return {"message": "CORS is working for rolling-bundle", "path": "/v1/records/rolling-bundle/test-cors"}

@router.get("/v1/records/rolling-bundle/{location}/{anchor}/preload")
async def rolling_bundle_preload(
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"] = Query(UNIT_GROUP_DEFAULT),
    month_mode: Literal["calendar", "rolling1m", "rolling30d"] = Query("rolling1m"),
):
    """
    Optimized preload endpoint that returns complete chart data, summary, trend, and average.
    Designed for website preloading with full data but optimized to prevent timeouts.
    Returns the same data structure as the main rolling-bundle endpoint but with optimizations.
    """
    try:
        return await asyncio.wait_for(
            _rolling_bundle_preload_impl(location, anchor, unit_group, month_mode),
            timeout=60.0  # 60 second timeout for preload (safer than 90s)
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail="Preload timeout - try again later or use individual endpoints"
        )

async def _rolling_bundle_preload_impl(
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"],
    month_mode: Literal["calendar", "rolling1m", "rolling30d"],
):
    """Optimized implementation that returns complete data for website preloading"""
    anchor_d = _safe_parse_date(anchor)
    
    # Get complete endpoint responses for weekly, monthly, and yearly
    # This gives you the full data structure with chart data, summary, trend, and average
    async def get_complete_period_response(period: str, anchor_date: date) -> Dict:
        """Get complete endpoint response for a given period with all data"""
        try:
            if period in ["weekly", "monthly", "yearly"]:
                mmdd = anchor_date.strftime("%m-%d")
                from main import get_temperature_data_v1
                return await get_temperature_data_v1(location, period, mmdd)
            return {"error": "Invalid period", "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}
        except Exception as e:
            return {"error": str(e), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}
    
    # Get all period responses concurrently (this is the key optimization)
    tasks = [
        get_complete_period_response("weekly", anchor_d),
        get_complete_period_response("monthly", anchor_d),
        get_complete_period_response("yearly", anchor_d)
    ]
    
    weekly_response, monthly_response, yearly_response = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    if isinstance(weekly_response, Exception):
        weekly_response = {"error": str(weekly_response), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}
    if isinstance(monthly_response, Exception):
        monthly_response = {"error": str(monthly_response), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}
    if isinstance(yearly_response, Exception):
        yearly_response = {"error": str(yearly_response), "values": [], "average": {}, "trend": {}, "summary": "", "metadata": {}}
    
    # Generate appropriate notes
    if month_mode == "calendar":
        notes = "Month uses full calendar month (1st to last day of anchor month)."
    elif month_mode == "rolling1m":
        notes = "Month uses calendar-aware 1-month window ending on anchor (EOM-clipped)."
    else:  # rolling30d
        notes = "Month uses fixed 30-day rolling window ending on anchor (consistent with /v1/records/monthly)."
    
    notes += " Preload endpoint - optimized for website data loading with complete chart data, summary, trend, and average."
    
    # Build response with complete data structure (same as main rolling-bundle)
    response_data = {
        "location": location,
        "anchor": anchor_d,
        "unit_group": unit_group,
        "metadata": {
            "anchor_date": anchor_d.isoformat(),
            "month_mode": month_mode,
            "endpoint": "preload",
            "optimized": True,
            "included_sections": ["week", "month", "year"],
            "data_sources": {
                "weekly": "timeline + local aggregation",
                "monthly": "timeline + local aggregation", 
                "yearly": "timeline + local aggregation"
            }
        },
        "notes": notes,
        # Include all the data your website needs
        "weekly": weekly_response,
        "monthly": monthly_response,
        "yearly": yearly_response,
    }
    
    return response_data

_client: Optional[aiohttp.ClientSession] = None
_sem = asyncio.Semaphore(1)  # Professional plan: concurrency=1 to avoid 429s

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


# --- New: timeline-based fetch + per-year rolling means ---

async def _vc_timeline_days(
    location: str,
    start_iso: str,
    end_iso: str,
    unit_group: str = UNIT_GROUP_DEFAULT,
    elements: str = "datetime,temp",
    include: str = "obs,stats,stations",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Fetch raw daily rows from the modern /timeline endpoint in one go.
    Returns Visual Crossing JSON (expects top-level 'days' list).
    """
    if not API_KEY or len(API_KEY) == 0:
        raise RuntimeError("VISUAL_CROSSING_API_KEY is not configured or is empty")

    # Validate that we have both start and end dates
    if not start_iso or not end_iso or start_iso == end_iso:
        raise ValueError(f"Timeline needs a start and end date, got: start={start_iso} end={end_iso}")

    url = f"{VC_BASE_URL}/timeline/{location}/{start_iso}/{end_iso}"
    params = {
        "unitGroup": _vc_unit_group(unit_group),
        "elements": elements,          # keep payload small
        "include": include,            # include stations if present
        "contentType": "json",
        "key": API_KEY,
    }
    
    # Log request details for debugging (without exposing API key)
    params_safe = {k: (v if k != "key" else f"{v[:10]}..." if v else "MISSING") for k, v in params.items()}
    logger.info(f"VC Request URL: {url}")
    logger.info(f"VC Request params: {params_safe}")
    logger.info(f"VC API Key present: {bool(API_KEY)}, length: {len(API_KEY) if API_KEY else 0}")
    
    # Verify params are being passed
    import urllib.parse
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    # Only mask the key if it exists and has content
    if API_KEY and len(API_KEY) > 0:
        logger.info(f"VC Full request URL (with masked key): {full_url.replace(API_KEY, 'KEY_HIDDEN')}")
    else:
        logger.error(f"VC Full request URL (KEY MISSING!): {full_url}")
    
    sess = await _client_session()
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            async with _sem:
                async with sess.get(url, params=params, headers={"Accept-Encoding": "gzip"}) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        logger.error(f"Visual Crossing API error {resp.status} for {location} ({start_iso} to {end_iso}): {text[:500]}")
                        
                        # Handle specific error cases
                        if resp.status == 400:
                            if "license level" in text.lower() or "not permitted" in text.lower():
                                # This is a permanent feature gate - don't retry
                                raise ValueError(f"Visual Crossing license limitation: {text[:200]}")
                            elif "bad request" in text.lower():
                                # Bad request - might be invalid parameters
                                raise ValueError(f"VC timeline bad request (400): {text[:200]}")
                            else:
                                # Other 400 errors might be temporary
                                if attempt < max_retries:
                                    logger.warning(f"Retrying 400 error for {location} (attempt {attempt + 1}/{max_retries + 1})")
                                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                    continue
                                raise ValueError(f"VC timeline 400: {text[:200]}")
                        elif resp.status == 429:
                            # Rate limit - retry with exponential backoff
                            if attempt < max_retries:
                                wait_time = 2 ** attempt
                                logger.warning(f"Rate limited for {location}, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries + 1})")
                                await asyncio.sleep(wait_time)
                                continue
                            raise ValueError(f"VC timeline rate limited (429): {text[:200]}")
                        elif resp.status == 500:
                            # Server error - retry with exponential backoff
                            if attempt < max_retries:
                                wait_time = 2 ** attempt
                                logger.warning(f"Server error for {location}, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries + 1})")
                                await asyncio.sleep(wait_time)
                                continue
                            raise ValueError(f"VC timeline server error (500): {text[:200]}")
                        else:
                            if attempt < max_retries:
                                wait_time = 2 ** attempt
                                logger.warning(f"HTTP {resp.status} for {location}, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries + 1})")
                                await asyncio.sleep(wait_time)
                                continue
                            raise ValueError(f"VC timeline {resp.status}: {text[:200]}")
                    
                    result = await resp.json()
                    logger.info(f"Successfully fetched timeline data for {location} ({start_iso} to {end_iso}) on attempt {attempt + 1}")
                    return result
                    
        except asyncio.TimeoutError as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.warning(f"Timeout for {location} ({start_iso} to {end_iso}), waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"Final timeout after {max_retries + 1} attempts for {location} ({start_iso} to {end_iso})")
                raise ValueError(f"VC timeline timeout after {max_retries + 1} attempts: {str(e)}")
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.warning(f"Request failed for {location} ({start_iso} to {end_iso}): {str(e)}, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"Final failure after {max_retries + 1} attempts for {location} ({start_iso} to {end_iso}): {str(e)}")
                raise
    
    # This should never be reached, but just in case
    raise ValueError(f"Unexpected error: exhausted all retry attempts for {location}")

def _split_date_range_optimally(start_date: date, end_date: date, max_days: int = 10000) -> List[Tuple[date, date]]:
    """
    Split a date range into optimal chunks to stay under the 10,000 record limit.
    For very large ranges (>20 years), use smaller chunks to avoid timeouts.
    Returns list of (start, end) date tuples.
    """
    total_days = (end_date - start_date).days + 1
    
    # For very large ranges, use smaller chunks to avoid timeouts
    if total_days > 7300:  # More than 20 years
        max_days = 5000  # Reduce chunk size for very large ranges
        logger.info(f"Large date range detected ({total_days} days), using smaller chunks ({max_days} days max)")
    elif total_days > 3650:  # More than 10 years
        max_days = 7000  # Slightly smaller chunks for large ranges
        logger.info(f"Medium-large date range detected ({total_days} days), using medium chunks ({max_days} days max)")
    
    if total_days <= max_days:
        return [(start_date, end_date)]
    
    # Calculate number of chunks needed
    num_chunks = (total_days + max_days - 1) // max_days  # Ceiling division
    
    chunks = []
    current_start = start_date
    
    for i in range(num_chunks):
        # Calculate end date for this chunk
        days_in_chunk = min(max_days, (end_date - current_start).days + 1)
        current_end = current_start + timedelta(days=days_in_chunk - 1)
        
        chunks.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
        
        # Stop if we've reached the end
        if current_start > end_date:
            break
    
    logger.info(f"Split {total_days} days into {len(chunks)} chunks (max {max_days} days per chunk)")
    return chunks

def _daterange_for_rolling_window(
    min_year: int,
    max_year: int,
    end_month: int,
    end_day: int,
    window_days: int,
) -> Tuple[date, date]:
    """
    Build a single continuous date range that covers every year's rolling window.
    For early-January windows (that spill into prev year), widen the start by one year.
    """
    # Earliest window starts on (min_year, end_md) - (window_days - 1).
    end_dt = date(max_year, end_month, end_day)
    start_dt = date(min_year, end_month, end_day) - timedelta(days=window_days - 1)
    if start_dt.year < min_year:  # handle windows that span previous year (e.g., Jan 02)
        start_dt = date(min_year - 1, 1, 1)
    return start_dt, end_dt

def _per_year_rolling_means(
    days: Iterable[Dict[str, Any]],
    end_month: int,
    end_day: int,
    window_days: int,
    min_days_required: int,
) -> Dict[int, Optional[float]]:
    """
    Compute per-year mean(temp) over the rolling window ending (MM-DD).
    Applies a completeness threshold (min_days_required) to avoid sparse artifacts.
    """
    # Index days by date for fast slicing
    by_date: Dict[date, float] = {}
    for d in days:
        # VC returns ISO 'datetime' (YYYY-MM-DD)
        s = d.get("datetime")
        t = d.get("temp")
        if not s or t is None:
            continue
        y, m, dd = map(int, s.split("-"))
        by_date[date(y, m, dd)] = float(t)

    logger.debug(f"Indexed {len(by_date)} temperature records for rolling means calculation")
    logger.debug(f"Window: {window_days} days, minimum required: {min_days_required} days")

    results: Dict[int, Optional[float]] = {}
    # Work out min/max year from payload keys (safer than assuming full range)
    if not by_date:
        logger.warning("No temperature data available for rolling means calculation")
        return results
    years = range(min(dt.year for dt in by_date), max(dt.year for dt in by_date) + 1)
    logger.debug(f"Processing years {min(years)} to {max(years)}")

    insufficient_data_years = []
    for y in years:
        try:
            end_dt = date(y, end_month, end_day)
        except ValueError:
            # Handles 2/29 in non-leap years etc.: skip that year
            logger.debug(f"Skipping year {y} due to invalid end date {end_month}-{end_day}")
            results[y] = None
            continue

        start_dt = end_dt - timedelta(days=window_days - 1)
        # Collect temps across the window; window can extend into prev year
        vals: List[float] = []
        cur = start_dt
        for _ in range(window_days):
            v = by_date.get(cur)
            if v is not None:
                vals.append(v)
            cur += timedelta(days=1)

        if len(vals) >= min_days_required:
            mean_temp = sum(vals) / len(vals)
            results[y] = mean_temp
            logger.debug(f"Year {y}: {len(vals)}/{window_days} days available, mean temp: {mean_temp:.1f}°C")
        else:
            results[y] = None
            insufficient_data_years.append(f"{y}({len(vals)}/{window_days})")
            logger.debug(f"Year {y}: insufficient data - {len(vals)}/{window_days} days (need {min_days_required})")

    if insufficient_data_years:
        logger.info(f"Insufficient data for {len(insufficient_data_years)} years: {', '.join(insufficient_data_years[:10])}{'...' if len(insufficient_data_years) > 10 else ''}")

    return results

def _filter_days_by_station(days: List[Dict[str, Any]], allowed: Set[str], min_completeness: float = 0.9) -> List[Dict[str, Any]]:
    """
    Filter days by station whitelist for continuity (e.g., Berlin airport changes).
    Falls back to all stations if data completeness is below min_completeness.
    """
    if not allowed:
        return days
    
    # First pass: filter by whitelisted stations
    filtered_days = []
    for d in days:
        st = d.get("stations")
        if st is None:
            filtered_days.append(d)
            continue
        if isinstance(st, list) and any(s in allowed for s in st):
            filtered_days.append(d)
        elif isinstance(st, dict) and any(s in allowed for s in st.keys()):
            filtered_days.append(d)
    
    # Check data completeness
    total_days = len(days)
    filtered_days_count = len(filtered_days)
    completeness = filtered_days_count / total_days if total_days > 0 else 0.0
    
    # If completeness is too low, fall back to all stations
    if completeness < min_completeness:
        return days
    
    return filtered_days

def _dict_to_values_list(d: Dict[int, Optional[float]]) -> List[Dict[str, Any]]:
    """Convert year->mean dict to values list format."""
    out = []
    for y in sorted(d):
        v = d[y]
        if v is None:
            continue  # or include with null if your UI expects holes
        out.append({"year": y, "temp": round(v, 2)})
    return out

async def _rolling_week_per_year_via_timeline(
    location: str,
    min_year: int,
    max_year: int,
    mm: int,
    dd: int,
    unit_group: str = UNIT_GROUP_DEFAULT,
) -> Dict[int, Optional[float]]:
    """Get rolling 7-day means per year using cached daily data."""
    # Get the full date range for all years
    start_dt, end_dt = _daterange_for_rolling_window(min_year, max_year, mm, dd, window_days=7)
    
    # Fetch daily data with caching
    daily_map = await _fetch_daily_data_cached(location, start_dt, end_dt, unit_group)
    
    # Convert daily map to the format expected by _per_year_rolling_means
    all_days = []
    for date_str, temp in daily_map.items():
        all_days.append({
            "datetime": date_str,
            "temp": temp
        })
    
    # Apply station filtering if available (with fallback for low completeness)
    station_whitelist = _get_station_whitelist(location)
    if station_whitelist:
        original_count = len(all_days)
        all_days = _filter_days_by_station(all_days, station_whitelist, min_completeness=0.9)
        filtered_count = len(all_days)
        logger.info(f"Station filtering for {location}: {original_count} -> {filtered_count} days")
    
    logger.info(f"Computing weekly rolling means for {location} (mm={mm}, dd={dd}, window=7 days, min_required=5)")
    return _per_year_rolling_means(all_days, mm, dd, window_days=7, min_days_required=5)

async def _rolling_30d_per_year_via_timeline(
    location: str,
    min_year: int,
    max_year: int,
    mm: int,
    dd: int,
    unit_group: str = UNIT_GROUP_DEFAULT,
) -> Dict[int, Optional[float]]:
    """Get rolling 30-day means per year using cached daily data."""
    # Get the full date range for all years
    start_dt, end_dt = _daterange_for_rolling_window(min_year, max_year, mm, dd, window_days=30)
    
    # Fetch daily data with caching
    daily_map = await _fetch_daily_data_cached(location, start_dt, end_dt, unit_group)
    
    # Convert daily map to the format expected by _per_year_rolling_means
    all_days = []
    for date_str, temp in daily_map.items():
        all_days.append({
            "datetime": date_str,
            "temp": temp
        })
    
    # Apply station filtering if available (with fallback for low completeness)
    station_whitelist = _get_station_whitelist(location)
    if station_whitelist:
        original_count = len(all_days)
        all_days = _filter_days_by_station(all_days, station_whitelist, min_completeness=0.9)
        filtered_count = len(all_days)
        logger.info(f"Station filtering for {location}: {original_count} -> {filtered_count} days")
    
    logger.info(f"Computing monthly rolling means for {location} (mm={mm}, dd={dd}, window=30 days, min_required=20)")
    return _per_year_rolling_means(all_days, mm, dd, window_days=30, min_days_required=20)

async def rolling_year_per_year_via_timeline(
    location: str,
    min_year: int,
    max_year: int,
    end_month: int,
    end_day: int,
    *,
    unit_group: str = "metric",
    window_days: int = 365,        # keep 365 for comparability across years
    min_days_required: int = 300,  # completeness guard (tune 300–330)
) -> Dict[int, Optional[float]]:
    """
    For each year Y in [min_year, max_year], compute the mean of 'temp' over the
    rolling window of 'window_days' ending on (Y-end_month-end_day), inclusive.
    Uses cached daily data to avoid duplicate VC calls.
    Returns {year: mean or None if insufficient data}.
    """
    # Build the full date range for all years
    last_end = date(max_year, end_month, end_day)
    first_end = date(min_year, end_month, end_day)
    global_start = first_end - timedelta(days=window_days - 1)
    
    # Fetch daily data with caching
    daily_map = await _fetch_daily_data_cached(location, global_start, last_end, unit_group)
    
    # Convert daily map to the format expected by the processing logic
    all_days = []
    for date_str, temp in daily_map.items():
        all_days.append({
            "datetime": date_str,
            "temp": temp
        })
    
    # Apply station filtering if available (with fallback for low completeness)
    station_whitelist = _get_station_whitelist(location)
    if station_whitelist:
        all_days = _filter_days_by_station(all_days, station_whitelist, min_completeness=0.9)

    # Index by date
    by_date: Dict[date, float] = {}
    for d in all_days:
        s = d.get("datetime")
        t = d.get("temp")
        if not s or t is None:
            continue
        y, m, dd = map(int, s.split("-"))
        by_date[date(y, m, dd)] = float(t)

    # Compute per-year rolling means with completeness threshold
    out: Dict[int, Optional[float]] = {}
    for y in range(min_year, max_year + 1):
        try:
            end_dt = date(y, end_month, end_day)
        except ValueError:
            # If the target end date doesn't exist (e.g., Feb 29 on non-leap year), skip
            out[y] = None
            continue

        start_dt = end_dt - timedelta(days=window_days - 1)
        vals: List[float] = []
        cur = start_dt
        for _ in range(window_days):
            v = by_date.get(cur)
            if v is not None:
                vals.append(v)
            cur += timedelta(days=1)

        out[y] = (sum(vals) / len(vals)) if len(vals) >= min_days_required else None

    return out


@router.get("/v1/records/monthly/{location}/{ym}/series")
async def monthly_series(location: str, ym: str, unit_group: str = UNIT_GROUP_DEFAULT):
    # ym is YYYY-MM (same month across all years)
    try:
        month = datetime.strptime(ym, "%Y-%m").month
    except ValueError:
        raise HTTPException(status_code=400, detail="Identifier must be YYYY-MM")
    
    # Try improved cache first
    try:
        # Convert YYYY-MM to MM-DD format for cache key (using last day of month)
        last_day = (datetime.strptime(ym, "%Y-%m") + relativedelta(months=1) - timedelta(days=1)).day
        cache_date = f"{month:02d}-{last_day:02d}"
        
        # Import redis_client from main module
        import redis
        REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379").strip()
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        
        cached_result = await cache_get(redis_client, "monthly", location, cache_date)
        if cached_result:
            payload, meta = cached_result
            logger.info(f"✅ SERVING IMPROVED CACHED MONTHLY: {location} | {ym}")
            if meta["approximate"]["temporal"]:
                logger.info(f"📅 TEMPORAL APPROXIMATION: Served from {meta['served_from']['end_date']} (Δ{meta['served_from']['temporal_delta_days']}d)")
            return payload
    except Exception as e:
        logger.error(f"Improved cache error for monthly {location}:{ym}: {e}")
    
    min_year, max_year = _years_range()
    
    # Use timeline-based approach for more reliable data
    try:
        # For monthly, we use the last day of the month as the anchor
        # This gives us a 30-day rolling window ending on the last day of the month
        last_day = (datetime.strptime(ym, "%Y-%m") + relativedelta(months=1) - timedelta(days=1)).day
        year_means = await _rolling_30d_per_year_via_timeline(
            location=location,
            min_year=min_year,
            max_year=max_year,
            mm=month,
            dd=last_day,
            unit_group=unit_group
        )
        items = _dict_to_values_list(year_means)
    except Exception as e:
        # Timeline approach failed - raise error
        raise RuntimeError(f"Failed to fetch monthly data: {str(e)}")
    
    # Calculate completeness metadata
    total_years = max_year - min_year + 1
    available_years = len(items)
    missing_years = []
    
    # Track missing years
    available_years_set = {item["year"] for item in items}
    for year in range(min_year, max_year + 1):
        if year not in available_years_set:
            missing_years.append({"year": year, "reason": "insufficient_data_timeline"})
    
    completeness = round(available_years / total_years * 100, 1) if total_years > 0 else 0.0
    
    result = {
        "period": "monthly", 
        "location": location, 
        "identifier": ym, 
        "unit_group": unit_group, 
        "values": items, 
        "count": len(items),
        "note": "timeline + local aggregation (30-day rolling window, min 20 days required)",
        "metadata": {
            "total_years": total_years,
            "available_years": available_years,
            "missing_years": missing_years,
            "completeness": completeness
        }
    }
    
    # Store in improved cache
    try:
        cache_date = f"{month:02d}-{last_day:02d}"
        await cache_set(redis_client, "monthly", location, cache_date, result)
        logger.info(f"💾 STORED IMPROVED CACHED MONTHLY: {location} | {ym}")
    except Exception as e:
        logger.error(f"Failed to store monthly data in improved cache: {e}")
    
    return result

@router.get("/v1/records/weekly/{location}/{week_start}/series")
async def weekly_series(location: str, week_start: str, unit_group: str = UNIT_GROUP_DEFAULT):
    """
    week_start = MM-DD or YYYY-MM-DD (anchor). Uses timeline endpoint for reliable data.
    """
    try:
        mmdd = week_start if len(week_start) == 5 else datetime.strptime(week_start, "%Y-%m-%d").strftime("%m-%d")
        mm, dd = map(int, mmdd.split("-"))
    except Exception:
        raise HTTPException(status_code=400, detail="Identifier must be MM-DD or YYYY-MM-DD")
    
    min_year, max_year = _years_range()
    
    # Use timeline-based approach for more reliable data
    try:
        year_means = await _rolling_week_per_year_via_timeline(
            location=location,
            min_year=min_year,
            max_year=max_year,
            mm=mm,
            dd=dd,
            unit_group=unit_group
        )
        items = _dict_to_values_list(year_means)
    except Exception as e:
        # Timeline approach failed - raise error
        raise RuntimeError(f"Failed to fetch weekly data: {str(e)}")
    
    # Calculate completeness metadata
    total_years = max_year - min_year + 1
    available_years = len(items)
    missing_years = []
    
    # Track missing years
    available_years_set = {item["year"] for item in items}
    for year in range(min_year, max_year + 1):
        if year not in available_years_set:
            missing_years.append({"year": year, "reason": "insufficient_data_timeline"})
    
    completeness = round(available_years / total_years * 100, 1) if total_years > 0 else 0.0
    
    return {
        "period": "weekly", 
        "location": location, 
        "identifier": mmdd, 
        "unit_group": unit_group, 
        "values": items, 
        "count": len(items), 
        "note": "timeline + local aggregation (7-day rolling window, min 5 days required)",
        "metadata": {
            "total_years": total_years,
            "available_years": available_years,
            "missing_years": missing_years,
            "completeness": completeness
        }
    }

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

# Station filtering for continuity (e.g., Berlin airport changes)
STATION_WHITELISTS = {
    "berlin": {"10382099999", "10385099999", "10386099999", "10395099999", "10379099999"},  # Berlin area stations
    "london": {"037720-99999", "037760-99999"},  # London area stations
    # Add more cities as needed
}

#STATION_WHITELISTS = {} # disable station filtering

def _get_station_whitelist(location: str) -> Optional[Set[str]]:
    """Get station whitelist for a location if available."""
    location_lower = location.lower()
    for city, stations in STATION_WHITELISTS.items():
        if city in location_lower:
            return stations
    return None

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

async def _fetch_daily_data_cached(
    location: str,
    start_date: date,
    end_date: date,
    unit_group: str = UNIT_GROUP_DEFAULT,
) -> Dict[str, float]:
    """
    Fetch daily temperature data with caching to avoid duplicate VC calls.
    Returns dict { 'YYYY-MM-DD': temp }.
    """
    # Create cache key based on location, date range, and unit group
    cache_key = f"daily:{location}:{start_date.isoformat()}:{end_date.isoformat()}:{unit_group}"
    
    logger.info(f"Fetching daily data for {location} from {start_date} to {end_date} (unit: {unit_group})")
    
    # Try to get from cache first
    cached_data = await daily_cache.get(cache_key)
    if cached_data:
        try:
            import json
            cached_map = json.loads(cached_data.decode())
            logger.info(f"Using cached data: {len(cached_map)} days for {location}")
            return cached_map
        except Exception as e:
            logger.warning(f"Cache data corrupted for {location}, fetching fresh data: {e}")
            # If cache data is corrupted, continue to fetch fresh data
            pass
    
    # Fetch fresh data from Visual Crossing
    date_chunks = _split_date_range_optimally(start_date, end_date, max_days=10000)
    logger.info(f"Split date range into {len(date_chunks)} chunks for {location}")
    
    all_days = []
    successful_chunks = 0
    failed_chunks = 0
    
    for i, (chunk_start, chunk_end) in enumerate(date_chunks):
        chunk_days_count = (chunk_end - chunk_start).days + 1
        logger.info(f"Fetching chunk {i+1}/{len(date_chunks)}: {chunk_start} to {chunk_end} ({chunk_days_count} days) for {location}")
        
        try:
            payload = await _vc_timeline_days(
                location=location,
                start_iso=chunk_start.isoformat(),
                end_iso=chunk_end.isoformat(),
                unit_group=unit_group,
                elements="datetime,temp",
                include="obs,stats,stations",
                max_retries=3  # Use retry logic
            )
            chunk_days = payload.get("days", [])
            all_days.extend(chunk_days)
            successful_chunks += 1
            logger.info(f"Chunk {i+1} successful: {len(chunk_days)}/{chunk_days_count} days retrieved for {location}")
        except Exception as e:
            failed_chunks += 1
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"Failed to fetch timeline data for {chunk_start} to {chunk_end} ({chunk_days_count} days) for {location}: {error_type}: {error_msg}")
            logger.error(f"Chunk {i+1}/{len(date_chunks)} error details: {repr(e)}")
            
            # Only log full traceback for unexpected errors, not timeout/retry errors
            if "timeout" not in error_msg.lower() and "retry" not in error_msg.lower():
                import traceback
                logger.error(f"Chunk error traceback:\n{traceback.format_exc()}")
            continue
    
    logger.info(f"Timeline fetch complete for {location}: {successful_chunks} successful, {failed_chunks} failed chunks, {len(all_days)} total days")
    
    # Convert to daily map
    daily_map = {}
    valid_temps = 0
    for day_data in all_days:
        dt = day_data.get("datetime")
        temp = day_data.get("temp")
        if isinstance(dt, str) and isinstance(temp, (int, float)):
            daily_map[dt] = float(temp)
            valid_temps += 1
    
    logger.info(f"Processed {len(all_days)} raw days into {len(daily_map)} valid temperature records for {location}")
    
    # Cache the result
    if daily_map:
        import json
        await daily_cache.set(cache_key, json.dumps(daily_map).encode())
        logger.debug(f"Cached {len(daily_map)} temperature records for {location}")
    
    return daily_map

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
        "unitGroup": _vc_unit_group(unit_group),
        "include": "obs,stats,stations",
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
    unit_group: Literal["celsius", "fahrenheit"] = Query(UNIT_GROUP_DEFAULT),
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
    try:
        # Add timeout protection to prevent 524 errors
        return await asyncio.wait_for(
            _rolling_bundle_impl(location, anchor, unit_group, month_mode, days_back, include, exclude),
            timeout=90.0  # 90 seconds timeout - should be under Cloudflare's 100s limit
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, 
            detail="Request timeout - try reducing the number of included sections or contact support"
        )

async def _rolling_bundle_impl(
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"],
    month_mode: Literal["calendar", "rolling1m", "rolling30d"],
    days_back: int,
    include: str | None,
    exclude: str | None,
):
    anchor_d = _safe_parse_date(anchor)
    
    # Parse CSV parameters and determine which sections to include
    def _parse_csv(s: str | None) -> Set[str]:
        return {p.strip() for p in s.split(",")} if s else set()

    inc = (_parse_csv(include) & ALLOWED_SECTIONS) if include else set()
    exc = (_parse_csv(exclude) & ALLOWED_SECTIONS) if exclude else set()

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
            from main import get_temperature_data_v1
            return await get_temperature_data_v1(location, "daily", mmdd)
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
                from main import get_temperature_data_v1
                return await get_temperature_data_v1(location, period, mmdd)
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
    
    notes += f" Includes {days_back} previous days. Weekly/monthly/yearly data uses timeline endpoints with optimal chunking for 50-year coverage. All API calls are optimized to run concurrently for better performance."

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
                "weekly": "timeline + local aggregation",
                "monthly": "timeline + local aggregation", 
                "yearly": "timeline + local aggregation"
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
    location: str,
    anchor: str,
    unit_group: Literal["celsius", "fahrenheit"] = Query(UNIT_GROUP_DEFAULT),
):
    """
    Check if rolling-bundle data is available in cache.
    Returns status and estimated completion time.
    """
    anchor_d = _safe_parse_date(anchor)
    
    # Check cache for recent data
    cache_key = f"rolling_bundle:{location}:{anchor_d.isoformat()}:{unit_group}"
    cached_data = await daily_cache.get(cache_key)
    
    if cached_data:
        return {
            "status": "ready",
            "location": location,
            "anchor": anchor_d,
            "cached": True,
            "message": "Data is available in cache"
        }
    else:
        return {
            "status": "not_cached",
            "location": location,
            "anchor": anchor_d,
            "cached": False,
            "message": "Data not in cache - request will take 30-90 seconds",
            "suggestion": "Use /preload endpoint for faster response"
        }

@router.get("/v1/records/rolling-bundle/preload-example")
async def preload_example():
    """
    Example of what the preload endpoint returns.
    Shows the complete data structure with all chart data, summary, trend, and average.
    """
    return {
        "description": "This is what the preload endpoint returns - complete data for your website",
        "endpoint": "GET /v1/records/rolling-bundle/{location}/{anchor}/preload",
        "example_url": "https://api.temphist.com/v1/records/rolling-bundle/London/2025-09-28/preload",
        "data_structure": {
            "location": "London, England, United Kingdom",
            "anchor": "2025-09-28",
            "unit_group": "celsius",
            "metadata": {
                "anchor_date": "2025-09-28",
                "month_mode": "rolling1m",
                "endpoint": "preload",
                "optimized": True,
                "included_sections": ["week", "month", "year"],
                "data_sources": {
                    "weekly": "timeline API with optimal chunking",
                    "monthly": "timeline API with optimal chunking", 
                    "yearly": "timeline API with optimal chunking"
                }
            },
            "notes": "Preload endpoint - optimized for website data loading with complete chart data, summary, trend, and average.",
            "weekly": {
                "description": "Complete weekly endpoint response with all data",
                "includes": ["values", "average", "trend", "summary", "metadata"],
                "example": {
                    "period": "weekly",
                    "location": "London",
                    "identifier": "09-28",
                    "unit_group": "celsius",
                    "values": [{"year": 2020, "temp": 15.2}, {"year": 2021, "temp": 16.1}],
                    "count": 50,
                    "average": {"mean": 15.8, "min": 12.3, "max": 19.2},
                    "trend": {"slope": 0.05, "r_squared": 0.23},
                    "summary": "Weekly temperatures show a slight warming trend...",
                    "metadata": {"data_source": "Visual Crossing", "years_covered": "1975-2024"}
                }
            },
            "monthly": {
                "description": "Complete monthly endpoint response with all data",
                "includes": ["values", "average", "trend", "summary", "metadata"],
                "example": {
                    "period": "monthly",
                    "location": "London",
                    "identifier": "09-28",
                    "unit_group": "celsius",
                    "values": [{"year": 2020, "temp": 15.2}, {"year": 2021, "temp": 16.1}],
                    "count": 50,
                    "average": {"mean": 15.8, "min": 12.3, "max": 19.2},
                    "trend": {"slope": 0.05, "r_squared": 0.23},
                    "summary": "Monthly temperatures show a slight warming trend...",
                    "metadata": {"data_source": "Visual Crossing", "years_covered": "1975-2024"}
                }
            },
            "yearly": {
                "description": "Complete yearly endpoint response with all data",
                "includes": ["values", "average", "trend", "summary", "metadata"],
                "example": {
                    "period": "yearly",
                    "location": "London",
                    "identifier": "09-28",
                    "unit_group": "celsius",
                    "values": [{"year": 2020, "temp": 15.2}, {"year": 2021, "temp": 16.1}],
                    "count": 50,
                    "average": {"mean": 15.8, "min": 12.3, "max": 19.2},
                    "trend": {"slope": 0.05, "r_squared": 0.23},
                    "summary": "Yearly temperatures show a slight warming trend...",
                    "metadata": {"data_source": "Visual Crossing", "years_covered": "1975-2024"}
                }
            }
        },
        "key_differences_from_main_endpoint": {
            "timeout": "60 seconds (vs 90 seconds for main endpoint)",
            "sections": "Always includes week, month, year (no include/exclude params)",
            "optimization": "Concurrent API calls instead of sequential",
            "purpose": "Designed specifically for website preloading",
            "data_completeness": "Same complete data structure as main endpoint"
        }
    }