"""V1 records endpoints."""
import json
import logging
import redis
import asyncio
from datetime import datetime, timedelta, timezone, date
from typing import Literal, Dict, List, Tuple
from fastapi import APIRouter, HTTPException, Path, Response, Request, Depends
from fastapi.responses import JSONResponse

from models import (
    RecordResponse, SubResourceResponse, UpdatedResponse,
    TemperatureValue, DateRange, AverageData, TrendData
)
from config import (
    CACHE_ENABLED, DEBUG, API_KEY,
)
from cache_utils import (
    normalize_location_for_cache, get_cache_updated_timestamp, _get_location_timezone,
    rec_key, bundle_key, rec_etag_key, get_records, assemble_and_cache,
    compute_bundle_etag, get_year_etags, TTL_STABLE, TTL_CURRENT_DAILY,
    TTL_CURRENT_WEEKLY, TTL_CURRENT_MONTHLY, TTL_CURRENT_YEARLY,
    get_job_manager
)

from utils.validation import validate_location_for_ssrf
from utils.location_validation import (
    is_location_likely_invalid, validate_location_response, InvalidLocationCache
)
from routers.dependencies import get_redis_client, get_invalid_location_cache
from utils.temperature import calculate_trend_slope, get_friendly_date, generate_summary
from utils.weather import get_year_range, track_missing_year, create_metadata
from utils.weather_data import get_temperature_series
from utils.cache_headers import set_weather_cache_headers
from utils.daily_temperature_store import (
    DailyTemperatureRecord,
    get_daily_temperature_store,
)
from utils.visual_crossing_timeline import fetch_timeline_days

logger = logging.getLogger(__name__)
router = APIRouter()


def parse_identifier(period: str, identifier: str) -> tuple:
    """Parse identifier based on period type. All periods use MM-DD format representing the end date."""
    # All periods use MM-DD format representing the end date of the period
    try:
        month, day = map(int, identifier.split("-"))
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        return month, day, period
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Identifier must be in MM-DD format: {str(e)}")


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


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _convert_c_to_unit(temp_c: float, unit_group: str) -> float:
    group = unit_group.lower()
    if group in ("celsius", "metric"):
        return temp_c
    if group in ("fahrenheit", "us"):
        return (temp_c * 9.0 / 5.0) + 32.0
    return temp_c


def _resolve_anchor_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        if month == 2 and day == 29:
            try:
                return date(year, 2, 28)
            except ValueError:
                return None
        # Fallback to last valid day of month
        for candidate_day in range(31, 27, -1):
            try:
                return date(year, month, candidate_day)
            except ValueError:
                continue
    return None


def _collapse_consecutive_dates(dates: List[date]) -> List[Tuple[date, date]]:
    if not dates:
        return []
    ordered = sorted(dates)
    ranges: List[Tuple[date, date]] = []
    start = prev = ordered[0]
    for current in ordered[1:]:
        if current == prev + timedelta(days=1):
            prev = current
            continue
        ranges.append((start, prev))
        start = prev = current
    ranges.append((start, prev))
    return ranges


WINDOW_DAYS = {
    "weekly": 7,
    "monthly": 31,
    "yearly": 365,
}


async def _collect_rolling_window_values(
    location: str,
    period: Literal["weekly", "monthly", "yearly"],
    month: int,
    day: int,
    unit_group: str,
    years: List[int],
) -> Tuple[List[TemperatureValue], List[float], List[Dict]]:
    store = await get_daily_temperature_store()

    values: List[TemperatureValue] = []
    aggregated: List[float] = []
    missing_years: List[Dict] = []
    window_days = WINDOW_DAYS[period]

    for year in years:
        anchor = _resolve_anchor_date(year, month, day)
        if anchor is None:
            track_missing_year(missing_years, year, "invalid_anchor_date")
            continue

        start_date = anchor - timedelta(days=window_days - 1)
        date_sequence = [start_date + timedelta(days=i) for i in range(window_days)]
        cache = await store.fetch(location, date_sequence)

        missing_dates = [d for d in date_sequence if d not in cache or cache[d].temp_c is None]
        timeline_failed = False

        if missing_dates:
            for range_start, range_end in _collapse_consecutive_dates(missing_dates):
                try:
                    timeline_days = await fetch_timeline_days(location, range_start, range_end)
                except Exception as exc:
                    logger.error(
                        "❌ timeline fetch failed for %s (%s to %s): %s",
                        location,
                        range_start,
                        range_end,
                        exc,
                    )
                    track_missing_year(missing_years, year, "timeline_error")
                    timeline_failed = True
                    break

                records_to_store: List[DailyTemperatureRecord] = []
                for day_payload in timeline_days:
                    dt_raw = day_payload.get("datetime") or day_payload.get("date")
                    if not dt_raw:
                        continue
                    dt_text = str(dt_raw)[:10]
                    try:
                        record_date = datetime.strptime(dt_text, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    if record_date < range_start or record_date > range_end:
                        continue
                    temp = _coerce_float(day_payload.get("temp"))
                    temp_max = _coerce_float(day_payload.get("tempmax") or day_payload.get("maxt"))
                    temp_min = _coerce_float(day_payload.get("tempmin") or day_payload.get("mint"))
                    filtered_payload = {
                        "datetime": record_date.isoformat(),
                        "temp": temp,
                        "tempmax": temp_max,
                        "tempmin": temp_min,
                    }
                    records_to_store.append(
                        DailyTemperatureRecord(
                            date=record_date,
                            temp_c=temp,
                            temp_max_c=temp_max,
                            temp_min_c=temp_min,
                            payload=filtered_payload,
                            source="timeline",
                        )
                    )

                if records_to_store:
                    await store.upsert(location, records_to_store)
                    for rec in records_to_store:
                        cache[rec.date] = rec

            if timeline_failed:
                continue

            remaining_missing = [d for d in date_sequence if d not in cache or cache[d].temp_c is None]
            if remaining_missing:
                track_missing_year(missing_years, year, "insufficient_daily_data")
                continue

        temps_converted = []
        incomplete = False
        for d in date_sequence:
            record = cache.get(d)
            if not record or record.temp_c is None:
                incomplete = True
                break
            temps_converted.append(_convert_c_to_unit(record.temp_c, unit_group))

        if incomplete or len(temps_converted) != len(date_sequence):
            track_missing_year(missing_years, year, "insufficient_daily_data")
            continue

        avg_temp = sum(temps_converted) / len(temps_converted)
        aggregated.append(avg_temp)
        values.append(
            TemperatureValue(
                date=anchor.strftime("%Y-%m-%d"),
                year=anchor.year,
                temperature=round(avg_temp, 1),
            )
        )

    return values, aggregated, missing_years

async def get_temperature_data_v1(
    location: str,
    period: str,
    identifier: str,
    unit_group: str = "celsius",
    redis_client: redis.Redis = None  # Can be None, will use dependency if not provided
) -> Dict:
    """Get temperature data for v1 API using rolling timeline windows for weekly/monthly/yearly."""
    # Parse identifier based on period (all use MM-DD format representing end date)
    month, day, period_type = parse_identifier(period, identifier)
    
    # Use 50 years of data ending at current year (consistent with other endpoints)
    current_year = datetime.now().year
    start_year = current_year - 50  # 50 years back + current year = 51 years total
    years = get_year_range(current_year)
    end_date = datetime(current_year, month, day)
    
    if period == "daily":
        # Single day across all years
        start_date = end_date
        date_range_days = 1
    elif period == "weekly":
        # 7 days ending on the specified date
        start_date = end_date - timedelta(days=6)
        date_range_days = 7
    elif period == "monthly":
        # Trailing month (31 days) ending on the specified date
        start_date = end_date - timedelta(days=30)
        date_range_days = 31
    elif period == "yearly":
        # 365 days ending on the specified date
        start_date = end_date - timedelta(days=364)
        date_range_days = 365
    else:
        raise HTTPException(status_code=400, detail="Invalid period. Must be daily, weekly, monthly, or yearly")
    
    # Get temperature data for the date range across all years
    values = []
    all_temps = []
    missing_years = []
    
    if period == "daily":
        # For daily, get data for just the specific day across all years
        weather_data = await get_temperature_series(location, month, day, redis_client)
        if weather_data and 'data' in weather_data:
            # Extract missing years from the series metadata
            if 'metadata' in weather_data and 'missing_years' in weather_data['metadata']:
                missing_years.extend(weather_data['metadata']['missing_years'])
            
            for data_point in weather_data['data']:
                year = int(data_point['x'])
                temp = data_point['y']
                if temp is not None:
                    all_temps.append(temp)
                    values.append(TemperatureValue(
                        date=f"{year}-{month:02d}-{day:02d}",
                        year=year,
                        temperature=temp
                    ))
    
    elif period in ["weekly", "monthly", "yearly"]:
        window_values, aggregated_values, window_missing = await _collect_rolling_window_values(
            location=location,
            period=period,  # type: ignore[arg-type]
            month=month,
            day=day,
            unit_group=unit_group,
            years=years,
        )
        values.extend(window_values)
        all_temps.extend(aggregated_values)
        missing_years.extend(window_missing)
    
    # Calculate date range
    if values:
        start_year_val = min(v.year for v in values)
        end_year_val = max(v.year for v in values)
        range_data = DateRange(
            start=f"{start_year_val}-{month:02d}-{day:02d}",
            end=f"{end_year_val}-{month:02d}-{day:02d}",
            years=end_year_val - start_year_val + 1
        )
    else:
        range_data = DateRange(start="", end="", years=0)
    
    # Calculate average
    if all_temps:
        avg_data = AverageData(
            mean=round(sum(all_temps) / len(all_temps), 1),
            unit=unit_group,
            data_points=len(all_temps)
        )
    else:
        avg_data = AverageData(mean=0.0, unit=unit_group, data_points=0)
    
    # Calculate trend
    if len(values) >= 2:
        trend_input = [{"x": v.year, "y": v.temperature} for v in values]
        slope = calculate_trend_slope(trend_input)
        trend_data = TrendData(
            slope=slope,
            unit="°C/decade" if unit_group == "celsius" else "°F/decade",
            data_points=len(values),
            r_squared=None
        )
    else:
        trend_data = TrendData(slope=0.0, unit="°C/decade" if unit_group == "celsius" else "°F/decade", data_points=len(values))
    
    # Generate summary using existing logic
    end_date_obj = datetime(current_year, month, day)
    
    # Create friendly date based on period
    if period == "daily":
        friendly_date = get_friendly_date(end_date_obj)
    elif period == "weekly":
        friendly_date = f"week ending {get_friendly_date(end_date_obj)}"
    elif period == "monthly":
        friendly_date = f"month ending {get_friendly_date(end_date_obj)}"
    elif period == "yearly":
        friendly_date = f"year ending {get_friendly_date(end_date_obj)}"
    else:
        friendly_date = get_friendly_date(end_date_obj)
    
    # Convert values to the format expected by generate_summary
    summary_data = []
    for value in values:
        summary_data.append({
            'x': value.year,
            'y': value.temperature
        })
    
    # Generate summary text
    summary_text = generate_summary(summary_data, end_date_obj, period)

    # Ensure metadata reflects missing current year when data is unavailable
    available_years = {v.year for v in values}
    if current_year not in available_years:
        if not any(entry.get("year") == current_year for entry in missing_years):
            track_missing_year(missing_years, current_year, "no_data_current_year")
    
    # Replace the friendly date in the summary with our period-specific version
    summary_text = summary_text.replace(get_friendly_date(end_date_obj), friendly_date)
    
    # Create comprehensive metadata
    additional_metadata = {
        "period_days": date_range_days, 
        "end_date": end_date_obj.strftime("%Y-%m-%d")
    }
    
    # Get timezone for the location
    timezone_str = None
    if redis_client:
        timezone_str = _get_location_timezone(location, redis_client)
    
    return {
        "period": period,
        "location": location,
        "identifier": identifier,
        "range": range_data.model_dump(),
        "unit_group": unit_group,
        "values": [v.model_dump() for v in values],
        "average": avg_data.model_dump(),
        "trend": trend_data.model_dump(),
        "summary": summary_text,
        "metadata": create_metadata(len(years), len(values), missing_years, additional_metadata),
        "timezone": timezone_str
    }


def _is_preapproved_location(slug: str) -> bool:
    """Check if a location slug is in the preapproved locations list.
    
    Args:
        slug: Normalized location slug
        
    Returns:
        True if location is preapproved, False otherwise
    """
    try:
        import os
        import json
        
        # Find project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        while project_root != os.path.dirname(project_root):
            if os.path.exists(os.path.join(project_root, "pyproject.toml")):
                break
            project_root = os.path.dirname(project_root)
        
        data_file = os.path.join(project_root, "data", "preapproved_locations.json")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        slug_lower = slug.lower()
        for item in data:
            item_slug = (item.get('slug') or item.get('id', '')).lower()
            if item_slug == slug_lower:
                return True
        
        return False
    except Exception:
        # If we can't check, assume not preapproved (safer)
        return False

def _rebuild_full_response_from_values(
    values: List[Dict],
    period: str,
    location: str,
    identifier: str,
    month: int,
    day: int,
    current_year: int,
    years: List[int],
    redis_client: redis.Redis
) -> Dict:
    """Rebuild full RecordResponse from list of year values.
    
    Args:
        values: List of TemperatureValue dicts
        period: Period type
        location: Location name
        identifier: Date identifier
        month: Month
        day: Day
        current_year: Current year
        years: List of all years in range
        redis_client: Redis client
        
    Returns:
        Full response dict with all fields
    """
    from utils.temperature import calculate_trend_slope, generate_summary
    from utils.weather import create_metadata
    
    all_temps = [v.get('temperature') for v in values if v.get('temperature') is not None]
    
    if all_temps:
        avg_data = {
            "mean": round(sum(all_temps) / len(all_temps), 1),
            "unit": "celsius",
            "data_points": len(all_temps)
        }
    else:
        avg_data = {"mean": 0.0, "unit": "celsius", "data_points": 0}
    
    if len(values) >= 2:
        trend_input = [{"x": v.get('year'), "y": v.get('temperature')} for v in values]
        slope = calculate_trend_slope(trend_input)
        trend_data = {
            "slope": slope,
            "unit": "°C/decade",
            "data_points": len(values),
            "r_squared": None
        }
    else:
        trend_data = {"slope": 0.0, "unit": "°C/decade", "data_points": len(values)}
    
    end_date_obj = datetime(current_year, month, day)
    summary_data = [{"x": v.get('year'), "y": v.get('temperature')} for v in values]
    summary_text = generate_summary(summary_data, end_date_obj, period)

    available_years = {v.get('year') for v in values if v.get('year') is not None}
    rebuilt_missing_years = []
    if current_year not in available_years:
        track_missing_year(rebuilt_missing_years, current_year, "no_data_current_year")
    
    return {
        "period": period,
        "location": location,
        "identifier": identifier,
        "range": {
            "start": f"{min(v.get('year') for v in values)}-{month:02d}-{day:02d}",
            "end": f"{max(v.get('year') for v in values)}-{month:02d}-{day:02d}",
            "years": max(v.get('year') for v in values) - min(v.get('year') for v in values) + 1
        },
        "unit_group": "celsius",
        "values": values,
        "average": avg_data,
        "trend": trend_data,
        "summary": summary_text,
        "metadata": create_metadata(len(years), len(values), rebuilt_missing_years, {"period_days": 1, "end_date": end_date_obj.strftime("%Y-%m-%d")}),
        "timezone": _get_location_timezone(location, redis_client),
        "updated": datetime.now(timezone.utc).isoformat()
    }

def _extract_per_year_records(full_data: Dict) -> Dict[int, Dict]:
    """Extract per-year records from full response data.
    
    Args:
        full_data: Full response from get_temperature_data_v1
        
    Returns:
        Dict mapping year -> per-year record (just the TemperatureValue for that year)
    """
    per_year = {}
    if 'values' in full_data:
        for value in full_data['values']:
            year = value.get('year')
            if year:
                per_year[year] = value
    return per_year

def _get_ttl_for_current_year(period: str) -> int:
    """Get TTL for current year based on period."""
    if period == "daily":
        return TTL_CURRENT_DAILY
    elif period == "weekly":
        return TTL_CURRENT_WEEKLY
    elif period == "monthly":
        return TTL_CURRENT_MONTHLY
    elif period == "yearly":
        return TTL_CURRENT_YEARLY
    else:
        return TTL_CURRENT_DAILY  # Default

async def _store_per_year_records(
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    per_year_records: Dict[int, Dict],
    current_year: int
):
    """Store per-year records in cache with appropriate TTLs."""
    from cache_utils import ETagGenerator
    
    for year, record_data in per_year_records.items():
        year_key = rec_key(scope, slug, identifier, year)
        etag_key = rec_etag_key(scope, slug, identifier, year)
        
        # Determine TTL
        if year < current_year:
            ttl = TTL_STABLE
        else:
            ttl = _get_ttl_for_current_year(scope)
        
        # Generate ETag for this year's record
        etag = ETagGenerator.generate_etag(record_data)
        
        try:
            json_data = json.dumps(record_data, sort_keys=True, separators=(',', ':'))
            redis_client.setex(year_key, ttl, json_data)
            redis_client.setex(etag_key, ttl, etag)
            if DEBUG:
                logger.debug(f"Cached per-year record: {year_key} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Error caching per-year record {year_key}: {e}")

async def _get_record_data_internal(
    period: str,
    location: str,
    identifier: str,
    redis_client: redis.Redis,
    invalid_location_cache: InvalidLocationCache
) -> Dict:
    """Internal helper to get record data using per-year caching (returns dict, not response)."""
    # This is essentially the same logic as get_record but returns the data dict
    # Parse identifier
    month, day, _ = parse_identifier(period, identifier)
    current_year = datetime.now(timezone.utc).year
    slug = normalize_location_for_cache(location)
    years = get_year_range(current_year)
    
    if CACHE_ENABLED:
        # Try bundle cache first
        bundle_key_str = bundle_key(period, slug, identifier)
        bundle_data = redis_client.get(bundle_key_str)
        
        if bundle_data:
            try:
                data_str = bundle_data.decode('utf-8') if isinstance(bundle_data, bytes) else bundle_data
                bundle_payload = json.loads(data_str)
                if 'records' in bundle_payload and len(bundle_payload['records']) > 0:
                    values = bundle_payload['records']
                    all_temps = [v.get('temperature') for v in values if v.get('temperature') is not None]
                    
                    from utils.temperature import calculate_trend_slope, generate_summary
                    from utils.weather import create_metadata
                    
                    if all_temps:
                        avg_data = {
                            "mean": round(sum(all_temps) / len(all_temps), 1),
                            "unit": "celsius",
                            "data_points": len(all_temps)
                        }
                    else:
                        avg_data = {"mean": 0.0, "unit": "celsius", "data_points": 0}
                    
                    if len(values) >= 2:
                        trend_input = [{"x": v.get('year'), "y": v.get('temperature')} for v in values]
                        slope = calculate_trend_slope(trend_input)
                        trend_data = {
                            "slope": slope,
                            "unit": "°C/decade",
                            "data_points": len(values),
                            "r_squared": None
                        }
                    else:
                        trend_data = {"slope": 0.0, "unit": "°C/decade", "data_points": len(values)}
                    
                    end_date_obj = datetime(current_year, month, day)
                    summary_data = [{"x": v.get('year'), "y": v.get('temperature')} for v in values]
                    summary_text = generate_summary(summary_data, end_date_obj, period)
                    
                    return {
                        "period": period,
                        "location": location,
                        "identifier": identifier,
                        "range": {
                            "start": f"{min(v.get('year') for v in values)}-{month:02d}-{day:02d}",
                            "end": f"{max(v.get('year') for v in values)}-{month:02d}-{day:02d}",
                            "years": max(v.get('year') for v in values) - min(v.get('year') for v in values) + 1
                        },
                        "unit_group": "celsius",
                        "values": values,
                        "average": avg_data,
                        "trend": trend_data,
                        "summary": summary_text,
                        "metadata": create_metadata(len(years), len(values), [], {"period_days": 1, "end_date": end_date_obj.strftime("%Y-%m-%d")}),
                        "timezone": _get_location_timezone(location, redis_client),
                        "updated": datetime.now(timezone.utc).isoformat()
                    }
            except Exception:
                pass
        
        # MGET all year keys
        year_data, missing_past, missing_current = await get_records(
            redis_client, period, slug, identifier, years
        )
        
        # Handle missing years (simplified - just fetch if needed)
        if missing_past or missing_current:
            full_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
            per_year_records = _extract_per_year_records(full_data)
            await _store_per_year_records(redis_client, period, slug, identifier, per_year_records, current_year)
            year_data = per_year_records
        
        # Assemble from year_data using helper
        if year_data:
            values = [year_data[y] for y in sorted(year_data.keys())]
            return _rebuild_full_response_from_values(
                values, period, location, identifier, month, day, current_year, years, redis_client
            )
    
    # Fallback: fetch fresh
    return await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)

@router.get("/v1/records/{period}/{location}/{identifier}", response_model=RecordResponse)
async def get_record(
    request: Request,
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None,
    redis_client: redis.Redis = Depends(get_redis_client),
    invalid_location_cache: InvalidLocationCache = Depends(get_invalid_location_cache)
):
    """Get temperature record data for a specific period, location, and identifier."""
    try:
        # Comprehensive SSRF validation for location
        try:
            location = validate_location_for_ssrf(location)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid location format. Please provide a valid location name."
            ) from e
        
        # Quick validation for obviously invalid locations (redundant but provides early feedback)
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now(timezone.utc).year
        end_date = datetime(current_year, month, day).date()
        
        # Normalize location to slug format
        slug = normalize_location_for_cache(location)
        
        # Get year range (50 years back + current year)
        years = get_year_range(current_year)
        
        # Check for ETag conditional request
        if_none_match = request.headers.get("if-none-match")
        
        # Initialize variables for cache status tracking
        year_data = {}
        cache_status = "MISS"
        bundle_etag_computed = None
        
        if CACHE_ENABLED:
            # Step 1: Try bundle cache first (fast path)
            bundle_key_str = bundle_key(period, slug, identifier)
            bundle_etag_key = f"{bundle_key_str}:etag"
            bundle_data = redis_client.get(bundle_key_str)
            bundle_etag = redis_client.get(bundle_etag_key)
            
            if bundle_data and bundle_etag:
                try:
                    data_str = bundle_data.decode('utf-8') if isinstance(bundle_data, bytes) else bundle_data
                    bundle_payload = json.loads(data_str)
                    bundle_etag_str = bundle_etag.decode('utf-8') if isinstance(bundle_etag, bytes) else bundle_etag
                    
                    # Check ETag conditional request
                    if if_none_match:
                        from cache_utils import ETagGenerator
                        if ETagGenerator.matches_etag(bundle_etag_str, if_none_match):
                            response.status_code = 304
                            response.headers["ETag"] = bundle_etag_str
                            response.headers["X-Cache-Status"] = "HIT"
                            return None
                    
                    # Reconstruct full response from bundle
                    if 'records' in bundle_payload and len(bundle_payload['records']) > 0:
                        values = bundle_payload['records']
                        data = _rebuild_full_response_from_values(
                            values, period, location, identifier, month, day, current_year, years, redis_client
                        )
                        
                        # Validate cached data
                        is_valid, error_msg = validate_location_response(data, location)
                        if not is_valid:
                            invalid_location_cache.mark_location_invalid(location, "no_data_cached")
                            raise HTTPException(status_code=400, detail=error_msg)
                        
                        # Set cache headers with stale-while-revalidate and ETag
                        json_response = JSONResponse(content=data)
                        json_response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
                        json_response.headers["ETag"] = bundle_etag_str
                        json_response.headers["X-Cache-Status"] = "HIT"
                        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1")
                        
                        if DEBUG:
                            logger.debug(f"✅ SERVING BUNDLE CACHE: {bundle_key_str}")
                        
                        return json_response
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    if DEBUG:
                        logger.debug(f"Error parsing bundle cache: {e}, falling through to per-year lookup")
            
            # Step 2: MGET all year keys
            year_data, missing_past, missing_current = await get_records(
                redis_client, period, slug, identifier, years
            )
            
            # Step 3: Handle missing years with guardrail for preapproved locations
            if missing_past or missing_current:
                # Guardrail: For preapproved locations, never trigger 50 external fetches
                # If bundle is MISS and location is preapproved, assemble from per-year keys
                # and only fetch current-year if missing
                is_preapproved = _is_preapproved_location(slug)
                
                # If only current year is missing, serve bundle immediately and enqueue refresh
                if missing_past == [] and missing_current:
                    # Try to serve last bundle if available (stale-while-revalidate)
                    bundle_data = redis_client.get(bundle_key_str)
                    bundle_etag = redis_client.get(f"{bundle_key_str}:etag")
                    if bundle_data:
                        try:
                            data_str = bundle_data.decode('utf-8') if isinstance(bundle_data, bytes) else bundle_data
                            bundle_payload = json.loads(data_str)
                            if 'records' in bundle_payload and len(bundle_payload['records']) > 0:
                                values = bundle_payload['records']
                                data = _rebuild_full_response_from_values(
                                    values, period, location, identifier, month, day, current_year, years, redis_client
                                )
                                
                                # Validate cached data
                                is_valid, error_msg = validate_location_response(data, location)
                                if not is_valid:
                                    invalid_location_cache.mark_location_invalid(location, "no_data_cached")
                                    raise HTTPException(status_code=400, detail=error_msg)
                                
                                # Compute bundle ETag from per-year ETags
                                year_etags = await get_year_etags(redis_client, period, slug, identifier, years)
                                bundle_etag_computed = compute_bundle_etag(year_etags)
                                
                                # Check ETag conditional request
                                if if_none_match:
                                    from cache_utils import ETagGenerator
                                    if ETagGenerator.matches_etag(bundle_etag_computed, if_none_match):
                                        response.status_code = 304
                                        response.headers["ETag"] = bundle_etag_computed
                                        response.headers["X-Cache-Status"] = "STALE"
                                        return None
                                
                                # Serve stale bundle with refresh job
                                json_response = JSONResponse(content=data)
                                json_response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
                                json_response.headers["ETag"] = bundle_etag_computed
                                json_response.headers["X-Cache-Status"] = "STALE"
                                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1")
                                
                                # Enqueue job to refresh current year only
                                try:
                                    job_manager = get_job_manager()
                                    if job_manager:
                                        job_id = job_manager.create_job("record_computation", {
                                            "scope": period,
                                            "slug": slug,
                                            "identifier": identifier,
                                            "year": current_year,
                                            "location": location
                                        })
                                        if DEBUG:
                                            logger.debug(f"Enqueued job to refresh current year: {job_id}")
                                except Exception as e:
                                    logger.warning(f"Failed to enqueue refresh job: {e}")
                                
                                return json_response
                        except Exception as e:
                            if DEBUG:
                                logger.debug(f"Error serving stale bundle: {e}")
                    
                    # If no stale bundle, fetch current year only
                    if is_preapproved:
                        # For preapproved locations, only fetch current year
                        if DEBUG:
                            logger.debug(f"Preapproved location {slug}: fetching only current year {current_year}")
                        full_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
                        per_year_records = _extract_per_year_records(full_data)
                        if current_year in per_year_records:
                            year_key = rec_key(period, slug, identifier, current_year)
                            etag_key = rec_etag_key(period, slug, identifier, current_year)
                            ttl = _get_ttl_for_current_year(period)
                            from cache_utils import ETagGenerator
                            etag = ETagGenerator.generate_etag(per_year_records[current_year])
                            try:
                                json_data = json.dumps(per_year_records[current_year], sort_keys=True, separators=(',', ':'))
                                redis_client.setex(year_key, ttl, json_data)
                                redis_client.setex(etag_key, ttl, etag)
                                year_data[current_year] = per_year_records[current_year]
                            except Exception as e:
                                logger.warning(f"Error caching current year: {e}")
                    else:
                        # For non-preapproved, enqueue job to refresh current year
                        try:
                            job_manager = get_job_manager()
                            if job_manager:
                                job_id = job_manager.create_job("record_computation", {
                                    "scope": period,
                                    "slug": slug,
                                    "identifier": identifier,
                                    "year": current_year,
                                    "location": location
                                })
                                if DEBUG:
                                    logger.debug(f"Enqueued job to refresh current year: {job_id}")
                        except Exception as e:
                            logger.warning(f"Failed to enqueue refresh job: {e}")
                
                # Fetch missing past years (should be rare after prewarming)
                # Guardrail: For preapproved locations, if past years are missing, 
                # this is unexpected - log warning but still fetch
                if missing_past:
                    if is_preapproved:
                        logger.warning(f"Preapproved location {slug} missing past years: {missing_past}. This should not happen after prewarming.")
                    
                    # Fetch full data (will get all years)
                    full_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
                    
                    # Extract per-year records
                    per_year_records = _extract_per_year_records(full_data)
                    
                    # Store missing past years only
                    for year in missing_past:
                        if year in per_year_records:
                            year_key = rec_key(period, slug, identifier, year)
                            etag_key = rec_etag_key(period, slug, identifier, year)
                            ttl = TTL_STABLE
                            from cache_utils import ETagGenerator
                            etag = ETagGenerator.generate_etag(per_year_records[year])
                            try:
                                json_data = json.dumps(per_year_records[year], sort_keys=True, separators=(',', ':'))
                                redis_client.setex(year_key, ttl, json_data)
                                redis_client.setex(etag_key, ttl, etag)
                                year_data[year] = per_year_records[year]
                            except Exception as e:
                                logger.warning(f"Error caching year {year}: {e}")
            
            # Step 4: Assemble response from per-year records
            if year_data:
                # Get per-year ETags
                year_etags = await get_year_etags(redis_client, period, slug, identifier, list(year_data.keys()))
                
                # Assemble bundle and store with ETag (returns payload and ETag)
                bundle_payload, bundle_etag_computed = await assemble_and_cache(
                    redis_client, period, slug, identifier, year_data, year_etags
                )
                
                # Check ETag conditional request
                if if_none_match:
                    from cache_utils import ETagGenerator
                    if ETagGenerator.matches_etag(bundle_etag_computed, if_none_match):
                        response.status_code = 304
                        response.headers["ETag"] = bundle_etag_computed
                        response.headers["X-Cache-Status"] = "HIT"
                        return None
                
                # Rebuild full response from year_data using helper
                values = [year_data[y] for y in sorted(year_data.keys())]
                data = _rebuild_full_response_from_values(
                    values, period, location, identifier, month, day, current_year, years, redis_client
                )
                
                # Set cache status
                cache_status = "HIT" if not missing_past and not missing_current else "PARTIAL"
            else:
                # No cached data, fetch fresh
                data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
                
                # Extract and store per-year records
                per_year_records = _extract_per_year_records(data)
                await _store_per_year_records(redis_client, period, slug, identifier, per_year_records, current_year)
                
                # Assemble bundle with ETag
                year_etags = await get_year_etags(redis_client, period, slug, identifier, list(per_year_records.keys()))
                bundle_payload, bundle_etag_computed = await assemble_and_cache(
                    redis_client, period, slug, identifier, per_year_records, year_etags
                )
                
                cache_status = "MISS"
        else:
            # Cache disabled, fetch fresh
            data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(data, location)
        if not is_valid:
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Ensure updated timestamp
        if "updated" not in data:
            data["updated"] = datetime.now(timezone.utc).isoformat()
        
        # Compute bundle ETag if not already computed
        if bundle_etag_computed is None:
            if CACHE_ENABLED:
                # Try to get from stored bundle ETag
                bundle_key_str = bundle_key(period, slug, identifier)
                bundle_etag_key = f"{bundle_key_str}:etag"
                bundle_etag_stored = redis_client.get(bundle_etag_key)
                if bundle_etag_stored:
                    bundle_etag_computed = bundle_etag_stored.decode('utf-8') if isinstance(bundle_etag_stored, bytes) else bundle_etag_stored
                else:
                    # Fallback: compute from data
                    from cache_utils import ETagGenerator
                    bundle_etag_computed = ETagGenerator.generate_etag(data)
            else:
                from cache_utils import ETagGenerator
                bundle_etag_computed = ETagGenerator.generate_etag(data)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=data)
        json_response.headers["ETag"] = bundle_etag_computed
        json_response.headers["X-Cache-Status"] = cache_status if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/average", response_model=SubResourceResponse)
async def get_record_average(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None,
    redis_client: redis.Redis = Depends(get_redis_client),
    invalid_location_cache: InvalidLocationCache = Depends(get_invalid_location_cache)
):
    """Get average temperature data for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Get full record data using per-year caching
        record_data = await _get_record_data_internal(period, location, identifier, redis_client, invalid_location_cache)
        
        # Extract average data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["average"],
            metadata=record_data["metadata"],
            timezone=record_data.get("timezone")
        )
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        json_response.headers["X-Cache-Status"] = "HIT" if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|average|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records average endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/trend", response_model=SubResourceResponse)
async def get_record_trend(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None,
    redis_client: redis.Redis = Depends(get_redis_client),
    invalid_location_cache: InvalidLocationCache = Depends(get_invalid_location_cache)
):
    """Get temperature trend data for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Get full record data using per-year caching
        record_data = await _get_record_data_internal(period, location, identifier, redis_client, invalid_location_cache)
        
        # Extract trend data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["trend"],
            metadata=record_data["metadata"],
            timezone=record_data.get("timezone")
        )
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        json_response.headers["X-Cache-Status"] = "HIT" if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|trend|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records trend endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/summary", response_model=SubResourceResponse)
async def get_record_summary(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    response: Response = None,
    redis_client: redis.Redis = Depends(get_redis_client),
    invalid_location_cache: InvalidLocationCache = Depends(get_invalid_location_cache)
):
    """Get temperature summary text for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )
        
        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}"
            )
        
        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Get full record data using per-year caching
        record_data = await _get_record_data_internal(period, location, identifier, redis_client, invalid_location_cache)
        
        # Extract summary data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["summary"],
            metadata=record_data["metadata"],
            timezone=record_data.get("timezone")
        )
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        json_response.headers["X-Cache-Status"] = "HIT" if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|summary|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records summary endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/updated", response_model=UpdatedResponse)
async def get_record_updated(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """
    Get the last updated timestamp for a specific record endpoint.
    
    Returns when the data was last updated (cached) or null if it's never been queried.
    This endpoint is designed for web apps that want to check if they need to refetch data.
    """
    try:
        # Create the same cache key that would be used by the main endpoint
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:values,average,trend,summary"
        
        # Get the updated timestamp from cache
        updated_timestamp = await get_cache_updated_timestamp(cache_key, redis_client)
        
        # Determine if data is cached
        is_cached = updated_timestamp is not None
        
        # Format timestamp as ISO string if available
        updated_iso = updated_timestamp.isoformat() if updated_timestamp else None
        
        return UpdatedResponse(
            period=period,
            location=location,
            identifier=identifier,
            updated=updated_iso,
            cached=is_cached,
            cache_key=cache_key
        )
    
    except Exception as e:
        logger.error(f"Error in v1 records updated endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
