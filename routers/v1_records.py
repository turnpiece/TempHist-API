"""V1 records endpoints."""
import json
import logging
import redis
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from typing import Literal, Dict, Tuple
from fastapi import APIRouter, HTTPException, Path, Response, Depends
from fastapi.responses import JSONResponse

from models import (
    RecordResponse, SubResourceResponse, UpdatedResponse,
    TemperatureValue, DateRange, AverageData, TrendData
)
from config import (
    CACHE_ENABLED, DEBUG, LONG_CACHE_DURATION, API_KEY,
    MAX_CONCURRENT_REQUESTS
)
from cache_utils import (
    get_cache_value, set_cache_value, generate_cache_key, get_cache_stats,
    normalize_location_for_cache, get_cache_updated_timestamp
)
from utils.validation import validate_location_for_ssrf
from utils.location_validation import (
    is_location_likely_invalid, validate_location_response, InvalidLocationCache
)
from routers.dependencies import get_redis_client, get_invalid_location_cache
from utils.temperature import calculate_trend_slope, get_friendly_date, generate_summary
from utils.weather import get_year_range, track_missing_year, create_metadata
from utils.weather_data import get_temperature_series, get_forecast_data
from utils.cache_headers import set_weather_cache_headers
from utils.firebase import verify_firebase_token
from constants import VC_BASE_URL

logger = logging.getLogger(__name__)
router = APIRouter()

# Global semaphore for Visual Crossing API requests
visual_crossing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


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


async def get_http_client():
    """Get configured httpx client."""
    from config import HTTP_TIMEOUT
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT)


async def _fetch_yearly_summary(
    location: str,
    start_year: int,
    end_year: int,
    unit_group: str = "metric"
):
    """Fetch yearly summary data from Visual Crossing historysummary endpoint."""
    url = f"{VC_BASE_URL}/weatherdata/historysummary"
    params = {
        "aggregateHours": 24,
        "minYear": start_year,
        "maxYear": end_year,
        "chronoUnit": "years",
        "breakBy": "years",
        "dailySummaries": "false",
        "contentType": "json",
        "unitGroup": unit_group,
        "locations": location,
        "maxStations": 8,
        "maxDistance": 120000,
        "key": API_KEY,
    }
    
    async with visual_crossing_semaphore:
        http = await get_http_client()
        async with http:
            r = await http.get(url, params=params, headers={"Accept-Encoding": "gzip"})
    r.raise_for_status()
    data = r.json()
    
    # Parse the response to extract yearly temperature data
    yearly_data = []
    if 'locations' in data and location in data['locations']:
        location_data = data['locations'][location]
        if 'values' in location_data:
            for value in location_data['values']:
                year = value.get('year')
                temp = value.get('temp')
                if year and temp is not None:
                    yearly_data.append((year, temp))
    
    return yearly_data


async def get_temperature_data_v1(
    location: str,
    period: str,
    identifier: str,
    unit_group: str = "celsius",
    redis_client: redis.Redis = None  # Can be None, will use dependency if not provided
) -> Dict:
    """Get temperature data for v1 API with unified logic using historysummary for weekly/monthly/yearly."""
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
        # 30 days ending on the specified date
        start_date = end_date - timedelta(days=29)
        date_range_days = 30
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
    
    elif period in ["weekly", "monthly"]:
        # Use historysummary endpoint for weekly/monthly data (much more efficient)
        try:
            from routers.records_agg import fetch_historysummary, _historysummary_values, _row_year, _row_mean_temp
            
            # Determine chrono_unit based on period
            chrono_unit = "weeks" if period == "weekly" else "months"
            
            if DEBUG:
                logger.debug(f"Fetching {period} summary for {location} from {start_year} to {current_year}")
            
            payload = await fetch_historysummary(
                location, 
                start_year, 
                current_year, 
                chrono_unit=chrono_unit, 
                break_by="years"
            )
            
            rows = _historysummary_values(payload)
            if DEBUG:
                logger.debug(f"Got {len(rows)} {period} data points")
            
            # For weekly, we need to match by ISO week number
            if period == "weekly":
                desired_week = datetime(current_year, month, day).isocalendar().week
                for r in rows:
                    y = _row_year(r)
                    t = _row_mean_temp(r)
                    if y is not None and t is not None:
                        # Check if this row corresponds to the desired week
                        try:
                            # Try to extract week info from the row
                            period_str = r.get('period') or r.get('datetimeStr') or r.get('start')
                            if period_str:
                                row_date = datetime.strptime(period_str[:10], "%Y-%m-%d")
                                row_week = row_date.isocalendar().week
                                if row_week == desired_week:
                                    all_temps.append(t)
                                    values.append(TemperatureValue(
                                        date=f"{y}-{month:02d}-{day:02d}",
                                        year=y,
                                        temperature=round(t, 1),
                                    ))
                        except Exception as e:
                            if DEBUG:
                                logger.debug(f"Error processing weekly row: {e}")
                            continue
            
            # For monthly, we need to match by month
            elif period == "monthly":
                for r in rows:
                    y = _row_year(r)
                    t = _row_mean_temp(r)
                    if y is not None and t is not None:
                        # Check if this row corresponds to the desired month
                        try:
                            # Try to extract month info from the row
                            period_str = r.get('period') or r.get('datetimeStr') or r.get('start')
                            if period_str:
                                row_month = int(period_str[5:7])
                                if row_month == month:
                                    all_temps.append(t)
                                    values.append(TemperatureValue(
                                        date=f"{y}-{month:02d}-{day:02d}",
                                        year=y,
                                        temperature=round(t, 1),
                                    ))
                        except Exception as e:
                            if DEBUG:
                                logger.debug(f"Error processing monthly row: {e}")
                            continue
            
            if not values:
                if DEBUG:
                    logger.debug(f"No {period} data found, falling back to sampling")
                raise Exception(f"No {period} data found")
                
        except Exception as e:
            # Only log on first occurrence to reduce log noise (historysummary often fails, fallback is expected)
            if DEBUG and "historysummary" not in str(e).lower():
                logger.debug(f"Error fetching {period} summary: {e}, falling back to sampling")
            # Fallback to sampling approach if historysummary fails
            sample_days = min(date_range_days, 7)  # Sample up to 7 days for efficiency
            step = max(1, date_range_days // sample_days)
            
            for year in range(start_year, current_year + 1):
                year_temps = []
                for day_offset in range(0, date_range_days, step):
                    current_date = start_date.replace(year=year) + timedelta(days=day_offset)
                    
                    try:
                        weather_data = await get_temperature_series(location, current_date.month, current_date.day, redis_client)
                        if weather_data and 'data' in weather_data:
                            # Extract missing years from the series metadata
                            if 'metadata' in weather_data and 'missing_years' in weather_data['metadata']:
                                for missing_year_info in weather_data['metadata']['missing_years']:
                                    if missing_year_info['year'] == year:
                                        track_missing_year(missing_years, year, f"{missing_year_info['reason']}_sampling")
                                        break
                            
                            for data_point in weather_data['data']:
                                if int(data_point['x']) == year and data_point['y'] is not None:
                                    year_temps.append(data_point['y'])
                                    break
                    except Exception as e:
                        if DEBUG:
                            logger.debug(f"Error getting data for {current_date}: {e}")
                        track_missing_year(missing_years, year, "sampling_error")
                        continue
                
                # Calculate average for the year (only add one value per year)
                if year_temps:
                    avg_temp = sum(year_temps) / len(year_temps)
                    all_temps.append(avg_temp)  # Add to all_temps only once per year
                    values.append(TemperatureValue(
                        date=end_date.replace(year=year).strftime("%Y-%m-%d"),
                        year=year,
                        temperature=round(avg_temp, 1),
                    ))
                else:
                    track_missing_year(missing_years, year, "no_data_sampling")
    
    elif period == "yearly":
        # For yearly, use the Visual Crossing historysummary endpoint for efficiency
        try:
            if DEBUG:
                logger.debug(f"Fetching yearly summary for {location} from {start_year} to {current_year}")
            yearly_data = await _fetch_yearly_summary(location, start_year, current_year)
            if DEBUG:
                logger.debug(f"Got {len(yearly_data)} yearly data points")
            
            if yearly_data:
                for year, temp in yearly_data:
                    all_temps.append(temp)
                    values.append(TemperatureValue(
                        date=f"{year}-{month:02d}-{day:02d}",
                        year=year,
                        temperature=round(temp, 1),
                    ))
            else:
                if DEBUG:
                    logger.debug("No yearly data returned, falling back to sampling")
                raise Exception("No yearly data returned")
        except Exception as e:
            # Historysummary endpoint often fails (400 errors), fallback is normal - don't log every occurrence
            pass
            # Fallback to simple sampling approach if historysummary fails
            # Just sample a few representative days for each year
            sample_dates = [
                (1, 15), (4, 15), (7, 15), (10, 15)  # Mid-month samples for each season
            ]
            
            for year in range(start_year, current_year + 1):
                year_values = []
                for sample_month, sample_day in sample_dates:
                    try:
                        weather_data = await get_temperature_series(location, sample_month, sample_day, redis_client)
                        if weather_data and 'data' in weather_data:
                            # Extract missing years from the series metadata
                            if 'metadata' in weather_data and 'missing_years' in weather_data['metadata']:
                                for missing_year_info in weather_data['metadata']['missing_years']:
                                    if missing_year_info['year'] == year:
                                        track_missing_year(missing_years, year, f"{missing_year_info['reason']}_yearly_sampling")
                                        break
                            
                            for data_point in weather_data['data']:
                                if int(data_point['x']) == year and data_point['y'] is not None:
                                    temp = data_point['y']
                                    year_values.append(temp)
                                    all_temps.append(temp)
                                    break
                    except Exception as e:
                        if DEBUG:
                            logger.debug(f"Error getting data for {year}-{sample_month:02d}-{sample_day:02d}: {e}")
                        track_missing_year(missing_years, year, "yearly_sampling_error")
                        continue
                
                # Calculate average for the year
                if year_values:
                    avg_temp = sum(year_values) / len(year_values)
                    values.append(TemperatureValue(
                        date=f"{year}-{month:02d}-{day:02d}",
                        year=year,
                        temperature=round(avg_temp, 1),
                    ))
                else:
                    track_missing_year(missing_years, year, "no_data_yearly_sampling")
    
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
    
    # Replace the friendly date in the summary with our period-specific version
    summary_text = summary_text.replace(get_friendly_date(end_date_obj), friendly_date)
    
    # Create comprehensive metadata
    additional_metadata = {
        "period_days": date_range_days, 
        "end_date": end_date_obj.strftime("%Y-%m-%d")
    }
    
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
        "metadata": create_metadata(len(years), len(values), missing_years, additional_metadata)
    }


@router.get("/v1/records/{period}/{location}/{identifier}", response_model=RecordResponse)
async def get_record(
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
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()
        
        # Create cache key for v1 endpoint with comprehensive format
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:values,average,trend,summary"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 RECORD: {cache_key}")
                data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                data = json.loads(data_str)
                
                # Validate cached data
                is_valid, error_msg = validate_location_response(data, location)
                if not is_valid:
                    # Mark location as invalid and return error
                    invalid_location_cache.mark_location_invalid(location, "no_data_cached")
                    raise HTTPException(status_code=400, detail=error_msg)
                
                # Add updated timestamp for cached data
                updated_timestamp = await get_cache_updated_timestamp(cache_key, redis_client)
                if updated_timestamp:
                    data["updated"] = updated_timestamp.isoformat()
                
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1")
                return json_response
        
        # Get data
        data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Add updated timestamp for newly computed data
        current_time = datetime.now(timezone.utc)
        data["updated"] = current_time.isoformat()
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION  # Use long cache for historical data
            set_cache_value(cache_key, cache_duration, json.dumps(data), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=data)
        set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1")
        return json_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records endpoint: {e}")
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
        
        # Create cache key for subresource
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:average"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records_average", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 AVERAGE: {cache_key}")
                data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                data = json.loads(data_str)
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|average|metric|v1")
                return json_response
        
        # Get full record data
        record_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(record_data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Extract average data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["average"],
            metadata=record_data["metadata"]
        )
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION
            set_cache_value(cache_key, cache_duration, json.dumps(response_data.model_dump()), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
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
        
        # Create cache key for subresource
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:trend"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records_trend", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 TREND: {cache_key}")
                data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                data = json.loads(data_str)
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|trend|metric|v1")
                return json_response
        
        # Get full record data
        record_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(record_data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Extract trend data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["trend"],
            metadata=record_data["metadata"]
        )
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION
            set_cache_value(cache_key, cache_duration, json.dumps(response_data.model_dump()), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
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
        
        # Create cache key for subresource
        normalized_location = normalize_location_for_cache(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:summary"
        
        # Check cache first
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "v1_records_summary", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED V1 SUMMARY: {cache_key}")
                data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                data = json.loads(data_str)
                # Set smart cache headers for cached data too
                json_response = JSONResponse(content=data)
                set_weather_cache_headers(json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|summary|metric|v1")
                return json_response
        
        # Get full record data
        record_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
        
        # Validate the response data
        is_valid, error_msg = validate_location_response(record_data, location)
        if not is_valid:
            # Mark location as invalid for future requests
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Extract summary data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["summary"],
            metadata=record_data["metadata"]
        )
        
        # Cache the result
        if CACHE_ENABLED:
            cache_duration = LONG_CACHE_DURATION
            set_cache_value(cache_key, cache_duration, json.dumps(response_data.model_dump()), redis_client)
        
        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
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
