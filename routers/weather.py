"""Weather endpoints."""
import json
import logging
import asyncio
import redis
from datetime import date as dt_date
from fastapi import APIRouter, HTTPException, Path, Response, Depends
from fastapi.responses import JSONResponse
from config import CACHE_ENABLED, DEBUG
from cache_utils import (
    get_cache_value, set_cache_value, generate_cache_key, get_cache_stats
)
from utils.weather_data import get_weather_for_date, get_forecast_data
from utils.sanitization import sanitize_for_logging
from utils.cache_headers import set_weather_cache_headers
from utils.weather import is_today_or_future, get_forecast_cache_duration

logger = logging.getLogger(__name__)
router = APIRouter()


from routers.dependencies import get_redis_client


@router.get("/weather/{location:path}/{date}")
def get_weather(
    location: str = Path(..., description="Location name", max_length=200),
    date: str = Path(..., description="Date in YYYY-MM-DD format"),
    response: Response = None,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get weather data for a specific location and date.
    
    Note: Location is validated by validate_location_for_ssrf() in build_visual_crossing_url() 
    which enforces max_length=200 and prevents SSRF attacks.
    """
    logger.info(f"[DEBUG] Weather endpoint called with location={sanitize_for_logging(location)}, date={date}")
    
    # Parse the date for cache headers
    try:
        req_date = dt_date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    cache_key = generate_cache_key("weather", location, date)
    logger.info(f"[DEBUG] Cache key: {cache_key}")
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        if DEBUG:
            logger.info(f"🔍 CACHE CHECK: {cache_key}")
        cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
        if cached_data:
            if DEBUG:
                logger.info(f"✅ SERVING CACHED DATA: {cache_key} | Location: {location} | Date: {date}")
            data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
            result = json.loads(data_str)
            # Set smart cache headers for cached data too
            set_weather_cache_headers(response, req_date=req_date, key_parts=f"{location}|{date}|metric|v1")
            return result
        if DEBUG:
            logger.info(f"❌ CACHE MISS: {cache_key} — fetching from API")
    else:
        if DEBUG:
            logger.info(f"⚠️  CACHING DISABLED: fetching from API")
    
    logger.info(f"[DEBUG] About to call get_weather_for_date...")
    # Use the shared async function for consistency
    # Since this is a sync endpoint, run the async function in the event loop
    result = asyncio.run(get_weather_for_date(location, date, redis_client))
    
    # Log the final response being returned to the client
    if DEBUG:
        logger.debug(f"🎯 FINAL RESPONSE TO CLIENT: {location} | {date}")
        logger.debug(f"📄 FINAL JSON RESPONSE: {json.dumps(result, indent=2)}")
    
    # Set smart cache headers based on data age
    set_weather_cache_headers(response, req_date=req_date, key_parts=f"{location}|{date}|metric|v1")
    
    # Only cache successful results if caching is enabled and not already cached
    if CACHE_ENABLED and "error" not in result:
        try:
            year, month, day = map(int, date.split("-")[:3])
            from config import SHORT_CACHE_DURATION, LONG_CACHE_DURATION
            cache_duration = SHORT_CACHE_DURATION if is_today_or_future(year, month, day) else LONG_CACHE_DURATION
        except Exception:
            cache_duration = LONG_CACHE_DURATION
        set_cache_value(cache_key, cache_duration, json.dumps(result), redis_client)
    return result


@router.get("/forecast/{location}")
async def get_forecast(
    location: str = Path(..., description="Location name", max_length=200),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get weather forecast for a location with time-based caching."""
    try:
        # Create cache key for forecast
        cache_key = generate_cache_key("forecast", location)
        
        # Check cache first if caching is enabled
        if CACHE_ENABLED:
            cached_forecast = get_cache_value(cache_key, redis_client, "forecast", location, get_cache_stats())
            if cached_forecast:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED FORECAST: {cache_key} | Location: {location}")
                data_str = cached_forecast.decode('utf-8') if isinstance(cached_forecast, bytes) else cached_forecast
                return JSONResponse(content=json.loads(data_str), headers={"Cache-Control": "public, max-age=1800"})
            elif DEBUG:
                logger.debug(f"❌ FORECAST CACHE MISS: {cache_key} — fetching fresh forecast")
        
        # Fetch fresh forecast data
        from datetime import datetime
        today = datetime.now().date()
        result = await get_forecast_data(location, today, redis_client)
        
        # Cache the result if caching is enabled and no error
        if CACHE_ENABLED and "error" not in result:
            cache_duration = get_forecast_cache_duration()
            # Convert date object to string for JSON serialization
            if "date" in result and hasattr(result["date"], 'strftime'):
                result["date"] = result["date"].strftime("%Y-%m-%d")
            set_cache_value(cache_key, cache_duration, json.dumps(result), redis_client)
            if DEBUG:
                current_hour = datetime.now().hour
                time_period = "stable" if current_hour >= 18 else "active"
                logger.debug(f"💾 CACHED FORECAST: {cache_key} | Duration: {cache_duration} | Time: {time_period}")
        
        # Convert date object to string for JSON response
        if "date" in result and hasattr(result["date"], 'strftime'):
            result["date"] = result["date"].strftime("%Y-%m-%d")
        
        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=1800"})
    
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}")
        return JSONResponse(
            content={"error": f"Forecast error: {str(e)}"},
            status_code=500
        )
