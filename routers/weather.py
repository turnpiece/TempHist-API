"""Weather endpoints."""

import json
import logging
from datetime import date as dt_date

import redis
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from fastapi.responses import JSONResponse

from cache.accessors import get_cache_stats
from cache.core import get_cache_value, set_cache_value
from cache.keys import generate_cache_key
from config import CACHE_ENABLED, DEBUG
from utils.cache_headers import set_weather_cache_headers
from utils.sanitization import sanitize_for_logging
from utils.weather import get_forecast_cache_duration, is_today_or_future
from utils.weather_data import _c_to_f, get_forecast_data, get_weather_for_date

logger = logging.getLogger(__name__)
router = APIRouter()

_TEMP_FIELDS = ("temp", "tempmin", "tempmax")


def _apply_unit_group_to_weather(result: dict, unit_group: str) -> dict:
    """Convert temp fields in a /weather response from Celsius to the requested unit."""
    if unit_group.lower() not in ("fahrenheit", "us"):
        return result
    if "days" not in result:
        return result
    converted_days = []
    for day in result["days"]:
        day = dict(day)
        for field in _TEMP_FIELDS:
            if field in day and day[field] is not None:
                day[field] = _c_to_f(day[field])
        converted_days.append(day)
    return {**result, "days": converted_days}


from routers.dependencies import get_redis_client  # noqa: E402


@router.get("/weather/{location:path}/{date}")
async def get_weather(
    location: str = Path(..., description="Location name", max_length=200),
    date: str = Path(..., description="Date in YYYY-MM-DD format"),
    unit_group: str = Query("celsius", description="Temperature unit: 'celsius' or 'fahrenheit'"),
    response: Response = None,
    redis_client: redis.Redis = Depends(get_redis_client),
):
    """Get weather data for a specific location and date.

    Note: Location is validated by validate_location_for_ssrf() which enforces
    max_length=200 and prevents SSRF attacks.
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
            data_str = cached_data.decode("utf-8") if isinstance(cached_data, bytes) else cached_data
            result = json.loads(data_str)
            result = _apply_unit_group_to_weather(result, unit_group)
            # Set smart cache headers for cached data too
            set_weather_cache_headers(response, req_date=req_date, key_parts=f"{location}|{date}|metric|v1")
            return result
        if DEBUG:
            logger.info(f"❌ CACHE MISS: {cache_key} — fetching from API")
    else:
        if DEBUG:
            logger.info("⚠️  CACHING DISABLED: fetching from API")

    logger.info("[DEBUG] About to call get_weather_for_date...")
    result = await get_weather_for_date(location, date, redis_client)

    # Log the final response being returned to the client
    if DEBUG:
        logger.debug(f"🎯 FINAL RESPONSE TO CLIENT: {location} | {date}")
        logger.debug(f"📄 FINAL JSON RESPONSE: {json.dumps(result, indent=2)}")

    # Set smart cache headers based on data age
    set_weather_cache_headers(response, req_date=req_date, key_parts=f"{location}|{date}|metric|v1")

    # Cache always stores Celsius; apply unit conversion after caching
    if CACHE_ENABLED and "error" not in result:
        try:
            year, month, day = map(int, date.split("-")[:3])
            from config import LONG_CACHE_DURATION, SHORT_CACHE_DURATION

            cache_duration = SHORT_CACHE_DURATION if is_today_or_future(year, month, day) else LONG_CACHE_DURATION
        except Exception:
            cache_duration = LONG_CACHE_DURATION
        set_cache_value(cache_key, cache_duration, json.dumps(result), redis_client)
    return _apply_unit_group_to_weather(result, unit_group)


@router.get("/forecast/{location}")
async def get_forecast(
    location: str = Path(..., description="Location name", max_length=200),
    unit_group: str = Query("celsius", description="Temperature unit: 'celsius' or 'fahrenheit'"),
    redis_client: redis.Redis = Depends(get_redis_client),
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
                data_str = cached_forecast.decode("utf-8") if isinstance(cached_forecast, bytes) else cached_forecast
                cached_result = json.loads(data_str)
                # Cache always stores Celsius; convert on output if needed
                use_fahrenheit = unit_group.lower() in ("fahrenheit", "us")
                if use_fahrenheit and "average_temperature" in cached_result:
                    cached_result = dict(cached_result)
                    cached_result["average_temperature"] = _c_to_f(cached_result["average_temperature"])
                    cached_result["unit"] = "fahrenheit"
                return JSONResponse(content=cached_result, headers={"Cache-Control": "public, max-age=1800"})
            elif DEBUG:
                logger.debug(f"❌ FORECAST CACHE MISS: {cache_key} — fetching fresh forecast")

        # Fetch fresh forecast data — always in Celsius for caching consistency
        from datetime import datetime

        today = datetime.now().date()
        result = await get_forecast_data(location, today, redis_client, unit_group="celsius")

        # Cache the Celsius result; unit conversion happens below before returning
        if CACHE_ENABLED and "error" not in result:
            cache_duration = get_forecast_cache_duration()
            # Convert date object to string for JSON serialization
            if "date" in result and hasattr(result["date"], "strftime"):
                result["date"] = result["date"].strftime("%Y-%m-%d")
            set_cache_value(cache_key, cache_duration, json.dumps(result), redis_client)
            if DEBUG:
                current_hour = datetime.now().hour
                time_period = "stable" if current_hour >= 18 else "active"
                logger.debug(f"💾 CACHED FORECAST: {cache_key} | Duration: {cache_duration} | Time: {time_period}")

        # Convert date object to string for JSON response
        if "date" in result and hasattr(result["date"], "strftime"):
            result["date"] = result["date"].strftime("%Y-%m-%d")

        # Apply unit conversion after caching (cache always stores Celsius)
        use_fahrenheit = unit_group.lower() in ("fahrenheit", "us")
        if use_fahrenheit and "average_temperature" in result and "error" not in result:
            result = dict(result)
            result["average_temperature"] = _c_to_f(result["average_temperature"])
            result["unit"] = "fahrenheit"

        return JSONResponse(content=result, headers={"Cache-Control": "public, max-age=1800"})

    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}")
        return JSONResponse(content={"error": f"Forecast error: {str(e)}"}, status_code=500)
