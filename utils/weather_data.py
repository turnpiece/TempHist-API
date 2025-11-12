"""Weather data fetching functions."""
import json
import logging
import asyncio
import aiohttp
import httpx
from typing import Dict
from datetime import datetime, timedelta
from config import (
    CACHE_ENABLED, FILTER_WEATHER_DATA, SHORT_CACHE_DURATION_SECONDS,
    LONG_CACHE_DURATION_SECONDS, SHORT_CACHE_DURATION, LONG_CACHE_DURATION,
    HTTP_TIMEOUT, HTTP_TIMEOUT_VISUAL_CROSSING, DEBUG, MAX_CONCURRENT_REQUESTS
)
from cache_utils import (
    get_cache_value, set_cache_value, get_weather_cache_key, generate_cache_key,
    get_cache_stats, store_location_timezone
)
from utils.validation import build_visual_crossing_url
from utils.sanitization import sanitize_url, sanitize_for_logging
from utils.weather import is_today
import redis

logger = logging.getLogger(__name__)


async def get_weather_for_date(
    location: str,
    date_str: str,
    redis_client: redis.Redis
) -> dict:
    """Fetch and cache weather data for a specific date.
    
    Implements fallback logic for remote data:
    1. First tries without remote data parameters
    2. If no temperature data and year >= 2005, retries with remote data parameters
    3. Never uses remote data for today's data
    """
    logger.debug(f"get_weather_for_date for {sanitize_for_logging(location)} on {date_str}")
    cache_key = get_weather_cache_key(location, date_str)
    if CACHE_ENABLED:
        cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
        if cached_data:
            if DEBUG:
                logger.debug(f"✅ SERVING CACHED WEATHER: {cache_key} | Location: {location} | Date: {date_str}")
            try:
                data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                return json.loads(data_str)
            except Exception as e:
                logger.error(f"Error decoding cached data for {cache_key}: {e}")

    logger.debug(f"Cache miss: {cache_key} — fetching from API")
    
    # Parse the date to determine the year
    try:
        year, month, day = map(int, date_str.split("-")[:3])
        is_today_date = is_today(year, month, day)
    except Exception:
        year = 2000  # Default fallback
        is_today_date = False
    
    # First attempt: without remote data parameters
    url = build_visual_crossing_url(location, date_str, remote=False)
    logger.debug(f"First attempt (no remote data): {sanitize_url(url)}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)  # Increased from 30s to 60s for better reliability
        logger.info(f"[DEBUG] Creating aiohttp session with 60-second timeout")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            logger.info(f"[DEBUG] Session created successfully, making GET request to: {sanitize_url(url)}")
            async with session.get(url, headers={"Accept-Encoding": "gzip"}) as resp:
                logger.info(f"[DEBUG] Response received: status={resp.status}, content-type={resp.headers.get('Content-Type')}")
                if resp.status == 200 and 'application/json' in resp.headers.get('Content-Type', ''):
                    logger.info(f"[DEBUG] Parsing JSON response...")
                    data = await resp.json()
                    logger.info(f"[DEBUG] JSON parsed successfully, checking for 'days' data")
                    
                    # Extract and store timezone from Visual Crossing response
                    if CACHE_ENABLED and redis_client:
                        timezone_str = data.get('timezone')
                        if timezone_str:
                            store_location_timezone(location, timezone_str, redis_client)
                    
                    days = data.get('days')
                    if days is not None and len(days) > 0:
                        # Check if we got valid temperature data
                        day_data = days[0]
                        temp = day_data.get('temp')
                        
                        if temp is not None:
                            # Success! Cache and return the data
                            if FILTER_WEATHER_DATA:
                                # Filter to only essential temperature data
                                filtered_days = []
                                for day_data in days:
                                    filtered_day = {
                                        'datetime': day_data.get('datetime'),
                                        'temp': day_data.get('temp'),
                                        'tempmin': day_data.get('tempmin'),
                                        'tempmax': day_data.get('tempmax')
                                    }
                                    filtered_days.append(filtered_day)
                                to_cache = {"days": filtered_days}
                            else:
                                # Return full data if filtering is disabled
                                to_cache = {"days": days}

                            if DEBUG:
                                logger.debug(f"🌤️ FRESH API RESPONSE: {location} | {date_str}")

                            if CACHE_ENABLED:
                                cache_duration = timedelta(seconds=SHORT_CACHE_DURATION_SECONDS) if is_today_date else timedelta(seconds=LONG_CACHE_DURATION_SECONDS)
                                set_cache_value(cache_key, cache_duration, json.dumps(to_cache), redis_client)
                            return to_cache
                        else:
                            logger.debug(f"No temperature data in first attempt for {date_str}")
                    else:
                        logger.debug(f"No 'days' data in first attempt for {date_str}")
                else:
                    logger.debug(f"First attempt failed with status {resp.status}")
                
                # If we reach here, the first attempt didn't provide valid temperature data
                # Check if we should try with remote data parameters
                if not is_today_date and year >= 2005:
                    logger.info(f"[DEBUG] First attempt failed, trying remote fallback for {date_str} (year {year})")
                    url_with_remote = build_visual_crossing_url(location, date_str, remote=True)
                    logger.info(f"[DEBUG] Remote fallback URL: {sanitize_url(url_with_remote)}")
                    
                    logger.info(f"[DEBUG] Making remote fallback request...")
                    async with session.get(url_with_remote, headers={"Accept-Encoding": "gzip"}) as remote_resp:
                        logger.info(f"[DEBUG] Remote response received: status={remote_resp.status}, content-type={remote_resp.headers.get('Content-Type')}")
                        if remote_resp.status == 200 and 'application/json' in remote_resp.headers.get('Content-Type', ''):
                            logger.info(f"[DEBUG] Parsing remote JSON response...")
                            remote_data = await remote_resp.json()
                            logger.info(f"[DEBUG] Remote JSON parsed successfully, checking for 'days' data")
                            
                            # Extract and store timezone from Visual Crossing response
                            if CACHE_ENABLED and redis_client:
                                timezone_str = remote_data.get('timezone')
                                if timezone_str:
                                    store_location_timezone(location, timezone_str, redis_client)
                            
                            remote_days = remote_data.get('days')
                            if remote_days is not None and len(remote_days) > 0:
                                remote_day_data = remote_days[0]
                                remote_temp = remote_day_data.get('temp')
                                
                                if remote_temp is not None:
                                    # Success with remote data! Cache and return
                                    logger.debug(f"Remote data fallback successful for {date_str}")
                                    to_cache = {"days": remote_days}
                                    if DEBUG:
                                        logger.debug(f"🌤️ REMOTE FALLBACK RESPONSE: {location} | {date_str}")
                                    if CACHE_ENABLED:
                                        set_cache_value(cache_key, timedelta(seconds=LONG_CACHE_DURATION_SECONDS), json.dumps(to_cache), redis_client)
                                    return to_cache
                                else:
                                    logger.debug(f"No temperature data in remote fallback for {date_str}")
                            else:
                                logger.debug(f"No 'days' data in remote fallback for {date_str}")
                        else:
                            logger.debug(f"Remote fallback failed with status {remote_resp.status}")
                
                # If we reach here, neither attempt provided valid temperature data
                logger.info(f"[DEBUG] Both attempts failed, returning error response")
                return {"error": "No temperature data available", "status": resp.status}
                
    except Exception as e:
        logger.error(f"[DEBUG] Exception occurred in get_weather_for_date for {date_str}: {str(e)}")
        logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return {"error": str(e)}


async def get_forecast_data(location: str, date, redis_client: redis.Redis = None) -> Dict:
    """Get forecast data for a location and date using httpx."""
    # Convert date to string format if it's a date object
    if hasattr(date, 'strftime'):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)
    
    # Debug logging to see what location we're processing
    logger.debug(f"🌐 Fetching forecast for location: '{location}' (repr: {repr(location)}), date: {date_str}")
    
    try:
        url = build_visual_crossing_url(location, date_str, remote=False)
        logger.debug(f"🔗 Built URL (sanitized): {sanitize_url(url)}")
    except Exception as url_error:
        logger.error(f"❌ Error building URL for location '{location}': {url_error}")
        raise
    
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_VISUAL_CROSSING) as client:
        response = await client.get(url, headers={"Accept-Encoding": "gzip"})
    if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
        data = response.json()
        
        # Extract and store timezone from Visual Crossing response
        if CACHE_ENABLED and redis_client:
            timezone_str = data.get('timezone')
            if timezone_str:
                store_location_timezone(location, timezone_str, redis_client)
        
        if data.get('days') and len(data['days']) > 0:
            day_data = data['days'][0]
            temp = day_data.get('temp')
            if temp is None:
                return {"error": "No temperature data in forecast response"}
            return {
                "location": location,
                "date": date,
                "average_temperature": round(temp, 1),
                "unit": "celsius"
            }
        else:
            return {"error": "No days data in forecast response"}
    return {"error": response.text, "status": response.status_code}


async def fetch_weather_batch(
    location: str,
    date_strs: list,
    redis_client: redis.Redis,
    max_concurrent: int = None,
    visual_crossing_semaphore=None
) -> dict:
    """
    Fetch weather data for multiple dates in parallel using httpx with concurrency control.
    Returns a dict mapping date_str to weather data.
    Now uses caching for each date and global concurrency semaphore.
    """
    logger.debug(f"fetch_weather_batch for {location}")
    from config import API_KEY
    if not API_KEY:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Visual Crossing API key not configured")
    
    # Use provided semaphore or create new one
    if visual_crossing_semaphore is not None:
        semaphore = visual_crossing_semaphore
    else:
        # Use global semaphore if max_concurrent not specified
        if max_concurrent is None:
            max_concurrent = MAX_CONCURRENT_REQUESTS
        semaphore = asyncio.Semaphore(max_concurrent)
    
    results = {}
    
    async def fetch_one(date_str):
        async with semaphore:
            return date_str, await get_weather_for_date(location, date_str, redis_client)
    
    tasks = [fetch_one(date_str) for date_str in date_strs]
    for fut in asyncio.as_completed(tasks):
        date_str, result = await fut
        results[date_str] = result
    return results


async def get_temperature_series(
    location: str,
    month: int,
    day: int,
    redis_client: redis.Redis
) -> Dict:
    """Get temperature series data for a location and date over multiple years."""
    from utils.weather import get_year_range, track_missing_year, create_metadata
    
    # Check for cached series first
    series_cache_key = generate_cache_key("series", location, f"{month:02d}_{day:02d}")
    if CACHE_ENABLED:
        cached_series = get_cache_value(series_cache_key, redis_client, "series", location, get_cache_stats())
        if cached_series:
            logger.debug(f"Cache hit: {series_cache_key}")
            try:
                data_str = cached_series.decode('utf-8') if isinstance(cached_series, bytes) else cached_series
                return json.loads(data_str)
            except Exception as e:
                logger.error(f"Error decoding cached series for {series_cache_key}: {e}")
    
    # Note: Removed is_valid_location() check - it makes unnecessary API calls.
    # The Visual Crossing API will naturally return an error if the location is invalid.
    logger.debug(f"get_temperature_series for {location} on {day}/{month}")
    today = datetime.now()
    current_year = today.year
    years = get_year_range(current_year)

    data = []
    missing_years = []
    uncached_years = []
    uncached_date_strs = []
    year_to_date_str = {}
    for year in years:
        logger.debug(f"get_temperature_series year: {year}")
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = generate_cache_key("weather", location, date_str)
        year_to_date_str[year] = date_str

        # Check cache for all dates (Visual Crossing API works the same for past/present/future)
        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(f"✅ SERVING CACHED TEMPERATURE: {cache_key} | Location: {location} | Year: {year}")
                data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                weather = json.loads(data_str)
                try:
                    temp = weather["days"][0]["temp"]
                    if temp is not None:
                        data.append({"x": year, "y": temp})
                    else:
                        logger.debug(f"Temperature is None for {year}, marking as missing.")
                        track_missing_year(missing_years, year, "no_temperature_data")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error processing cached data for {date_str}: {str(e)}")
                    track_missing_year(missing_years, year, "data_processing_error")
                continue
            else:
                uncached_years.append(year)
                uncached_date_strs.append(date_str)
        else:
            uncached_years.append(year)
            uncached_date_strs.append(date_str)

    # Batch fetch for all uncached years
    if uncached_date_strs:
        # Import here to avoid circular dependency
        from utils.weather_data import fetch_weather_batch
        batch_results = await fetch_weather_batch(location, uncached_date_strs, redis_client)
        for year, date_str in zip(uncached_years, uncached_date_strs):
            weather = batch_results.get(date_str)
            if weather and "error" not in weather:
                try:
                    temp = weather["days"][0]["temp"]
                    logger.debug(f"Got {year} temperature = {temp}")
                    if temp is not None:
                        data.append({"x": year, "y": temp})
                        # Cache the result if caching is enabled
                        if CACHE_ENABLED:
                            cache_key = generate_cache_key("weather", location, date_str)
                            # Determine cache duration based on how recent/future the date is
                            target_date = datetime(year, month, day).date()
                            days_diff = (target_date - today.date()).days
                            # Use short cache for today, future dates, and recent past (last 7 days)
                            # These may be forecasts or recently updated data
                            if -7 <= days_diff <= 365:  # Last 7 days or future dates
                                cache_duration = SHORT_CACHE_DURATION
                            else:  # Historical data (8+ days old)
                                cache_duration = LONG_CACHE_DURATION
                            set_cache_value(cache_key, cache_duration, json.dumps(weather), redis_client)
                    else:
                        logger.debug(f"Temperature is None for {year}, marking as missing.")
                        track_missing_year(missing_years, year, "no_temperature_data")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Error processing batch data for {date_str}: {str(e)}")
                    track_missing_year(missing_years, year, "data_processing_error")
            else:
                track_missing_year(missing_years, year, weather.get("error", "api_error") if weather else "api_error")

    # Print summary of collected data
    logger.debug("Data summary:")
    logger.debug(f"Total data points: {len(data)}")
    if data:
        temps = [d['y'] for d in data]
        if temps:  # Add check to ensure temps list is not empty
            logger.debug(f"Temperature range: {min(temps):.1f}°C to {max(temps):.1f}°C")
            logger.debug(f"Average temperature: {sum(temps)/len(temps):.1f}°C")

    data_list = sorted(data, key=lambda d: d['x'])

    if data_list:
        latest_year = data_list[-1]['x']
        if isinstance(latest_year, str):
            try:
                latest_year = int(latest_year)
            except ValueError:
                latest_year = None
        if latest_year is not None and latest_year != current_year:
            if not any(entry.get("year") == current_year for entry in missing_years):
                track_missing_year(missing_years, current_year, "no_data_current_year")
    elif current_year not in [entry.get("year") for entry in missing_years]:
        track_missing_year(missing_years, current_year, "no_data_current_year")

    if not data_list:
        logger.warning(f"No valid temperature data found for {location} on {month}-{day}")
        return {
            "data": [],
            "metadata": create_metadata(len(years), 0, [{"year": y, "reason": "no_data"} for y in years]),
            "error": f"Invalid location: {location}"
        }

    # Cache the entire series for a short duration
    if CACHE_ENABLED:
        set_cache_value(series_cache_key, SHORT_CACHE_DURATION, json.dumps({
            "data": data_list,
            "metadata": create_metadata(len(years), len(data), missing_years)
        }), redis_client)

    return {
        "data": data_list,
        "metadata": create_metadata(len(years), len(data), missing_years)
    }
