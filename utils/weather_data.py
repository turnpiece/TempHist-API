"""Weather data fetching functions — delegates to the configured weather provider."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict

import redis

from cache.accessors import get_cache_stats
from cache.core import get_cache_value, set_cache_value
from cache.keys import generate_cache_key, get_weather_cache_key
from config import (
    CACHE_ENABLED,
    DEBUG,
    FILTER_WEATHER_DATA,
    LONG_CACHE_DURATION,
    LONG_CACHE_DURATION_SECONDS,
    MAX_CONCURRENT_REQUESTS,
    SHORT_CACHE_DURATION,
    SHORT_CACHE_DURATION_SECONDS,
)
from utils.sanitization import sanitize_for_logging
from utils.weather import is_today
from utils.weather_provider import fetch_single_date

logger = logging.getLogger(__name__)


def _c_to_f(value) -> float | None:
    """Convert Celsius to Fahrenheit, rounded to 2dp."""
    if value is None:
        return None
    return round(float(value) * 9 / 5 + 32, 2)


async def get_weather_for_date(
    location: str,
    date_str: str,
    redis_client: redis.Redis,
) -> dict:
    """Fetch and cache weather data for a specific date via the configured weather provider."""
    logger.debug("get_weather_for_date for %s on %s", sanitize_for_logging(location), date_str)
    cache_key = get_weather_cache_key(location, date_str)
    if CACHE_ENABLED:
        cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
        if cached_data:
            if DEBUG:
                logger.debug(
                    "✅ SERVING CACHED WEATHER: %s | Location: %s | Date: %s",
                    cache_key,
                    location,
                    date_str,
                )
            try:
                data_str = cached_data.decode("utf-8") if isinstance(cached_data, bytes) else cached_data
                return json.loads(data_str)
            except Exception as e:
                logger.error("Error decoding cached data for %s: %s", cache_key, e)

    logger.debug("Cache miss: %s — fetching from weather provider", cache_key)

    try:
        year, month, day = map(int, date_str.split("-")[:3])
        is_today_date = is_today(year, month, day)
    except Exception:
        is_today_date = False

    try:
        result = await fetch_single_date(location, date_str)
    except Exception as exc:
        logger.error("Weather fetch failed for %s on %s: %s", location, date_str, exc)
        return {"error": str(exc)}

    if "error" in result:
        return result

    days = result.get("days", [])
    if not days:
        return {"error": "No temperature data available"}

    day_data = days[0]
    temp = day_data.get("temp")

    if temp is None:
        return {"error": "No temperature data available"}

    if FILTER_WEATHER_DATA:
        to_cache = {
            "days": [
                {
                    "datetime": day_data.get("datetime"),
                    "temp": day_data.get("temp"),
                    "tempmin": day_data.get("tempmin"),
                    "tempmax": day_data.get("tempmax"),
                }
            ]
        }
    else:
        to_cache = {"days": [day_data]}

    if DEBUG:
        logger.debug("🌤️ FRESH WEATHER RESPONSE: %s | %s", location, date_str)

    if CACHE_ENABLED:
        cache_duration = (
            timedelta(seconds=SHORT_CACHE_DURATION_SECONDS)
            if is_today_date
            else timedelta(seconds=LONG_CACHE_DURATION_SECONDS)
        )
        set_cache_value(cache_key, cache_duration, json.dumps(to_cache), redis_client)

    return to_cache


async def get_forecast_data(location: str, date, redis_client: redis.Redis = None, unit_group: str = "celsius") -> Dict:
    """Get forecast data for a location and date via the configured weather provider."""
    if hasattr(date, "strftime"):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)

    logger.debug("🌐 Fetching forecast for location: %r, date: %s", location, date_str)

    try:
        result = await fetch_single_date(location, date_str)
    except Exception as exc:
        logger.error("Weather forecast fetch failed for %s on %s: %s", location, date_str, exc)
        return {"error": str(exc)}

    if "error" in result:
        return result

    days = result.get("days", [])
    if not days:
        return {"error": "No days data in forecast response"}

    day_data = days[0]
    temp_c = day_data.get("temp")
    if temp_c is None:
        return {"error": "No temperature data in forecast response"}

    use_fahrenheit = unit_group.lower() in ("fahrenheit", "us")
    return {
        "location": location,
        "date": date,
        "average_temperature": _c_to_f(temp_c) if use_fahrenheit else temp_c,
        "unit": "fahrenheit" if use_fahrenheit else "celsius",
    }


async def fetch_weather_batch(
    location: str,
    date_strs: list,
    redis_client: redis.Redis,
    max_concurrent: int = None,
    semaphore_override=None,
) -> dict:
    """Fetch weather data for multiple dates in parallel via the configured weather provider.

    Returns a dict mapping date_str to weather data.
    """
    logger.debug("fetch_weather_batch for %s", location)

    semaphore = semaphore_override or asyncio.Semaphore(max_concurrent or MAX_CONCURRENT_REQUESTS)

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
    redis_client: redis.Redis,
) -> Dict:
    """Get temperature series data for a location and date over multiple years."""
    from utils.weather import create_metadata, get_year_range, track_missing_year

    series_cache_key = generate_cache_key("series", location, f"{month:02d}_{day:02d}")
    if CACHE_ENABLED:
        cached_series = get_cache_value(series_cache_key, redis_client, "series", location, get_cache_stats())
        if cached_series:
            logger.debug("Cache hit: %s", series_cache_key)
            try:
                data_str = cached_series.decode("utf-8") if isinstance(cached_series, bytes) else cached_series
                return json.loads(data_str)
            except Exception as e:
                logger.error("Error decoding cached series for %s: %s", series_cache_key, e)

    logger.debug("get_temperature_series for %s on %d/%d", location, day, month)
    today = datetime.now()
    current_year = today.year
    years = get_year_range(current_year)

    data = []
    missing_years = []
    uncached_years = []
    uncached_date_strs = []
    year_to_date_str = {}
    for year in years:
        logger.debug("get_temperature_series year: %d", year)
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = generate_cache_key("weather", location, date_str)
        year_to_date_str[year] = date_str

        if CACHE_ENABLED:
            cached_data = get_cache_value(cache_key, redis_client, "weather", location, get_cache_stats())
            if cached_data:
                if DEBUG:
                    logger.debug(
                        "✅ SERVING CACHED TEMPERATURE: %s | Location: %s | Year: %d",
                        cache_key,
                        location,
                        year,
                    )
                data_str = cached_data.decode("utf-8") if isinstance(cached_data, bytes) else cached_data
                weather = json.loads(data_str)
                try:
                    temp = weather["days"][0]["temp"]
                    if temp is not None:
                        data.append({"x": year, "y": temp})
                    else:
                        logger.debug("Temperature is None for %d, marking as missing.", year)
                        track_missing_year(missing_years, year, "no_temperature_data")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error("Error processing cached data for %s: %s", date_str, e)
                    track_missing_year(missing_years, year, "data_processing_error")
                continue
            else:
                uncached_years.append(year)
                uncached_date_strs.append(date_str)
        else:
            uncached_years.append(year)
            uncached_date_strs.append(date_str)

    if uncached_date_strs:
        batch_results = await fetch_weather_batch(location, uncached_date_strs, redis_client)
        for year, date_str in zip(uncached_years, uncached_date_strs):
            weather = batch_results.get(date_str)
            if weather and "error" not in weather:
                try:
                    temp = weather["days"][0]["temp"]
                    logger.debug("Got %d temperature = %s", year, temp)
                    if temp is not None:
                        data.append({"x": year, "y": temp})
                        if CACHE_ENABLED:
                            cache_key = generate_cache_key("weather", location, date_str)
                            target_date = datetime(year, month, day).date()
                            days_diff = (target_date - today.date()).days
                            cache_duration = SHORT_CACHE_DURATION if -7 <= days_diff <= 365 else LONG_CACHE_DURATION
                            set_cache_value(cache_key, cache_duration, json.dumps(weather), redis_client)
                    else:
                        logger.debug("Temperature is None for %d, marking as missing.", year)
                        track_missing_year(missing_years, year, "no_temperature_data")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error("Error processing batch data for %s: %s", date_str, e)
                    track_missing_year(missing_years, year, "data_processing_error")
            else:
                track_missing_year(
                    missing_years,
                    year,
                    weather.get("error", "api_error") if weather else "api_error",
                )

    data_list = sorted(data, key=lambda d: d["x"])

    if data_list:
        latest_year = data_list[-1]["x"]
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
        logger.warning("No valid temperature data found for %s on %d-%d", location, month, day)
        return {
            "data": [],
            "metadata": create_metadata(len(years), 0, [{"year": y, "reason": "no_data"} for y in years]),
            "error": f"Invalid location: {location}",
        }

    if CACHE_ENABLED:
        set_cache_value(
            series_cache_key,
            SHORT_CACHE_DURATION,
            json.dumps(
                {
                    "data": data_list,
                    "metadata": create_metadata(len(years), len(data), missing_years),
                }
            ),
            redis_client,
        )

    return {
        "data": data_list,
        "metadata": create_metadata(len(years), len(data), missing_years),
    }
