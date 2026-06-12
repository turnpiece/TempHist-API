"""Open-Meteo weather client — replaces Visual Crossing as the data source.

Open-Meteo is free (no API key), uses ERA5 data, and has no daily record budget.
Two endpoints cover all use cases:
  - archive-api.open-meteo.com/v1/archive  — dates older than ~7 days
  - api.open-meteo.com/v1/forecast         — recent, today, and forecast dates
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp

from utils.weather_types import LocationNotFoundError  # noqa: F401

logger = logging.getLogger(__name__)


# ── Constants (imported lazily to avoid circular import at module load) ────────

_FORECAST_PAST_DAYS = 7  # matches Open-Meteo past_days parameter
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0

# Semaphore(2) at ~285ms/req ≈ 7 req/s — well under OM's 600 req/min free-tier cap.
# Initialized lazily to avoid binding to a closed event loop at import time.
_sem: Optional[asyncio.Semaphore] = None

# ── Shared HTTP session ───────────────────────────────────────────────────────

_client: Optional[aiohttp.ClientSession] = None
_client_lock: Optional[asyncio.Lock] = None


async def _get_client() -> aiohttp.ClientSession:
    global _client, _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    if _client is not None and not _client.closed:
        return _client
    async with _client_lock:
        if _client is None or _client.closed:
            _client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0, connect=10.0))
        return _client


async def close_client() -> None:
    global _client
    if _client and not _client.closed:
        await _client.close()
    _client = None


# ── URL builders ──────────────────────────────────────────────────────────────


def _om_archive_url(lat: float, lon: float, start: date, end: date) -> str:
    from config import OPEN_METEO_ARCHIVE_URL

    return (
        f"{OPEN_METEO_ARCHIVE_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        f"&timezone=auto"
    )


def _om_forecast_url(lat: float, lon: float) -> str:
    from config import OPEN_METEO_FORECAST_URL

    return (
        f"{OPEN_METEO_FORECAST_URL}?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        f"&current=temperature_2m&timezone=auto"
        f"&past_days={_FORECAST_PAST_DAYS}&forecast_days=1"
    )


def _om_requests_for_range(lat: float, lon: float, start: date, end: date) -> List[Tuple[str, date, date]]:
    """Return (url, filter_start, filter_end) tuples covering [start, end].

    Splits at the archive/forecast boundary for windows that straddle it.
    All-archive: end < boundary.
    All-forecast: start >= boundary.
    Straddle: two requests, merged by the caller.
    """
    boundary = date.today() - timedelta(days=_FORECAST_PAST_DAYS)

    if end < boundary:
        return [(_om_archive_url(lat, lon, start, end), start, end)]

    if start >= boundary:
        return [(_om_forecast_url(lat, lon), start, end)]

    # Straddles boundary — split into archive + forecast
    archive_end = boundary - timedelta(days=1)
    return [
        (_om_archive_url(lat, lon, start, archive_end), start, archive_end),
        (_om_forecast_url(lat, lon), boundary, end),
    ]


# ── Response normalisation ────────────────────────────────────────────────────


def _process_om_response(payload: dict, filter_start: date, filter_end: date) -> List[Dict]:
    """Normalise OM's array-based response to [{datetime, temp, tempmax, tempmin}],
    filtered to [filter_start, filter_end] inclusive.
    """
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    means = daily.get("temperature_2m_mean", [])
    maxs = daily.get("temperature_2m_max", [])
    mins = daily.get("temperature_2m_min", [])

    start_s = filter_start.isoformat()
    end_s = filter_end.isoformat()

    days = []
    for t, m, mx, mn in zip(times, means, maxs, mins):
        if not t or not (start_s <= t <= end_s):
            continue
        if m is None and mx is not None and mn is not None:
            m = round((mx + mn) / 2, 1)
        days.append({"datetime": t, "temp": m, "tempmax": mx, "tempmin": mn})
    return days


def _extract_metadata(payload: dict) -> Dict[str, Any]:
    """Extract timezone and coordinates from an OM response payload."""
    return {
        "resolvedAddress": None,  # OM doesn't provide a resolved address string
        "timezone": payload.get("timezone"),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
    }


# ── Geocoding ─────────────────────────────────────────────────────────────────


async def _get_coordinates_from_store(
    location: str,
) -> Optional[Tuple[float, float, Optional[str]]]:
    """Look up coordinates from Postgres locations table or preapproved list."""
    try:
        from utils.daily_temperature_store import get_daily_temperature_store

        store = await get_daily_temperature_store()
        return await store.get_coordinates(location)
    except Exception as exc:
        logger.debug("Coordinate store lookup failed for %r: %s", location, exc)
        return None


async def _geocode_mapbox(
    query: str,
) -> Optional[Tuple[float, float, Optional[str]]]:
    """Geocode a location string via Mapbox, returning (lat, lon, None).

    Returns None if MAPBOX_TOKEN is not configured or the query fails.
    The third element (timezone) is always None — Mapbox doesn't return timezone.
    """
    from config import MAPBOX_TOKEN

    if not MAPBOX_TOKEN:
        return None

    encoded = quote(query.strip(), safe="")
    url = (
        f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded}.json"
        f"?access_token={MAPBOX_TOKEN}&limit=1&types=place,locality,district,region"
    )

    session = await _get_client()
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning("Mapbox geocoding returned status %s for %r", resp.status, query)
                return None
            data = await resp.json()
            features = data.get("features", [])
            if not features:
                return None
            center = features[0].get("center", [])
            if len(center) < 2:
                return None
            lon, lat = center[0], center[1]
            return float(lat), float(lon), None
    except Exception as exc:
        logger.warning("Mapbox geocoding error for %r: %s", query, exc)
        return None


async def geocode_location(location: str) -> Tuple[float, float, Optional[str]]:
    """Resolve a location string to (lat, lon, timezone).

    Resolution order:
    1. Postgres locations table / preapproved_locations.json
    2. Mapbox geocoding API
    3. LocationNotFoundError
    """
    coords = await _get_coordinates_from_store(location)
    if coords is not None:
        return coords

    coords = await _geocode_mapbox(location)
    if coords is not None:
        logger.info("Geocoded %r via Mapbox: lat=%.4f lon=%.4f", location, coords[0], coords[1])
        return coords

    raise LocationNotFoundError(f"Location not found: {location!r}")


# ── Core fetch ────────────────────────────────────────────────────────────────


async def fetch_days(
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Fetch temperature data for a date range from OM archive/forecast APIs.

    Returns (days, metadata) where:
      - days: [{datetime, temp, tempmax, tempmin}] sorted by date, temperatures in °C
      - metadata: {resolvedAddress, timezone, latitude, longitude}

    Handles archive/forecast splitting, retries, and 429 back-off automatically.
    """
    requests = _om_requests_for_range(lat, lon, start, end)
    session = await _get_client()
    all_days: List[Dict] = []
    metadata: Dict[str, Any] = {
        "resolvedAddress": None,
        "latitude": lat,
        "longitude": lon,
        "timezone": None,
    }

    async def fetch_one(url: str, fs: date, fe: date) -> None:
        global _sem
        if _sem is None:
            _sem = asyncio.Semaphore(2)
        attempt = 0
        while True:
            attempt += 1
            try:
                async with _sem:
                    async with session.get(url, headers={"Accept-Encoding": "gzip"}) as resp:
                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", _RETRY_BASE_DELAY * attempt))
                            if attempt < _RETRY_ATTEMPTS:
                                logger.warning(
                                    "OM rate limited, retrying in %.1fs (attempt %d)",
                                    retry_after,
                                    attempt,
                                )
                                await asyncio.sleep(retry_after)
                                continue
                            logger.error("OM rate limit exceeded after %d attempts: %s", attempt, url)
                            return
                        if resp.status != 200:
                            logger.warning("OM returned status %s for %s", resp.status, url)
                            return
                        payload = await resp.json()
                        if payload.get("error"):
                            reason = str(payload.get("reason", "")).lower()
                            logger.warning("OM error for %s: %s", url, payload.get("reason"))
                            if ("rate" in reason or "limit" in reason) and attempt < _RETRY_ATTEMPTS:
                                await asyncio.sleep(_RETRY_BASE_DELAY * attempt)
                                continue
                            return
                        days = _process_om_response(payload, fs, fe)
                        all_days.extend(days)
                        if metadata["timezone"] is None:
                            metadata.update(_extract_metadata(payload))
                        return
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                if attempt >= _RETRY_ATTEMPTS:
                    logger.error("OM fetch failed after %d attempts for %s: %s", attempt, url, exc)
                    return
                delay = _RETRY_BASE_DELAY * attempt
                logger.warning("OM fetch attempt %d failed, retrying in %.1fs: %s", attempt, delay, exc)
                await asyncio.sleep(delay)

    await asyncio.gather(*[fetch_one(url, fs, fe) for url, fs, fe in requests])
    all_days.sort(key=lambda d: d.get("datetime", ""))
    return all_days, metadata


# ── Public API ────────────────────────────────────────────────────────────────


async def fetch_timeline_for_location(
    location: str,
    start: date,
    end: date,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Geocode a location string and fetch temperature data for a date range.

    Returns (days, metadata) matching the legacy fetch_timeline_days contract.
    Raises LocationNotFoundError if the location cannot be resolved.
    """
    if start > end:
        raise ValueError("start date must be before end date")

    lat, lon, timezone_hint = await geocode_location(location)
    logger.debug("🌡️ OM fetch %s → lat=%.4f lon=%.4f [%s – %s]", location, lat, lon, start.isoformat(), end.isoformat())
    days, metadata = await fetch_days(lat, lon, start, end)
    if timezone_hint and not metadata.get("timezone"):
        metadata["timezone"] = timezone_hint

    _cache_location_timezone(location, metadata.get("timezone"))

    return days, metadata


def _cache_location_timezone(location: str, tz: Optional[str]) -> None:
    """Persist a resolved timezone to Redis so non-preapproved locations
    surface a populated `timezone` field in `/v1/records` responses.
    """
    if not tz:
        return
    try:
        from cache.keys import store_location_timezone

        store_location_timezone(location, tz)
    except Exception as exc:
        logger.debug("Could not cache timezone for %r: %s", location, exc)


async def fetch_single_date(location: str, date_str: str) -> Dict:
    """Fetch weather data for a single date.

    Returns {"days": [{datetime, temp, tempmax, tempmin}]} on success,
    or {"error": "..."} if no data is available.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    lat, lon, timezone_hint = await geocode_location(location)
    days, metadata = await fetch_days(lat, lon, target, target)
    if timezone_hint and not metadata.get("timezone"):
        metadata["timezone"] = timezone_hint

    _cache_location_timezone(location, metadata.get("timezone"))

    day = next((d for d in days if d.get("datetime") == date_str), None)
    if day is None:
        return {"error": "No temperature data available"}
    return {"days": [day]}
