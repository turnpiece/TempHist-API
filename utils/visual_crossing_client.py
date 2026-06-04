"""Visual Crossing weather client.

Alternative to Open-Meteo — enabled by setting WEATHER_PROVIDER=visual_crossing.
Requires VISUAL_CROSSING_API_KEY to be set.

Exposes the same public interface as open_meteo_client so weather_provider.py
can dispatch to either without callers knowing which is active:
  - fetch_timeline_for_location(location, start, end) → (days, metadata)
  - fetch_single_date(location, date_str) → {"days": [...]} | {"error": "..."}
  - close_client()
"""

import asyncio
import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp

from utils.validation import validate_location_for_ssrf
from utils.weather_types import LocationNotFoundError  # noqa: F401

logger = logging.getLogger(__name__)

VC_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services"

# Configuration from environment
_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
_VC_TIMEOUT = float(os.getenv("HTTP_TIMEOUT_VISUAL_CROSSING", "30.0"))
_RETRY_ATTEMPTS = int(os.getenv("VC_TIMELINE_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.getenv("VC_TIMELINE_RETRY_DELAY", "1.0"))

# Concurrency limit — match OM's semaphore so switching doesn't change behaviour
_sem = asyncio.Semaphore(2)

# ── Shared HTTP session ───────────────────────────────────────────────────────

_client: Optional[aiohttp.ClientSession] = None
_client_lock = asyncio.Lock()


async def _get_client() -> aiohttp.ClientSession:
    global _client
    if _client is not None and not _client.closed:
        return _client
    async with _client_lock:
        if _client is None or _client.closed:
            connect_timeout = min(10.0, _VC_TIMEOUT / 3)
            read_timeout = min(_VC_TIMEOUT - 5, _VC_TIMEOUT * 0.8)
            _client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=_VC_TIMEOUT,
                    connect=connect_timeout,
                    sock_read=read_timeout,
                )
            )
        return _client


async def close_client() -> None:
    global _client
    if _client and not _client.closed:
        await _client.close()
    _client = None


# ── URL builder ───────────────────────────────────────────────────────────────


def _build_timeline_url(location: str, start: date, end: date) -> str:
    api_key = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("VISUAL_CROSSING_API_KEY is not set — required when WEATHER_PROVIDER=visual_crossing")
    validated = validate_location_for_ssrf(location)
    encoded = quote(validated, safe="")
    return (
        f"{VC_BASE_URL}/timeline/{encoded}/{start.isoformat()}/{end.isoformat()}"
        f"?unitGroup=us&include=days&elements=datetime,temp,tempmax,tempmin"
        f"&contentType=json&key={api_key}"
    )


# ── Unit conversion ───────────────────────────────────────────────────────────


def _f_to_c(value) -> Optional[float]:
    if value is None:
        return None
    return round((float(value) - 32) * 5 / 9, 2)


def _convert_days_to_celsius(days: list) -> list:
    result = []
    for day in days:
        d = dict(day)
        for field in ("temp", "tempmin", "tempmax"):
            if field in d:
                d[field] = _f_to_c(d[field])
        result.append(d)
    return result


# ── Core fetch ────────────────────────────────────────────────────────────────


async def fetch_timeline_for_location(
    location: str,
    start: date,
    end: date,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Fetch temperature data for a contiguous date range (inclusive).

    Returns (days, metadata) where days is [{datetime, temp, tempmax, tempmin}]
    with temperatures in °C. Raises LocationNotFoundError if the location
    cannot be resolved by Visual Crossing.
    """
    if start > end:
        raise ValueError("start date must be before end date")

    url = _build_timeline_url(location, start, end)
    api_key = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
    redacted_url = url.replace(api_key, "[redacted]") if api_key else url
    session = await _get_client()
    logger.debug("🌡️ VC timeline GET %s", redacted_url)

    attempt = 0
    while True:
        attempt += 1
        try:
            async with _sem:
                async with session.get(url, headers={"Accept-Encoding": "gzip"}) as response:
                    if response.status >= 400:
                        text = await response.text()
                        logger.error(
                            "VC timeline error: status=%s url=%s body=%s",
                            response.status,
                            redacted_url,
                            text[:200],
                        )
                        if response.status == 400:
                            # VC returns 400 when the location string is unresolvable.
                            # Retrying won't help — raise immediately.
                            raise LocationNotFoundError(f"Location not found: {text[:200]}")
                        raise RuntimeError(f"Visual Crossing error: {response.status}")
                    payload = await response.json()
                    days = payload.get("days") or []
                    if not isinstance(days, list):
                        raise RuntimeError("Unexpected VC response payload")
                    metadata = {
                        "resolvedAddress": payload.get("resolvedAddress"),
                        "address": payload.get("address"),
                        "timezone": payload.get("timezone"),
                        "latitude": payload.get("latitude"),
                        "longitude": payload.get("longitude"),
                    }
                    return _convert_days_to_celsius(days), metadata

        except LocationNotFoundError:
            raise  # never retry — the location string itself is wrong
        except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError) as exc:
            if attempt >= _RETRY_ATTEMPTS:
                logger.error(
                    "VC timeline fetch failed after %d attempts: %s (%s)",
                    attempt,
                    exc,
                    redacted_url,
                )
                raise
            delay = _RETRY_BASE_DELAY * attempt
            logger.warning(
                "VC timeline attempt %d failed for %s — retrying in %.1fs (%s)",
                attempt,
                redacted_url,
                delay,
                exc,
            )
            await asyncio.sleep(delay)


# ── Public single-date helper (mirrors open_meteo_client) ────────────────────


async def fetch_single_date(location: str, date_str: str) -> Dict:
    """Fetch weather data for a single date.

    Returns {"days": [{datetime, temp, tempmax, tempmin}]} on success,
    or {"error": "..."} if no data is available.
    """
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    days, _ = await fetch_timeline_for_location(location, target, target)
    day = next((d for d in days if d.get("datetime") == date_str), None)
    if day is None:
        return {"error": "No temperature data available"}
    return {"days": [day]}
