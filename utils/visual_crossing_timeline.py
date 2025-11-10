import asyncio
import logging
import os
from datetime import date
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import aiohttp

from constants import VC_BASE_URL
from utils.validation import validate_location_for_ssrf

logger = logging.getLogger(__name__)

from typing import Optional

API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()

_client: Optional[aiohttp.ClientSession] = None
_client_lock = asyncio.Lock()
_sem = asyncio.Semaphore(2)
_RETRY_ATTEMPTS = int(os.getenv("VC_TIMELINE_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.getenv("VC_TIMELINE_RETRY_DELAY", "1.0"))


async def _get_client() -> aiohttp.ClientSession:
    global _client
    if _client is not None and not _client.closed:
        return _client
    async with _client_lock:
        if _client is None or _client.closed:
            timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=90)
            _client = aiohttp.ClientSession(timeout=timeout)
        return _client


async def close_client_session() -> None:
    global _client
    if _client and not _client.closed:
        await _client.close()
    _client = None


def _build_timeline_url(location: str, start: date, end: date) -> str:
    if not API_KEY:
        raise RuntimeError("VISUAL_CROSSING_API_KEY is not configured")
    validated_location = validate_location_for_ssrf(location)
    encoded_location = quote(validated_location, safe="")
    return (
        f"{VC_BASE_URL}/timeline/{encoded_location}/{start.isoformat()}/{end.isoformat()}"
        f"?unitGroup=metric&include=days&elements=datetime,temp,tempmax,tempmin&contentType=json&key={API_KEY}"
    )


async def fetch_timeline_days(location: str, start: date, end: date) -> Tuple[List[Dict], Dict[str, Any]]:
    """Fetch timeline data for a contiguous date range (inclusive)."""
    if start > end:
        raise ValueError("start date must be before end date")

    url = _build_timeline_url(location, start, end)
    redacted_url = url.replace(API_KEY, "[redacted]") if API_KEY else url
    session = await _get_client()
    logger.debug("🌡️ timeline GET %s", redacted_url)
    attempt = 0
    while True:
        attempt += 1
        try:
            async with _sem:
                async with session.get(url, headers={"Accept-Encoding": "gzip"}) as response:
                    if response.status >= 400:
                        text = await response.text()
                        logger.error(
                            "Visual Crossing timeline error: status=%s url=%s body=%s",
                            response.status,
                            redacted_url,
                            text[:200],
                        )
                        raise RuntimeError(f"Visual Crossing timeline error: {response.status}")
                    payload = await response.json()
                    days = payload.get("days") or []
                    if not isinstance(days, list):
                        raise RuntimeError("Unexpected timeline response payload")
                    metadata = {
                        "resolvedAddress": payload.get("resolvedAddress"),
                        "address": payload.get("address"),
                        "timezone": payload.get("timezone"),
                        "tz": payload.get("tz"),
                        "latitude": payload.get("latitude"),
                        "longitude": payload.get("longitude"),
                    }
                    return days, metadata
        except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError) as exc:
            if attempt >= _RETRY_ATTEMPTS:
                logger.error("Timeline fetch failed after %s attempts: %s (%s)", attempt, exc, redacted_url)
                raise
            delay = _RETRY_BASE_DELAY * attempt
            logger.warning(
                "Timeline fetch attempt %s failed for %s – retrying in %.1fs (%s)",
                attempt,
                redacted_url,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

