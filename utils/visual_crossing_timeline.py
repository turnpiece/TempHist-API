import asyncio
import logging
import os
from datetime import date
from typing import Dict, List
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


async def fetch_timeline_days(location: str, start: date, end: date) -> List[Dict]:
    """Fetch timeline data for a contiguous date range (inclusive)."""
    if start > end:
        raise ValueError("start date must be before end date")

    url = _build_timeline_url(location, start, end)
    session = await _get_client()
    logger.debug("🌡️ timeline GET %s", url)
    async with _sem:
        async with session.get(url, headers={"Accept-Encoding": "gzip"}) as response:
            if response.status >= 400:
                text = await response.text()
                logger.error("Visual Crossing timeline error: %s %s", response.status, text[:200])
                raise RuntimeError(f"Visual Crossing timeline error: {response.status}")
            payload = await response.json()
            days = payload.get("days") or []
            if not isinstance(days, list):
                raise RuntimeError("Unexpected timeline response payload")
            return days

