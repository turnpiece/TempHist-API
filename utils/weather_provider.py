"""Provider-agnostic weather client interface.

Dispatches to the configured weather backend based on the WEATHER_PROVIDER
environment variable:

  WEATHER_PROVIDER=open_meteo        (default) — Open-Meteo, free, no API key
  WEATHER_PROVIDER=visual_crossing   — Visual Crossing, requires VISUAL_CROSSING_API_KEY

Module roles:
  utils.weather_types         — shared error/type definitions (no provider deps)
  utils.weather_provider      — this file: the common public interface
  utils.open_meteo_client     — OM-specific implementation
  utils.visual_crossing_client — VC-specific implementation
"""

import logging
from datetime import date
from typing import Any, Dict, List, Tuple

from utils.weather_types import LocationNotFoundError  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["LocationNotFoundError", "fetch_timeline_days", "fetch_single_date", "close_client_session"]


def _get_provider():
    """Return the active provider module, logging once on first selection."""
    from config import WEATHER_PROVIDER

    if WEATHER_PROVIDER == "visual_crossing":
        import utils.visual_crossing_client as _mod
    else:
        if WEATHER_PROVIDER != "open_meteo":
            logger.warning("Unknown WEATHER_PROVIDER=%r — falling back to open_meteo", WEATHER_PROVIDER)
        import utils.open_meteo_client as _mod  # type: ignore[no-redef]

    return _mod


async def fetch_timeline_days(
    location: str,
    start: date,
    end: date,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Fetch temperature data for a contiguous date range (inclusive).

    Returns (days, metadata) where days is [{datetime, temp, tempmax, tempmin}]
    with temperatures in °C. Raises LocationNotFoundError if the location
    cannot be geocoded.
    """
    return await _get_provider().fetch_timeline_for_location(location, start, end)


async def fetch_single_date(location: str, date_str: str) -> Dict:
    """Fetch weather data for a single date.

    Returns {"days": [{datetime, temp, tempmax, tempmin}]} on success,
    or {"error": "..."} if no data is available.
    """
    return await _get_provider().fetch_single_date(location, date_str)


async def close_client_session() -> None:
    """Close the shared HTTP session used by the active weather provider."""
    await _get_provider().close_client()
