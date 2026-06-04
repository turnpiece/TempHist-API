"""Provider-agnostic weather client interface.

Currently backed by Open-Meteo. To swap providers, update the
implementation functions below — callers need no changes.

Module roles:
  utils.weather_types    — shared error/type definitions (no provider deps)
  utils.weather_provider — this file: the common public interface
  utils.open_meteo_client — OM-specific implementation details
"""

from datetime import date
from typing import Any, Dict, List, Tuple

from utils.open_meteo_client import close_client, fetch_timeline_for_location
from utils.weather_types import LocationNotFoundError  # noqa: F401

__all__ = ["LocationNotFoundError", "fetch_timeline_days", "close_client_session"]


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
    return await fetch_timeline_for_location(location, start, end)


async def close_client_session() -> None:
    """Close the shared HTTP session used by the active weather provider."""
    await close_client()
