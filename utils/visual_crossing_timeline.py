"""Thin compatibility shim — delegates to open_meteo_client.

Preserves the public surface (`fetch_timeline_days`, `close_client_session`,
`LocationNotFoundError`) so callers need no changes.
"""

from datetime import date
from typing import Any, Dict, List, Tuple

# Re-export LocationNotFoundError so importers still work unchanged.
from utils.open_meteo_client import (  # noqa: F401
    LocationNotFoundError,
    fetch_timeline_for_location,
)
from utils.open_meteo_client import (
    close_client as _close_client,
)


async def fetch_timeline_days(location: str, start: date, end: date) -> Tuple[List[Dict], Dict[str, Any]]:
    """Fetch temperature data for a contiguous date range (inclusive).

    Returns (days, metadata) where days is a list of
    {datetime, temp, tempmax, tempmin} dicts with temperatures in °C.

    Raises LocationNotFoundError if the location cannot be geocoded.
    """
    return await fetch_timeline_for_location(location, start, end)


async def close_client_session() -> None:
    """Close the shared HTTP session used by the Open-Meteo client."""
    await _close_client()
