"""URL path helpers for middleware location extraction."""

from __future__ import annotations

from typing import Optional, Tuple
from urllib.parse import unquote


def extract_location_from_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (location, endpoint) from a billable API path, or (None, None).

    Supported shapes:
      /weather/{location}/{date}
      /forecast/{location}
      /v1/records/{period}/{location}/{identifier}[...]
    """
    parts = path.split("/")

    if path.startswith("/weather/") and len(parts) >= 4:
        return unquote(parts[2]), "weather"

    if path.startswith("/forecast/") and len(parts) >= 3:
        return unquote(parts[2]), "forecast"

    if path.startswith("/v1/records/") and len(parts) >= 6:
        return unquote(parts[4]), parts[3]

    return None, None
