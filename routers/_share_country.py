"""Work out which country a share is in, so clients can show its flag.

Resolution order for a share (see resolve_share_country_code):
  1. a country code the client supplied (validated),
  2. a preapproved location (by ID, or by name + country),
  3. the country at the end of a "City, Region, Country" string,
  4. the geocoder, for bare names such as "Madrid".
"""

import logging
import math
from typing import List, Optional

import pycountry

import routers.locations as locations_module
from routers.locations import COUNTRY_CODE_ALIASES, _geocode_open_meteo

logger = logging.getLogger(__name__)

# Abbreviations people write after a city that aren't a country's name or ISO code ("London, UK")
_COUNTRY_TEXT_ALIASES = {"uk": "GB", "usa": "US", "uae": "AE"}

# A geocoder hit this close to the share's coordinates is taken to be the same place.
_MAX_GEOCODE_DISTANCE_KM = 100.0


def normalise_country_code(code: str) -> str:
    """Return the ISO 3166-1 alpha-2 code for `code`, or raise ValueError.

    Accepts the aliases the rest of the API accepts (UK -> GB). The "EU" grouping is not a country.
    """
    upper = code.strip().upper()
    upper = COUNTRY_CODE_ALIASES.get(upper, upper)
    if len(upper) != 2 or pycountry.countries.get(alpha_2=upper) is None:
        raise ValueError(f"Unknown country code '{code}'. Use a 2-letter ISO 3166-1 code such as 'GB' or 'NO'.")
    return upper


def _country_code_from_preapproved(location: str) -> Optional[str]:
    """Country of a preapproved location, matched by ID or by name (and country name when the text has one)."""
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if not parts:
        return None
    first = parts[0].lower()
    country_part = parts[-1].lower() if len(parts) > 1 else None
    by_name = None
    # Read through the module: initialize_locations_data() rebinds locations_data at startup
    for loc in locations_module.locations_data:
        if len(parts) == 1 and loc.id == first:
            return loc.country_code
        if loc.name.lower() != first:
            continue
        if country_part and loc.country_name.lower() == country_part:
            return loc.country_code
        # A bare name ("London") is taken as the preapproved place, as the web app already does
        if country_part is None:
            by_name = loc.country_code
    return by_name


def _country_code_from_text(location: str) -> Optional[str]:
    """Country named at the end of "City, Region, Country", or None.

    Exact name lookup only (no fuzzy search, no ISO codes) plus a few well-known abbreviations
    (UK, USA, UAE): "Finnmark" or "IL" at the end of a string must not be mistaken for a country.
    """
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    alias = _COUNTRY_TEXT_ALIASES.get(parts[-1].lower())
    if alias:
        return alias
    if len(parts[-1]) <= 3:
        return None
    try:
        return pycountry.countries.lookup(parts[-1]).alpha_2
    except LookupError:
        return None


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _pick_geocoder_country(
    city: str, places: List[dict], latitude: Optional[float], longitude: Optional[float]
) -> Optional[str]:
    """Choose a country from geocoder hits for `city`, or None when the answer would be a guess.

    With coordinates: the nearest hit within _MAX_GEOCODE_DISTANCE_KM. Without: only when every
    exact-name hit is in the same country (so "Madrid" is Spain but "Birmingham" is left alone).
    """
    named = [p for p in places if (p.get("name") or "").strip().lower() == city.lower() and p.get("country_code")]
    if latitude is not None and longitude is not None:
        nearby = [
            (_distance_km(latitude, longitude, p["latitude"], p["longitude"]), p["country_code"])
            for p in named
            if p.get("latitude") is not None and p.get("longitude") is not None
        ]
        nearby = [n for n in nearby if n[0] <= _MAX_GEOCODE_DISTANCE_KM]
        return min(nearby)[1] if nearby else None
    countries = {p["country_code"] for p in named}
    return countries.pop() if len(countries) == 1 else None


async def _country_code_from_geocoder(
    location: str, latitude: Optional[float], longitude: Optional[float]
) -> Optional[str]:
    city = location.split(",")[0].strip()
    if not city:
        return None
    try:
        places = await _geocode_open_meteo(city, limit=10)
    except Exception as exc:  # the geocoder is best-effort; never fail a share over it
        logger.warning("Share country geocode failed: %s", type(exc).__name__)
        return None
    return _pick_geocoder_country(city, places, latitude, longitude)


async def resolve_share_country_code(
    location: str,
    client_code: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Optional[str]:
    """Return the share's ISO 3166-1 alpha-2 country code, or None if it can't be determined confidently.

    `client_code` must already be normalised (see normalise_country_code).
    """
    if client_code:
        return client_code
    return (
        _country_code_from_preapproved(location)
        or _country_code_from_text(location)
        or await _country_code_from_geocoder(location, latitude, longitude)
    )
