"""Work out which country a share is in, so clients can show its flag.

Resolution order for a share (see resolve_share_country):
  1. a country code the client supplied (validated),
  2. a preapproved location (by ID, or by name + country),
  3. the country at the end of a "City, Region, Country" string,
  4. Open-Meteo's geocoder, for bare names such as "Madrid",
  5. Mapbox's top exact-name match, to break ties the previous step won't guess at
     ("Kyoto" also exists in Tanzania; Mapbox ranks the Japanese one first).
"""

import logging
import math
from typing import List, NamedTuple, Optional

import pycountry

import routers.locations as locations_module
from routers.locations import COUNTRY_CODE_ALIASES, _fold_name, _geocode_mapbox, _geocode_open_meteo

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


def _explain_geocoder_miss(
    city: str, places: List[dict], latitude: Optional[float], longitude: Optional[float]
) -> str:
    """Why _pick_geocoder_country found nothing, for the backfill's report."""
    named = [p for p in places if (p.get("name") or "").strip().lower() == city.lower() and p.get("country_code")]
    if not named:
        return "Open-Meteo has no exact-name match"
    if latitude is not None and longitude is not None:
        return f"no Open-Meteo match within {_MAX_GEOCODE_DISTANCE_KM:.0f} km of the share's coordinates"
    countries = ", ".join(sorted({p["country_code"] for p in named}))
    return f"ambiguous without coordinates (Open-Meteo matches in {countries})"


def _pick_mapbox_country(
    city: str, places: List[dict], latitude: Optional[float], longitude: Optional[float]
) -> tuple:
    """Return (country_code, why_not) from Mapbox results, which arrive ranked by relevance.

    Takes the top result whose name is exactly `city` (ignoring case and accents). With the
    share's coordinates, that result must also be within _MAX_GEOCODE_DISTANCE_KM of them,
    otherwise it's a different place with the same name.
    """
    if not places:
        return None, "Mapbox returned nothing"
    folded = _fold_name(city)
    for place in places:
        if not place.get("country_code") or _fold_name(place.get("name") or "") != folded:
            continue
        has_coordinates = place.get("latitude") is not None and place.get("longitude") is not None
        if latitude is not None and longitude is not None and has_coordinates:
            if _distance_km(latitude, longitude, place["latitude"], place["longitude"]) > _MAX_GEOCODE_DISTANCE_KM:
                return None, f"Mapbox's match is more than {_MAX_GEOCODE_DISTANCE_KM:.0f} km from the share's coordinates"
        return place["country_code"], ""
    return None, f"Mapbox has no exact-name match (top result: {places[0].get('name')})"


class CountryResolution(NamedTuple):
    """A share's country, or None. `detail` is where the code came from, or why there isn't one."""

    code: Optional[str]
    detail: str


async def _resolve_with_geocoders(
    location: str, latitude: Optional[float], longitude: Optional[float]
) -> CountryResolution:
    city = location.split(",")[0].strip()
    if not city:
        return CountryResolution(None, "no place name in the location")

    notes: List[str] = []
    try:
        places = await _geocode_open_meteo(city, limit=10)
    except Exception as exc:  # the geocoders are best-effort; never fail a share over them
        logger.warning("Share country geocode failed: %s", type(exc).__name__)
        notes.append("Open-Meteo lookup failed")
    else:
        code = _pick_geocoder_country(city, places, latitude, longitude)
        if code:
            return CountryResolution(code, "Open-Meteo geocoder")
        notes.append(_explain_geocoder_miss(city, places, latitude, longitude))

    try:
        mapbox_places = await _geocode_mapbox(city, limit=5)
    except Exception as exc:
        logger.warning("Share country Mapbox geocode failed: %s", type(exc).__name__)
        notes.append("Mapbox lookup failed")
    else:
        code, why_not = _pick_mapbox_country(city, mapbox_places, latitude, longitude)
        if code:
            return CountryResolution(code, "Mapbox top match")
        notes.append(why_not)
    return CountryResolution(None, "; ".join(notes))


async def resolve_share_country(
    location: str,
    client_code: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> CountryResolution:
    """Work out the share's ISO 3166-1 alpha-2 country code, or None if it can't be determined confidently.

    `client_code` must already be normalised (see normalise_country_code).
    """
    if client_code:
        return CountryResolution(client_code, "supplied by the client")
    code = _country_code_from_preapproved(location)
    if code:
        return CountryResolution(code, "preapproved location")
    code = _country_code_from_text(location)
    if code:
        return CountryResolution(code, "country named in the location")
    return await _resolve_with_geocoders(location, latitude, longitude)


async def resolve_share_country_code(
    location: str,
    client_code: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Optional[str]:
    """Like resolve_share_country, but just the code."""
    return (await resolve_share_country(location, client_code, latitude, longitude)).code
