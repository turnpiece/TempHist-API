"""
Preapproved locations endpoint router.

Provides access to a curated list of preapproved locations with filtering,
caching, and rate limiting capabilities.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Tuple, Union
from urllib.parse import quote

import aiohttp
import anyio
import pycountry
import redis
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from cache.accessors import get_usage_tracker
from config import BASE_URL, MAPBOX_TOKEN, POPULARITY_MAX_LOCATIONS, POPULARITY_WINDOW_DAYS

# Configure logging
logger = logging.getLogger(__name__)

if not MAPBOX_TOKEN:
    logger.warning("MAPBOX_TOKEN is not set — location search will only cover the preapproved list")

# Constants
CACHE_TTL = 604800  # 7 days (data changes infrequently)
POPULAR_CACHE_TTL = 3600  # 1 hour — rebuilt from live selection signal
RATE_LIMIT_REQUESTS = 60  # requests per minute
RATE_LIMIT_WINDOW = 60  # 1 minute window
CACHE_PREFIX = "preapproved:v1"
POPULAR_CACHE_PREFIX = "popular:v1"
MAX_LIMIT = 500

# Country code aliases: non-ISO codes that users commonly try
COUNTRY_CODE_ALIASES: Dict[str, str] = {
    "UK": "GB",  # UK is colloquial; the ISO 3166-1 code for the United Kingdom is GB
}

# EU member state codes for the "EU" convenience grouping
EU_MEMBER_CODES: FrozenSet[str] = frozenset(
    [
        "AT",
        "BE",
        "BG",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "HU",
        "IE",
        "IT",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
    ]
)


# Pydantic Models
class ImageUrl(BaseModel):
    """Image URL with format options."""

    webp: str = Field(..., description="WebP format image URL")
    jpeg: str = Field(..., description="JPEG format image URL")


class ImageAttribution(BaseModel):
    """Image attribution details."""

    title: Optional[str] = Field(None, description="Title or description of the image")
    photographerName: Optional[str] = Field(None, description="Name of the photographer or image creator")
    sourceName: Optional[str] = Field(None, description="Name of the source (e.g., Wikimedia Commons, Pexels)")
    sourceUrl: Optional[str] = Field(None, description="URL to the original source or license information")
    licenseName: Optional[str] = Field(None, description="Name of the license (e.g., CC BY-SA 4.0, Pexels License)")
    licenseUrl: Optional[str] = Field(None, description="URL to the full license text")
    attributionRequired: Optional[bool] = Field(None, description="Whether attribution is required by the license")


class LocationItem(BaseModel):
    """Individual location item."""

    id: str = Field(..., description="Unique location identifier")
    slug: str = Field(..., description="URL-friendly location slug")
    name: str = Field(..., description="Human-readable location name")
    admin1: str = Field(..., description="First-level administrative division")
    country_name: str = Field(..., description="Full country name")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    timezone: str = Field(..., description="IANA timezone identifier")
    tier: str = Field(..., description="Location tier classification")
    imageUrl: ImageUrl = Field(..., description="Location image URLs")
    imageAlt: str = Field(..., description="Alt text for location image")
    imageAttribution: Optional[ImageAttribution] = Field(None, description="Location image attribution")

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v):
        """Validate country code format (ISO 3166-1 alpha-2)."""
        if not re.match(r"^[A-Z]{2}$", v):
            raise ValueError("Country code must be a 2-letter ISO 3166-1 alpha-2 code")
        return v


class PreapprovedResponse(BaseModel):
    """Response model for preapproved locations endpoint."""

    version: int = Field(default=1, description="API version")
    count: int = Field(..., description="Number of locations returned")
    generated_at: datetime = Field(..., description="Response generation timestamp")
    locations: List[LocationItem] = Field(..., description="List of location items")


class PopularResponse(BaseModel):
    """Response model for popular locations endpoint."""

    version: int = Field(default=1, description="API version")
    count: int = Field(..., description="Number of locations returned")
    generated_at: datetime = Field(..., description="Response generation timestamp")
    locations: List[LocationItem] = Field(..., description="List of location items")


class SelectionRequest(BaseModel):
    """Request body for recording a location selection.

    Supply either ``location_id`` (canonical ID from the API) or the human
    fields ``name`` + optionally ``admin1`` / ``country_code``.  At least one
    of ``location_id`` or ``name`` is required.
    """

    location_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Canonical location ID returned by the API (id field on location objects)",
    )
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="City or place name",
    )
    admin1: Optional[str] = Field(
        None,
        max_length=200,
        description="First-level administrative division (state, county, region, etc.)",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    country_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Full country name (e.g. 'India'). Takes precedence over country_code lookup.",
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90,
        le=90,
        description="Latitude of the selected location, if known (e.g. from geocoding search results).",
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180,
        le=180,
        description="Longitude of the selected location, if known (e.g. from geocoding search results).",
    )

    @model_validator(mode="after")
    def require_location_or_name(self) -> "SelectionRequest":
        if not self.location_id and not self.name:
            raise ValueError("Provide either 'location_id' or 'name' (with optional admin1 / country_code)")
        return self


# Global state
locations_data: List[LocationItem] = []
locations_etag: str = ""
locations_last_modified: str = ""
redis_client: Optional[redis.Redis] = None

# Rate limiting state
rate_limit_requests: Dict[str, List[float]] = defaultdict(list)
rate_limit_lock = asyncio.Lock()

# Router
router = APIRouter()


def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis client not initialized")
    return redis_client


def validate_country_code(country_code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a country code input. Returns (is_valid, error_message).

    Accepts ISO 3166-1 alpha-2 codes, known aliases (e.g. UK→GB), and the
    special grouping 'EU'. Input is expected to already be uppercased.
    """
    if country_code == "EU":
        return True, None
    if country_code in COUNTRY_CODE_ALIASES:
        return True, None
    if not re.match(r"^[A-Z]{2}$", country_code):
        return (
            False,
            f"Invalid country code '{country_code}'. Use a 2-letter ISO 3166-1 code (e.g. 'GB', 'US'), 'EU' for all EU members, or 'UK' as an alias for 'GB'.",
        )
    if pycountry.countries.get(alpha_2=country_code) is None:
        suggestion = COUNTRY_CODE_ALIASES.get(country_code)
        msg = f"Unknown country code '{country_code}'."
        if suggestion:
            msg += f" Did you mean '{suggestion}'?"
        return False, msg
    return True, None


def resolve_country_code(country_code: str) -> Union[str, FrozenSet[str]]:
    """Resolve alias or EU grouping. Returns a single ISO code or a frozenset of codes."""
    if country_code == "EU":
        return EU_MEMBER_CODES
    return COUNTRY_CODE_ALIASES.get(country_code, country_code)


def validate_limit(limit: int) -> bool:
    """Validate limit parameter."""
    return 1 <= limit <= MAX_LIMIT


def generate_etag(data: str) -> str:
    """Generate ETag from data content using SHA256 (32 chars for 128-bit security)."""
    return f'"{hashlib.sha256(data.encode()).hexdigest()[:32]}"'


def get_cache_key(country_code: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Generate cache key for filtered data."""
    if country_code and tier:
        return f"{CACHE_PREFIX}:country:{country_code}:tier:{tier}"
    elif country_code:
        return f"{CACHE_PREFIX}:country:{country_code}"
    elif tier:
        return f"{CACHE_PREFIX}:tier:{tier}"
    else:
        return f"{CACHE_PREFIX}:all"


def get_popular_cache_key(country_code: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Generate cache key for popular locations filtered data."""
    if country_code and tier:
        return f"{POPULAR_CACHE_PREFIX}:country:{country_code}:tier:{tier}"
    elif country_code:
        return f"{POPULAR_CACHE_PREFIX}:country:{country_code}"
    elif tier:
        return f"{POPULAR_CACHE_PREFIX}:tier:{tier}"
    else:
        return f"{POPULAR_CACHE_PREFIX}:all"


async def check_rate_limit(ip: str) -> Tuple[bool, str]:
    """Check if IP is within rate limits."""
    async with rate_limit_lock:
        current_time = time.time()
        window_start = current_time - RATE_LIMIT_WINDOW

        # Clean old requests outside the window
        rate_limit_requests[ip] = [req_time for req_time in rate_limit_requests[ip] if req_time > window_start]

        # Check if limit exceeded
        if len(rate_limit_requests[ip]) >= RATE_LIMIT_REQUESTS:
            return False, f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per minute"

        # Add current request
        rate_limit_requests[ip].append(current_time)
        return True, "OK"


def filter_locations(
    locations: List[LocationItem],
    country_code: Optional[Union[str, FrozenSet[str]]] = None,
    tier: Optional[str] = None,
    limit: Optional[int] = None,
    preserve_order: bool = False,
) -> List[LocationItem]:
    """Filter locations based on criteria."""
    filtered = locations

    if country_code:
        if isinstance(country_code, frozenset):
            filtered = [loc for loc in filtered if loc.country_code in country_code]
        else:
            filtered = [loc for loc in filtered if loc.country_code == country_code]

    if tier:
        filtered = [loc for loc in filtered if loc.tier == tier]

    if not preserve_order:
        # Sort by name for deterministic output (preapproved endpoint)
        filtered = sorted(filtered, key=lambda x: x.name)

    if limit:
        filtered = filtered[:limit]

    return filtered


_IMAGE_FIELDS = frozenset({"imageUrl", "imageAlt", "imageAttribution"})


def _loc_to_dict(loc: LocationItem, include_images: bool) -> dict:
    """Serialise a LocationItem, optionally stripping image fields."""
    d = loc.model_dump()
    if not include_images:
        for field in _IMAGE_FIELDS:
            d.pop(field, None)
    return d


def _find_preapproved_id(name: str, country_code: str) -> Optional[str]:
    """Return the canonical location ID if name+country_code matches a preapproved location."""
    target = (name.lower(), country_code.upper())
    for loc in locations_data:
        if (loc.name.lower(), loc.country_code) == target:
            return loc.id
    return None


def _resolve_country_fields(country_code: Optional[str], country_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a (country_code, country_name) pair, filling in whichever is missing.

    - If both are supplied, they're returned as-is (normalised).
    - If only country_code is supplied, country_name is looked up via pycountry.
    - If only country_name is supplied, country_code is looked up via a fuzzy
      pycountry search (best match by relevance), so clients that only know
      the country's display name (e.g. from Mapbox) still get a usable code
      for things like flag rendering.
    """
    code = country_code.upper() if country_code else None
    name = country_name or None

    if code and not name:
        country = pycountry.countries.get(alpha_2=code)
        if country:
            name = country.name

    if name and not code:
        try:
            matches = pycountry.countries.search_fuzzy(name)
            if matches:
                code = matches[0].alpha_2
        except LookupError:
            pass

    return code, name


def _resolve_canonical_id(body: "SelectionRequest") -> str:
    """Resolve a SelectionRequest to a canonical location ID.

    Resolution order:
    1. ``location_id`` if explicitly supplied.
    2. Preapproved list match on name + country_code (so submissions without
       a location_id still accumulate signal toward the popular ranking when
       the location is in the preapproved list).
    3. Slug generated from name — lowercase, non-alphanumeric runs replaced
       by underscores.  Allows tracking any location even if not preapproved.
    """
    if body.location_id:
        return body.location_id

    name = body.name or ""

    if body.country_code:
        preapproved_id = _find_preapproved_id(name, body.country_code)
        if preapproved_id:
            return preapproved_id

    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def convert_image_urls(image_urls: Dict[str, str]) -> Dict[str, str]:
    """Convert relative image URLs to full URLs."""
    return {format_type: f"{BASE_URL}{url}" if url.startswith("/") else url for format_type, url in image_urls.items()}


async def load_locations_data() -> Tuple[List[LocationItem], str, str]:
    """Load and validate locations data from JSON file."""
    # Find project root by looking for pyproject.toml
    project_root = anyio.Path(__file__).parent
    while project_root != project_root.parent:  # Stop at filesystem root
        if await (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    data_file = project_root / "data" / "preapproved_locations.json"

    try:
        raw_data = await data_file.read_text(encoding="utf-8")

        # Parse JSON
        json_data = json.loads(raw_data)

        # Transform relative URLs to full URLs
        for item in json_data:
            if "imageUrl" in item:
                item["imageUrl"] = convert_image_urls(item["imageUrl"])

        # Validate against schema
        locations = [LocationItem(**item) for item in json_data]

        # Generate ETag and Last-Modified
        etag = generate_etag(raw_data)
        file_stat = await data_file.stat()
        last_modified = datetime.fromtimestamp(file_stat.st_mtime).strftime("%a, %d %b %Y %H:%M:%S GMT")

        logger.info(f"Loaded {len(locations)} preapproved locations from {data_file}")
        return locations, etag, last_modified

    except FileNotFoundError:
        logger.error(f"Locations data file not found: {data_file}")
        raise HTTPException(status_code=500, detail="Locations data not available")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in locations data file: {e}")
        raise HTTPException(status_code=500, detail="Invalid locations data format")
    except Exception as e:
        logger.error(f"Error loading locations data: {e}")
        raise HTTPException(status_code=500, detail="Error loading locations data")


async def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response from Redis."""
    try:
        redis = get_redis_client()
        cached = redis.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    return None


async def set_cached_response(cache_key: str, data: Dict, ttl: int = CACHE_TTL) -> None:
    """Cache response in Redis."""
    try:
        redis = get_redis_client()
        redis.setex(cache_key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.warning(f"Cache set error: {e}")


async def warm_cache() -> None:
    """Warm the cache with all locations data."""
    try:
        response_data = {
            "version": 1,
            "count": len(locations_data),
            "generated_at": datetime.now().isoformat(),
            "locations": [loc.model_dump() for loc in locations_data],
        }
        await set_cached_response(get_cache_key(), response_data)
        logger.info("Warmed preapproved locations cache")
    except Exception as e:
        logger.error(f"Error warming cache: {e}")


# ---------------------------------------------------------------------------
# Mapbox geocoding
# ---------------------------------------------------------------------------
MAPBOX_GEOCODE_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places"
SEARCH_CACHE_TTL = 86400  # 24 hours for geocoding results

_mapbox_client: Optional[aiohttp.ClientSession] = None
_mapbox_client_lock = asyncio.Lock()


async def _get_mapbox_client() -> aiohttp.ClientSession:
    """Return (or lazily create) a shared aiohttp session for Mapbox calls."""
    global _mapbox_client
    if _mapbox_client is not None and not _mapbox_client.closed:
        return _mapbox_client
    async with _mapbox_client_lock:
        if _mapbox_client is None or _mapbox_client.closed:
            timeout = aiohttp.ClientTimeout(total=10.0, connect=5.0)
            _mapbox_client = aiohttp.ClientSession(timeout=timeout)
        return _mapbox_client


async def _geocode_mapbox(query: str, limit: int = 10) -> List[Dict]:
    """
    Call the Mapbox Geocoding v5 API and return a list of dicts with keys:
      name, admin1, country_name, country_code
    Results are cached in Redis for SEARCH_CACHE_TTL seconds.
    """
    cache_key = f"geocode:mapbox:v1:{query.strip().lower()}:{limit}"
    cached = await get_cached_response(cache_key)
    if cached is not None:
        return cached.get("results", [])

    encoded = quote(query.strip(), safe="")
    url = f"{MAPBOX_GEOCODE_URL}/{encoded}.json?types=place&language=en&limit={limit}&access_token={MAPBOX_TOKEN}"

    session = await _get_mapbox_client()
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning("Mapbox returned status %s for query %r", resp.status, query)
                return []
            data = await resp.json()
    except Exception as exc:
        logger.warning("Mapbox request failed: %s", exc)
        return []

    results: List[Dict] = []
    for feature in data.get("features", []):
        name = feature.get("text", "").strip()
        if not name:
            continue
        admin1 = ""
        country_name = ""
        country_code = ""
        for ctx in feature.get("context", []):
            ctx_id = ctx.get("id", "")
            if ctx_id.startswith("region."):
                admin1 = ctx.get("text", "").strip()
            elif ctx_id.startswith("country."):
                country_name = ctx.get("text", "").strip()
                sc = ctx.get("short_code", "")
                country_code = sc[:2].upper() if sc else ""
        if not country_name:
            # Top-level feature may itself be a country-level result; skip it
            continue
        results.append(
            {
                "name": name,
                "admin1": admin1,
                "country_name": country_name,
                "country_code": country_code,
            }
        )

    await set_cached_response(cache_key, {"results": results}, ttl=SEARCH_CACHE_TTL)
    return results


# ---------------------------------------------------------------------------
# Preapproved locations endpoint
# ---------------------------------------------------------------------------


@router.get("/v1/locations/preapproved", response_model=PreapprovedResponse)
async def get_preapproved_locations(
    request: Request,
    response: Response,
    country_code: Optional[str] = Query(None, description="Filter by ISO 3166-1 alpha-2 country code"),
    tier: Optional[str] = Query(None, description="Filter by location tier"),
    limit: Optional[int] = Query(None, ge=1, le=MAX_LIMIT, description=f"Limit results (max {MAX_LIMIT})"),
):
    """
    Get preapproved locations with optional filtering.

    Returns a curated list of preapproved locations that can be used with the weather API.
    Supports filtering by country code and tier, with optional result limiting.
    """
    # Rate limiting
    client_ip = request.client.host
    allowed, reason = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    # Normalise, validate, and resolve country code
    resolved_country = None
    cache_country_key = None
    if country_code:
        country_code = country_code.upper()
        valid, error_msg = validate_country_code(country_code)
        if not valid:
            raise HTTPException(status_code=400, detail=error_msg)
        resolved_country = resolve_country_code(country_code)
        # Aliases (e.g. UK→GB) share the canonical code's cache entry; EU keeps its own
        cache_country_key = country_code if isinstance(resolved_country, frozenset) else resolved_country

    if limit and not validate_limit(limit):
        raise HTTPException(status_code=400, detail=f"Invalid limit. Must be between 1 and {MAX_LIMIT}")

    # Check for ETag match
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == locations_etag:
        return Response(status_code=304)

    # Check for Last-Modified match
    if_modified_since = request.headers.get("if-modified-since")
    if if_modified_since and if_modified_since == locations_last_modified:
        return Response(status_code=304)

    cache_key = get_cache_key(cache_country_key, tier)
    cached_response = await get_cached_response(cache_key)

    if cached_response:
        response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800"
        response.headers["ETag"] = locations_etag
        response.headers["Last-Modified"] = locations_last_modified
        return PreapprovedResponse(**cached_response)

    filtered_locations = filter_locations(locations_data, resolved_country, tier, limit)

    # Build response
    response_data = {
        "version": 1,
        "count": len(filtered_locations),
        "generated_at": datetime.now().isoformat(),
        "locations": [loc.model_dump() for loc in filtered_locations],
    }

    # Cache the response
    await set_cached_response(cache_key, response_data)

    # Set cache headers (7 days for CDN/shared caches, 1 hour for browsers)
    response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800"
    response.headers["ETag"] = locations_etag
    response.headers["Last-Modified"] = locations_last_modified

    return PreapprovedResponse(**response_data)


@router.get("/v1/locations/search")
async def search_locations(
    request: Request,
    q: str = Query(..., min_length=2, max_length=100, description="City name search query"),
    limit: int = Query(10, ge=1, le=20, description="Maximum number of results"),
):
    """
    Search for locations by city name.

    When MAPBOX_TOKEN is configured, delegates to the Mapbox Geocoding API and
    returns global results.  Without a token (dev / CI), falls back to a
    ranked search over the preapproved list.

    Each result includes:
      - name          city name
      - admin1        first-level subdivision (state, province, etc.)
      - country_name  full country name
      - country_code  ISO 3166-1 alpha-2 code
      - location_id   canonical ID if the result matches a preapproved location,
                      otherwise null. Clients should pass this value to
                      POST /v1/locations/selections when non-null.
    """
    # Rate limiting (shared bucket with preapproved endpoint)
    client_ip = request.client.host
    allowed, reason = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    # --- Mapbox path ---
    if MAPBOX_TOKEN:
        results = await _geocode_mapbox(q, limit=limit)
        # _geocode_mapbox returns fresh dicts (JSON-deserialised on each call),
        # so mutation here is safe.
        for result in results:
            result["location_id"] = _find_preapproved_id(result.get("name", ""), result.get("country_code", ""))
        return {
            "count": len(results),
            "locations": results,
        }

    # --- Fallback: search the preapproved list ---
    if not locations_data:
        raise HTTPException(status_code=503, detail="Locations data not yet loaded")

    query_lower = q.strip().lower()

    exact: List[LocationItem] = []
    starts: List[LocationItem] = []
    contains_name: List[LocationItem] = []
    contains_country: List[LocationItem] = []

    for loc in locations_data:
        name_lower = loc.name.lower()
        country_lower = loc.country_name.lower()
        if name_lower == query_lower:
            exact.append(loc)
        elif name_lower.startswith(query_lower):
            starts.append(loc)
        elif query_lower in name_lower:
            contains_name.append(loc)
        elif query_lower in country_lower:
            contains_country.append(loc)

    for group in (exact, starts, contains_name, contains_country):
        group.sort(key=lambda x: x.name)

    ranked = (exact + starts + contains_name + contains_country)[:limit]

    return {
        "count": len(ranked),
        "locations": [
            {
                "name": loc.name,
                "admin1": loc.admin1,
                "country_name": loc.country_name,
                "country_code": loc.country_code,
                "location_id": loc.id,  # fallback results are always preapproved
            }
            for loc in ranked
        ],
    }


@router.get("/v1/locations/preapproved/status")
async def get_locations_status():
    """Get status information about the preapproved locations service."""
    return {
        "status": "healthy",
        "locations_loaded": len(locations_data),
        "etag": locations_etag,
        "last_modified": locations_last_modified,
        "cache_enabled": redis_client is not None,
        "rate_limit": {"requests_per_minute": RATE_LIMIT_REQUESTS, "window_seconds": RATE_LIMIT_WINDOW},
    }


# ---------------------------------------------------------------------------
# Popular locations helpers
# ---------------------------------------------------------------------------


def _build_popular_locations(limit: int) -> List[dict]:
    """
    Return ranked location dicts for the popular endpoint.

    Preapproved locations are returned with full metadata (including image
    fields).  Non-preapproved locations are returned with whatever minimal
    metadata was stored at selection time (name, country, etc.; no images).

    Selection-ranked results are used whenever any signal exists; preapproved
    locations always pad the list so there are at least as many results as
    the preapproved list contains.  Falls back to preapproved (alphabetical)
    only when the usage tracker is unavailable.
    """
    tracker = get_usage_tracker()
    loc_by_id = {loc.id: loc for loc in locations_data}

    if tracker is None:
        return [loc.model_dump() for loc in locations_data[:limit]]

    # Fetch extra to cover IDs that may not be in the preapproved list
    ranked = tracker.get_popular_from_selections(limit=limit * 2, days=POPULARITY_WINDOW_DAYS)

    result: List[dict] = []
    for location_id, _score in ranked:
        if location_id in loc_by_id:
            result.append(loc_by_id[location_id].model_dump())
        else:
            meta = tracker.get_location_metadata(location_id)
            if meta:
                result.append(meta)
        if len(result) >= limit:
            break

    # Pad with preapproved fallback if not enough signal-derived results
    if len(result) < limit:
        seen_ids = {loc["id"] for loc in result}
        for loc in locations_data:
            if loc.id not in seen_ids:
                result.append(loc.model_dump())
            if len(result) >= limit:
                break

    return result[:limit]


# ---------------------------------------------------------------------------
# Popular locations endpoint
# ---------------------------------------------------------------------------


@router.get("/v1/locations/popular")
async def get_popular_locations(
    request: Request,
    response: Response,
    country_code: Optional[str] = Query(None, description="Filter by ISO 3166-1 alpha-2 country code"),
    tier: Optional[str] = Query(None, description="Filter by location tier"),
    limit: Optional[int] = Query(None, ge=1, le=MAX_LIMIT, description=f"Limit results (max {MAX_LIMIT})"),
    include_images: bool = Query(
        False, description="Include imageUrl, imageAlt, imageAttribution fields (default false)"
    ),
):
    """
    Get popular locations with optional filtering.

    Returns the most popular locations ranked by selection frequency, falling
    back to the preapproved list until sufficient usage signal exists.

    Image fields (imageUrl, imageAlt, imageAttribution) are omitted by default;
    pass include_images=true if you need them.

    Response shape:
      { version, count, generated_at, locations: [{id, slug, name, admin1,
        country_name, country_code, latitude, longitude, timezone, tier,
        [imageUrl, imageAlt, imageAttribution if include_images=true]}] }
    """
    client_ip = request.client.host
    allowed, reason = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    resolved_country = None
    cache_country_key = None
    if country_code:
        country_code = country_code.upper()
        valid, error_msg = validate_country_code(country_code)
        if not valid:
            raise HTTPException(status_code=400, detail=error_msg)
        resolved_country = resolve_country_code(country_code)
        cache_country_key = country_code if isinstance(resolved_country, frozenset) else resolved_country

    if limit and not validate_limit(limit):
        raise HTTPException(status_code=400, detail=f"Invalid limit. Must be between 1 and {MAX_LIMIT}")

    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == locations_etag:
        return Response(status_code=304)

    if_modified_since = request.headers.get("if-modified-since")
    if if_modified_since and if_modified_since == locations_last_modified:
        return Response(status_code=304)

    # Cache always stores full location data (with images); strip at response time.
    cache_key = get_popular_cache_key(cache_country_key, tier)
    cached_response = await get_cached_response(cache_key)

    if cached_response:
        response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800"
        response.headers["ETag"] = locations_etag
        response.headers["Last-Modified"] = locations_last_modified
        if not include_images:
            for loc in cached_response.get("locations", []):
                for field in _IMAGE_FIELDS:
                    loc.pop(field, None)
        return cached_response

    # Build ranked list (popularity order preserved), then apply filters.
    # _build_popular_locations returns dicts (mixed preapproved + non-preapproved),
    # so we filter inline rather than via filter_locations (which expects LocationItem).
    popular = _build_popular_locations(limit or POPULARITY_MAX_LOCATIONS)
    filtered: List[dict] = popular
    if resolved_country:
        if isinstance(resolved_country, frozenset):
            filtered = [loc for loc in filtered if loc.get("country_code") in resolved_country]
        else:
            filtered = [loc for loc in filtered if loc.get("country_code") == resolved_country]
    if tier:
        filtered = [loc for loc in filtered if loc.get("tier") == tier]
    if limit:
        filtered = filtered[:limit]

    # Cache stores full data (images present for preapproved; absent for non-preapproved)
    full_data = {
        "version": 1,
        "count": len(filtered),
        "generated_at": datetime.now().isoformat(),
        "locations": filtered,
    }
    await set_cached_response(cache_key, full_data, ttl=POPULAR_CACHE_TTL)

    response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800"
    response.headers["ETag"] = locations_etag
    response.headers["Last-Modified"] = locations_last_modified

    if include_images:
        return full_data

    slim_locs = [{k: v for k, v in loc.items() if k not in _IMAGE_FIELDS} for loc in filtered]
    return {**full_data, "locations": slim_locs}


@router.get("/v1/locations/popular/status")
async def get_popular_locations_status():
    """Get status information about the popular locations service."""
    return {
        "status": "healthy",
        "locations_loaded": len(locations_data),
        "etag": locations_etag,
        "last_modified": locations_last_modified,
        "cache_enabled": redis_client is not None,
        "fallback": "preapproved",
        "rate_limit": {"requests_per_minute": RATE_LIMIT_REQUESTS, "window_seconds": RATE_LIMIT_WINDOW},
    }


@router.get("/v1/locations/popular/stats")
async def get_popular_locations_stats():
    """
    Debug/ops endpoint: per-location selection counts from the rolling window.

    Returns each location ID ranked by total selections, the window size,
    the minimum-signal threshold, and whether live signal is currently being
    served (vs. the preapproved fallback).

    No authentication required. Rate-limit exempt.
    """
    tracker = get_usage_tracker()
    if tracker is None:
        return {
            "status": "unavailable",
            "window_days": POPULARITY_WINDOW_DAYS,
            "total_selections": 0,
            "using_signal": False,
            "locations": [],
        }

    ranked = tracker.get_popular_from_selections(limit=500, days=POPULARITY_WINDOW_DAYS)
    total = tracker.get_total_selections(days=POPULARITY_WINDOW_DAYS)
    loc_by_id = {loc.id: loc for loc in locations_data}

    return {
        "status": "ok",
        "window_days": POPULARITY_WINDOW_DAYS,
        "total_selections": total,
        "using_signal": total > 0,
        "locations": [
            {
                "location_id": loc_id,
                "name": loc_by_id[loc_id].name if loc_id in loc_by_id else None,
                "count": count,
                "in_preapproved": loc_id in loc_by_id,
            }
            for loc_id, count in ranked
        ],
    }


def _build_display_string(body: "SelectionRequest") -> Optional[str]:
    """Derive a human-readable display string from a SelectionRequest.

    Tries the preapproved list first (for canonical IDs), then falls back to
    constructing a string from raw name/admin1/country_code fields.
    Returns None when there is not enough information.
    """
    # Preapproved lookup by ID
    if body.location_id:
        loc_by_id = {loc.id: loc for loc in locations_data}
        loc = loc_by_id.get(body.location_id)
        if loc:
            parts = [loc.name]
            if loc.admin1:
                parts.append(loc.admin1)
            parts.append(loc.country_name)
            return ", ".join(parts)

    # Build from raw fields (non-preapproved or fallback)
    if body.name:
        parts = [body.name]
        if body.admin1:
            parts.append(body.admin1)
        if body.country_code:
            country = pycountry.countries.get(alpha_2=body.country_code.upper())
            if country:
                parts.append(country.name)
        return ", ".join(parts)

    return None


@router.post("/v1/locations/selections", status_code=204)
async def record_location_selection(request: Request, body: SelectionRequest):
    """
    Record a canonical location ID selected by the authenticated user.

    Used to build usage-derived popular locations over time.
    Requires Firebase authentication (anonymous users are accepted).
    Silently no-ops when the usage tracker is unavailable.
    """
    if not getattr(request.state, "user", None):
        raise HTTPException(status_code=401, detail="Authentication required.")

    client_ip = request.client.host
    allowed, reason = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    uid = request.state.user.get("uid", "anonymous")
    canonical_id = _resolve_canonical_id(body)
    logger.debug(
        "selection resolving: input=%r → canonical_id=%s uid=%s",
        body.location_id or body.name,
        canonical_id,
        uid,
    )

    tracker = get_usage_tracker()
    if tracker is not None:
        tracker.record_selection(canonical_id, uid)

        loc_by_id = {loc.id: loc for loc in locations_data}
        if canonical_id not in loc_by_id and body.name:
            country_code, country_name = _resolve_country_fields(body.country_code, body.country_name)
            tracker.store_location_metadata(canonical_id, {
                "id": canonical_id,
                "slug": canonical_id,
                "name": body.name,
                "admin1": body.admin1,
                "country_name": country_name,
                "country_code": country_code,
                "latitude": body.latitude,
                "longitude": body.longitude,
            })

        display = _build_display_string(body)
        if display:
            tracker.store_location_display(canonical_id, display)

    return Response(status_code=204)


@router.get("/v1/locations/popular/display-strings")
async def get_popular_display_strings(
    request: Request,
    limit: Optional[int] = Query(
        None, ge=1, le=MAX_LIMIT, description=f"Max locations to return (default: {POPULARITY_MAX_LOCATIONS})"
    ),
):
    """
    Return display strings for the most-selected locations, including
    non-preapproved ones.  Intended for use by the cache prewarmer.

    Unlike /v1/locations/popular (which returns full LocationItem objects for
    preapproved locations only), this endpoint resolves every ranked location
    ID to its stored display string, so locations like 'Budapest, Budapest,
    Hungary' appear here as soon as they accumulate selection signal.

    Falls back to the preapproved list (alphabetical) when there is
    insufficient selection signal (no selections recorded yet).

    Response: { "count": N, "locations": ["City, Region, Country", ...] }
    """
    client_ip = request.client.host
    allowed, reason = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    effective_limit = limit or POPULARITY_MAX_LOCATIONS
    tracker = get_usage_tracker()

    if tracker is not None:
        strings = tracker.get_popular_display_strings(limit=effective_limit, days=POPULARITY_WINDOW_DAYS)
        if strings:
            return {"count": len(strings), "locations": strings}

    # Fallback: build display strings from preapproved list
    fallback = []
    for loc in locations_data[:effective_limit]:
        parts = [loc.name]
        if loc.admin1:
            parts.append(loc.admin1)
        parts.append(loc.country_name)
        fallback.append(", ".join(parts))

    return {"count": len(fallback), "locations": fallback}


async def initialize_locations_data(redis: redis.Redis):
    """Initialize locations data and warm cache."""
    global locations_data, locations_etag, locations_last_modified, redis_client

    redis_client = redis
    locations_data, locations_etag, locations_last_modified = await load_locations_data()
    await warm_cache()
