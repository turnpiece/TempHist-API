"""
Preapproved locations endpoint router.

Provides access to a curated list of preapproved locations with filtering,
caching, and rate limiting capabilities.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from urllib.parse import quote

import aiohttp
import redis
from fastapi import APIRouter, HTTPException, Query, Request, Response, Depends
from pydantic import BaseModel, Field, field_validator

from config import BASE_URL, MAPBOX_TOKEN

# Configure logging
logger = logging.getLogger(__name__)

# Constants
CACHE_TTL = 604800  # 7 days (data changes infrequently)
RATE_LIMIT_REQUESTS = 60  # requests per minute
RATE_LIMIT_WINDOW = 60  # 1 minute window
CACHE_PREFIX = "preapproved:v1"
MAX_LIMIT = 500

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

    @field_validator('country_code')
    @classmethod
    def validate_country_code(cls, v):
        """Validate country code format (ISO 3166-1 alpha-2)."""
        if not re.match(r'^[A-Z]{2}$', v):
            raise ValueError('Country code must be a 2-letter ISO 3166-1 alpha-2 code')
        return v

class PreapprovedResponse(BaseModel):
    """Response model for preapproved locations endpoint."""
    version: int = Field(default=1, description="API version")
    count: int = Field(..., description="Number of locations returned")
    generated_at: datetime = Field(..., description="Response generation timestamp")
    locations: List[LocationItem] = Field(..., description="List of location items")

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

def validate_country_code(country_code: str) -> bool:
    """Validate ISO 3166-1 alpha-2 country code format."""
    return bool(re.match(r'^[A-Z]{2}$', country_code))

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

async def check_rate_limit(ip: str) -> Tuple[bool, str]:
    """Check if IP is within rate limits."""
    async with rate_limit_lock:
        current_time = time.time()
        window_start = current_time - RATE_LIMIT_WINDOW
        
        # Clean old requests outside the window
        rate_limit_requests[ip] = [
            req_time for req_time in rate_limit_requests[ip]
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(rate_limit_requests[ip]) >= RATE_LIMIT_REQUESTS:
            return False, f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per minute"
        
        # Add current request
        rate_limit_requests[ip].append(current_time)
        return True, "OK"

def filter_locations(
    locations: List[LocationItem],
    country_code: Optional[str] = None,
    tier: Optional[str] = None,
    limit: Optional[int] = None
) -> List[LocationItem]:
    """Filter locations based on criteria."""
    filtered = locations
    
    if country_code:
        filtered = [loc for loc in filtered if loc.country_code == country_code]
    
    if tier:
        filtered = [loc for loc in filtered if loc.tier == tier]
    
    # Sort by name for deterministic output
    filtered.sort(key=lambda x: x.name)
    
    if limit:
        filtered = filtered[:limit]
    
    return filtered

def convert_image_urls(image_urls: Dict[str, str]) -> Dict[str, str]:
    """Convert relative image URLs to full URLs."""
    return {
        format_type: f"{BASE_URL}{url}" if url.startswith('/') else url
        for format_type, url in image_urls.items()
    }

async def load_locations_data() -> Tuple[List[LocationItem], str, str]:
    """Load and validate locations data from JSON file."""
    # Find project root by looking for pyproject.toml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    while project_root != os.path.dirname(project_root):  # Stop at filesystem root
        if os.path.exists(os.path.join(project_root, "pyproject.toml")):
            break
        project_root = os.path.dirname(project_root)

    data_file = os.path.join(project_root, "data", "preapproved_locations.json")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        # Parse JSON
        json_data = json.loads(raw_data)

        # Transform relative URLs to full URLs
        for item in json_data:
            if 'imageUrl' in item:
                item['imageUrl'] = convert_image_urls(item['imageUrl'])

        # Validate against schema
        locations = [LocationItem(**item) for item in json_data]

        # Generate ETag and Last-Modified
        etag = generate_etag(raw_data)
        file_stat = os.stat(data_file)
        last_modified = datetime.fromtimestamp(file_stat.st_mtime).strftime('%a, %d %b %Y %H:%M:%S GMT')

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
        cache_key = get_cache_key()
        response_data = {
            "version": 1,
            "count": len(locations_data),
            "generated_at": datetime.now().isoformat(),
            "locations": [loc.model_dump() for loc in locations_data]
        }
        await set_cached_response(cache_key, response_data)
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
    url = (
        f"{MAPBOX_GEOCODE_URL}/{encoded}.json"
        f"?types=place&language=en&limit={limit}&access_token={MAPBOX_TOKEN}"
    )

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
        results.append({
            "name": name,
            "admin1": admin1,
            "country_name": country_name,
            "country_code": country_code,
        })

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
    limit: Optional[int] = Query(None, ge=1, le=MAX_LIMIT, description=f"Limit results (max {MAX_LIMIT})")
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
    
    # Validate parameters
    if country_code and not validate_country_code(country_code):
        raise HTTPException(status_code=400, detail="Invalid country code format. Must be ISO 3166-1 alpha-2")
    
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
    
    # Try cache first
    cache_key = get_cache_key(country_code, tier)
    cached_response = await get_cached_response(cache_key)
    
    if cached_response:
        # Set cache headers (7 days for CDN/shared caches, 1 hour for browsers)
        response.headers["Cache-Control"] = "public, max-age=3600, s-maxage=604800"
        response.headers["ETag"] = locations_etag
        response.headers["Last-Modified"] = locations_last_modified

        return PreapprovedResponse(**cached_response)
    
    # Filter locations
    filtered_locations = filter_locations(locations_data, country_code, tier, limit)
    
    # Build response
    response_data = {
        "version": 1,
        "count": len(filtered_locations),
        "generated_at": datetime.now().isoformat(),
        "locations": [loc.model_dump() for loc in filtered_locations]
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
    """
    # Rate limiting (shared bucket with preapproved endpoint)
    client_ip = request.client.host
    allowed, reason = await check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    # --- Mapbox path ---
    if MAPBOX_TOKEN:
        results = await _geocode_mapbox(q, limit=limit)
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
        "rate_limit": {
            "requests_per_minute": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW
        }
    }

async def initialize_locations_data(redis: redis.Redis):
    """Initialize locations data and warm cache."""
    global locations_data, locations_etag, locations_last_modified, redis_client
    
    redis_client = redis
    locations_data, locations_etag, locations_last_modified = await load_locations_data()
    await warm_cache()
