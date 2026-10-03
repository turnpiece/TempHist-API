"""Social share endpoints — POST /v1/shares, GET /v1/shares, GET /v1/shares/{id}."""

import json
import logging
from typing import Annotated, List, Literal, Optional

import redis
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from routers._params import ShareIdParam
from routers._responses import error_responses
from routers.dependencies import get_redis_client
from routers.locations import locations_data
from utils.share_store import get_share_store
from utils.weather import is_today

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Shares"])

_SHARE_CACHE_TTL = 30 * 24 * 3600  # 30 days — share records never change
_UNAVAILABLE_RETRY_AFTER = 30  # seconds to suggest when the share store (Postgres) is unreachable


def _share_cache_key(share_id: str) -> str:
    return f"share:{share_id}"


def _compute_is_today(share: dict, redis_client: redis.Redis) -> bool:
    """Whether the share's reference date (identifier + ref_year) is "today" in the
    share's location's local timezone. Must be computed fresh on every read — a share
    for today becomes a complete past day tomorrow, but the share record itself (and
    its Redis cache entry) never changes.
    """
    try:
        month_str, day_str = share["identifier"].split("-")
        return is_today(share["ref_year"], int(month_str), int(day_str), share["location"], redis_client)
    except Exception as exc:
        logger.warning("Failed to compute is_today for share %s: %s", share.get("id"), exc)
        return False


def _resolve_location_name(location: str) -> str:
    """Return a human-readable display name for a location string.

    If the value matches a preapproved location ID (case-insensitive), return
    the canonical display name (e.g. "CAPE_TOWN" → "Cape Town, Western Cape,
    South Africa"). Otherwise return the original string unchanged.
    """
    normalised = location.lower()
    for loc in locations_data:
        if loc.id == normalised:
            parts = [loc.name]
            if loc.admin1:
                parts.append(loc.admin1)
            parts.append(loc.country_name)
            return ", ".join(parts)
    return location


class ShareCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "location": "london",
                    "period": "daily",
                    "identifier": "01-15",
                    "ref_year": 2025,
                    "unit": "celsius",
                    "latitude": 51.5074,
                    "longitude": -0.1278,
                }
            ]
        }
    )

    location: str = Field(
        ..., min_length=1, max_length=200, description="Location name or preapproved location ID", examples=["london"]
    )
    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Aggregation period")
    identifier: str = Field(..., pattern=r"^\d{2}-\d{2}$", description="Period end date as `MM-DD`")  # MM-dd
    ref_year: int = Field(..., ge=1970, le=2100, description="The year the shared record highlights")
    unit: Literal["celsius", "fahrenheit"] = Field("celsius", description="Temperature unit shown on the share")
    latitude: Optional[float] = Field(
        None, ge=-90, le=90, description="Latitude of the location, used to merge near-duplicate shares in listings"
    )
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude of the location")


class ShareCreatedResponse(BaseModel):
    """A newly created share."""

    id: str = Field(..., description="Share ID (8 alphanumeric characters)", examples=["aB3dE5gH"])
    url: str = Field(
        ..., description="Relative path of the share page. Prepend your own origin.", examples=["/s/aB3dE5gH"]
    )


class ShareRecord(BaseModel):
    """The stored parameters of a share."""

    id: str = Field(..., description="Share ID", examples=["aB3dE5gH"])
    location: str = Field(..., description="Display name of the location", examples=["London, England, United Kingdom"])
    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Aggregation period")
    identifier: str = Field(..., description="Period end date as `MM-DD`", examples=["01-15"])
    ref_year: int = Field(..., description="The year the shared record highlights", examples=[2025])
    unit: Literal["celsius", "fahrenheit"] = Field(..., description="Temperature unit shown on the share")
    created_at: str = Field(..., description="When the share was created, as an ISO 8601 timestamp")


class ShareSummary(ShareRecord):
    """A share as it appears in listings."""

    og_image_url: str = Field(
        ...,
        description="Relative path of the preview image. Prepend your own origin.",
        examples=["/v1/og/aB3dE5gH.png"],
    )
    share_url: str = Field(
        ..., description="Relative path of the share page. Prepend your own origin.", examples=["/s/aB3dE5gH"]
    )


class ShareListResponse(BaseModel):
    """A page of recent shares."""

    shares: List[ShareSummary] = Field(..., description="Shares, most recent first")
    limit: int = Field(..., description="Page size that was applied")
    offset: int = Field(..., description="Offset that was applied")


class ShareResponse(ShareRecord):
    """A share, with whether its date is currently today."""

    is_today: bool = Field(
        ...,
        description="Whether the shared date is today in the location's timezone. Evaluated on every request.",
    )


@router.get(
    "/v1/shares",
    responses={200: {"model": ShareListResponse}, **error_responses(503)},
)
async def list_shares(
    period: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = Query(
        None, description="Only return shares for this period"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of shares to return"),
    offset: int = Query(0, ge=0, description="Number of shares to skip"),
):
    """List recent share records, deduplicated by location+period+identifier. Public — no auth required."""
    store = get_share_store()
    shares = await store.list_shares(period=period, limit=limit, offset=offset)
    if shares is None:
        raise HTTPException(
            status_code=503,
            detail="Share service unavailable.",
            headers={"Retry-After": str(_UNAVAILABLE_RETRY_AFTER)},
        )
    return {"shares": shares, "limit": limit, "offset": offset}


@router.post(
    "/v1/shares",
    status_code=201,
    responses={201: {"model": ShareCreatedResponse, "description": "The share was created"}, **error_responses(503)},
)
async def create_share(
    request: Request,
    body: ShareCreate,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)],
):
    """Create a share record and return a short URL. Requires Firebase auth."""
    # Auth is enforced by the middleware for all non-public paths.
    # This guard is a belt-and-suspenders check in case middleware config changes.
    if not getattr(request.state, "user", None):
        raise HTTPException(status_code=401, detail="Authentication required.")

    store = get_share_store()
    result = await store.create_share(
        location=_resolve_location_name(body.location),
        period=body.period,
        identifier=body.identifier,
        ref_year=body.ref_year,
        unit=body.unit,
        latitude=body.latitude,
        longitude=body.longitude,
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Share service unavailable.",
            headers={"Retry-After": str(_UNAVAILABLE_RETRY_AFTER)},
        )
    return result


@router.get(
    "/v1/shares/{share_id}",
    responses={200: {"model": ShareResponse}, **error_responses(404)},
)
async def get_share(
    share_id: ShareIdParam,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)],
):
    """Retrieve share parameters by ID. Public — no auth required."""
    if len(share_id) != 8 or not share_id.isalnum():
        raise HTTPException(status_code=404, detail="Share not found.")

    cache_key = _share_cache_key(share_id)
    share = None

    # Check Redis first
    try:
        cached = redis_client.get(cache_key)
        if cached:
            data_str = cached.decode("utf-8") if isinstance(cached, bytes) else cached
            share = json.loads(data_str)
    except Exception as exc:
        logger.warning("Redis read failed for share %s: %s", share_id, exc)

    if share is None:
        # Fall back to Postgres
        store = get_share_store()
        share = await store.get_share(share_id)
        if share is None:
            raise HTTPException(status_code=404, detail="Share not found.")

        # Populate cache for future requests
        try:
            redis_client.setex(cache_key, _SHARE_CACHE_TTL, json.dumps(share))
        except Exception as exc:
            logger.warning("Redis write failed for share %s: %s", share_id, exc)

    # Computed fresh on every request, never cached — see _compute_is_today.
    # Copy rather than mutate: `share` may be a dict owned by the store/cache layer.
    return {**share, "is_today": _compute_is_today(share, redis_client)}
