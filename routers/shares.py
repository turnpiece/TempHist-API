"""Social share endpoints — POST /v1/shares, GET /v1/shares, GET /v1/shares/{id}."""

import json
import logging
from typing import Annotated, Literal, Optional

import redis
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from routers._responses import error_responses
from routers.dependencies import get_redis_client
from utils.share_store import get_share_store

logger = logging.getLogger(__name__)
router = APIRouter()

_SHARE_CACHE_TTL = 30 * 24 * 3600  # 30 days — share records never change


def _share_cache_key(share_id: str) -> str:
    return f"share:{share_id}"


class ShareCreate(BaseModel):
    location: str = Field(..., min_length=1, max_length=200)
    period: Literal["daily", "weekly", "monthly", "yearly"]
    identifier: str = Field(..., pattern=r"^\d{2}-\d{2}$")  # MM-dd
    ref_year: int = Field(..., ge=1970, le=2100)
    unit: Literal["celsius", "fahrenheit"] = "celsius"
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)


@router.get("/v1/shares", responses=error_responses(503))
async def list_shares(
    period: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List recent share records, deduplicated by location+period+identifier. Public — no auth required."""
    store = get_share_store()
    shares = await store.list_shares(period=period, limit=limit, offset=offset)
    if shares is None:
        raise HTTPException(status_code=503, detail="Share service unavailable.")
    return {"shares": shares, "limit": limit, "offset": offset}


@router.post("/v1/shares", status_code=201, responses=error_responses(401, 503))
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
        location=body.location,
        period=body.period,
        identifier=body.identifier,
        ref_year=body.ref_year,
        unit=body.unit,
        latitude=body.latitude,
        longitude=body.longitude,
    )
    if result is None:
        raise HTTPException(status_code=503, detail="Share service unavailable.")
    return result


@router.get("/v1/shares/{share_id}", responses=error_responses(404))
async def get_share(
    share_id: str,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)],
):
    """Retrieve share parameters by ID. Public — no auth required."""
    if len(share_id) != 8 or not share_id.isalnum():
        raise HTTPException(status_code=404, detail="Share not found.")

    cache_key = _share_cache_key(share_id)

    # Check Redis first
    try:
        cached = redis_client.get(cache_key)
        if cached:
            data_str = cached.decode("utf-8") if isinstance(cached, bytes) else cached
            return json.loads(data_str)
    except Exception as exc:
        logger.warning("Redis read failed for share %s: %s", share_id, exc)

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

    return share
