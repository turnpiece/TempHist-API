"""Persistent store for social share records, backed by Postgres."""
import json
import logging
import math
import os
import secrets
import string
from typing import Optional, List

import asyncpg

logger = logging.getLogger(__name__)

_ALPHABET = string.ascii_letters + string.digits  # 62 chars, ~218 trillion combinations at 8 chars

# Proximity threshold for location deduplication in list_shares.
# Two shares with the same (period, identifier) are considered duplicates when
# their coordinates are within this distance.  50 km keeps e.g. "London" and
# "Greater London" as one entry while still distinguishing separate cities.
_DEDUP_PROXIMITY_KM = 50.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))

_share_store_instance: Optional["ShareStore"] = None


def get_share_store() -> "ShareStore":
    global _share_store_instance
    if _share_store_instance is None:
        _share_store_instance = ShareStore()
    return _share_store_instance


def _generate_id(length: int = 8) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


class ShareStore:
    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or os.getenv("TEMPHIST_PG_DSN") or os.getenv("DATABASE_URL")
        self._disabled = not self._dsn
        self._pool: Optional[asyncpg.Pool] = None
        if self._disabled:
            logger.warning("ShareStore disabled: no Postgres DSN configured.")

    async def _ensure_pool(self) -> Optional[asyncpg.Pool]:
        if self._disabled:
            return None
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=2,
                max_size=10,
                command_timeout=10.0,
                max_inactive_connection_lifetime=300.0,
            )
            await self._ensure_table()
        return self._pool

    async def _ensure_table(self):
        pool = self._pool
        if not pool:
            return
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS shares (
                    id          VARCHAR(8) PRIMARY KEY,
                    location    TEXT NOT NULL,
                    period      VARCHAR(10) NOT NULL,
                    identifier  VARCHAR(5) NOT NULL,
                    ref_year    INTEGER NOT NULL,
                    unit        VARCHAR(12) NOT NULL DEFAULT 'celsius',
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS shares_created_at_idx ON shares (created_at);
                ALTER TABLE shares ADD COLUMN IF NOT EXISTS latitude  DOUBLE PRECISION;
                ALTER TABLE shares ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;
            """)

    async def create_share(
        self,
        location: str,
        period: str,
        identifier: str,
        ref_year: int,
        unit: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Optional[dict]:
        pool = await self._ensure_pool()
        if not pool:
            return None

        # Retry on the rare collision (probability ~1 in 218 trillion per attempt)
        for _ in range(5):
            share_id = _generate_id()
            async with pool.acquire() as conn:
                try:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO shares
                            (id, location, period, identifier, ref_year, unit, latitude, longitude)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        RETURNING id, location, period, identifier, ref_year, unit, created_at
                        """,
                        share_id, location, period, identifier, ref_year, unit,
                        latitude, longitude,
                    )
                    return {
                        "id": row["id"],
                        "url": f"/s/{row['id']}",
                    }
                except asyncpg.UniqueViolationError:
                    logger.warning("Share ID collision on %s, retrying", share_id)
                    continue

        logger.error("Failed to generate a unique share ID after 5 attempts")
        return None

    async def list_shares(
        self,
        period: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Optional[List[dict]]:
        pool = await self._ensure_pool()
        if not pool:
            return None

        # Fetch a generous candidate set so Python-side proximity deduplication
        # still has enough rows to fill (offset + limit) after removing duplicates.
        fetch_limit = (offset + limit) * 10

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, location, period, identifier, ref_year, unit,
                       created_at, latitude, longitude
                FROM shares
                WHERE ($1::text IS NULL OR period = $1)
                ORDER BY created_at DESC
                LIMIT $2
                """,
                period, fetch_limit,
            )

        # Proximity deduplication: for each candidate (most-recent first), keep it
        # only if no already-accepted row has the same (period, identifier) AND a
        # location within _DEDUP_PROXIMITY_KM km.  When either row lacks coordinates,
        # fall back to exact location string comparison.
        accepted: List[dict] = []
        for row in rows:
            r_lat = row["latitude"]
            r_lon = row["longitude"]
            r_period = row["period"]
            r_identifier = row["identifier"]
            r_location = row["location"]

            duplicate = False
            for acc in accepted:
                if acc["period"] != r_period or acc["identifier"] != r_identifier:
                    continue
                # Same period + identifier — check location proximity
                a_lat = acc["_lat"]
                a_lon = acc["_lon"]
                if r_lat is not None and r_lon is not None and a_lat is not None and a_lon is not None:
                    if _haversine_km(r_lat, r_lon, a_lat, a_lon) <= _DEDUP_PROXIMITY_KM:
                        duplicate = True
                        break
                else:
                    # No coordinates on one or both rows — exact string fallback
                    if acc["location"] == r_location:
                        duplicate = True
                        break

            if not duplicate:
                accepted.append({
                    "id": row["id"],
                    "location": r_location,
                    "period": r_period,
                    "identifier": r_identifier,
                    "ref_year": row["ref_year"],
                    "unit": row["unit"],
                    "created_at": row["created_at"].isoformat(),
                    "og_image_url": f"/v1/og/{row['id']}.png",
                    "share_url": f"/s/{row['id']}",
                    # Private fields used only during dedup — stripped before return
                    "_lat": r_lat,
                    "_lon": r_lon,
                })

        # Apply pagination, then strip the private coordinate fields
        page = accepted[offset: offset + limit]
        for entry in page:
            entry.pop("_lat", None)
            entry.pop("_lon", None)
        return page

    async def get_share(self, share_id: str) -> Optional[dict]:
        pool = await self._ensure_pool()
        if not pool:
            return None

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, location, period, identifier, ref_year, unit, created_at FROM shares WHERE id = $1",
                share_id,
            )
        if row is None:
            return None
        return {
            "id": row["id"],
            "location": row["location"],
            "period": row["period"],
            "identifier": row["identifier"],
            "ref_year": row["ref_year"],
            "unit": row["unit"],
            "created_at": row["created_at"].isoformat(),
        }
