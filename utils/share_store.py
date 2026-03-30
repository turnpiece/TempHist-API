"""Persistent store for social share records, backed by Postgres."""
import json
import logging
import os
import secrets
import string
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

_ALPHABET = string.ascii_letters + string.digits  # 62 chars, ~218 trillion combinations at 8 chars
_SHARE_BASE_URL = os.getenv("SHARE_BASE_URL", "https://temphist.com")

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
            """)

    async def create_share(
        self,
        location: str,
        period: str,
        identifier: str,
        ref_year: int,
        unit: str,
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
                        INSERT INTO shares (id, location, period, identifier, ref_year, unit)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING id, location, period, identifier, ref_year, unit, created_at
                        """,
                        share_id, location, period, identifier, ref_year, unit,
                    )
                    return {
                        "id": row["id"],
                        "url": f"{_SHARE_BASE_URL}/s/{row['id']}",
                    }
                except asyncpg.UniqueViolationError:
                    logger.warning("Share ID collision on %s, retrying", share_id)
                    continue

        logger.error("Failed to generate a unique share ID after 5 attempts")
        return None

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
