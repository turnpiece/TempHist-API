import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional

import asyncpg  # type: ignore[import-untyped]

from cache_utils import normalize_location_for_cache


@dataclass(frozen=True)
class DailyTemperatureRecord:
    """Represents a cached daily temperature observation stored in the database."""

    date: date
    temp_c: Optional[float]
    temp_max_c: Optional[float]
    temp_min_c: Optional[float]
    payload: Dict[str, object]
    source: str


class DailyTemperatureStore:
    """Persistent cache for daily temperature data backed by Postgres."""

    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or os.getenv("TEMPHIST_PG_DSN") or os.getenv("DATABASE_URL")
        if not self._dsn:
            raise RuntimeError(
                "Configure TEMPHIST_PG_DSN (or DATABASE_URL) with the Postgres connection string."
            )
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool:
            return self._pool
        async with self._init_lock:
            if self._pool:
                return self._pool
            self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=5)
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_temperatures (
                        location TEXT NOT NULL,
                        day DATE NOT NULL,
                        temp_c DOUBLE PRECISION,
                        temp_max_c DOUBLE PRECISION,
                        temp_min_c DOUBLE PRECISION,
                        payload JSONB NOT NULL,
                        source TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        PRIMARY KEY (location, day)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_daily_temperatures_location_day
                    ON daily_temperatures (location, day)
                    """
                )
            return self._pool

    async def fetch(
        self,
        location: str,
        dates: Iterable[date],
    ) -> Dict[date, DailyTemperatureRecord]:
        """Fetch cached records for a location and list of dates."""
        date_list = sorted({d for d in dates})
        if not date_list:
            return {}

        pool = await self._ensure_pool()
        normalized_location = normalize_location_for_cache(location)
        result: Dict[date, DailyTemperatureRecord] = {}

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT day, temp_c, temp_max_c, temp_min_c, payload, source
                FROM daily_temperatures
                WHERE location = $1 AND day = ANY($2::date[])
                """,
                normalized_location,
                date_list,
            )

        for row in rows:
            record_date: date = row["day"]
            payload = row["payload"] or {}
            result[record_date] = DailyTemperatureRecord(
                date=record_date,
                temp_c=row["temp_c"],
                temp_max_c=row["temp_max_c"],
                temp_min_c=row["temp_min_c"],
                payload=payload,
                source=row["source"],
            )
        return result

    async def upsert(
        self,
        location: str,
        records: List[DailyTemperatureRecord],
    ) -> None:
        """Insert or update a batch of records for a location."""
        if not records:
            return

        pool = await self._ensure_pool()
        normalized_location = normalize_location_for_cache(location)
        now_ts = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO daily_temperatures (
                    location,
                    day,
                    temp_c,
                    temp_max_c,
                    temp_min_c,
                    payload,
                    source,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                ON CONFLICT (location, day) DO UPDATE SET
                    temp_c = EXCLUDED.temp_c,
                    temp_max_c = EXCLUDED.temp_max_c,
                    temp_min_c = EXCLUDED.temp_min_c,
                    payload = EXCLUDED.payload,
                    source = EXCLUDED.source,
                    updated_at = EXCLUDED.updated_at
                """,
                [
                    (
                        normalized_location,
                        record.date,
                        record.temp_c,
                        record.temp_max_c,
                        record.temp_min_c,
                        json.dumps(record.payload, separators=(",", ":"), sort_keys=True),
                        record.source,
                        now_ts,
                    )
                    for record in records
                ],
            )


_store: Optional[DailyTemperatureStore] = None
_store_lock = asyncio.Lock()


async def get_daily_temperature_store() -> DailyTemperatureStore:
    """Return a singleton DailyTemperatureStore instance."""
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is None:
            _store = DailyTemperatureStore()
        return _store

