import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import asyncpg  # type: ignore[import-untyped]

from cache_utils import normalize_location_for_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyTemperatureRecord:
    """Represents a cached daily temperature observation stored in the database."""

    date: date
    temp_c: Optional[float]
    temp_max_c: Optional[float]
    temp_min_c: Optional[float]
    payload: Dict[str, Any]
    source: str


class DailyTemperatureStore:
    """Persistent cache for daily temperature data backed by Postgres."""

    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or os.getenv("TEMPHIST_PG_DSN") or os.getenv("DATABASE_URL")
        self._disabled = False
        if not self._dsn:
            logger.warning(
                "DailyTemperatureStore disabled: TEMPHIST_PG_DSN (or DATABASE_URL) is not configured."
            )
            self._disabled = True
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()
        self._schema_lock = asyncio.Lock()
        self._schema_initialized = False

    async def _ensure_pool(self) -> Optional[asyncpg.Pool]:
        if self._disabled:
            return None
        if self._pool:
            return self._pool
        async with self._pool_lock:
            if self._pool or self._disabled:
                return self._pool
            try:
                pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=5)
            except Exception as exc:  # asyncpg raises multiple subclasses
                logger.error(
                    "DailyTemperatureStore: unable to create Postgres pool (%s). Disabling persistent cache.",
                    exc,
                )
                self._disabled = True
                return None
            await self._initialize_schema(pool)
            self._pool = pool
            return self._pool

    async def _initialize_schema(self, pool: asyncpg.Pool) -> None:
        if self._schema_initialized or self._disabled:
            return
        async with self._schema_lock:
            if self._schema_initialized or self._disabled:
                return
            async with pool.acquire() as conn:
                await self._ensure_locations_table(conn)
                await self._ensure_daily_temperatures_table(conn)
            self._schema_initialized = True

    async def _ensure_locations_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS locations (
                id BIGSERIAL PRIMARY KEY,
                original_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE,
                resolved_name TEXT,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                timezone TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_locations_normalized
            ON locations (normalized_name)
            """
        )

    async def _ensure_daily_temperatures_table(self, conn: asyncpg.Connection) -> None:
        table_exists = await conn.fetchval(
            "SELECT to_regclass('public.daily_temperatures') IS NOT NULL"
        )
        if not table_exists:
            await self._create_daily_temperatures_table(conn)
            return

        has_location_id = await conn.fetchval(
            """
            SELECT COUNT(*) > 0
            FROM information_schema.columns
            WHERE table_name = 'daily_temperatures' AND column_name = 'location_id'
            """
        )
        if has_location_id:
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_temperatures_location_day
                ON daily_temperatures (location_id, day)
                """
            )
            return

        logger.info("Migrating legacy daily_temperatures schema to use location_id.")
        await self._migrate_legacy_daily_temperatures(conn)

    async def _create_daily_temperatures_table(self, conn: asyncpg.Connection) -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_temperatures (
                location_id BIGINT NOT NULL REFERENCES locations (id) ON DELETE CASCADE,
                day DATE NOT NULL,
                temp_c DOUBLE PRECISION,
                temp_max_c DOUBLE PRECISION,
                temp_min_c DOUBLE PRECISION,
                payload JSONB NOT NULL,
                source TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (location_id, day)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_daily_temperatures_location_day
            ON daily_temperatures (location_id, day)
            """
        )

    async def _migrate_legacy_daily_temperatures(self, conn: asyncpg.Connection) -> None:
        await conn.execute("ALTER TABLE daily_temperatures RENAME TO daily_temperatures_legacy")
        await self._create_daily_temperatures_table(conn)

        legacy_locations = await conn.fetch(
            "SELECT DISTINCT location FROM daily_temperatures_legacy"
        )
        location_id_map: Dict[str, int] = {}
        for record in legacy_locations:
            original_name = record["location"]
            if not original_name:
                continue
            normalized = normalize_location_for_cache(original_name)
            row = await conn.fetchrow(
                """
                INSERT INTO locations (
                    original_name,
                    normalized_name,
                    resolved_name
                )
                VALUES ($1, $2, $3)
                ON CONFLICT (normalized_name) DO UPDATE SET
                    original_name = EXCLUDED.original_name,
                    resolved_name = COALESCE(EXCLUDED.resolved_name, locations.resolved_name),
                    updated_at = NOW()
                RETURNING id
                """,
                original_name,
                normalized,
                original_name,
            )
            if row:
                location_id_map[normalized] = row["id"]

        legacy_rows = await conn.fetch(
            """
            SELECT location, day, temp_c, temp_max_c, temp_min_c, payload, source, updated_at
            FROM daily_temperatures_legacy
            """
        )
        batch: List[tuple] = []
        for row in legacy_rows:
            original_name = row["location"]
            if not original_name:
                continue
            normalized = normalize_location_for_cache(original_name)
            location_id = location_id_map.get(normalized)
            if location_id is None:
                continue
            payload = row["payload"]
            if isinstance(payload, dict):
                payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            else:
                payload_json = payload
            batch.append(
                (
                    location_id,
                    row["day"],
                    row["temp_c"],
                    row["temp_max_c"],
                    row["temp_min_c"],
                    payload_json,
                    row["source"],
                    row["updated_at"],
                )
            )

        if batch:
            await conn.executemany(
                """
                INSERT INTO daily_temperatures (
                    location_id,
                    day,
                    temp_c,
                    temp_max_c,
                    temp_min_c,
                    payload,
                    source,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                """,
                batch,
            )

        await conn.execute("DROP TABLE daily_temperatures_legacy")

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
        if pool is None:
            return {}

        normalized_location = normalize_location_for_cache(location)
        result: Dict[date, DailyTemperatureRecord] = {}

        try:
            async with pool.acquire() as conn:
                location_row = await conn.fetchrow(
                    "SELECT id FROM locations WHERE normalized_name = $1",
                    normalized_location,
                )
                if not location_row:
                    return {}

                rows = await conn.fetch(
                    """
                    SELECT day, temp_c, temp_max_c, temp_min_c, payload, source
                    FROM daily_temperatures
                    WHERE location_id = $1 AND day = ANY($2::date[])
                    """,
                    location_row["id"],
                    date_list,
                )
        except Exception as exc:
            logger.warning(
                "DailyTemperatureStore.fetch failed for %s due to %s. Returning empty result.",
                location,
                exc,
            )
            return {}

        for row in rows:
            record_date: date = row["day"]
            payload = row["payload"] or {}
            result[record_date] = DailyTemperatureRecord(
                date=record_date,
                temp_c=row["temp_c"],
                temp_max_c=row["temp_max_c"],
                temp_min_c=row["temp_min_c"],
                payload=payload if isinstance(payload, dict) else {},
                source=row["source"],
            )
        return result

    async def upsert(
        self,
        location: str,
        records: List[DailyTemperatureRecord],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a batch of records for a location."""
        if not records:
            return

        pool = await self._ensure_pool()
        if pool is None:
            return

        normalized_location = normalize_location_for_cache(location)
        now_ts = datetime.now(timezone.utc)
        prepared_records = [
            (
                record.date,
                record.temp_c,
                record.temp_max_c,
                record.temp_min_c,
                json.dumps(record.payload, separators=(",", ":"), sort_keys=True),
                record.source,
            )
            for record in records
        ]

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    location_id = await self._get_or_create_location_id(
                        conn, location, normalized_location, metadata
                    )
                    if location_id is None:
                        return
                    await conn.executemany(
                        """
                        INSERT INTO daily_temperatures (
                            location_id,
                            day,
                            temp_c,
                            temp_max_c,
                            temp_min_c,
                            payload,
                            source,
                            updated_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                        ON CONFLICT (location_id, day) DO UPDATE SET
                            temp_c = EXCLUDED.temp_c,
                            temp_max_c = EXCLUDED.temp_max_c,
                            temp_min_c = EXCLUDED.temp_min_c,
                            payload = EXCLUDED.payload,
                            source = EXCLUDED.source,
                            updated_at = EXCLUDED.updated_at
                        """,
                        [
                            (
                                location_id,
                                day,
                                temp_c,
                                temp_max_c,
                                temp_min_c,
                                payload,
                                source,
                                now_ts,
                            )
                            for day, temp_c, temp_max_c, temp_min_c, payload, source in prepared_records
                        ],
                    )
        except Exception as exc:
            logger.warning(
                "DailyTemperatureStore.upsert failed for %s due to %s. Skipping persistence.",
                location,
                exc,
            )

    async def _get_or_create_location_id(
        self,
        conn: asyncpg.Connection,
        original_name: str,
        normalized_name: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        resolved_name = self._extract_metadata_value(metadata, ["resolvedAddress", "resolved_address"])
        latitude = self._extract_numeric_metadata(metadata, ["latitude", "lat"])
        longitude = self._extract_numeric_metadata(metadata, ["longitude", "lon", "lng"])
        timezone_name = self._extract_metadata_value(metadata, ["timezone", "tz"])

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO locations (
                    original_name,
                    normalized_name,
                    resolved_name,
                    latitude,
                    longitude,
                    timezone
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (normalized_name) DO UPDATE SET
                    original_name = EXCLUDED.original_name,
                    resolved_name = COALESCE(EXCLUDED.resolved_name, locations.resolved_name),
                    latitude = COALESCE(EXCLUDED.latitude, locations.latitude),
                    longitude = COALESCE(EXCLUDED.longitude, locations.longitude),
                    timezone = COALESCE(EXCLUDED.timezone, locations.timezone),
                    updated_at = NOW()
                RETURNING id
                """,
                original_name,
                normalized_name,
                resolved_name or original_name,
                latitude,
                longitude,
                timezone_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to upsert location '%s' (%s): %s",
                original_name,
                normalized_name,
                exc,
            )
            return None

        return row["id"] if row else None

    @staticmethod
    def _extract_metadata_value(
        metadata: Optional[Dict[str, Any]],
        keys: List[str],
    ) -> Optional[str]:
        if not metadata:
            return None
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _extract_numeric_metadata(
        metadata: Optional[Dict[str, Any]],
        keys: List[str],
    ) -> Optional[float]:
        if not metadata:
            return None
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return None


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

