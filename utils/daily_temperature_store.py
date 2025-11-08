import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import aiosqlite

from cache_utils import normalize_location_for_cache


@dataclass(frozen=True)
class DailyTemperatureRecord:
    """Represents a cached daily temperature observation stored in SQLite."""

    date: date
    temp_c: Optional[float]
    temp_max_c: Optional[float]
    temp_min_c: Optional[float]
    payload: Dict[str, object]
    source: str

    def as_storage_tuple(self) -> Tuple[str, Optional[float], Optional[float], Optional[float], str, str]:
        """Return values suitable for SQLite insertion."""
        payload_json = json.dumps(self.payload, separators=(",", ":"), sort_keys=True)
        return (
            self.date.isoformat(),
            self.temp_c,
            self.temp_max_c,
            self.temp_min_c,
            payload_json,
            self.source,
        )


class DailyTemperatureStore:
    """Persistent cache for daily temperature data using SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        default_path = Path(__file__).resolve().parents[1] / "weather_cache" / "daily_temperatures.db"
        self._db_path = Path(db_path or os.getenv("TEMPHIST_DAILY_CACHE_DB", default_path))
        self._init_lock = asyncio.Lock()
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def _ensure_initialized(self) -> None:
        """Create the SQLite database and schema on first use."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self._db_path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_temperatures (
                        location TEXT NOT NULL,
                        date TEXT NOT NULL,
                        temp_c REAL,
                        temp_max_c REAL,
                        temp_min_c REAL,
                        payload TEXT NOT NULL,
                        source TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (location, date)
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_daily_temperatures_location_date
                    ON daily_temperatures (location, date)
                    """
                )
                await conn.commit()
            self._initialized = True

    async def fetch(
        self,
        location: str,
        dates: Iterable[date],
    ) -> Dict[date, DailyTemperatureRecord]:
        """Fetch cached records for a location and list of dates."""
        await self._ensure_initialized()
        normalized_location = normalize_location_for_cache(location)
        date_list = sorted({d.isoformat() for d in dates})
        if not date_list:
            return {}

        placeholders = ",".join(["?"] * len(date_list))
        query = (
            f"SELECT date, temp_c, temp_max_c, temp_min_c, payload, source "
            f"FROM daily_temperatures WHERE location = ? AND date IN ({placeholders})"
        )

        result: Dict[date, DailyTemperatureRecord] = {}
        async with aiosqlite.connect(self._db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(query, [normalized_location, *date_list])
            async for row in cursor:
                row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
                payload = json.loads(row["payload"])
                result[row_date] = DailyTemperatureRecord(
                    date=row_date,
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
        await self._ensure_initialized()
        normalized_location = normalize_location_for_cache(location)
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        values = [
            (
                normalized_location,
                record.date.isoformat(),
                record.temp_c,
                record.temp_max_c,
                record.temp_min_c,
                json.dumps(record.payload, separators=(",", ":"), sort_keys=True),
                record.source,
                now_iso,
            )
            for record in records
        ]

        async with aiosqlite.connect(self._db_path) as conn:
            await conn.executemany(
                """
                INSERT INTO daily_temperatures (
                    location,
                    date,
                    temp_c,
                    temp_max_c,
                    temp_min_c,
                    payload,
                    source,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(location, date) DO UPDATE SET
                    temp_c=excluded.temp_c,
                    temp_max_c=excluded.temp_max_c,
                    temp_min_c=excluded.temp_min_c,
                    payload=excluded.payload,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                values,
            )
            await conn.commit()


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

