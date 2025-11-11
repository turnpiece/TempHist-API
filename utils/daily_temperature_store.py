import asyncio
import asyncpg  # type: ignore[import-untyped]
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _calculate_insert_fields(records: List["DailyTemperatureRecord"]) -> Tuple[bool, bool, bool]:
    """Determine which temperature columns are present to build a typed insert."""
    has_temp = any(record.temp_c is not None for record in records)
    has_max = any(record.temp_max_c is not None for record in records)
    has_min = any(record.temp_min_c is not None for record in records)
    return has_temp, has_max, has_min

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


_PREAPPROVED_LOCATION_MAP: Optional[Dict[str, Dict[str, Any]]] = None


def _register_preapproved_candidate(
    mapping: Dict[str, Dict[str, Any]],
    candidate: Optional[str],
    info: Dict[str, Any],
) -> None:
    """Register a preapproved location candidate with normalized variations.

    Args:
        mapping: Dictionary mapping normalized names to location info
        candidate: Location name candidate to register
        info: Location metadata (coordinates, timezone, etc.)
    """
    if not candidate:
        return
    normalized = normalize_location_for_cache(candidate)
    if not normalized or normalized in mapping:
        return
    mapping[normalized] = info

    if "-" in candidate:
        _register_preapproved_candidate(mapping, candidate.replace("-", " "), info)
        _register_preapproved_candidate(mapping, candidate.replace("-", "_"), info)


def _load_preapproved_location_map() -> Dict[str, Dict[str, Any]]:
    """Load preapproved locations from JSON file and create normalized mapping.

    Returns:
        Dictionary mapping normalized location names to location metadata
    """
    global _PREAPPROVED_LOCATION_MAP
    if _PREAPPROVED_LOCATION_MAP is not None:
        return _PREAPPROVED_LOCATION_MAP

    mapping: Dict[str, Dict[str, Any]] = {}
    try:
        current_path = Path(__file__).resolve()
        project_root = current_path.parent
        while project_root != project_root.parent:
            if (project_root / "pyproject.toml").exists():
                break
            project_root = project_root.parent
        data_file = project_root / "data" / "preapproved_locations.json"
        with data_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("Preapproved locations file not found; location metadata backfill disabled.")
        _PREAPPROVED_LOCATION_MAP = {}
        return _PREAPPROVED_LOCATION_MAP
    except Exception as exc:
        logger.warning("Failed to load preapproved locations: %s", exc)
        _PREAPPROVED_LOCATION_MAP = {}
        return _PREAPPROVED_LOCATION_MAP

    for entry in data:
        name = entry.get("name")
        admin1 = entry.get("admin1")
        country = entry.get("country_name")
        latitude = entry.get("latitude")
        longitude = entry.get("longitude")
        timezone_name = entry.get("timezone")
        slug = entry.get("slug")
        identifier = entry.get("id")

        components = [part for part in [name, admin1, country] if part]
        full_name = ", ".join(components) if components else name
        if not full_name:
            continue

        info = {
            "full_name": full_name,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone_name,
            "slug": slug,
            "id": identifier,
        }

        candidates = {full_name, name, identifier, slug}
        if country and name:
            candidates.add(f"{name}, {country}")
        if slug:
            candidates.add(slug.replace("-", " "))

        for candidate in list(candidates):
            _register_preapproved_candidate(mapping, candidate, info)

    _PREAPPROVED_LOCATION_MAP = mapping
    return _PREAPPROVED_LOCATION_MAP


def _get_preapproved_location_info(normalized_name: str) -> Optional[Dict[str, Any]]:
    """Get preapproved location info for a normalized location name.

    Args:
        normalized_name: Normalized location string

    Returns:
        Location metadata dict if found, None otherwise
    """
    mapping = _load_preapproved_location_map()
    return mapping.get(normalized_name)


class DailyTemperatureStore:
    """Persistent cache for daily temperature data backed by Postgres."""

    def __init__(self, dsn: Optional[str] = None):
        """Initialize the DailyTemperatureStore with optional DSN override.

        Args:
            dsn: PostgreSQL connection string. If None, uses TEMPHIST_PG_DSN or DATABASE_URL env vars.
        """
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
        """Ensure the connection pool is initialized, creating it if necessary.

        Returns:
            asyncpg.Pool instance or None if disabled/failed
        """
        if self._disabled:
            return None
        if self._pool:
            return self._pool
        async with self._pool_lock:
            if self._pool or self._disabled:
                return self._pool
            try:
                pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=5,  # Keep 5 connections ready to avoid connection overhead
                    max_size=20,  # Support up to 20 concurrent requests
                    command_timeout=10.0,  # Timeout queries after 10s
                    max_inactive_connection_lifetime=300.0  # Recycle idle connections after 5min
                )
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
        """Initialize database schema tables (locations, aliases, temperatures).

        Args:
            pool: asyncpg connection pool
        """
        if self._schema_initialized or self._disabled:
            return
        async with self._schema_lock:
            if self._schema_initialized or self._disabled:
                return
            async with pool.acquire() as conn:
                await self._ensure_locations_table(conn)
                await self._ensure_location_aliases_table(conn)
                await self._ensure_daily_temperatures_table(conn)
                await self._backfill_locations_metadata(conn)
            self._schema_initialized = True

    async def _ensure_locations_table(self, conn: asyncpg.Connection) -> None:
        """Create locations table and indexes if they don't exist.

        Args:
            conn: asyncpg database connection
        """
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
        # Index for coordinate-based nearby location searches
        # Only index rows with coordinates to keep index small
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_locations_coordinates
            ON locations (latitude, longitude)
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """
        )

    async def _ensure_location_aliases_table(self, conn: asyncpg.Connection) -> None:
        """Ensure the location_aliases table exists.

        This maps incoming normalized location strings onto a canonical locations.id.
        """
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS location_aliases (
                alias_normalized_name TEXT PRIMARY KEY,
                location_id BIGINT NOT NULL REFERENCES locations(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        # Index for fast lookups by location_id (for finding all aliases of a location)
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_location_aliases_location_id
            ON location_aliases (location_id)
            """
        )

    async def _ensure_daily_temperatures_table(self, conn: asyncpg.Connection) -> None:
        """Create daily_temperatures table or migrate from legacy schema if needed.

        Args:
            conn: asyncpg database connection
        """
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

            # Drop old indexes with non-immutable predicates (if they exist)
            await conn.execute("DROP INDEX IF EXISTS idx_daily_temperatures_current_year")

            # Partial indexes for optimized recent data queries
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_temperatures_recent
                ON daily_temperatures (location_id, day)
                WHERE day >= '2020-01-01'::date
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_temperatures_day_desc
                ON daily_temperatures (location_id, day DESC)
                WHERE day >= '2020-01-01'::date
                """
            )
            return

        logger.info("Migrating legacy daily_temperatures schema to use location_id.")
        await self._migrate_legacy_daily_temperatures(conn)

    async def _create_daily_temperatures_table(self, conn: asyncpg.Connection) -> None:
        """Create the daily_temperatures table with proper schema and indexes.

        Args:
            conn: asyncpg database connection
        """
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

        # Drop old indexes with non-immutable predicates (if they exist)
        await conn.execute("DROP INDEX IF EXISTS idx_daily_temperatures_current_year")

        # Partial index for recent queries (2020 onwards)
        # This smaller index improves performance for recent data lookups
        # Using fixed year instead of CURRENT_DATE to satisfy IMMUTABLE requirement
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_daily_temperatures_recent
            ON daily_temperatures (location_id, day)
            WHERE day >= '2020-01-01'::date
            """
        )

        # Index with descending order for recent data queries
        # Optimizes queries that fetch latest data first
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_daily_temperatures_day_desc
            ON daily_temperatures (location_id, day DESC)
            WHERE day >= '2020-01-01'::date
            """
        )

    async def _backfill_locations_metadata(self, conn: asyncpg.Connection) -> None:
        """Backfill missing metadata for existing locations using preapproved data.

        Args:
            conn: asyncpg database connection
        """
        preapproved_map = _load_preapproved_location_map()
        if not preapproved_map:
            return

        for normalized_name, info in preapproved_map.items():
            await conn.execute(
                """
                UPDATE locations
                SET
                    original_name = CASE
                        WHEN (POSITION(',' IN original_name) = 0 OR original_name = normalized_name) AND $2::text IS NOT NULL
                        THEN $2::text
                        ELSE original_name
                    END,
                    resolved_name = COALESCE(resolved_name, $2::text),
                    latitude = COALESCE(latitude, $3::double precision),
                    longitude = COALESCE(longitude, $4::double precision),
                    timezone = COALESCE(timezone, $5::text),
                    updated_at = CASE
                        WHEN (resolved_name IS NULL AND $2::text IS NOT NULL)
                             OR (latitude IS NULL AND $3::double precision IS NOT NULL)
                             OR (longitude IS NULL AND $4::double precision IS NOT NULL)
                             OR (timezone IS NULL AND $5::text IS NOT NULL)
                        THEN NOW()
                        ELSE updated_at
                    END
                WHERE normalized_name = $1::text
                """,
                normalized_name,
                info.get("full_name"),
                info.get("latitude"),
                info.get("longitude"),
                info.get("timezone"),
            )

    async def _migrate_legacy_daily_temperatures(self, conn: asyncpg.Connection) -> None:
        """Migrate legacy daily_temperatures table to new schema with location_id foreign key.

        Args:
            conn: asyncpg database connection
        """
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
            preapproved = _get_preapproved_location_info(normalized)
            resolved_name = (
                preapproved.get("full_name") if preapproved and preapproved.get("full_name") else original_name
            )
            latitude = preapproved.get("latitude") if preapproved else None
            longitude = preapproved.get("longitude") if preapproved else None
            timezone_name = preapproved.get("timezone") if preapproved else None
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
                VALUES ($1::text, $2::text, $3::text, $4::double precision, $5::double precision, $6::text)
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
                normalized,
                resolved_name,
                latitude,
                longitude,
                timezone_name,
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
                VALUES ($1::bigint, $2::date, $3::double precision, $4::double precision, $5::double precision, $6::jsonb, $7::text, $8::timestamptz)
                """,
                batch,
            )

        await conn.execute("DROP TABLE daily_temperatures_legacy")

    async def fetch(
        self,
        location: str,
        dates: Iterable[date],
    ) -> Dict[date, DailyTemperatureRecord]:
        """Fetch cached temperature records for a location and list of dates.

        Uses location aliases to resolve the canonical location_id, ensuring that
        different location variations (e.g., nearby locations within 25km) share
        the same cached data.

        Args:
            location: Location name string (will be normalized)
            dates: Iterable of date objects to fetch data for

        Returns:
            Dictionary mapping dates to DailyTemperatureRecord objects
        """
        normalized_dates: List[date] = []
        for candidate in dates:
            if isinstance(candidate, datetime):
                normalized_dates.append(candidate.date())
            elif isinstance(candidate, date):
                normalized_dates.append(candidate)
        date_list = sorted(set(normalized_dates))
        if not date_list:
            return {}

        date_texts = [d.isoformat() for d in date_list]
        date_set = set(date_list)
        start_date = date_list[0]
        end_date = date_list[-1]

        pool = await self._ensure_pool()
        if pool is None:
            return {}

        normalized_location = normalize_location_for_cache(location)
        result: Dict[date, DailyTemperatureRecord] = {}

        # Use = ANY() for exact date matching (more efficient than BETWEEN when dates are sparse)
        # For large consecutive ranges, BETWEEN might be slightly faster, but = ANY() handles both well
        query = """
            SELECT day, temp_c, temp_max_c, temp_min_c, payload, source
            FROM daily_temperatures
            WHERE location_id = $1
              AND day = ANY($2::date[])
        """

        logger.debug("Fetch cached records for a location and list of dates")
        try:
            logger.debug("Acquiring connection from pool")
            async with pool.acquire() as conn:
                logger.debug("Connection acquired; fetching id for %s", normalized_location)
                # Resolve location_id through aliases first, then direct lookup
                location_row = await conn.fetchrow(
                    """
                    SELECT location_id as id FROM location_aliases WHERE alias_normalized_name = $1
                    UNION ALL
                    SELECT id FROM locations WHERE normalized_name = $1
                        AND NOT EXISTS (SELECT 1 FROM location_aliases WHERE alias_normalized_name = $1)
                    LIMIT 1
                    """,
                    normalized_location,
                )
                if not location_row:
                    return {}

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "DailyTemperatureStore.fetch executing SQL for %s (location_id=%s) with %d dates: %s",
                        location,
                        location_row["id"],
                        len(date_texts),
                        query.replace("\n", " ").strip(),
                    )
                    logger.debug(
                        "DailyTemperatureStore.fetch dates requested: %s",
                        date_texts,
                    )

                parameters = (location_row["id"], date_list)
                rows = await conn.fetch(query, *parameters)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "DailyTemperatureStore.fetch returned %d rows for %s",
                        len(rows),
                        location,
                    )
            logger.debug("Connection released")
        except Exception as exc:
            if isinstance(exc, asyncpg.PostgresError):
                logger.error(
                    "DailyTemperatureStore.fetch SQL error for %s: sqlstate=%s detail=%s hint=%s context=%s",
                    location,
                    getattr(exc, "sqlstate", None),
                    getattr(exc, "detail", None),
                    getattr(exc, "hint", None),
                    getattr(exc, "context", None),
                )
                logger.error(
                    "DailyTemperatureStore.fetch query: %s | date_range=(%s -> %s) | requested_dates=%s",
                    query.replace("\n", " ").strip(),
                    start_date,
                    end_date,
                    date_texts,
                )
                logger.error("DailyTemperatureStore.fetch parameters: %s", parameters if 'parameters' in locals() else None)
            else:
                logger.warning(
                    "DailyTemperatureStore.fetch failed for %s due to %s. Returning empty result.",
                    location,
                    exc,
                )
            return {}

        for row in rows:
            record_date: date = row["day"]
            if record_date not in date_set:
                continue
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
        """Insert or update a batch of temperature records for a location.

        Automatically resolves the canonical location_id using aliases and nearby
        location matching. Creates new locations and aliases as needed.

        Args:
            location: Location name string (will be normalized)
            records: List of DailyTemperatureRecord objects to store
            metadata: Optional Visual Crossing metadata (coordinates, timezone, etc.)
        """
        if not records:
            return

        pool = await self._ensure_pool()
        if pool is None:
            return

        normalized_location = normalize_location_for_cache(location)
        now_ts = datetime.now(timezone.utc)
        has_temp, has_max, has_min = _calculate_insert_fields(records)

        column_clauses = ["location_id", "day"]
        value_placeholders = ["$1::bigint", "$2::date"]
        update_assignments = []

        param_index = 3  # 1 and 2 are reserved for location_id and day in executemany loop

        if has_temp:
            column_clauses.append("temp_c")
            value_placeholders.append(f"${param_index}::double precision")
            update_assignments.append("temp_c = EXCLUDED.temp_c")
            param_index += 1
        if has_max:
            column_clauses.append("temp_max_c")
            value_placeholders.append(f"${param_index}::double precision")
            update_assignments.append("temp_max_c = EXCLUDED.temp_max_c")
            param_index += 1
        if has_min:
            column_clauses.append("temp_min_c")
            value_placeholders.append(f"${param_index}::double precision")
            update_assignments.append("temp_min_c = EXCLUDED.temp_min_c")
            param_index += 1

        # payload and source are always present
        column_clauses.extend(["payload", "source", "updated_at"])
        value_placeholders.extend(
            [
                f"${param_index}::jsonb",
                f"${param_index + 1}::text",
                f"${param_index + 2}::timestamptz",
            ]
        )
        update_assignments.extend(
            [
                "payload = EXCLUDED.payload",
                "source = EXCLUDED.source",
                "updated_at = EXCLUDED.updated_at",
            ]
        )

        insert_sql = f"""
            INSERT INTO daily_temperatures (
                {", ".join(column_clauses)}
            )
            VALUES ({", ".join(value_placeholders)})
            ON CONFLICT (location_id, day) DO UPDATE SET
                {", ".join(update_assignments)}
        """

        logger.debug(
            "DailyTemperatureStore.upsert SQL (has_temp=%s, has_max=%s, has_min=%s): %s",
            has_temp,
            has_max,
            has_min,
            insert_sql.replace("\n", " ").strip(),
        )

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    location_id = await self._get_or_create_location_id(
                        conn, location, normalized_location, metadata
                    )
                    if location_id is None:
                        return
                    param_rows = [
                        self._build_insert_params(
                            location_id,
                            record,
                            has_temp,
                            has_max,
                            has_min,
                            now_ts,
                        )
                        for record in records
                    ]
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "DailyTemperatureStore.upsert executing SQL for %s with %d rows: %s",
                            location,
                            len(param_rows),
                            insert_sql.replace("\n", " ").strip(),
                        )
                    await conn.executemany(insert_sql, param_rows)
        except Exception as exc:
            if isinstance(exc, asyncpg.PostgresError):
                logger.error(
                    "DailyTemperatureStore.upsert SQL error for %s: sqlstate=%s detail=%s hint=%s context=%s",
                    location,
                    getattr(exc, "sqlstate", None),
                    getattr(exc, "detail", None),
                    getattr(exc, "hint", None),
                    getattr(exc, "context", None),
                )
                if "daily_temperatures" in str(exc):
                    logger.error(
                        "Failed SQL: %s",
                        insert_sql.replace("\n", " ").strip() if "insert_sql" in locals() else "<unknown>",
                    )
            else:
                logger.warning(
                    "DailyTemperatureStore.upsert failed for %s due to %s. Skipping persistence.",
                    location,
                    exc,
                )

    @staticmethod
    def _build_insert_params(
        location_id: int,
        record: "DailyTemperatureRecord",
        has_temp: bool,
        has_max: bool,
        has_min: bool,
        now_ts: datetime,
    ) -> Tuple[Any, ...]:
        """Build parameter tuple for database insert based on available temperature fields.

        Args:
            location_id: Location ID foreign key
            record: Temperature record to insert
            has_temp: Whether temp_c field is present in dataset
            has_max: Whether temp_max_c field is present in dataset
            has_min: Whether temp_min_c field is present in dataset
            now_ts: Timestamp for updated_at field

        Returns:
            Tuple of parameters for SQL insert
        """
        params: List[Any] = [location_id, record.date]
        if has_temp:
            params.append(record.temp_c)
        if has_max:
            params.append(record.temp_max_c)
        if has_min:
            params.append(record.temp_min_c)
        payload_json = json.dumps(record.payload, separators=(",", ":"), sort_keys=True)
        params.extend([payload_json, record.source, now_ts])
        return tuple(params)

    async def _find_nearby_location(
        self,
        conn: asyncpg.Connection,
        latitude: Optional[float],
        longitude: Optional[float],
        max_distance_km: float = 25.0,
    ) -> Optional[asyncpg.Record]:
        """Return closest existing location within max_distance_km km of (lat, lng), or None.

        Uses bounding box pre-filter for performance before calculating exact distance.
        """
        if latitude is None or longitude is None:
            return None

        # Calculate bounding box for fast pre-filtering
        # Approximation: 1 degree latitude ≈ 111 km, longitude varies by latitude
        import math
        lat_delta = max_distance_km / 111.0
        # At equator, 1 degree longitude ≈ 111 km, narrows at higher latitudes
        lon_delta = max_distance_km / (111.0 * abs(math.cos(math.radians(latitude))))

        min_lat = latitude - lat_delta
        max_lat = latitude + lat_delta
        min_lon = longitude - lon_delta
        max_lon = longitude + lon_delta

        row = await conn.fetchrow(
            """
            SELECT
                id,
                original_name,
                normalized_name,
                resolved_name,
                latitude,
                longitude,
                timezone,
                (
                    6371.0 * acos(
                        cos(radians($1)) * cos(radians(latitude)) *
                        cos(radians(longitude) - radians($2)) +
                        sin(radians($1)) * sin(radians(latitude))
                    )
                ) AS distance_km
            FROM locations
            WHERE latitude IS NOT NULL
              AND longitude IS NOT NULL
              AND latitude BETWEEN $3 AND $4
              AND longitude BETWEEN $5 AND $6
            ORDER BY distance_km
            LIMIT 1
            """,
            latitude,
            longitude,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )
        if not row:
            return None

        distance = row["distance_km"]
        if distance is None:
            return None

        if float(distance) <= max_distance_km:
            return row

        return None

    async def _get_or_create_location_id(
        self,
        conn: asyncpg.Connection,
        original_name: str,
        normalized_name: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """Resolve a location to a canonical locations.id.

        Resolution order:
        1. Existing alias in location_aliases.
        2. Existing locations.normalized_name.
        3. Preapproved location metadata.
        4. Nearby existing location based on Visual Crossing coordinates.
        5. Create a new locations row.
        """

        # 1. Alias lookup
        row = await conn.fetchrow(
            """
            SELECT location_id
            FROM location_aliases
            WHERE alias_normalized_name = $1
            """,
            normalized_name,
        )
        if row:
            return int(row["location_id"])

        # 2. Direct locations match
        row = await conn.fetchrow(
            """
            SELECT id
            FROM locations
            WHERE normalized_name = $1
            """,
            normalized_name,
        )
        if row:
            location_id = int(row["id"])
            await conn.execute(
                """
                INSERT INTO location_aliases (alias_normalized_name, location_id)
                VALUES ($1, $2)
                ON CONFLICT (alias_normalized_name) DO NOTHING
                """,
                normalized_name,
                location_id,
            )
            return location_id

        # 3. Extract Visual Crossing metadata
        resolved_name = self._extract_metadata_value(
            metadata, ["resolvedAddress", "resolved_address"]
        )
        latitude = self._extract_numeric_metadata(
            metadata, ["latitude", "lat"]
        )
        longitude = self._extract_numeric_metadata(
            metadata, ["longitude", "lon", "lng"]
        )
        timezone_name = self._extract_metadata_value(
            metadata, ["timezone", "tz"]
        )

        # Apply preapproved info if available (do not remove existing logic)
        preapproved = _get_preapproved_location_info(normalized_name)
        if preapproved:
            if not resolved_name:
                resolved_name = preapproved.get("full_name")
            if latitude is None:
                latitude = preapproved.get("latitude")
            if longitude is None:
                longitude = preapproved.get("longitude")
            if not timezone_name:
                timezone_name = preapproved.get("timezone")

        # 4. Try snapping to nearby existing location
        if latitude is not None and longitude is not None:
            nearby = await self._find_nearby_location(conn, latitude, longitude)
            if nearby:
                location_id = int(nearby["id"])
                await conn.execute(
                    """
                    INSERT INTO location_aliases (alias_normalized_name, location_id)
                    VALUES ($1, $2)
                    ON CONFLICT (alias_normalized_name) DO NOTHING
                    """,
                    normalized_name,
                    location_id,
                )
                return location_id

        # 5. Fall back to creating a new canonical location
        if not resolved_name:
            resolved_name = original_name

        original_to_store = original_name
        if resolved_name and "," not in original_name:
            original_to_store = resolved_name

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
            VALUES (
                $1::text,
                $2::text,
                $3::text,
                $4::double precision,
                $5::double precision,
                $6::text
            )
            RETURNING id
            """,
            original_to_store,
            normalized_name,
            resolved_name,
            latitude,
            longitude,
            timezone_name,
        )
        if not row:
            return None

        location_id = int(row["id"])

        await conn.execute(
            """
            INSERT INTO location_aliases (alias_normalized_name, location_id)
            VALUES ($1, $2)
            ON CONFLICT (alias_normalized_name) DO NOTHING
            """,
            normalized_name,
            location_id,
        )

        return location_id

    @staticmethod
    def _extract_metadata_value(
        metadata: Optional[Dict[str, Any]],
        keys: List[str],
    ) -> Optional[str]:
        """Extract a string value from metadata by trying multiple key names.

        Args:
            metadata: Metadata dictionary (e.g., from Visual Crossing API)
            keys: List of possible key names to try in order

        Returns:
            First non-empty string value found, or None
        """
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
        """Extract a numeric value from metadata by trying multiple key names.

        Args:
            metadata: Metadata dictionary (e.g., from Visual Crossing API)
            keys: List of possible key names to try in order

        Returns:
            First numeric value found (as float), or None
        """
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

