#!/usr/bin/env python3
"""Re-fetch stored daily temperatures from Open-Meteo on one pinned model (P1-173, #123).

Stored history is a mix of sources, which distorts long-term trends:

  - Rows loaded before the 2026-06-03 move to Open-Meteo came from Visual Crossing.
    Manchester, Dublin and London are essentially all Visual Crossing, and their
    series differ from ERA5 by amounts that drift or step over time.
  - Rows loaded from Open-Meteo before the archive model was pinned (see
    OPEN_METEO_ARCHIVE_MODEL) used its default `best_match`, which switches model on
    2017-01-01 and puts a location-dependent step into the series (about -0.9 degC in
    Hong Kong).

Reads never re-fetch days that are already stored, so neither problem corrects itself.
This script re-fetches each location's whole stored range from the pinned archive model
and upserts it over the stored rows, tagging them with a distinct `source`
(`open-meteo:<model>`) so backfilled rows can be told apart afterwards.

It stops at today minus 8 days. Newer days come from the forecast endpoint, which cannot
serve the pinned model, and the daily top-up keeps refreshing them.

Modes (nothing is written unless --execute is given):

  (default)   Read-only report from the database: rows per location, how many predate
              the migration, how many are not yet on the pinned source, and an estimate
              of the Open-Meteo calls a full backfill would use. No network access.
  --compare   Also fetch each selected location from Open-Meteo and show the 50-year
              trend before and after. Read-only, but uses Open-Meteo quota.
  --execute   Fetch, back up the rows about to change to --backup-dir, then upsert.

Open-Meteo counts a request by its length, and a 51-year request is very large: roughly
2,600 weighted calls per location by the pricing page's formula, against free-tier limits
of 5,000/hour and 10,000/day. A full run needs a paid plan (OPEN_METEO_API_KEY plus the
customer-* URLs, as for the API) or many days. The script stops at the first incomplete
fetch and is resumable: locations already fully on the pinned source are skipped.

Requires TEMPHIST_PG_DSN (or DATABASE_URL). The production DSN is on Railway's private
network, so run it from inside Railway. Try it against staging first.

Usage:
    python -m scripts.backfill_open_meteo                               # report only
    python -m scripts.backfill_open_meteo --compare --ids 1 2 9 18      # preview 4 locations
    python -m scripts.backfill_open_meteo --execute --backup-dir ./bk --ids 1 2 9 18
    python -m scripts.backfill_open_meteo --execute --backup-dir ./bk --limit 3

To restore a location, load its CSV from the backup directory into a temporary table and
upsert it back over daily_temperatures on (location_id, day).
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.open_meteo_client import _FORECAST_PAST_DAYS  # noqa: E402
from utils.temperature import calculate_trend_slope  # noqa: E402

LEGACY_CUTOFF = "2026-06-03"  # the Open-Meteo migration commit (02b4952)
MIN_COVERAGE = 0.95  # fetched days / expected days below this means the fetch failed
MIN_WINDOW_DAYS = 300  # of 365, for a rolling year to count towards the trend
CHANGED_THRESHOLD_C = 0.05
TREND_YEARS = 51


class FetchIncomplete(RuntimeError):
    """Open-Meteo returned too few days (fetch_days swallows errors and rate limits)."""


@dataclass(frozen=True)
class LocationSummary:
    id: int
    name: str
    latitude: Optional[float]
    longitude: Optional[float]
    total_rows: int
    first_day: date
    last_day: date
    legacy_rows: int
    pending_rows: int
    future_rows: int
    first_pending_day: Optional[date] = None

    @property
    def refresh_from(self) -> date:
        """First day to re-fetch: the oldest row not on the target source, else the whole history."""
        return self.first_pending_day or self.first_day


def archive_end_date(today: date) -> date:
    """Last day served by the archive endpoint (the forecast endpoint takes over after)."""
    return today - timedelta(days=_FORECAST_PAST_DAYS + 1)


def estimate_open_meteo_calls(days: int) -> int:
    """Rough weighted-call cost of one request spanning `days`.

    From Open-Meteo's pricing page: up to two weeks is 1 call and 4 weeks is 3, which
    fits weeks - 1. An estimate only; check the pricing page before a large run.
    """
    return max(1, round(days / 7) - 1)


def rolling_year_means(daily: Dict[date, float], anchor: date, years: int = TREND_YEARS) -> List[Dict[str, float]]:
    """Mean temperature of the 365 days ending on anchor's month/day, one point per year.

    Mirrors the yearly records endpoint, so the resulting trend is the one the API shows.
    """
    points: List[Dict[str, float]] = []
    for year in range(anchor.year - years + 1, anchor.year + 1):
        try:
            end = date(year, anchor.month, anchor.day)
        except ValueError:  # 29 February in a non-leap year
            end = date(year, 2, 28)
        values = [daily[d] for d in (end - timedelta(days=i) for i in range(365)) if d in daily]
        if len(values) >= MIN_WINDOW_DAYS:
            points.append({"x": year, "y": sum(values) / len(values)})
    return points


def build_rows(location_id: int, days: Sequence[dict], source: str) -> List[tuple]:
    """Insert parameters for fetched days, skipping any without a mean temperature."""
    rows = []
    for day in days:
        if day.get("temp") is None:
            continue
        payload = {
            "datetime": day["datetime"],
            "temp": day["temp"],
            "tempmax": day.get("tempmax"),
            "tempmin": day.get("tempmin"),
        }
        rows.append(
            (
                location_id,
                date.fromisoformat(day["datetime"]),
                day["temp"],
                day.get("tempmax"),
                day.get("tempmin"),
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                source,
            )
        )
    return rows


def check_coverage(fetched_days: int, start: date, end: date) -> None:
    expected = (end - start).days + 1
    if fetched_days < MIN_COVERAGE * expected:
        raise FetchIncomplete(
            f"got {fetched_days} of {expected} days; Open-Meteo probably rate-limited or failed the request"
        )


def compare_series(stored: Dict[date, float], fetched: Dict[date, float], anchor: date) -> dict:
    """Trend and daily differences before and after replacing stored days with fetched ones."""
    before = calculate_trend_slope(rolling_year_means(stored, anchor))
    after = calculate_trend_slope(rolling_year_means({**stored, **fetched}, anchor))
    diffs = [abs(fetched[d] - stored[d]) for d in fetched if d in stored]
    return {
        "trend_before": before[0],
        "trend_after": after[0],
        "r2_before": before[1],
        "r2_after": after[1],
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "days_changed": sum(1 for x in diffs if x > CHANGED_THRESHOLD_C),
        "days_compared": len(diffs),
    }


SUMMARY_SQL = """
    SELECT l.id, l.normalized_name, l.latitude, l.longitude,
           count(*) AS total_rows,
           min(d.day) AS first_day,
           max(d.day) AS last_day,
           count(*) FILTER (WHERE d.updated_at < $1::timestamptz) AS legacy_rows,
           count(*) FILTER (WHERE d.source <> $2::text AND d.day <= $3::date) AS pending_rows,
           count(*) FILTER (WHERE d.day > $4::date) AS future_rows,
           min(d.day) FILTER (WHERE d.source <> $2::text AND d.day <= $3::date) AS first_pending_day
    FROM locations l
    JOIN daily_temperatures d ON d.location_id = l.id
    WHERE ($5::bigint[] IS NULL OR l.id = ANY($5::bigint[]))
    GROUP BY l.id
    ORDER BY l.id
"""

UPSERT_SQL = """
    INSERT INTO daily_temperatures (
        location_id, day, temp_c, temp_max_c, temp_min_c, payload, source, updated_at
    )
    VALUES (
        $1::bigint, $2::date, $3::double precision, $4::double precision, $5::double precision,
        $6::jsonb, $7::text, NOW()
    )
    ON CONFLICT (location_id, day) DO UPDATE SET
        temp_c = EXCLUDED.temp_c,
        temp_max_c = EXCLUDED.temp_max_c,
        temp_min_c = EXCLUDED.temp_min_c,
        payload = EXCLUDED.payload,
        source = EXCLUDED.source,
        updated_at = EXCLUDED.updated_at
"""

BACKUP_COLUMNS = ["location_id", "day", "temp_c", "temp_max_c", "temp_min_c", "payload", "source", "updated_at"]


async def load_summaries(
    conn: asyncpg.Connection, source: str, today: date, end: date, ids: Optional[List[int]]
) -> List[LocationSummary]:
    cutoff = datetime.fromisoformat(LEGACY_CUTOFF).replace(tzinfo=timezone.utc)
    rows = await conn.fetch(SUMMARY_SQL, cutoff, source, end, today, ids)
    return [
        LocationSummary(
            id=r["id"],
            name=r["normalized_name"],
            latitude=r["latitude"],
            longitude=r["longitude"],
            total_rows=r["total_rows"],
            first_day=r["first_day"],
            last_day=r["last_day"],
            legacy_rows=r["legacy_rows"],
            pending_rows=r["pending_rows"],
            future_rows=r["future_rows"],
            first_pending_day=r["first_pending_day"],
        )
        for r in rows
    ]


async def fetch_series(loc: LocationSummary, end: date) -> List[dict]:
    from utils.open_meteo_client import fetch_days

    start = loc.refresh_from
    days, _meta = await fetch_days(loc.latitude, loc.longitude, start, end)
    usable = [d for d in days if d.get("temp") is not None]
    check_coverage(len(usable), start, end)
    return usable


def backup_path(backup_dir: Path, location_id: int, name: str) -> Path:
    """Backup file for a location, guaranteed to sit directly inside backup_dir.

    The name comes from the database, so it is reduced to a safe slug rather than trusted.
    """
    root = backup_dir.resolve()
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:80]
    path = (root / f"location_{int(location_id)}_{slug}.csv").resolve()
    if path.parent != root:
        raise ValueError(f"backup path escapes {root}: {path}")
    return path


def _write_csv(path: Path, rows: Sequence[asyncpg.Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BACKUP_COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in BACKUP_COLUMNS])


async def write_backup(conn: asyncpg.Connection, loc: LocationSummary, end: date, backup_dir: Path) -> Path:
    rows = await conn.fetch(
        """
        SELECT location_id, day, temp_c, temp_max_c, temp_min_c, payload::text AS payload, source, updated_at
        FROM daily_temperatures
        WHERE location_id = $1 AND day BETWEEN $2 AND $3
        ORDER BY day
        """,
        loc.id,
        loc.refresh_from,
        end,
    )
    path = backup_path(backup_dir, loc.id, loc.name)
    _write_csv(path, rows)
    return path


def describe(loc: LocationSummary) -> str:
    return (
        f"[{loc.id}] {loc.name} ({loc.latitude}, {loc.longitude}) {loc.first_day}..{loc.last_day} "
        f"rows={loc.total_rows} pre-migration={loc.legacy_rows} not-pinned={loc.pending_rows} future-dated={loc.future_rows} "
        f"refresh-from={loc.refresh_from}"
    )


async def process_location(
    conn: asyncpg.Connection,
    loc: LocationSummary,
    args: argparse.Namespace,
    today: date,
    end: date,
    source: str,
) -> None:
    print(describe(loc))
    fetched = await fetch_series(loc, end)
    fetched_by_day = {date.fromisoformat(d["datetime"]): d["temp"] for d in fetched}

    stored_rows = await conn.fetch(
        "SELECT day, temp_c FROM daily_temperatures WHERE location_id = $1 AND day BETWEEN $2 AND $3 AND temp_c IS NOT NULL",
        loc.id,
        loc.first_day,
        end,
    )
    stored = {r["day"]: r["temp_c"] for r in stored_rows}
    result = compare_series(stored, fetched_by_day, today)
    print(
        f"    trend {result['trend_before']:+.2f} -> {result['trend_after']:+.2f} degC/decade "
        f"(r2 {result['r2_before']} -> {result['r2_after']}); mean |diff| {result['mean_abs_diff']:.2f} degC; "
        f"{result['days_changed']}/{result['days_compared']} days differ by more than {CHANGED_THRESHOLD_C} degC"
    )

    if not args.execute:
        return

    backup = await write_backup(conn, loc, end, Path(args.backup_dir))
    rows = build_rows(loc.id, fetched, source)
    async with conn.transaction():
        await conn.executemany(UPSERT_SQL, rows)
    print(f"    backed up to {backup}; upserted {len(rows)} rows as source={source!r}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--execute", action="store_true", help="fetch and write (default: read-only)")
    parser.add_argument("--compare", action="store_true", help="fetch and show trends before/after, no writes")
    parser.add_argument("--ids", type=int, nargs="+", help="only these locations.id values")
    parser.add_argument("--limit", type=int, help="process at most this many locations that need work")
    parser.add_argument("--delay", type=float, default=10.0, help="seconds between Open-Meteo fetches")
    parser.add_argument("--force", action="store_true", help="also process locations already on the pinned source")
    parser.add_argument(
        "--backup-dir", help="where to write CSV backups of rows about to change (required with --execute)"
    )
    args = parser.parse_args(argv)
    if args.execute and not args.backup_dir:
        parser.error("--execute requires --backup-dir")
    return args


def _mode_label(args: argparse.Namespace) -> str:
    if args.execute:
        return "EXECUTE"
    if args.compare:
        return "COMPARE (read-only)"
    return "REPORT (read-only)"


def select_work(summaries: Sequence[LocationSummary], args: argparse.Namespace) -> Tuple[List[LocationSummary], int]:
    """Locations still needing work (honouring --limit) and how many were already on the pinned source."""
    todo = [s for s in summaries if args.force or s.pending_rows > 0]
    skipped = len(summaries) - len(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    return todo, skipped


def print_report(todo: Sequence[LocationSummary], skipped: int, end: date, source: str) -> None:
    total = 0
    for loc in todo:
        calls = estimate_open_meteo_calls((end - loc.refresh_from).days + 1)
        total += calls
        print(f"{describe(loc)} est-calls={calls}")
    print(
        f"\n{len(todo)} locations need work ({skipped} already on {source!r}); "
        f"about {total:,} weighted Open-Meteo calls (estimate)."
    )


async def process_all(
    conn: asyncpg.Connection,
    todo: Sequence[LocationSummary],
    skipped: int,
    args: argparse.Namespace,
    today: date,
    end: date,
    source: str,
) -> int:
    """Fetch (and with --execute, write) each location. Returns the process exit code."""
    done = 0
    for loc in todo:
        if loc.latitude is None or loc.longitude is None:
            print(f"[{loc.id}] {loc.name}: no coordinates, skipped")
            continue
        if done:
            await asyncio.sleep(args.delay)
        try:
            await process_location(conn, loc, args, today, end, source)
        except FetchIncomplete as exc:
            print(f"[{loc.id}] {loc.name}: {exc}. Stopping; re-run later to resume.", file=sys.stderr)
            return 2
        done += 1
    print(f"\nProcessed {done} locations ({skipped} already on {source!r}).")
    if args.execute and done:
        print(
            "Cached yearly/trend responses are now stale: clear them for the locations above "
            "(DELETE /cache/invalidate/location/{location}, routers/cache.py)."
        )
    return 0


def configured_model() -> str:
    from config import OPEN_METEO_ARCHIVE_MODEL

    return OPEN_METEO_ARCHIVE_MODEL


async def run(args: argparse.Namespace) -> int:
    dsn = os.getenv("TEMPHIST_PG_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        print("TEMPHIST_PG_DSN (or DATABASE_URL) is not set", file=sys.stderr)
        return 1

    needs_fetch = args.execute or args.compare
    model = configured_model()
    if needs_fetch and not model:
        print("OPEN_METEO_ARCHIVE_MODEL is empty; refusing to fetch unpinned data", file=sys.stderr)
        return 1
    source = f"open-meteo:{model or 'unpinned'}"

    today = date.today()
    end = archive_end_date(today)
    print(f"{_mode_label(args)}; model={model or '(unset)'}; range ends {end}; target source={source!r}")

    conn = await asyncpg.connect(dsn)
    try:
        summaries = await load_summaries(conn, source, today, end, args.ids)
        todo, skipped = select_work(summaries, args)
        if not needs_fetch:
            print_report(todo, skipped, end, source)
            return 0
        return await process_all(conn, todo, skipped, args, today, end, source)
    finally:
        await conn.close()
        if needs_fetch:
            from utils.open_meteo_client import close_client

            await close_client()


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
