#!/usr/bin/env python3
"""
compare_apis.py — Benchmark Visual Crossing vs Open-Meteo response times.

Replicates the exact HTTP call patterns and processing steps used by the
v1 records endpoint, testing three variants across four period types.

Usage (from the api/ directory):
    python benchmarks/compare_apis.py
    python benchmarks/compare_apis.py --locations 3 --runs 2
    python benchmarks/compare_apis.py --skip-vc --csv results.csv
    python benchmarks/compare_apis.py --periods daily weekly

Requires:
    VISUAL_CROSSING_API_KEY  in env or api/.env  (for VC tests)
    MAPBOX_TOKEN             in env or api/.env  (for OM-cold geocoding tests)

Variants tested:
    VC       — Visual Crossing (current production). Accepts location string;
               VC geocodes internally. Returns °F; script converts to °C.
    OM-warm  — Open-Meteo with pre-loaded lat/lon from preapproved_locations.json.
               Represents steady-state (location already seen by the system).
    OM-cold  — Open-Meteo with a Mapbox geocoding call first.
               Represents first-ever request for a location.
    OM-batch — Open-Meteo single archive call spanning the full 50-year range,
               filtered in-process. Daily period only; tests latency/bandwidth tradeoff.

URL patterns matched to production code:
    VC daily   → timeline/{location}/{date}?unitGroup=us&include=days&key=...
                 (matches weather_data.build_visual_crossing_url, no elements filter)
    VC range   → timeline/{location}/{start}/{end}?unitGroup=us&include=days
                 &elements=datetime,temp,tempmax,tempmin&contentType=json&key=...
                 (matches visual_crossing_timeline._build_timeline_url)
    OM archive → archive-api.open-meteo.com/v1/archive?...&timezone=auto
    OM forecast→ api.open-meteo.com/v1/forecast?...&past_days=7&forecast_days=1
"""

import argparse
import asyncio
import csv as csv_module
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp


# ── Rate limiter ──────────────────────────────────────────────────────────────
class _RateLimiter:
    """Token-bucket rate limiter shared across all OM requests."""

    def __init__(self, rate_per_second: float):
        self._interval = 1.0 / rate_per_second
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.perf_counter()
            wait = self._interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.perf_counter()


# 9.5 req/s = 570/min — safely under Open-Meteo's free-tier 600/min cap.
# Applied only to OM archive/forecast requests (not Mapbox geocoding or VC).
_OM_RATE_LIMITER = _RateLimiter(9.5)

# ── Environment setup ─────────────────────────────────────────────────────────
_API_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _API_DIR / ".env"

if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=_ENV_FILE, override=False)
    except ImportError:
        pass  # dotenv not available; env vars must be set manually

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_FILE = _API_DIR / "data" / "preapproved_locations.json"

VC_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
MAPBOX_BASE = "https://api.mapbox.com/geocoding/v5/mapbox.places"

# Dates older than this many days use the OM archive endpoint; newer use forecast.
# past_days=7 on the forecast endpoint covers dates from today-7 onwards.
FORECAST_PAST_DAYS = 7

ALL_PERIODS = ["daily", "weekly", "monthly", "yearly"]

# Window sizes matching production WINDOW_DAYS in v1_records.py
WINDOW_DAYS = {"daily": 1, "weekly": 7, "monthly": 31, "yearly": 365}

# 51 data points: current year + 50 prior years, matching get_year_range()
YEARS_BACK = 50

# ── Fresh test locations (not in preapproved_locations.json) ──────────────────
FRESH_LOCATIONS = [
    {"name": "Paris, France", "query": "Paris, France"},
    {"name": "Berlin, Germany", "query": "Berlin, Germany"},
    {"name": "Tokyo, Japan", "query": "Tokyo, Japan"},
    {"name": "Rio de Janeiro, Brazil", "query": "Rio de Janeiro, Brazil"},
    {"name": "Mumbai, India", "query": "Mumbai, India"},
    {"name": "Cairo, Egypt", "query": "Cairo, Egypt"},
    {"name": "Buenos Aires, Argentina", "query": "Buenos Aires, Argentina"},
    {"name": "Nairobi, Kenya", "query": "Nairobi, Kenya"},
    {"name": "Seoul, South Korea", "query": "Seoul, South Korea"},
    {"name": "Mexico City, Mexico", "query": "Mexico City, Mexico"},
]


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    variant: str  # "VC" | "OM-warm" | "OM-cold" | "OM-batch"
    period: str  # "daily" | "weekly" | "monthly" | "yearly"
    location_name: str
    location_set: str  # "warm" | "fresh"
    run: int  # 0-based run index
    total_time_s: float
    geocode_ms: float  # Mapbox call duration (0 for VC and OM-warm)
    fetch_time_s: float  # Wall-clock time for all HTTP requests (incl. OM pacing)
    avg_http_ms: float  # Mean per-request HTTP round-trip (excl. pacing; fair VC comparison)
    process_ms: float  # temp conversion / field mapping after fetch
    request_count: int  # actual HTTP requests made
    total_bytes: int  # compressed bytes received
    data_points: int  # non-null temperature values in final result
    error: Optional[str] = None

    @property
    def avg_req_ms(self) -> float:
        """Pure HTTP time per request — use this to compare VC vs OM fairly."""
        return self.avg_http_ms

    @property
    def data_kb(self) -> float:
        return self.total_bytes / 1024


# ── URL builders ──────────────────────────────────────────────────────────────
def _vc_daily_url(location: str, target: date, api_key: str) -> str:
    """Single-date VC URL — matches weather_data.build_visual_crossing_url(remote=False).
    No elements filter: returns all ~50 fields per day."""
    encoded = quote(location, safe="")
    return f"{VC_BASE}/{encoded}/{target.isoformat()}?unitGroup=us&include=days&key={api_key}"


def _vc_range_url(location: str, start: date, end: date, api_key: str) -> str:
    """Date-range VC URL — matches visual_crossing_timeline._build_timeline_url.
    Uses elements filter: returns only the 4 fields we need."""
    encoded = quote(location, safe="")
    return (
        f"{VC_BASE}/{encoded}/{start.isoformat()}/{end.isoformat()}"
        f"?unitGroup=us&include=days"
        f"&elements=datetime,temp,tempmax,tempmin"
        f"&contentType=json&key={api_key}"
    )


def _om_archive_url(lat: float, lon: float, start: date, end: date) -> str:
    return (
        f"{OM_ARCHIVE}?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        f"&timezone=auto"
    )


def _om_forecast_url(lat: float, lon: float) -> str:
    return (
        f"{OM_FORECAST}?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min"
        f"&current=temperature_2m&timezone=auto"
        f"&past_days={FORECAST_PAST_DAYS}&forecast_days=1"
    )


def _om_requests_for_range(lat: float, lon: float, start: date, end: date) -> List[Tuple[str, date, date]]:
    """
    Return (url, filter_start, filter_end) tuples covering [start, end].
    Splits at the archive/forecast boundary for windows that straddle it.
    All historical years → 1 archive request.
    Current year's short windows → 1 forecast request.
    Current year's long windows (monthly/yearly) → 2 requests.
    """
    boundary = date.today() - timedelta(days=FORECAST_PAST_DAYS)

    if end < boundary:
        # Entirely archive
        return [(_om_archive_url(lat, lon, start, end), start, end)]

    if start >= boundary:
        # Entirely within forecast window
        return [(_om_forecast_url(lat, lon), start, end)]

    # Straddles boundary — split
    archive_end = boundary - timedelta(days=1)
    return [
        (_om_archive_url(lat, lon, start, archive_end), start, archive_end),
        (_om_forecast_url(lat, lon), boundary, end),
    ]


def _mapbox_url(query: str, token: str) -> str:
    encoded = quote(query, safe="")
    return f"{MAPBOX_BASE}/{encoded}.json?access_token={token}&limit=1&types=place,locality,district,region"


# ── Temperature processing (matching production code) ────────────────────────
def _f_to_c(value) -> Optional[float]:
    """Convert °F to °C — matches weather_data._f_to_c."""
    if value is None:
        return None
    return round((float(value) - 32) * 5 / 9, 2)


def _process_vc_days(days: list) -> Tuple[list, int]:
    """
    Extract and convert VC day records.
    Returns (processed_days, conversion_count).
    Matches the F→C conversion in visual_crossing_timeline._convert_days_to_celsius.
    """
    out = []
    conversions = 0
    for day in days:
        d = {
            "datetime": day.get("datetime"),
            "temp": _f_to_c(day.get("temp")),
            "tempmax": _f_to_c(day.get("tempmax")),
            "tempmin": _f_to_c(day.get("tempmin")),
        }
        conversions += sum(1 for v in (day.get("temp"), day.get("tempmax"), day.get("tempmin")) if v is not None)
        out.append(d)
    return out, conversions


def _process_om_response(payload: dict, filter_start: date, filter_end: date) -> list:
    """
    Normalise OM array-based response to [{datetime, temp, tempmax, tempmin}]
    and filter to the requested date window.
    """
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    means = daily.get("temperature_2m_mean", [])
    maxs = daily.get("temperature_2m_max", [])
    mins = daily.get("temperature_2m_min", [])

    start_s = filter_start.isoformat()
    end_s = filter_end.isoformat()

    return [
        {"datetime": t, "temp": m, "tempmax": mx, "tempmin": mn}
        for t, m, mx, mn in zip(times, means, maxs, mins)
        if t and start_s <= t <= end_s
    ]


# ── Date utilities ────────────────────────────────────────────────────────────
def _resolve_anchor(year: int, month: int, day: int) -> Optional[date]:
    """Resolve date, adjusting Feb 29 → Feb 28 in non-leap years."""
    try:
        return date(year, month, day)
    except ValueError:
        if month == 2 and day == 29:
            try:
                return date(year, 2, 28)
            except ValueError:
                pass
    return None


def _year_ranges(years: List[int], month: int, day: int, period: str) -> List[Tuple[int, date, date]]:
    """Return [(year, window_start, anchor_date)] for the given period."""
    window = WINDOW_DAYS[period]
    result = []
    for year in years:
        anchor = _resolve_anchor(year, month, day)
        if anchor is None:
            continue
        start = anchor - timedelta(days=window - 1)
        result.append((year, start, anchor))
    return result


# ── Core async fetch: Visual Crossing ────────────────────────────────────────
async def _fetch_vc(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    period: str,
    location_str: str,
    ranges: List[Tuple[int, date, date]],
    api_key: str,
) -> Tuple[float, float, int, int, list]:
    """
    Make all VC requests with concurrency control, then process responses.
    Returns (fetch_time_s, process_ms, request_count, total_bytes, days).
    """
    raw_days_by_year: Dict[int, list] = {}
    total_bytes = 0

    async def fetch_one(year: int, start: date, end: date) -> None:
        nonlocal total_bytes
        if period == "daily":
            url = _vc_daily_url(location_str, end, api_key)
        else:
            url = _vc_range_url(location_str, start, end, api_key)
        async with sem:
            try:
                async with session.get(url, headers={"Accept-Encoding": "gzip"}) as resp:
                    raw = await resp.read()
                    total_bytes += len(raw)
                    if resp.status != 200:
                        return
                    payload = json.loads(raw)
                    days = payload.get("days")
                    if isinstance(days, list) and days:
                        raw_days_by_year[year] = days
            except Exception:
                pass

    t_fetch = time.perf_counter()
    await asyncio.gather(*[fetch_one(y, s, e) for y, s, e in ranges])
    fetch_time = time.perf_counter() - t_fetch

    n = len(ranges)
    avg_http = (fetch_time / n * 1000) if n else 0.0  # VC has no pacing; wall ≈ HTTP time

    t_proc = time.perf_counter()
    all_days: list = []
    for year in sorted(raw_days_by_year):
        converted, _ = _process_vc_days(raw_days_by_year[year])
        all_days.extend(converted)
    process_ms = (time.perf_counter() - t_proc) * 1000

    return fetch_time, avg_http, process_ms, len(ranges), total_bytes, all_days


# ── Core async fetch: Open-Meteo ─────────────────────────────────────────────
async def _fetch_om(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    lat: float,
    lon: float,
    ranges: List[Tuple[int, date, date]],
) -> Tuple[float, float, int, int, list]:
    """
    Make all OM requests with concurrency control, then normalise responses.
    Handles archive/forecast boundary splitting automatically.
    Returns (fetch_time_s, process_ms, request_count, total_bytes, days).
    """
    # Build all (year, url, filter_start, filter_end) tasks
    tasks: List[Tuple[int, str, date, date]] = []
    for year, start, end in ranges:
        for url, fs, fe in _om_requests_for_range(lat, lon, start, end):
            tasks.append((year, url, fs, fe))

    raw_days_by_year: Dict[int, list] = {}
    total_bytes = 0
    actual_requests = 0
    http_times: List[float] = []  # per-request HTTP round-trip (excl. pacing)
    http_lock = asyncio.Lock()

    async def fetch_one(year: int, url: str, fs: date, fe: date) -> None:
        nonlocal total_bytes, actual_requests
        retry_wait = 0.0
        for attempt in range(4):  # up to 3 retries on 429
            if retry_wait > 0:
                await asyncio.sleep(retry_wait)
            await _OM_RATE_LIMITER.acquire()  # global 570/min cap; wait excluded from HTTP metric
            async with sem:
                try:
                    t_http = time.perf_counter()
                    async with session.get(url, headers={"Accept-Encoding": "gzip"}) as resp:
                        raw = await resp.read()
                    http_elapsed = time.perf_counter() - t_http
                    total_bytes += len(raw)
                    actual_requests += 1
                    async with http_lock:
                        http_times.append(http_elapsed)
                    if resp.status == 429:
                        retry_wait = float(resp.headers.get("Retry-After", 2.0 * (attempt + 1)))
                        continue  # release sem, wait, retry
                    if resp.status != 200:
                        return
                    payload = json.loads(raw)
                    err = payload.get("error")
                    if err:
                        reason = str(payload.get("reason", "")).lower()
                        if "rate" in reason or "limit" in reason:
                            retry_wait = 2.0 * (attempt + 1)
                            continue  # rate-limit in JSON body, retry
                        return
                    days = _process_om_response(payload, fs, fe)
                    raw_days_by_year.setdefault(year, []).extend(days)
                    return
                except Exception:
                    return

    t_fetch = time.perf_counter()
    await asyncio.gather(*[fetch_one(y, u, fs, fe) for y, u, fs, fe in tasks])
    fetch_time = time.perf_counter() - t_fetch

    avg_http = (sum(http_times) / len(http_times) * 1000) if http_times else 0.0

    t_proc = time.perf_counter()
    all_days: list = []
    for year in sorted(raw_days_by_year):
        # Sort within each year's window to maintain chronological order
        year_days = sorted(raw_days_by_year[year], key=lambda d: d["datetime"])
        all_days.extend(year_days)
    process_ms = (time.perf_counter() - t_proc) * 1000

    return fetch_time, avg_http, process_ms, actual_requests, total_bytes, all_days


# ── Core async fetch: OM batched (daily, single call) ────────────────────────
async def _fetch_om_batch(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
    month: int,
    day: int,
    years: List[int],
) -> Tuple[float, float, int, int, list]:
    """
    Single OM archive call spanning the full year range; filter in-process.
    Compares against 51 individual calls (OM-warm daily) to test the
    latency-vs-bandwidth tradeoff.
    Returns (fetch_time_s, process_ms, request_count, total_bytes, days).
    """
    # Cap end_date at yesterday to stay within archive coverage
    yesterday = date.today() - timedelta(days=1)
    start = _resolve_anchor(min(years), month, day) or date(min(years), 1, 1)
    end = min(_resolve_anchor(max(years), month, day) or yesterday, yesterday)

    url = _om_archive_url(lat, lon, start, end)

    total_bytes = 0
    raw_payload: dict = {}

    await _OM_RATE_LIMITER.acquire()
    t_fetch = time.perf_counter()
    try:
        async with session.get(url, headers={"Accept-Encoding": "gzip"}) as resp:
            raw = await resp.read()
            total_bytes = len(raw)
            if resp.status == 200:
                raw_payload = json.loads(raw)
    except Exception:
        pass
    fetch_time = time.perf_counter() - t_fetch

    # Processing: normalise all rows, then filter to the target MM-DD
    t_proc = time.perf_counter()
    target_suffix = f"{month:02d}-{day:02d}"
    all_rows = _process_om_response(raw_payload, start, end)
    filtered = [d for d in all_rows if (d.get("datetime") or "")[-5:] == target_suffix]
    process_ms = (time.perf_counter() - t_proc) * 1000

    return fetch_time, process_ms, 1, total_bytes, filtered


# ── Mapbox geocoding ──────────────────────────────────────────────────────────
async def _geocode(
    session: aiohttp.ClientSession,
    query: str,
    token: str,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Geocode a location string via Mapbox.
    Returns (geocode_ms, latitude, longitude) — or (ms, None, None) on failure.
    """
    url = _mapbox_url(query, token)
    t0 = time.perf_counter()
    try:
        async with session.get(url) as resp:
            raw = await resp.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if resp.status != 200:
                return elapsed_ms, None, None
            payload = json.loads(raw)
            features = payload.get("features", [])
            if not features:
                return elapsed_ms, None, None
            center = features[0].get("center", [])
            if len(center) < 2:
                return elapsed_ms, None, None
            lon, lat = center[0], center[1]  # GeoJSON order: [lon, lat]
            return elapsed_ms, lat, lon
    except Exception:
        return (time.perf_counter() - t0) * 1000, None, None


# ── High-level benchmark runners ──────────────────────────────────────────────
def _count_valid(days: list) -> int:
    return sum(1 for d in days if d.get("temp") is not None)


async def run_vc(session, sem, period, location, month, day, api_key, run_idx, location_set) -> BenchmarkResult:
    years = list(range(date.today().year - YEARS_BACK, date.today().year + 1))
    ranges = _year_ranges(years, month, day, period)
    t0 = time.perf_counter()
    fetch_t, avg_http, proc_ms, req_count, nbytes, days = await _fetch_vc(
        session, sem, period, location["name"], ranges, api_key
    )
    return BenchmarkResult(
        variant="VC",
        period=period,
        location_name=location["name"],
        location_set=location_set,
        run=run_idx,
        total_time_s=time.perf_counter() - t0,
        geocode_ms=0.0,
        fetch_time_s=fetch_t,
        avg_http_ms=avg_http,
        process_ms=proc_ms,
        request_count=req_count,
        total_bytes=nbytes,
        data_points=_count_valid(days),
    )


async def run_om_warm(session, sem, period, location, month, day, run_idx) -> BenchmarkResult:
    lat, lon = location["lat"], location["lon"]
    years = list(range(date.today().year - YEARS_BACK, date.today().year + 1))
    ranges = _year_ranges(years, month, day, period)
    t0 = time.perf_counter()
    fetch_t, avg_http, proc_ms, req_count, nbytes, days = await _fetch_om(session, sem, lat, lon, ranges)
    return BenchmarkResult(
        variant="OM-warm",
        period=period,
        location_name=location["name"],
        location_set="warm",
        run=run_idx,
        total_time_s=time.perf_counter() - t0,
        geocode_ms=0.0,
        fetch_time_s=fetch_t,
        avg_http_ms=avg_http,
        process_ms=proc_ms,
        request_count=req_count,
        total_bytes=nbytes,
        data_points=_count_valid(days),
    )


async def run_om_cold(session, sem, period, location, month, day, mapbox_token, run_idx) -> BenchmarkResult:
    # Step 1 — Mapbox geocoding (timed separately)
    geo_ms, lat, lon = await _geocode(session, location["query"], mapbox_token)

    if lat is None or lon is None:
        return BenchmarkResult(
            variant="OM-cold",
            period=period,
            location_name=location["name"],
            location_set="fresh",
            run=run_idx,
            total_time_s=geo_ms / 1000,
            geocode_ms=geo_ms,
            fetch_time_s=0.0,
            avg_http_ms=0.0,
            process_ms=0.0,
            request_count=0,
            total_bytes=0,
            data_points=0,
            error="geocoding failed",
        )

    # Step 2 — OM weather fetch
    years = list(range(date.today().year - YEARS_BACK, date.today().year + 1))
    ranges = _year_ranges(years, month, day, period)
    t_fetch_start = time.perf_counter()
    fetch_t, avg_http, proc_ms, req_count, nbytes, days = await _fetch_om(session, sem, lat, lon, ranges)
    total_t = geo_ms / 1000 + (time.perf_counter() - t_fetch_start)

    return BenchmarkResult(
        variant="OM-cold",
        period=period,
        location_name=location["name"],
        location_set="fresh",
        run=run_idx,
        total_time_s=total_t,
        geocode_ms=geo_ms,
        fetch_time_s=fetch_t,
        avg_http_ms=avg_http,
        process_ms=proc_ms,
        request_count=req_count,
        total_bytes=nbytes,
        data_points=_count_valid(days),
    )


async def run_om_batch(session, period, location, month, day, run_idx) -> BenchmarkResult:
    """OM single-call spanning the full date range (daily period only)."""
    lat, lon = location["lat"], location["lon"]
    years = list(range(date.today().year - YEARS_BACK, date.today().year + 1))
    t0 = time.perf_counter()
    fetch_t, proc_ms, req_count, nbytes, days = await _fetch_om_batch(session, lat, lon, month, day, years)
    return BenchmarkResult(
        variant="OM-batch",
        period="daily",
        location_name=location["name"],
        location_set="warm",
        run=run_idx,
        total_time_s=time.perf_counter() - t0,
        geocode_ms=0.0,
        fetch_time_s=fetch_t,
        avg_http_ms=fetch_t * 1000,
        process_ms=proc_ms,
        request_count=req_count,
        total_bytes=nbytes,
        data_points=_count_valid(days),
    )


# ── Results aggregation ───────────────────────────────────────────────────────
def _average_runs(runs: List[BenchmarkResult]) -> dict:
    """Average multiple runs of the same (variant, period, location) key."""
    ok = [r for r in runs if not r.error]
    ref = runs[0]
    if not ok:
        return {**asdict(ref), "runs_ok": 0, "avg_req_ms": 0.0, "data_kb": 0.0}
    n = len(ok)
    return {
        "variant": ref.variant,
        "period": ref.period,
        "location_name": ref.location_name,
        "location_set": ref.location_set,
        "runs_ok": n,
        "total_time_s": round(sum(r.total_time_s for r in ok) / n, 3),
        "geocode_ms": round(sum(r.geocode_ms for r in ok) / n, 1),
        "fetch_time_s": round(sum(r.fetch_time_s for r in ok) / n, 3),
        "avg_http_ms": round(sum(r.avg_http_ms for r in ok) / n, 1),
        "process_ms": round(sum(r.process_ms for r in ok) / n, 2),
        "request_count": ok[0].request_count,
        "avg_req_ms": round(sum(r.avg_req_ms for r in ok) / n, 1),
        "data_kb": round(sum(r.data_kb for r in ok) / n, 1),
        "data_points": round(sum(r.data_points for r in ok) / n),
        "error": None,
    }


# ── Output ────────────────────────────────────────────────────────────────────
_HEADERS = [
    "Variant",
    "Period",
    "Location",
    "Set",
    "Runs",
    "Wall(s)",
    "Geocode(ms)",
    "AvgHTTP(ms)",
    "Proc(ms)",
    "Requests",
    "Data(KB)",
    "Points",
]
# Note: "Wall(s)" includes OM rate-limiter pacing; "AvgHTTP(ms)" is pure per-request
# HTTP time (excludes pacing) and is the fair metric for VC vs OM comparison.


def _row_values(r: dict) -> List[str]:
    return [
        r["variant"],
        r["period"],
        r["location_name"][:22],
        r["location_set"],
        str(r["runs_ok"]),
        f"{r['total_time_s']:.2f}",
        f"{r['geocode_ms']:.0f}" if r["geocode_ms"] > 0 else "—",
        f"{r['avg_http_ms']:.0f}",
        f"{r['process_ms']:.1f}",
        str(r["request_count"]),
        f"{r['data_kb']:.1f}",
        str(r["data_points"]),
    ]


def _print_table(rows: List[dict]) -> None:
    widths = [max(len(_HEADERS[i]), *[len(_row_values(r)[i]) for r in rows]) for i in range(len(_HEADERS))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    hdr = "|" + "|".join(f" {h:<{w}} " for h, w in zip(_HEADERS, widths)) + "|"

    def _line(r):
        vals = _row_values(r)
        return "|" + "|".join(f" {v:<{w}} " for v, w in zip(vals, widths)) + "|"

    print(sep)
    print(hdr)
    print(sep)
    last_period = None
    for r in rows:
        if r["period"] != last_period and last_period is not None:
            print(sep)
        last_period = r["period"]
        print(_line(r))
    print(sep)


def _print_summary(rows: List[dict]) -> None:
    print("\n── OM-warm vs VC, warm locations (geocoding excluded) ──────────────")
    for period in ALL_PERIODS:
        vc = [
            r
            for r in rows
            if r["variant"] == "VC" and r["period"] == period and r["location_set"] == "warm" and r["runs_ok"] > 0
        ]
        om = [r for r in rows if r["variant"] == "OM-warm" and r["period"] == period and r["runs_ok"] > 0]
        if not vc or not om:
            continue
        # Use avg_http_ms (pure HTTP time, excludes OM pacing) for fair comparison
        vc_t = sum(r["avg_http_ms"] for r in vc) / len(vc)
        om_t = sum(r["avg_http_ms"] for r in om) / len(om)
        vc_kb = sum(r["data_kb"] for r in vc) / len(vc)
        om_kb = sum(r["data_kb"] for r in om) / len(om)
        t_ratio = om_t / vc_t if vc_t else float("nan")
        kb_ratio = om_kb / vc_kb if vc_kb else float("nan")
        verdict = "OM faster ✓" if t_ratio < 1 else "VC faster"
        print(f"  {period:8s}: {t_ratio:.2f}× AvgHTTP  |  {kb_ratio:.2f}× data  ({verdict})")
    print("  (AvgHTTP = pure per-request HTTP time, excl. OM's 9.5 req/s rate-limiter pacing)")
    print("  (Wall(s) in table includes pacing; AvgHTTP is the fair speed comparison)")

    print("\n── OM-cold vs VC, fresh locations (first-access, includes geocoding) ─")
    for period in ALL_PERIODS:
        vc = [
            r
            for r in rows
            if r["variant"] == "VC" and r["period"] == period and r["location_set"] == "fresh" and r["runs_ok"] > 0
        ]
        cold = [r for r in rows if r["variant"] == "OM-cold" and r["period"] == period and r["runs_ok"] > 0]
        if not vc or not cold:
            continue
        vc_t = sum(r["avg_http_ms"] for r in vc) / len(vc)
        cold_t = sum(r["avg_http_ms"] for r in cold) / len(cold)
        geo_t = sum(r["geocode_ms"] for r in cold) / len(cold)
        t_ratio = (cold_t + geo_t) / vc_t if vc_t else float("nan")
        verdict = "OM-cold faster ✓" if t_ratio < 1 else "VC faster"
        print(
            f"  {period:8s}: {t_ratio:.2f}× (avg {cold_t:.0f}ms/req + {geo_t:.0f}ms geocode vs {vc_t:.0f}ms/req)  ({verdict})"
        )

    cold_all = [r for r in rows if r["variant"] == "OM-cold" and r["runs_ok"] > 0 and r["geocode_ms"] > 0]
    if cold_all:
        avg_geo = sum(r["geocode_ms"] for r in cold_all) / len(cold_all)
        print(f"\n  Mapbox geocoding: {avg_geo:.0f} ms avg (one-time per location)")

    print("\n── OM-batch vs OM-warm (daily, single call vs 51 calls) ────────────")
    batch = [r for r in rows if r["variant"] == "OM-batch" and r["runs_ok"] > 0]
    warm = [r for r in rows if r["variant"] == "OM-warm" and r["period"] == "daily" and r["runs_ok"] > 0]
    if batch and warm:
        bt = sum(r["avg_http_ms"] for r in batch) / len(batch) / 1000
        wt = sum(r["avg_http_ms"] for r in warm) / len(warm) / 1000 * 51
        bk = sum(r["data_kb"] for r in batch) / len(batch)
        wk = sum(r["data_kb"] for r in warm) / len(warm)
        verdict = "batch faster ✓" if bt < wt else "51-call approach faster"
        print(f"  HTTP time: batch ~{bt:.2f}s vs 51-call ~{wt:.2f}s  ({bt / wt:.2f}×, {verdict})")
        print(f"  Data     : batch {bk:.1f} KB vs warm {wk:.1f} KB  ({bk / wk:.2f}× data)")


# ── Main orchestration ────────────────────────────────────────────────────────
async def _run_all(args, vc_key: str, mapbox_token: str) -> List[BenchmarkResult]:
    connector = aiohttp.TCPConnector(limit=30, enable_cleanup_closed=True)
    timeout = aiohttp.ClientTimeout(total=120, connect=15, sock_read=90)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(args.concurrency)

        # Load preapproved warm locations
        with open(DATA_FILE) as f:
            raw_locs = json.load(f)
        warm_locs = [
            {
                "name": ", ".join(filter(None, [l.get("name"), l.get("admin1"), l.get("country_name")])),
                "lat": l["latitude"],
                "lon": l["longitude"],
                "tz": l.get("timezone"),
            }
            for l in raw_locs[: args.locations]
        ]
        fresh_locs = FRESH_LOCATIONS[: args.locations]

        # Parse anchor date
        today = date.today()
        month, day = today.month, today.day
        if args.date:
            try:
                month, day = map(int, args.date.split("-"))
            except ValueError:
                print(f"ERROR: Invalid --date '{args.date}'. Use MM-DD format.")
                sys.exit(1)

        periods = args.periods
        results: List[BenchmarkResult] = []

        done = [0]

        def _tick(label: str) -> None:
            done[0] += 1
            print(f"  [{done[0]:>3}] {label[:72]}", flush=True)

        # ── Warm locations: VC vs OM-warm ────────────────────────────────────
        if not args.skip_warm:
            print(f"\n{'─' * 60}")
            print(f" WARM LOCATIONS  ({len(warm_locs)} locations × {len(periods)} periods × {args.runs} runs)")
            print(f"{'─' * 60}")
            for loc in warm_locs:
                for period in periods:
                    for run in range(args.runs):
                        if not args.skip_vc:
                            _tick(f"VC          / {period:8s} / {loc['name'][:30]} / run {run + 1}")
                            r = await run_vc(session, sem, period, loc, month, day, vc_key, run, "warm")
                            results.append(r)

                        _tick(f"OM-warm     / {period:8s} / {loc['name'][:30]} / run {run + 1}")
                        r = await run_om_warm(session, sem, period, loc, month, day, run)
                        results.append(r)

        # ── Fresh locations: VC vs OM-cold ───────────────────────────────────
        if not args.skip_cold:
            print(f"\n{'─' * 60}")
            print(f" FRESH LOCATIONS ({len(fresh_locs)} locations × {len(periods)} periods × {args.runs} runs)")
            print(f"{'─' * 60}")
            for loc in fresh_locs:
                for period in periods:
                    for run in range(args.runs):
                        if not args.skip_vc:
                            _tick(f"VC(fresh)   / {period:8s} / {loc['name'][:30]} / run {run + 1}")
                            r = await run_vc(session, sem, period, loc, month, day, vc_key, run, "fresh")
                            results.append(r)

                        _tick(f"OM-cold     / {period:8s} / {loc['name'][:30]} / run {run + 1}")
                        r = await run_om_cold(session, sem, period, loc, month, day, mapbox_token, run)
                        results.append(r)

        # ── OM-batch: single-call daily, first 3 warm locations ──────────────
        if not args.no_batch and not args.skip_warm:
            print(f"\n{'─' * 60}")
            print(" OM-BATCH (daily, single archive call, first 3 warm locations)")
            print(f"{'─' * 60}")
            for loc in warm_locs[:3]:
                _tick(f"OM-batch    / daily    / {loc['name'][:30]} / run 1")
                r = await run_om_batch(session, "daily", loc, month, day, 0)
                results.append(r)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Visual Crossing vs Open-Meteo response times.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--date", metavar="MM-DD", default=None, help="Anchor month-day (default: today)")
    parser.add_argument("--locations", type=int, default=5, metavar="N", help="Locations per set, 1–20 (default: 5)")
    parser.add_argument("--runs", type=int, default=3, metavar="N", help="Repetitions per test (default: 3)")
    parser.add_argument(
        "--concurrency", type=int, default=2, metavar="N", help="Semaphore limit, matches production (default: 2)"
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=ALL_PERIODS,
        default=ALL_PERIODS,
        metavar="PERIOD",
        help="Periods to test (default: all four)",
    )
    parser.add_argument("--csv", metavar="PATH", default=None, help="Write averaged results to CSV")
    parser.add_argument("--raw-csv", metavar="PATH", default=None, help="Write every individual run to CSV")
    parser.add_argument("--skip-vc", action="store_true", help="Skip VC tests (saves API record budget)")
    parser.add_argument("--skip-cold", action="store_true", help="Skip OM-cold / Mapbox geocoding tests")
    parser.add_argument("--skip-warm", action="store_true", help="Skip warm-location tests")
    parser.add_argument("--no-batch", action="store_true", help="Skip OM-batched-daily bonus test")
    args = parser.parse_args()
    args.locations = max(1, min(20, args.locations))

    vc_key = os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
    mapbox_token = os.getenv("MAPBOX_TOKEN", "").strip()

    if not args.skip_vc and not vc_key:
        print("ERROR: VISUAL_CROSSING_API_KEY not set. Use --skip-vc to run OM tests only.")
        sys.exit(1)
    if not args.skip_cold and not mapbox_token:
        print("ERROR: MAPBOX_TOKEN not set. Use --skip-cold to skip geocoding tests.")
        sys.exit(1)

    # Rough call estimate (VC records consumed)
    n_years = YEARS_BACK + 1
    if not args.skip_vc and not args.skip_warm:
        vc_records = args.locations * args.runs * sum(n_years * WINDOW_DAYS[p] for p in args.periods)
        print(f"\nNote: VC tests will consume ~{vc_records:,} VC records from your daily budget.")

    print(f"\n{'=' * 64}")
    print("  TempHist API Benchmark: Visual Crossing vs Open-Meteo")
    print(f"{'=' * 64}")
    print(f"  Anchor date  : {args.date or date.today().strftime('%m-%d') + ' (today)'}")
    print(f"  Locations/set: {args.locations}")
    print(f"  Runs         : {args.runs}")
    print(f"  Concurrency  : {args.concurrency}  (matches production)")
    print(f"  Periods      : {', '.join(args.periods)}")
    print(f"  VC tests     : {'yes' if not args.skip_vc else 'SKIPPED'}")
    print(f"  OM-cold      : {'yes' if not args.skip_cold else 'SKIPPED'}")
    print(f"  OM-batch     : {'yes' if not args.no_batch else 'SKIPPED'}")
    print(f"{'=' * 64}")

    t_wall = time.perf_counter()
    all_results = asyncio.run(_run_all(args, vc_key, mapbox_token))
    elapsed = time.perf_counter() - t_wall

    # Average across runs, grouped by (variant, period, location_name)
    groups: Dict[tuple, List[BenchmarkResult]] = {}
    for r in all_results:
        key = (r.variant, r.period, r.location_name)
        groups.setdefault(key, []).append(r)

    averaged = [_average_runs(v) for v in groups.values()]

    # Sort: period order, then location_set, then variant, then location
    period_order = {p: i for i, p in enumerate(ALL_PERIODS)}
    variant_order = {"VC": 0, "OM-warm": 1, "OM-cold": 2, "OM-batch": 3}
    averaged.sort(
        key=lambda r: (
            period_order.get(r["period"], 99),
            r["location_set"],
            variant_order.get(r["variant"], 9),
            r["location_name"],
        )
    )

    print(f"\n\n{'=' * 64}")
    print(f"  Results  (total wall time: {elapsed:.1f}s)")
    print(f"{'=' * 64}\n")
    _print_table(averaged)
    _print_summary(averaged)

    if args.csv:
        fieldnames = list(averaged[0].keys()) if averaged else []
        with open(args.csv, "w", newline="") as f:
            w = csv_module.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(averaged)
        print(f"\nAveraged results → {args.csv}")

    if args.raw_csv:
        fieldnames = [f for f in asdict(all_results[0]).keys()] if all_results else []
        with open(args.raw_csv, "w", newline="") as f:
            w = csv_module.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(asdict(r) for r in all_results)
        print(f"Raw per-run results → {args.raw_csv}")


if __name__ == "__main__":
    main()
