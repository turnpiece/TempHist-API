"""V1 records endpoints."""

import asyncio
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import redis
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse

from app.cache_utils import (
    cache_get as temporal_cache_get,
)
from app.cache_utils import (
    cache_set as temporal_cache_set,
)
from cache.accessors import get_job_manager
from cache.core import (
    TTL_CURRENT_DAILY,
    TTL_CURRENT_MONTHLY,
    TTL_CURRENT_WEEKLY,
    TTL_CURRENT_YEARLY,
    TTL_STABLE,
    get_cache_updated_timestamp,
)
from cache.keys import (
    _get_location_timezone,
    assemble_and_cache,
    bundle_key,
    compute_bundle_etag,
    get_bundle_with_slug_fallback,
    get_local_today,
    get_records,
    get_year_etags,
    rec_etag_key,
    rec_key,
)
from config import (
    CACHE_ENABLED,
    DEBUG,
    MAX_CONCURRENT_REQUESTS,
)
from models import (
    AverageData,
    DateRange,
    MetaData,
    MetaResponse,
    RankingData,
    RecordResponse,
    SubResourceResponse,
    TemperatureValue,
    TrendData,
    UpdatedResponse,
)
from routers.dependencies import get_invalid_location_cache, get_redis_client
from utils.cache_headers import set_weather_cache_headers
from utils.daily_temperature_store import (
    DailyTemperatureRecord,
    get_daily_temperature_store,
    resolve_location_cache_identity,
    resolve_location_cache_slug,
)
from utils.location_validation import InvalidLocationCache, is_location_likely_invalid, validate_location_response
from utils.sanitization import sanitize_for_logging
from utils.temperature import (
    calculate_gradient_factor,
    calculate_standard_deviation,
    calculate_trend_slope,
    generate_summary,
    get_friendly_date,
)
from utils.validation import validate_location_for_ssrf
from utils.weather import create_metadata, get_year_range, track_missing_year
from utils.weather_data import get_temperature_series
from utils.weather_provider import LocationNotFoundError, fetch_timeline_days

logger = logging.getLogger(__name__)
router = APIRouter()


def parse_identifier(period: str, identifier: str) -> tuple:
    """Parse identifier based on period type. All periods use MM-DD format representing the end date."""
    # All periods use MM-DD format representing the end date of the period
    try:
        month, day = map(int, identifier.split("-"))
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        return month, day, period
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Identifier must be in MM-DD format: {str(e)}")


def _coerce_float(value) -> float | None:
    """Coerce a value to float, returning None if conversion fails.

    Args:
        value: Value to convert to float (can be int, float, str, or None)

    Returns:
        Float value or None if conversion fails
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _convert_c_to_unit(temp_c: float, unit_group: str) -> float:
    """Convert temperature from Celsius to the requested unit.

    Args:
        temp_c: Temperature in Celsius
        unit_group: Target unit group ('celsius', 'metric', 'fahrenheit', or 'us')

    Returns:
        Temperature in the requested unit
    """
    group = unit_group.lower()
    if group in ("celsius", "metric"):
        return temp_c
    if group in ("fahrenheit", "us"):
        return round((temp_c * 9.0 / 5.0) + 32.0, 2)
    return temp_c


def _resolve_anchor_date(year: int, month: int, day: int) -> date | None:
    """Resolve a date, adjusting for invalid dates like Feb 29 in non-leap years.

    Args:
        year: Year value
        month: Month value (1-12)
        day: Day value (1-31)

    Returns:
        Valid date object or None if no valid date can be resolved
    """
    try:
        return date(year, month, day)
    except ValueError:
        if month == 2 and day == 29:
            try:
                return date(year, 2, 28)
            except ValueError:
                return None
        # Fallback to last valid day of month
        for candidate_day in range(31, 27, -1):
            try:
                return date(year, month, candidate_day)
            except ValueError:
                continue
    return None


def _collapse_consecutive_dates(dates: List[date]) -> List[Tuple[date, date]]:
    """Collapse a list of dates into consecutive date ranges.

    Args:
        dates: List of date objects

    Returns:
        List of tuples representing date ranges (start_date, end_date)
    """
    if not dates:
        return []
    ordered = sorted(dates)
    ranges: List[Tuple[date, date]] = []
    start = prev = ordered[0]
    for current in ordered[1:]:
        if current == prev + timedelta(days=1):
            prev = current
            continue
        ranges.append((start, prev))
        start = prev = current
    ranges.append((start, prev))
    return ranges


def _timeline_days_to_records(timeline_days: List[Dict[str, Any]], range_start: date, range_end: date) -> List[DailyTemperatureRecord]:
    records_to_store: List[DailyTemperatureRecord] = []
    for day_payload in timeline_days:
        dt_raw = day_payload.get("datetime") or day_payload.get("date")
        if not dt_raw:
            continue
        dt_text = str(dt_raw)[:10]
        try:
            record_date = datetime.strptime(dt_text, "%Y-%m-%d").date()
        except ValueError:
            continue
        if record_date < range_start or record_date > range_end:
            continue
        temp = _coerce_float(day_payload.get("temp"))
        temp_max = _coerce_float(day_payload.get("tempmax") or day_payload.get("maxt"))
        temp_min = _coerce_float(day_payload.get("tempmin") or day_payload.get("mint"))
        filtered_payload = {
            "datetime": record_date.isoformat(),
            "temp": temp,
            "tempmax": temp_max,
            "tempmin": temp_min,
        }
        records_to_store.append(
            DailyTemperatureRecord(
                date=record_date,
                temp_c=temp,
                temp_max_c=temp_max,
                temp_min_c=temp_min,
                payload=filtered_payload,
                source="timeline",
            )
        )
    return records_to_store


async def _fetch_timeline_ranges(
    location: str,
    ranges: List[Tuple[int, date, date]],
) -> List[Tuple[int, date, date, List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Exception]]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def fetch_one(
        year: int,
        range_start: date,
        range_end: date,
    ) -> Tuple[int, date, date, List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Exception]]:
        async with semaphore:
            try:
                timeline_days, timeline_metadata = await fetch_timeline_days(location, range_start, range_end)
                return year, range_start, range_end, timeline_days, timeline_metadata, None
            except Exception as exc:
                return year, range_start, range_end, [], None, exc

    return await asyncio.gather(*(fetch_one(year, range_start, range_end) for year, range_start, range_end in ranges))


WINDOW_DAYS = {
    "daily": 1,
    "weekly": 7,
    "monthly": 31,
    "yearly": 365,
}

# Minimum coverage requirements (fraction of expected days) for using partially cached datasets.
COVERAGE_TOLERANCE = {
    # Allow up to one missing day in a week (≈86% coverage).
    "weekly": {"min_ratio": 6 / 7, "min_days": 6},
    # Require at least 28 of 31 days for monthly aggregations (≈90% coverage).
    "monthly": {"min_ratio": 0.9, "min_days": 28},
    # Yearly aggregates can tolerate larger gaps, but still require strong coverage.
    "yearly": {"min_ratio": 0.9, "min_days": 330},
}

# Lower coverage requirements for the current year (incomplete by nature)
COVERAGE_TOLERANCE_CURRENT_YEAR = {
    # For current year, accept more sparse data since year is incomplete
    "weekly": {"min_ratio": 0.5, "min_days": 4},  # At least 4 of 7 days
    "monthly": {"min_ratio": 0.6, "min_days": 19},  # At least 19 of 31 days
    "yearly": {"min_ratio": 0.8, "min_days": 292},  # At least 80% of the year
}


def _evaluate_coverage(
    period: str, available_days: int, expected_days: int, year: Optional[int] = None
) -> Tuple[bool, float]:
    """Determine whether the available data meets tolerance for the requested period.

    Args:
        period: Period type (daily, weekly, monthly, yearly)
        available_days: Number of days with data
        expected_days: Total number of days expected
        year: Optional year to check if current year (uses relaxed thresholds)

    Returns:
        Tuple of (coverage_ok, coverage_ratio)
    """
    if expected_days == 0:
        return False, 0.0

    # Use relaxed coverage requirements for current year
    current_year = datetime.now().year
    is_current_year = year == current_year

    if is_current_year:
        tolerance = COVERAGE_TOLERANCE_CURRENT_YEAR.get(period, {"min_ratio": 0.5, "min_days": 1})
    else:
        tolerance = COVERAGE_TOLERANCE.get(period, {"min_ratio": 1.0, "min_days": expected_days})

    ratio = available_days / expected_days
    return (
        available_days >= tolerance.get("min_days", expected_days) and ratio >= tolerance.get("min_ratio", 1.0)
    ), ratio


async def _enqueue_backfill_job(
    period: str,
    location: str,
    identifier: str,
    year: int,
    slug: str | None = None,
) -> None:
    """Request the background worker to backfill missing daily data for a specific year.

    Uses a per-key cooldown so that rapid client retries don't flood the job
    queue with duplicate backfill requests (even beyond the job-level dedup).
    """
    try:
        job_manager = get_job_manager()
        if not job_manager:
            return

        if slug is None:
            slug = await resolve_location_cache_slug(location)

        # Skip years known to have no data (marked by the worker after a failed DB pre-check
        # or after the full VC computation also returns no data for that year)
        skip_key = f"backfill:skip:{period}:{slug}:{year}"
        if job_manager.redis.exists(skip_key):
            return

        # Router-level cooldown: skip enqueue if we already requested this
        # backfill recently.  60 s is enough for the worker to pick it up.
        cooldown_key = f"backfill:cd:{period}:{slug}:{identifier}:{year}"
        if job_manager.redis.set(cooldown_key, 1, nx=True, ex=60):
            # Key was newly set — proceed with enqueue
            job_manager.create_job(
                "record_computation",
                {
                    "scope": period,
                    "slug": slug,
                    "identifier": identifier,
                    "year": year,
                    "location": location,
                },
            )
        elif DEBUG:
            logger.debug("Backfill cooldown active for %s %s %s %s", period, slug, identifier, year)
    except Exception as exc:  # Best-effort enqueue; log at debug level when verbose.
        if DEBUG:
            logger.debug("Failed to enqueue backfill job for %s %s %s: %s", period, location, year, exc)


def _plan_year_date_sequences(
    years: List[int], month: int, day: int, window_days: int, missing_years: List[Dict]
) -> Tuple[Dict[int, List[date]], set[date]]:
    """Build per-year date windows and a deduplicated set of all dates needed."""
    year_to_date_sequence: Dict[int, List[date]] = {}
    all_dates_needed: set[date] = set()
    for year in years:
        anchor = _resolve_anchor_date(year, month, day)
        if anchor is None:
            track_missing_year(missing_years, year, "invalid_anchor_date")
            continue
        start_date = anchor - timedelta(days=window_days - 1)
        date_sequence = [start_date + timedelta(days=i) for i in range(window_days)]
        year_to_date_sequence[year] = date_sequence
        all_dates_needed.update(date_sequence)
    return year_to_date_sequence, all_dates_needed


async def _enqueue_backfill_and_collect_ranges(
    location: str,
    period: str,
    month: int,
    day: int,
    slug: Optional[str],
    year_to_date_sequence: Dict[int, List[date]],
    cache: Dict[date, DailyTemperatureRecord],
    current_year_now: int,
) -> List[Tuple[int, date, date]]:
    """Per year, enqueue a backfill job if anything is missing and (if coverage is below threshold) return the dates to fetch from the provider."""
    ranges_to_fetch: List[Tuple[int, date, date]] = []
    for year, date_sequence in year_to_date_sequence.items():
        expected_days = len(date_sequence)
        missing_dates = [d for d in date_sequence if d not in cache or cache[d].temp_c is None]
        available_count = expected_days - len(missing_dates)
        coverage_ok, coverage_ratio = _evaluate_coverage(period, available_count, expected_days, year=year)

        if year == current_year_now and missing_dates:
            logger.info(
                "Current year [%s] coverage for %s: %d/%d days (%.1f%%)",
                year,
                sanitize_for_logging(location),
                available_count,
                expected_days,
                coverage_ratio * 100,
            )

        if not missing_dates:
            continue

        await _enqueue_backfill_job(period, location, f"{month:02d}-{day:02d}", year, slug=slug)
        if not coverage_ok:
            ranges_to_fetch.extend(
                (year, range_start, range_end)
                for range_start, range_end in _collapse_consecutive_dates(missing_dates)
            )
    return ranges_to_fetch


async def _fetch_and_store_timeline_ranges(
    location: str,
    ranges_to_fetch: List[Tuple[int, date, date]],
    cache: Dict[date, DailyTemperatureRecord],
    missing_years: List[Dict],
    timeline_failed_years: set[int],
) -> None:
    """Hit the weather provider for missing ranges and merge records back into ``cache``."""
    store = await get_daily_temperature_store()
    fetch_results = await _fetch_timeline_ranges(location, ranges_to_fetch)
    records_to_store: List[DailyTemperatureRecord] = []
    upsert_metadata: Optional[Dict[str, Any]] = None

    for year, range_start, range_end, timeline_days, timeline_metadata, exc in fetch_results:
        if isinstance(exc, LocationNotFoundError):
            logger.warning("Location not found, aborting fetch for %s: %s", location, exc)
            raise HTTPException(status_code=422, detail="location_not_found")
        if exc is not None:
            logger.error(
                "❌ timeline fetch failed for %s (%s to %s): %s",
                sanitize_for_logging(location),
                range_start.strftime("%Y-%m-%d"),
                range_end.strftime("%Y-%m-%d"),
                exc,
            )
            track_missing_year(missing_years, year, "timeline_error")
            timeline_failed_years.add(year)
            continue
        if timeline_metadata and upsert_metadata is None:
            upsert_metadata = timeline_metadata
        records_to_store.extend(_timeline_days_to_records(timeline_days, range_start, range_end))

    if records_to_store:
        await store.upsert(location, records_to_store, metadata=upsert_metadata)
        for rec in records_to_store:
            cache[rec.date] = rec


def _aggregate_year(
    year: int,
    date_sequence: List[date],
    cache: Dict[date, DailyTemperatureRecord],
    period: str,
    unit_group: str,
    location: str,
    current_year_now: int,
    timeline_failed: bool,
    missing_years: List[Dict],
) -> Optional[Tuple[TemperatureValue, float, Dict]]:
    """Return (value, aggregated temp, coverage entry) for a year, or None if it should be skipped."""
    anchor = _resolve_anchor_date(year, *_anchor_components(date_sequence))
    expected_days = len(date_sequence)
    final_missing_dates = [d for d in date_sequence if d not in cache or cache[d].temp_c is None]
    available_count = expected_days - len(final_missing_dates)
    coverage_ok, _ = _evaluate_coverage(period, available_count, expected_days, year=year)
    if timeline_failed:
        coverage_ok = False

    if final_missing_dates and not coverage_ok:
        if year == current_year_now:
            available_days = expected_days - len(final_missing_dates)
            logger.warning(
                "Current year [%s] skipped for %s: Coverage %d/%d days (%.1f%%) below threshold",
                year,
                sanitize_for_logging(location),
                available_days,
                expected_days,
                (available_days / expected_days) * 100,
            )
        if not timeline_failed:
            track_missing_year(missing_years, year, "coverage_below_threshold")
        return None

    temps_converted = [
        _convert_c_to_unit(cache[d].temp_c, unit_group)
        for d in date_sequence
        if d in cache and cache[d].temp_c is not None
    ]
    if not temps_converted:
        track_missing_year(missing_years, year, "no_daily_data")
        return None

    avg_temp = sum(temps_converted) / len(temps_converted)
    value = TemperatureValue(
        date=anchor.strftime("%Y-%m-%d"),
        year=anchor.year,
        temperature=round(avg_temp, 2),
    )
    if year == current_year_now:
        logger.info(
            "Current year [%s] included for %s: %d/%d days",
            year,
            sanitize_for_logging(location),
            len(temps_converted),
            expected_days,
        )
    coverage_entry = {
        "year": anchor.year,
        "available_days": len(temps_converted),
        "expected_days": expected_days,
        "missing_days": expected_days - len(temps_converted),
        "coverage_ratio": round(len(temps_converted) / expected_days, 4) if expected_days else 0.0,
        "approximate": len(temps_converted) < expected_days,
    }
    return value, avg_temp, coverage_entry


def _anchor_components(date_sequence: List[date]) -> Tuple[int, int]:
    """Return (month, day) for the anchor (the last date in the window)."""
    anchor = date_sequence[-1]
    return anchor.month, anchor.day


async def _collect_rolling_window_values(
    location: str,
    period: Literal["daily", "weekly", "monthly", "yearly"],
    month: int,
    day: int,
    unit_group: str,
    years: List[int],
    slug: str | None = None,
) -> Tuple[List[TemperatureValue], List[float], List[Dict], List[Dict]]:
    store = await get_daily_temperature_store()

    values: List[TemperatureValue] = []
    aggregated: List[float] = []
    missing_years: List[Dict] = []
    coverage_details: List[Dict] = []
    timeline_failed_years: set[int] = set()
    window_days = WINDOW_DAYS[period]
    current_year_now = datetime.now().year

    year_to_date_sequence, all_dates_needed = _plan_year_date_sequences(
        years, month, day, window_days, missing_years
    )

    if all_dates_needed:
        logger.info(
            "Fetching %d unique dates for %s across %d years in single query",
            len(all_dates_needed),
            sanitize_for_logging(location),
            len(year_to_date_sequence),
        )
    cache = await store.fetch(location, list(all_dates_needed))

    ranges_to_fetch = await _enqueue_backfill_and_collect_ranges(
        location, period, month, day, slug,
        year_to_date_sequence, cache, current_year_now,
    )
    if ranges_to_fetch:
        await _fetch_and_store_timeline_ranges(
            location, ranges_to_fetch, cache, missing_years, timeline_failed_years
        )

    for year, date_sequence in year_to_date_sequence.items():
        result = _aggregate_year(
            year=year,
            date_sequence=date_sequence,
            cache=cache,
            period=period,
            unit_group=unit_group,
            location=location,
            current_year_now=current_year_now,
            timeline_failed=year in timeline_failed_years,
            missing_years=missing_years,
        )
        if result is None:
            continue
        value, avg_temp, coverage_entry = result
        values.append(value)
        aggregated.append(avg_temp)
        coverage_details.append(coverage_entry)

    return values, aggregated, missing_years, coverage_details


_PERIOD_DATE_RANGE_DAYS = {"daily": 1, "weekly": 7, "monthly": 31, "yearly": 365}

_PERIOD_FRIENDLY_OVERRIDE = {
    "weekly": "week ending ",
    "monthly": "month ending ",
    "yearly": "year ending ",
}


def _date_range_for_period(period: str) -> int:
    if period not in _PERIOD_DATE_RANGE_DAYS:
        raise HTTPException(status_code=400, detail="Invalid period. Must be daily, weekly, monthly, or yearly")
    return _PERIOD_DATE_RANGE_DAYS[period]


def _build_range_block(values: List, month: int, day: int) -> DateRange:
    if not values:
        return DateRange(start="", end="", years=0)
    start_year_val = min(v.year for v in values)
    end_year_val = max(v.year for v in values)
    return DateRange(
        start=f"{start_year_val}-{month:02d}-{day:02d}",
        end=f"{end_year_val}-{month:02d}-{day:02d}",
        years=end_year_val - start_year_val + 1,
    )


def _build_average_block_v1(
    all_temps: List[float], unit_group: str
) -> Tuple[AverageData, float, Optional[float]]:
    if not all_temps:
        return AverageData(mean=0.0, unit=unit_group, data_points=0), 0.0, None
    series_mean = sum(all_temps) / len(all_temps)
    series_std_dev = calculate_standard_deviation(all_temps)
    avg = AverageData(
        mean=round(series_mean, 2),
        unit=unit_group,
        data_points=len(all_temps),
        standard_deviation=series_std_dev,
    )
    return avg, series_mean, series_std_dev


def _build_trend_block_v1(values: List, unit_group: str) -> TrendData:
    trend_unit = "°C/decade" if unit_group == "celsius" else "°F/decade"
    if len(values) < 2:
        return TrendData(slope=0.0, unit=trend_unit, data_points=len(values))
    trend_input = [{"x": v.year, "y": v.temperature} for v in values]
    slope, r_squared, slope_error = calculate_trend_slope(trend_input)
    gf = calculate_gradient_factor(slope, slope_error, unit_group) if slope_error is not None else None
    return TrendData(
        slope=slope,
        unit=trend_unit,
        data_points=len(values),
        r_squared=r_squared,
        slope_error=slope_error,
        gradient_factor=gf,
    )


def _make_summary_v1(
    values: List,
    period: str,
    location: str,
    unit_group: str,
    series_mean: float,
    end_date_obj: datetime,
    redis_client: Optional[redis.Redis],
) -> str:
    summary_data = [{"x": v.year, "y": v.temperature} for v in values]
    local_today = get_local_today(location, redis_client)
    text = generate_summary(
        summary_data, end_date_obj, period, unit_group, mean=series_mean, local_today=local_today
    )
    prefix = _PERIOD_FRIENDLY_OVERRIDE.get(period)
    if prefix:
        friendly = get_friendly_date(end_date_obj)
        text = text.replace(friendly, prefix + friendly)
    return text


def _coverage_metadata(coverage_details: List[Dict]) -> Dict:
    if not coverage_details:
        return {}
    total_expected = sum(item["expected_days"] for item in coverage_details)
    total_available = sum(item["available_days"] for item in coverage_details)
    overall_ratio = round(total_available / total_expected, 4) if total_expected else 0.0
    return {
        "overall_available_days": total_available,
        "overall_expected_days": total_expected,
        "overall_coverage_ratio": overall_ratio,
        "approximate_years": [item for item in coverage_details if item["approximate"]],
        "per_year": coverage_details,
    }


async def get_temperature_data_v1(
    location: str,
    period: str,
    identifier: str,
    unit_group: str = "celsius",
    redis_client: redis.Redis = None,  # Can be None, will use dependency if not provided
) -> Dict:
    """Get temperature data for v1 API using rolling timeline windows for weekly/monthly/yearly."""
    month, day, _ = parse_identifier(period, identifier)
    slug = await resolve_location_cache_slug(location)

    current_year = datetime.now().year
    years = get_year_range(current_year)

    date_range_days = _date_range_for_period(period)

    window_values, aggregated_values, window_missing, coverage_details = await _collect_rolling_window_values(
        location=location,
        period=period,  # type: ignore[arg-type]
        month=month,
        day=day,
        unit_group=unit_group,
        years=years,
        slug=slug,
    )
    values = list(window_values)
    all_temps = list(aggregated_values)
    missing_years = list(window_missing)

    range_data = _build_range_block(values, month, day)
    avg_data, series_mean, _series_std_dev = _build_average_block_v1(all_temps, unit_group)

    values = [v.model_copy(update={"anomaly": round(v.temperature - series_mean, 2)}) for v in values]

    trend_data = _build_trend_block_v1(values, unit_group)

    end_date_obj = datetime(current_year, month, day)
    summary_text = _make_summary_v1(
        values, period, location, unit_group, series_mean, end_date_obj, redis_client
    )

    available_years = {v.year for v in values}
    if current_year not in available_years and not any(
        entry.get("year") == current_year for entry in missing_years
    ):
        track_missing_year(missing_years, current_year, "no_data_current_year")

    additional_metadata: Dict = {
        "period_days": date_range_days,
        "end_date": end_date_obj.strftime("%Y-%m-%d"),
    }
    coverage = _coverage_metadata(coverage_details)
    if coverage:
        additional_metadata["coverage"] = coverage

    timezone_str = _get_location_timezone(location, redis_client) if redis_client else None

    return {
        "period": period,
        "location": location,
        "identifier": identifier,
        "range": range_data.model_dump(),
        "unit_group": unit_group,
        "values": [v.model_dump() for v in values],
        "average": avg_data.model_dump(),
        "trend": trend_data.model_dump(),
        "summary": summary_text,
        "metadata": create_metadata(len(years), len(values), missing_years, additional_metadata),
        "timezone": timezone_str,
    }


def _convert_values_to_unit(values: List[Dict], unit_group: str) -> List[Dict]:
    """Return shallow copies of ``values`` with temperatures converted to ``unit_group``."""
    converted = []
    for v in values:
        out = dict(v)
        if out.get("temperature") is not None:
            out["temperature"] = _convert_c_to_unit(out["temperature"], unit_group)
        converted.append(out)
    return converted


def _build_average_block(temps: List[float], unit_group: str, std_dev: float) -> Dict:
    if not temps:
        return {"mean": 0.0, "unit": unit_group, "data_points": 0}
    return {
        "mean": round(sum(temps) / len(temps), 2),
        "unit": unit_group,
        "data_points": len(temps),
        "standard_deviation": std_dev,
    }


def _build_trend_block(values: List[Dict], unit_group: str) -> Dict:
    from utils.temperature import calculate_gradient_factor, calculate_trend_slope

    trend_unit = "°F/decade" if unit_group.lower() == "fahrenheit" else "°C/decade"
    if len(values) < 2:
        return {"slope": 0.0, "unit": trend_unit, "data_points": len(values)}

    trend_input = [{"x": v.get("year"), "y": v.get("temperature")} for v in values]
    slope, r_squared, slope_error = calculate_trend_slope(trend_input)
    gf = calculate_gradient_factor(slope, slope_error, unit_group) if slope_error is not None else None
    return {
        "slope": slope,
        "unit": trend_unit,
        "data_points": len(values),
        "r_squared": r_squared,
        "slope_error": slope_error,
        "gradient_factor": gf,
    }


_PERIOD_FRIENDLY_PREFIX = {"weekly": "week ending ", "monthly": "month ending ", "yearly": "year ending "}


def _build_ranking_block(values: List[Dict]) -> Dict:
    """Compute warm/cold rank of the most recent year against the full series."""
    scored = [(v["year"], v["temperature"]) for v in values if v.get("temperature") is not None]
    if not scored:
        return {"warm": 1, "cold": 1, "total": 0}
    current_year = max(y for y, _ in scored)
    by_warm = sorted(scored, key=lambda x: x[1], reverse=True)
    by_cold = sorted(scored, key=lambda x: x[1])
    warm_rank = next(i + 1 for i, (y, _) in enumerate(by_warm) if y == current_year)
    cold_rank = next(i + 1 for i, (y, _) in enumerate(by_cold) if y == current_year)
    return {"warm": warm_rank, "cold": cold_rank, "total": len(scored)}


def _build_summary_text(
    values: List[Dict],
    period: str,
    location: str,
    unit_group: str,
    series_mean: float,
    end_date_obj: datetime,
    redis_client: redis.Redis,
) -> str:
    from utils.temperature import generate_summary

    summary_data = [{"x": v.get("year"), "y": v.get("temperature")} for v in values]
    local_today = get_local_today(location, redis_client)
    text = generate_summary(
        summary_data, end_date_obj, period, unit_group, mean=series_mean, local_today=local_today
    )

    prefix = _PERIOD_FRIENDLY_PREFIX.get(period)
    if prefix:
        friendly = get_friendly_date(end_date_obj)
        text = text.replace(friendly, prefix + friendly)
    return text


def _collect_missing_years(values: List[Dict], years: List[int], current_year: int) -> List:
    available = {v.get("year") for v in values if v.get("year") is not None}
    missing: List = []
    for y in years:
        if y in available:
            continue
        reason = "no_data_current_year" if y == current_year else "no_data"
        track_missing_year(missing, y, reason)
    return missing


def _rebuild_full_response_from_values(
    values: List[Dict],
    period: str,
    location: str,
    identifier: str,
    month: int,
    day: int,
    current_year: int,
    years: List[int],
    redis_client: redis.Redis,
    unit_group: str = "celsius",
) -> Dict:
    """Rebuild full RecordResponse from list of year values (Celsius inputs)."""
    from utils.temperature import calculate_standard_deviation
    from utils.weather import create_metadata

    converted_values = _convert_values_to_unit(values, unit_group)
    all_temps = [v.get("temperature") for v in converted_values if v.get("temperature") is not None]

    series_mean = sum(all_temps) / len(all_temps) if all_temps else 0.0
    series_std_dev = calculate_standard_deviation(all_temps)
    for v in converted_values:
        temp = v.get("temperature")
        v["anomaly"] = round(temp - series_mean, 2) if temp is not None else None

    end_date_obj = datetime(current_year, month, day)
    avg_data = _build_average_block(all_temps, unit_group, series_std_dev)
    trend_data = _build_trend_block(converted_values, unit_group)
    summary_text = _build_summary_text(
        converted_values, period, location, unit_group, series_mean, end_date_obj, redis_client
    )
    rebuilt_missing_years = _collect_missing_years(converted_values, years, current_year)
    period_days = WINDOW_DAYS.get(period, 1)

    min_year = min(v.get("year") for v in converted_values)
    max_year = max(v.get("year") for v in converted_values)
    return {
        "period": period,
        "location": location,
        "identifier": identifier,
        "range": {
            "start": f"{min_year}-{month:02d}-{day:02d}",
            "end": f"{max_year}-{month:02d}-{day:02d}",
            "years": max_year - min_year + 1,
        },
        "unit_group": unit_group,
        "values": converted_values,
        "average": avg_data,
        "trend": trend_data,
        "summary": summary_text,
        "metadata": create_metadata(
            len(years),
            len(converted_values),
            rebuilt_missing_years,
            {"period_days": period_days, "end_date": end_date_obj.strftime("%Y-%m-%d")},
        ),
        "timezone": _get_location_timezone(location, redis_client),
        "updated": datetime.now(timezone.utc).isoformat(),
    }


def _extract_per_year_records(full_data: Dict) -> Dict[int, Dict]:
    """Extract per-year records from full response data.

    Args:
        full_data: Full response from get_temperature_data_v1

    Returns:
        Dict mapping year -> per-year record (just the TemperatureValue for that year)
    """
    per_year = {}
    if "values" in full_data:
        for value in full_data["values"]:
            year = value.get("year")
            if year:
                per_year[year] = value
    return per_year


async def compute_per_year_records(
    location: str,
    period: str,
    identifier: str,
    years_to_fetch: List[int],
    unit_group: str = "celsius",
    redis_client: redis.Redis = None,
) -> Tuple[Dict[int, Dict], List[Dict], List[Dict]]:
    """Compute per-year records for a specific subset of years.

    Narrow-path counterpart to :func:`get_temperature_data_v1` for the
    partial-cache-miss case: when callers already know which years are
    missing, this avoids recomputing the full 50-year window. The returned
    ``per_year_records`` shape matches :func:`_extract_per_year_records`
    (each value is a ``TemperatureValue.model_dump()`` dict). Anomalies are
    not populated here — they depend on the full-series mean and are
    recomputed at bundle assembly by ``_rebuild_full_response_from_values``.

    Returns:
        (per_year_records, missing_years, coverage_details)
    """
    if not years_to_fetch:
        return {}, [], []

    month, day, _ = parse_identifier(period, identifier)
    slug = await resolve_location_cache_slug(location)

    per_year: Dict[int, Dict] = {}
    missing_years: List[Dict] = []
    coverage_details: List[Dict] = []

    if period == "daily":
        weather_data = await get_temperature_series(
            location, month, day, redis_client, years=list(years_to_fetch)
        )
        if weather_data and "data" in weather_data:
            if "metadata" in weather_data and "missing_years" in weather_data["metadata"]:
                missing_years.extend(weather_data["metadata"]["missing_years"])

            for data_point in weather_data["data"]:
                year = int(data_point["x"])
                temp = data_point["y"]
                if temp is None:
                    continue
                converted = _convert_c_to_unit(temp, unit_group)
                per_year[year] = TemperatureValue(
                    date=f"{year}-{month:02d}-{day:02d}",
                    year=year,
                    temperature=round(converted, 2),
                ).model_dump()
    elif period in ("weekly", "monthly", "yearly"):
        (
            window_values,
            _aggregated,
            window_missing,
            window_coverage,
        ) = await _collect_rolling_window_values(
            location=location,
            period=period,  # type: ignore[arg-type]
            month=month,
            day=day,
            unit_group=unit_group,
            years=list(years_to_fetch),
            slug=slug,
        )
        missing_years.extend(window_missing)
        coverage_details.extend(window_coverage)
        for value in window_values:
            per_year[value.year] = value.model_dump()
    else:
        raise HTTPException(status_code=400, detail="Invalid period. Must be daily, weekly, monthly, or yearly")

    return per_year, missing_years, coverage_details


def _get_ttl_for_current_year(period: str) -> int:
    """Get TTL for current year based on period."""
    if period == "daily":
        return TTL_CURRENT_DAILY
    elif period == "weekly":
        return TTL_CURRENT_WEEKLY
    elif period == "monthly":
        return TTL_CURRENT_MONTHLY
    elif period == "yearly":
        return TTL_CURRENT_YEARLY
    else:
        return TTL_CURRENT_DAILY  # Default


async def _store_per_year_records(
    redis_client: redis.Redis,
    scope: str,
    slug: str,
    identifier: str,
    per_year_records: Dict[int, Dict],
    current_year: int,
):
    """Store per-year records in cache with appropriate TTLs."""
    from cache.core import ETagGenerator

    for year, record_data in per_year_records.items():
        year_key = rec_key(scope, slug, identifier, year)
        etag_key = rec_etag_key(scope, slug, identifier, year)

        # Determine TTL
        if year < current_year:
            ttl = TTL_STABLE
        else:
            ttl = _get_ttl_for_current_year(scope)

        # Generate ETag for this year's record
        etag = ETagGenerator.generate_etag(record_data)

        try:
            json_data = json.dumps(record_data, sort_keys=True, separators=(",", ":"))
            redis_client.setex(year_key, ttl, json_data)
            redis_client.setex(etag_key, ttl, etag)
            if DEBUG:
                logger.debug(f"Cached per-year record: {year_key} (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Error caching per-year record {year_key}: {e}")


async def _get_record_data_internal(
    period: str,
    location: str,
    identifier: str,
    redis_client: redis.Redis,
    invalid_location_cache: InvalidLocationCache,
    unit_group: str = "celsius",
) -> Dict:
    """Internal helper to get record data using per-year caching (returns dict, not response)."""
    # This is essentially the same logic as get_record but returns the data dict
    # Parse identifier
    month, day, _ = parse_identifier(period, identifier)
    current_year = datetime.now(timezone.utc).year
    location_identity = await resolve_location_cache_identity(location)
    slug = location_identity.redis_slug
    lookup_slugs = location_identity.lookup_slugs or (slug,)
    years = get_year_range(current_year)

    if CACHE_ENABLED:
        bundle_data, _bundle_etag, _hit_slug = get_bundle_with_slug_fallback(
            redis_client, period, lookup_slugs, identifier
        )

        if bundle_data:
            try:
                data_str = bundle_data.decode("utf-8") if isinstance(bundle_data, bytes) else bundle_data
                bundle_payload = json.loads(data_str)
                if "records" in bundle_payload and len(bundle_payload["records"]) > 0:
                    values = bundle_payload["records"]
                    return _rebuild_full_response_from_values(
                        values, period, location, identifier, month, day, current_year, years, redis_client, unit_group
                    )
            except Exception as _e:
                logger.debug("Could not read bundle cache: %s", _e)

        # MGET all year keys
        year_data, missing_past, missing_current = await get_records(
            redis_client, period, slug, identifier, years, lookup_slugs=lookup_slugs
        )

        # Handle missing years (simplified - just fetch if needed)
        if missing_past or missing_current:
            years_to_fetch = list(missing_past) + ([current_year] if missing_current else [])
            new_records, _, _ = await compute_per_year_records(
                location, period, identifier, years_to_fetch, "celsius", redis_client
            )
            if new_records:
                await _store_per_year_records(redis_client, period, slug, identifier, new_records, current_year)
                year_data = {**year_data, **new_records}

        # Assemble from year_data using helper
        if year_data:
            values = [year_data[y] for y in sorted(year_data.keys())]
            return _rebuild_full_response_from_values(
                values, period, location, identifier, month, day, current_year, years, redis_client, unit_group
            )

    # Fallback: fetch fresh and rebuild with correct unit
    fallback_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)

    # Convert the fallback data to the requested unit
    values = fallback_data.get("values", [])
    if values:
        return _rebuild_full_response_from_values(
            values, period, location, identifier, month, day, current_year, years, redis_client, unit_group
        )
    else:
        return fallback_data  # Return as-is if no values


def _validate_record_location(location: str, invalid_location_cache: InvalidLocationCache) -> str:
    """Run SSRF check, format check, and the invalid-location cache lookup; return normalised location."""
    try:
        location = validate_location_for_ssrf(location)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Invalid location format. Please provide a valid location name."
        ) from e

    if is_location_likely_invalid(location):
        raise HTTPException(
            status_code=400, detail=f"Invalid location format: '{location}'. Please provide a valid location name."
        )

    if invalid_location_cache.is_invalid_location(location):
        invalid_info = invalid_location_cache.get_invalid_location_info(location)
        reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
        raise HTTPException(
            status_code=400,
            detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}",
        )
    return location


def _enqueue_record_refresh_job(
    period: str, slug: str, identifier: str, year: int, location: str, log_label: str
) -> None:
    """Enqueue a record_computation job for the given (period, slug, year). Failures are logged, not raised."""
    try:
        job_manager = get_job_manager()
        if not job_manager:
            return
        job_id = job_manager.create_job(
            "record_computation",
            {
                "scope": period,
                "slug": slug,
                "identifier": identifier,
                "year": year,
                "location": location,
            },
        )
        if DEBUG:
            logger.debug("Enqueued job to %s: %s", log_label, job_id)
    except Exception as e:
        logger.warning("Failed to enqueue refresh job: %s", e)


def _try_serve_bundle_cache(
    *,
    redis_client: redis.Redis,
    period: str,
    location: str,
    identifier: str,
    slug: str,
    lookup_slugs: Tuple[str, ...],
    month: int,
    day: int,
    current_year: int,
    years: List[int],
    unit_group: str,
    if_none_match: Optional[str],
    invalid_location_cache: InvalidLocationCache,
    end_date,
    response: Response,
) -> Optional[Response]:
    """Return a JSONResponse / 304 / None — None means caller should fall through to other strategies."""
    bundle_data, bundle_etag, _bundle_hit_slug = get_bundle_with_slug_fallback(
        redis_client, period, lookup_slugs, identifier
    )
    bundle_key_str = bundle_key(period, slug, identifier)

    if not (bundle_data and bundle_etag):
        return None

    try:
        data_str = bundle_data.decode("utf-8") if isinstance(bundle_data, bytes) else bundle_data
        bundle_payload = json.loads(data_str)
        bundle_etag_str = bundle_etag.decode("utf-8") if isinstance(bundle_etag, bytes) else bundle_etag

        if if_none_match:
            from cache.core import ETagGenerator

            if ETagGenerator.matches_etag(bundle_etag_str, if_none_match):
                response.status_code = 304
                response.headers["ETag"] = bundle_etag_str
                response.headers["X-Cache-Status"] = "HIT"
                return None  # FastAPI will serve the empty Response with the headers

        if not bundle_payload.get("records"):
            return None

        data = _rebuild_full_response_from_values(
            bundle_payload["records"], period, location, identifier, month, day, current_year, years,
            redis_client, unit_group,
        )

        is_valid, error_msg = validate_location_response(data, location)
        if not is_valid:
            invalid_location_cache.mark_location_invalid(location, "no_data_cached")
            raise HTTPException(status_code=400, detail=error_msg)

        json_response = JSONResponse(content=data)
        json_response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
        json_response.headers["ETag"] = bundle_etag_str
        json_response.headers["X-Cache-Status"] = "HIT"
        set_weather_cache_headers(
            json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1"
        )
        if DEBUG:
            logger.debug("✅ SERVING BUNDLE CACHE: %s", bundle_key_str)
        return json_response
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        if DEBUG:
            logger.debug("Error parsing bundle cache: %s, falling through to per-year lookup", e)
        return None


def _try_serve_temporal_cache(
    *,
    redis_client: redis.Redis,
    period: str,
    location: str,
    identifier: str,
    month: int,
    day: int,
    current_year: int,
    years: List[int],
    unit_group: str,
    if_none_match: Optional[str],
    invalid_location_cache: InvalidLocationCache,
    location_identity,
    end_date,
    response: Response,
) -> Optional[Response]:
    temporal_hit = temporal_cache_get(
        redis_client,
        agg=period,
        original_location=location,
        end_date=end_date,
        canonical_name=location_identity.canonical_name,
    )
    if not temporal_hit:
        return None

    data = temporal_hit["data"]
    meta = temporal_hit["meta"]

    if "records" in data:
        data = _rebuild_full_response_from_values(
            data["records"], period, location, identifier, month, day, current_year, years, redis_client, unit_group,
        )
    data["meta"] = meta

    is_valid, error_msg = validate_location_response(data, location)
    if not is_valid:
        invalid_location_cache.mark_location_invalid(location, "no_data_cached")
        raise HTTPException(status_code=400, detail=error_msg)

    from cache.core import ETagGenerator

    etag = ETagGenerator.generate_etag(data)
    if if_none_match and ETagGenerator.matches_etag(etag, if_none_match):
        response.status_code = 304
        response.headers["ETag"] = etag
        response.headers["X-Cache-Status"] = "HIT"
        return None

    json_response = JSONResponse(content=data)
    json_response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
    json_response.headers["ETag"] = etag
    cache_label = "APPROX" if meta["approximate"]["temporal"] else "HIT"
    json_response.headers["X-Cache-Status"] = cache_label
    set_weather_cache_headers(
        json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1"
    )
    if DEBUG:
        logger.debug("✅ SERVING TEMPORAL CACHE (%s): %s/%s/%s", cache_label, period, location, identifier)
    return json_response


async def _try_serve_stale_bundle_for_current_year(
    *,
    redis_client: redis.Redis,
    period: str,
    location: str,
    identifier: str,
    slug: str,
    lookup_slugs: Tuple[str, ...],
    month: int,
    day: int,
    current_year: int,
    years: List[int],
    unit_group: str,
    if_none_match: Optional[str],
    invalid_location_cache: InvalidLocationCache,
    end_date,
    response: Response,
) -> Optional[Response]:
    """When only the current year is missing, serve the previous bundle stale and enqueue a refresh."""
    bundle_data, _bundle_etag, _stale_hit_slug = get_bundle_with_slug_fallback(
        redis_client, period, lookup_slugs, identifier
    )
    if not bundle_data:
        return None

    try:
        data_str = bundle_data.decode("utf-8") if isinstance(bundle_data, bytes) else bundle_data
        bundle_payload = json.loads(data_str)
        if not bundle_payload.get("records"):
            return None

        data = _rebuild_full_response_from_values(
            bundle_payload["records"], period, location, identifier, month, day, current_year, years,
            redis_client, unit_group,
        )
        is_valid, error_msg = validate_location_response(data, location)
        if not is_valid:
            invalid_location_cache.mark_location_invalid(location, "no_data_cached")
            raise HTTPException(status_code=400, detail=error_msg)

        year_etags = await get_year_etags(
            redis_client, period, slug, identifier, years, lookup_slugs=lookup_slugs
        )
        bundle_etag_computed = compute_bundle_etag(year_etags)

        if if_none_match:
            from cache.core import ETagGenerator

            if ETagGenerator.matches_etag(bundle_etag_computed, if_none_match):
                response.status_code = 304
                response.headers["ETag"] = bundle_etag_computed
                response.headers["X-Cache-Status"] = "STALE"
                return None

        json_response = JSONResponse(content=data)
        json_response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
        json_response.headers["ETag"] = bundle_etag_computed
        json_response.headers["X-Cache-Status"] = "STALE"
        set_weather_cache_headers(
            json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1"
        )
        _enqueue_record_refresh_job(period, slug, identifier, current_year, location, "refresh current year")
        return json_response
    except HTTPException:
        raise
    except Exception as e:
        if DEBUG:
            logger.debug("Error serving stale bundle: %s", e)
        return None


def _resolve_bundle_etag(
    redis_client: redis.Redis, period: str, slug: str, identifier: str, data: Dict
) -> str:
    """Return the stored bundle ETag if present, else hash the response data."""
    from cache.core import ETagGenerator

    if CACHE_ENABLED:
        bundle_etag_key = f"{bundle_key(period, slug, identifier)}:etag"
        stored = redis_client.get(bundle_etag_key)
        if stored:
            return stored.decode("utf-8") if isinstance(stored, bytes) else stored
    return ETagGenerator.generate_etag(data)


async def _inline_fetch_missing_years(
    *,
    redis_client: redis.Redis,
    period: str,
    location: str,
    identifier: str,
    slug: str,
    current_year: int,
    missing_past: List[int],
    missing_current: bool,
    year_data: Dict[int, Dict],
) -> None:
    """Synchronously compute records for missing years and write them into cache and ``year_data``."""
    years_to_cache = list(missing_past) + ([current_year] if missing_current else [])
    per_year_records, _, _ = await compute_per_year_records(
        location, period, identifier, years_to_cache, "celsius", redis_client
    )
    from cache.core import ETagGenerator

    for year in years_to_cache:
        if year not in per_year_records:
            continue
        year_key = rec_key(period, slug, identifier, year)
        etag_key = rec_etag_key(period, slug, identifier, year)
        ttl = _get_ttl_for_current_year(period) if year == current_year else TTL_STABLE
        etag = ETagGenerator.generate_etag(per_year_records[year])
        try:
            json_data = json.dumps(per_year_records[year], sort_keys=True, separators=(",", ":"))
            redis_client.setex(year_key, ttl, json_data)
            redis_client.setex(etag_key, ttl, etag)
            year_data[year] = per_year_records[year]
        except Exception as e:
            logger.warning("Error caching year %s: %s", year, e)


@router.get("/v1/records/{period}/{location}/{identifier}", response_model=RecordResponse)
async def get_record(
    request: Request,
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius", description="Temperature unit for response"),
    response: Response = None,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)] = None,
    invalid_location_cache: Annotated[InvalidLocationCache, Depends(get_invalid_location_cache)] = None,
):
    """Get temperature record data for a specific period, location, and identifier."""
    try:
        location = _validate_record_location(location, invalid_location_cache)

        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now(timezone.utc).year
        end_date = datetime(current_year, month, day).date()

        location_identity = await resolve_location_cache_identity(location)
        slug = location_identity.redis_slug
        lookup_slugs = location_identity.lookup_slugs or (slug,)

        # Get year range (50 years back + current year)
        years = get_year_range(current_year)

        # Check for ETag conditional request
        if_none_match = request.headers.get("if-none-match")

        # Initialize variables for cache status tracking
        year_data = {}
        cache_status = "MISS"
        bundle_etag_computed = None
        celsius_records_for_temporal_cache = []

        if CACHE_ENABLED:
            bundle_resp = _try_serve_bundle_cache(
                redis_client=redis_client, period=period, location=location, identifier=identifier,
                slug=slug, lookup_slugs=lookup_slugs, month=month, day=day, current_year=current_year,
                years=years, unit_group=unit_group, if_none_match=if_none_match,
                invalid_location_cache=invalid_location_cache, end_date=end_date, response=response,
            )
            if bundle_resp is not None or response.status_code == 304:
                return bundle_resp

            temporal_resp = _try_serve_temporal_cache(
                redis_client=redis_client, period=period, location=location, identifier=identifier,
                month=month, day=day, current_year=current_year, years=years, unit_group=unit_group,
                if_none_match=if_none_match, invalid_location_cache=invalid_location_cache,
                location_identity=location_identity, end_date=end_date, response=response,
            )
            if temporal_resp is not None or response.status_code == 304:
                return temporal_resp

            # Step 2: MGET all year keys
            year_data, missing_past, missing_current = await get_records(
                redis_client, period, slug, identifier, years, lookup_slugs=lookup_slugs
            )

            if missing_past or missing_current:
                # If only the current year is missing, try to serve a stale bundle while a refresh job runs.
                if not missing_past and missing_current:
                    stale_resp = await _try_serve_stale_bundle_for_current_year(
                        redis_client=redis_client, period=period, location=location, identifier=identifier,
                        slug=slug, lookup_slugs=lookup_slugs, month=month, day=day,
                        current_year=current_year, years=years, unit_group=unit_group,
                        if_none_match=if_none_match, invalid_location_cache=invalid_location_cache,
                        end_date=end_date, response=response,
                    )
                    if stale_resp is not None or response.status_code == 304:
                        return stale_resp

                    # No stale bundle — enqueue job to fetch current year
                    _enqueue_record_refresh_job(
                        period, slug, identifier, current_year, location, "fetch current year"
                    )

                # Fetch missing years inline. When only the current year is missing
                # and there is no stale bundle to serve, we still need to fetch it
                # inline — otherwise it gets left out of the assembled bundle.
                await _inline_fetch_missing_years(
                    redis_client=redis_client, period=period, location=location, identifier=identifier,
                    slug=slug, current_year=current_year, missing_past=missing_past,
                    missing_current=missing_current, year_data=year_data,
                )

            # Step 4: Assemble response from per-year records
            if year_data:
                # Get per-year ETags
                year_etags = await get_year_etags(
                    redis_client, period, slug, identifier, list(year_data.keys()), lookup_slugs=lookup_slugs
                )

                # Assemble bundle and store with ETag (returns payload and ETag)
                bundle_payload, bundle_etag_computed = await assemble_and_cache(
                    redis_client, period, slug, identifier, year_data, year_etags
                )

                # Check ETag conditional request
                if if_none_match:
                    from cache.core import ETagGenerator

                    if ETagGenerator.matches_etag(bundle_etag_computed, if_none_match):
                        response.status_code = 304
                        response.headers["ETag"] = bundle_etag_computed
                        response.headers["X-Cache-Status"] = "HIT"
                        return None

                # Rebuild full response from year_data using helper
                values = [year_data[y] for y in sorted(year_data.keys())]
                celsius_records_for_temporal_cache = values
                data = _rebuild_full_response_from_values(
                    values, period, location, identifier, month, day, current_year, years, redis_client, unit_group
                )

                # Set cache status
                cache_status = "HIT" if not missing_past and not missing_current else "PARTIAL"
            else:
                # No cached data — fetch fresh synchronously (sync fallback path;
                # normal clients use the /async endpoint and only land here if
                # the async flow is unavailable).
                data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)

                # Extract and store per-year records (celsius)
                per_year_records = _extract_per_year_records(data)
                await _store_per_year_records(redis_client, period, slug, identifier, per_year_records, current_year)

                # Don't cache the bundle if current year is missing (incomplete data)
                current_year_missing = False
                if "metadata" in data and "missing_years" in data["metadata"]:
                    current_year_missing = any(
                        entry.get("year") == current_year for entry in data["metadata"]["missing_years"]
                    )

                if current_year_missing:
                    if DEBUG:
                        logger.debug(
                            f"Not caching: Current year {current_year} is missing from {location} {period}/{identifier}"
                        )
                    bundle_etag_computed = None
                else:
                    year_etags = await get_year_etags(
                        redis_client,
                        period,
                        slug,
                        identifier,
                        list(per_year_records.keys()),
                        lookup_slugs=lookup_slugs,
                    )
                    bundle_payload, bundle_etag_computed = await assemble_and_cache(
                        redis_client, period, slug, identifier, per_year_records, year_etags
                    )

                values_list = [per_year_records[y] for y in sorted(per_year_records.keys())]
                celsius_records_for_temporal_cache = values_list
                if values_list:
                    data = _rebuild_full_response_from_values(
                        values_list,
                        period,
                        location,
                        identifier,
                        month,
                        day,
                        current_year,
                        years,
                        redis_client,
                        unit_group,
                    )

                cache_status = "MISS"
        else:
            # Cache disabled, fetch fresh in celsius then convert for response
            celsius_data = await get_temperature_data_v1(location, period, identifier, "celsius", redis_client)
            per_year = _extract_per_year_records(celsius_data)
            values_list = [per_year[y] for y in sorted(per_year.keys())]
            if values_list:
                data = _rebuild_full_response_from_values(
                    values_list, period, location, identifier, month, day, current_year, years, redis_client, unit_group
                )
            else:
                data = celsius_data

        # Validate the response data
        is_valid, error_msg = validate_location_response(data, location)
        if not is_valid:
            invalid_location_cache.mark_location_invalid(location, "no_data")
            raise HTTPException(status_code=400, detail=error_msg)

        # Ensure updated timestamp
        if "updated" not in data:
            data["updated"] = datetime.now(timezone.utc).isoformat()

        if bundle_etag_computed is None:
            bundle_etag_computed = _resolve_bundle_etag(redis_client, period, slug, identifier, data)

        if CACHE_ENABLED and celsius_records_for_temporal_cache:
            try:
                temporal_payload = {"records": celsius_records_for_temporal_cache}
                temporal_cache_set(
                    redis_client,
                    agg=period,
                    original_location=location,
                    end_date=end_date,
                    payload=temporal_payload,
                    canonical_name=location_identity.canonical_name,
                )
            except Exception as e:
                logger.warning("Failed to store in temporal cache: %s", e)

        json_response = JSONResponse(content=data)
        json_response.headers["ETag"] = bundle_etag_computed
        json_response.headers["X-Cache-Status"] = cache_status if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(
            json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|metric|v1"
        )
        return json_response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in v1 records endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/average", response_model=SubResourceResponse)
async def get_record_average(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius", description="Temperature unit for response"),
    response: Response = None,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)] = None,
    invalid_location_cache: Annotated[InvalidLocationCache, Depends(get_invalid_location_cache)] = None,
):
    """Get average temperature data for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )

        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}",
            )

        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()

        # Get full record data using per-year caching
        record_data = await _get_record_data_internal(
            period, location, identifier, redis_client, invalid_location_cache, unit_group
        )

        # Extract average data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["average"],
            metadata=record_data["metadata"],
            timezone=record_data.get("timezone"),
        )

        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        json_response.headers["X-Cache-Status"] = "HIT" if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(
            json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|average|metric|v1"
        )
        return json_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records average endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/trend", response_model=SubResourceResponse)
async def get_record_trend(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius", description="Temperature unit for response"),
    response: Response = None,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)] = None,
    invalid_location_cache: Annotated[InvalidLocationCache, Depends(get_invalid_location_cache)] = None,
):
    """Get temperature trend data for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )

        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}",
            )

        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()

        # Get full record data using per-year caching
        record_data = await _get_record_data_internal(
            period, location, identifier, redis_client, invalid_location_cache, unit_group
        )

        # Extract trend data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["trend"],
            metadata=record_data["metadata"],
            timezone=record_data.get("timezone"),
        )

        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        json_response.headers["X-Cache-Status"] = "HIT" if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(
            json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|trend|metric|v1"
        )
        return json_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records trend endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/summary", response_model=SubResourceResponse)
async def get_record_summary(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius", description="Temperature unit for response"),
    response: Response = None,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)] = None,
    invalid_location_cache: Annotated[InvalidLocationCache, Depends(get_invalid_location_cache)] = None,
):
    """Get temperature summary text for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )

        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}",
            )

        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()

        # Get full record data using per-year caching
        record_data = await _get_record_data_internal(
            period, location, identifier, redis_client, invalid_location_cache, unit_group
        )

        # Extract summary data
        response_data = SubResourceResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=record_data["summary"],
            metadata=record_data["metadata"],
            timezone=record_data.get("timezone"),
        )

        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        json_response.headers["X-Cache-Status"] = "HIT" if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(
            json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|summary|metric|v1"
        )
        return json_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records summary endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/meta", response_model=MetaResponse)
async def get_record_meta(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    unit_group: Literal["celsius", "fahrenheit"] = Query("celsius", description="Temperature unit for response"),
    response: Response = None,
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)] = None,
    invalid_location_cache: Annotated[InvalidLocationCache, Depends(get_invalid_location_cache)] = None,
):
    """Get combined summary, average and trend data for a specific record."""
    try:
        # Quick validation for obviously invalid locations
        if is_location_likely_invalid(location):
            raise HTTPException(
                status_code=400, detail=f"Invalid location format: '{location}'. Please provide a valid location name."
            )

        # Check if location is known to be invalid
        if invalid_location_cache.is_invalid_location(location):
            invalid_info = invalid_location_cache.get_invalid_location_info(location)
            reason = invalid_info.get("reason", "invalid location") if invalid_info else "invalid location"
            raise HTTPException(
                status_code=400,
                detail=f"Location '{location}' is not supported by the weather service. Reason: {reason}",
            )

        # Parse identifier to get the end date for cache headers
        month, day, _ = parse_identifier(period, identifier)
        current_year = datetime.now().year
        end_date = datetime(current_year, month, day).date()

        # Get full record data using per-year caching
        record_data = await _get_record_data_internal(
            period, location, identifier, redis_client, invalid_location_cache, unit_group
        )

        # Combine summary, average, trend and ranking into a single response
        values = record_data.get("values", [])
        ranking_data = _build_ranking_block(values)
        current_val = next((v for v in values if v.get("year") == current_year), None)
        current_anomaly = current_val.get("anomaly") if current_val else None
        response_data = MetaResponse(
            period=record_data["period"],
            location=record_data["location"],
            identifier=record_data["identifier"],
            data=MetaData(
                summary=record_data["summary"],
                average=record_data["average"],
                trend=record_data["trend"],
                ranking=ranking_data,
                current_anomaly=current_anomaly,
            ),
            metadata=record_data["metadata"],
            timezone=record_data.get("timezone"),
        )

        # Create response with smart cache headers
        json_response = JSONResponse(content=response_data.model_dump())
        json_response.headers["X-Cache-Status"] = "HIT" if CACHE_ENABLED else "DISABLED"
        set_weather_cache_headers(
            json_response, req_date=end_date, key_parts=f"{location}|{identifier}|{period}|meta|metric|v1"
        )
        return json_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1 records meta endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/v1/records/{period}/{location}/{identifier}/updated", response_model=UpdatedResponse)
async def get_record_updated(
    period: Literal["daily", "weekly", "monthly", "yearly"] = Path(..., description="Data period"),
    location: str = Path(..., description="Location name", max_length=200),
    identifier: str = Path(..., description="Date identifier"),
    redis_client: redis.Redis = Depends(get_redis_client),
):
    """
    Get the last updated timestamp for a specific record endpoint.

    Returns when the data was last updated (cached) or null if it's never been queried.
    This endpoint is designed for web apps that want to check if they need to refetch data.
    """
    try:
        # Create the same cache key that would be used by the main endpoint
        normalized_location = await resolve_location_cache_slug(location)
        cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:values,average,trend,summary"

        # Get the updated timestamp from cache
        updated_timestamp = await get_cache_updated_timestamp(cache_key, redis_client)

        # Determine if data is cached
        is_cached = updated_timestamp is not None

        # Format timestamp as ISO string if available
        updated_iso = updated_timestamp.isoformat() if updated_timestamp else None

        return UpdatedResponse(
            period=period,
            location=location,
            identifier=identifier,
            updated=updated_iso,
            cached=is_cached,
            cache_key=cache_key,
        )

    except Exception as e:
        logger.error(f"Error in v1 records updated endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
