"""Temperature calculation and summary generation utilities."""

import math
from datetime import date as dt_date
from datetime import datetime, timedelta
from typing import Dict, List, Optional


def calculate_standard_deviation(values: List[float]) -> Optional[float]:
    """Calculate population standard deviation of a list of temperature values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(variance**0.5, 2)


def calculate_historical_average(data: List[Dict[str, float]]) -> float:
    """
    Calculate the average temperature using only historical data (excluding current year).
    Returns the average temperature rounded to 1 decimal place.
    """
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not data or len(data) < 2:
        return 0.0
    # Filter out None values
    historical_data = [p for p in data[:-1] if p.get("y") is not None]
    if not historical_data:
        return 0.0
    avg_temp = sum(p["y"] for p in historical_data) / len(historical_data)
    return round(avg_temp, 2)


def calculate_trend_slope(data: List[Dict[str, float]]) -> tuple[float, Optional[float], Optional[float]]:
    """Calculate temperature trend slope, R², and slope standard error using linear regression.

    Args:
        data: List of dictionaries with 'x' (year) and 'y' (temperature) keys

    Returns:
        Tuple of (slope in °C/decade, R², slope_error in °C/decade), rounded to 2 decimal places.
        R² and slope_error are None when they cannot be computed.
    """
    data = [d for d in data if d.get("y") is not None]
    n = len(data)
    if n < 2:
        return 0.0, None, None

    data = sorted(data, key=lambda d: d["x"])

    sum_x = sum(p["x"] for p in data)
    sum_y = sum(p["y"] for p in data)
    sum_xy = sum(p["x"] * p["y"] for p in data)
    sum_xx = sum(p["x"] ** 2 for p in data)

    denominator = n * sum_xx - sum_x**2
    if denominator == 0:
        return 0.0, None, None

    slope_per_year = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope_per_year * sum_x) / n

    ss_res = sum((p["y"] - (slope_per_year * p["x"] + intercept)) ** 2 for p in data)
    mean_y = sum_y / n
    ss_tot = sum((p["y"] - mean_y) ** 2 for p in data)

    r_squared = round(1 - ss_res / ss_tot, 2) if ss_tot != 0 else None

    # SE of slope per year = sqrt(SS_res / ((n-2) * SS_xx)); SS_xx = denominator / n
    slope_error = None
    if n > 2:
        ss_xx = denominator / n
        slope_error = (ss_res / ((n - 2) * ss_xx)) ** 0.5 * 10.0
        # Inflate SE for missing years: sparse data over a long span has more
        # uncertainty than OLS on the observed points alone can express.
        year_span = data[-1]["x"] - data[0]["x"] + 1
        if year_span > n:
            slope_error *= (year_span / n) ** 0.5
        slope_error = round(slope_error, 2)

    return round(slope_per_year * 10.0, 2), r_squared, slope_error


def calculate_gradient_factor(
    slope: float,
    slope_error: float,
    unit_group: str = "celsius",
) -> float:
    """Return a [-1, 1] UI intensity factor for the colour gradient engine.

    Penalises the slope by its standard error, then compresses via tanh so
    even extreme slopes are bounded. scale_factor is unit-aware: 0.5 for
    °C/decade, 0.9 for °F/decade (≈ 0.5 × 1.8).

    A higher value for Z will increase suppression of trends with high error margins.
    """
    Z = 0.95
    scale_factor = 0.9 if unit_group.lower() in ("fahrenheit", "us") else 0.5
    adjusted_abs = max(0.0, abs(slope) - Z * slope_error)
    intensity = math.tanh(adjusted_abs / scale_factor)
    return round(math.copysign(intensity, slope), 4)


def get_friendly_date(date: datetime) -> str:
    """Get a friendly date string with ordinal suffix."""
    day = date.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {date.strftime('%B')}"


# ---------------------------------------------------------------------------
# Private helpers for generate_summary
# ---------------------------------------------------------------------------


def _ordinal_suffix(day: int) -> str:
    return "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _period_includes_today(period: str, target_date: dt_date, today: dt_date) -> bool:
    if period == "weekly":
        return target_date - timedelta(days=6) <= today <= target_date
    if period == "monthly":
        month_start = target_date.replace(day=1)
        month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        return month_start <= today <= month_end
    if period == "yearly":
        return target_date.year == today.year
    return target_date == today


def _get_tense(period: str, target_date: dt_date, today: dt_date) -> str:
    """Return the appropriate tense string for a period and date."""
    if period == "daily":
        return "is" if target_date == today else "was"
    return "has been" if _period_includes_today(period, target_date, today) else "was"


def _get_period_context(period: str, target_date: dt_date, tense: str, today: dt_date) -> tuple[str, str]:
    """Return (period_context, period_context_alt) strings for summary phrasing."""
    if period == "daily":
        yesterday = today - timedelta(days=1)
        if target_date == today:
            return "today", "the time of year"
        if target_date == yesterday:
            return "yesterday", "the time of year"
        return "it", "the time of year"

    if tense == "has been":
        labels = {"weekly": "this week", "monthly": "this month", "yearly": "this year"}
        label = labels.get(period, "this period")
        return label, label

    day = target_date.day
    suffix = _ordinal_suffix(day)
    if period == "weekly":
        ctx = f"the week ending {day}{suffix} {target_date.strftime('%B')}"
    elif period == "monthly":
        ctx = f"the month ending {day}{suffix} {target_date.strftime('%B')}"
    elif period == "yearly":
        ctx = f"the year ending {day}{suffix} {target_date.strftime('%B %Y')}"
    else:
        return "that period", "that period"
    return ctx, ctx


def _summarise_warmer_than_last(
    latest: dict, previous: list, tense: str, friendly_date: str, is_above_average: bool
) -> tuple[str, str]:
    """Summaries for the case where the current year is warmer than the previous year."""
    last_warmer = next((p["x"] for p in reversed(previous) if p["y"] > latest["y"]), None)
    if last_warmer is None:
        if is_above_average:
            return "It's warmer than last year.", ""
        return "", "It's not as cold as last year."
    years_since = int(latest["x"] - last_warmer)
    if years_since == 2:
        if is_above_average:
            return f"It's warmer than last year but not as warm as {last_warmer}.", ""
        return "", f"It's not as cold as last year but cooler than {last_warmer}."
    if years_since <= 10:
        return f"This {tense} the warmest {friendly_date} since {last_warmer}.", ""
    return f"This {tense} the warmest {friendly_date} in {years_since} years.", ""


def _summarise_colder_than_last(
    latest: dict, previous: list, tense: str, friendly_date: str, is_above_average: bool
) -> tuple[str, str]:
    """Summaries for the case where the current year is colder than the previous year."""
    last_colder = next((p["x"] for p in reversed(previous) if p["y"] < latest["y"]), None)
    if last_colder is None:
        if is_above_average:
            return "It's not as warm as last year.", ""
        return "", "It's colder than last year."
    years_since = int(latest["x"] - last_colder)
    if years_since == 2:
        if is_above_average:
            return f"It's not as warm as last year but warmer than {last_colder}.", ""
        return "", f"It's colder than last year but not as cold as {last_colder}."
    if years_since <= 10:
        return "", f"This {tense} the coldest {friendly_date} since {last_colder}."
    return "", f"This {tense} the coldest {friendly_date} in {years_since} years."


def _compare_against_years(
    latest: dict, previous: list, tense: str, friendly_date: str, is_above_average: bool
) -> tuple[str, str]:
    """Return (warm_summary, cold_summary) from year-by-year comparison."""
    is_warmest = bool(previous) and all(latest["y"] > p["y"] for p in previous)
    is_coldest = bool(previous) and all(latest["y"] < p["y"] for p in previous)
    if is_warmest:
        return f"This {tense} the warmest {friendly_date} on record.", ""
    if is_coldest:
        return "", f"This {tense} the coldest {friendly_date} on record."
    last_year_temp = next((p["y"] for p in reversed(previous) if p["x"] == latest["x"] - 1), None)
    if last_year_temp is None:
        return "", ""
    if latest["y"] > last_year_temp:
        return _summarise_warmer_than_last(latest, previous, tense, friendly_date, is_above_average)
    if latest["y"] < last_year_temp:
        return _summarise_colder_than_last(latest, previous, tense, friendly_date, is_above_average)
    return "", ""


def _build_avg_summary(
    diff: float,
    rounded_diff,
    period: str,
    period_context: str,
    period_context_alt: str,
    tense: str,
    warm_summary: str,
    cold_summary: str,
    unit_symbol: str,
) -> str:
    """Build the sentence comparing current temperature to the historical average."""
    if rounded_diff == 0:
        if period == "yearly":
            return f"It {tense} an average year."
        return f"It {tense} about average for {period_context_alt}."

    amount = abs(rounded_diff)
    direction = "warmer" if diff > 0 else "cooler"
    contrary = cold_summary if diff > 0 else warm_summary

    if period in ("weekly", "monthly", "yearly"):
        prefix = "However, it" if contrary else "It"
        return f"{prefix} was {amount}{unit_symbol} {direction} than average."

    # daily
    if contrary:
        return f"However, {period_context.lower()} {tense} {amount}{unit_symbol} {direction} than average for the time of year."
    return f"{period_context.capitalize()} {tense} {amount}{unit_symbol} {direction} than average for the time of year."


def generate_summary(
    data: List[Dict[str, float]],
    date: datetime,
    period: str = "daily",
    unit_group: str = "celsius",
    mean: Optional[float] = None,
    local_today: Optional[dt_date] = None,
) -> str:
    """Generate a summary text for temperature data in the requested unit.

    Args:
        data: List of dicts with 'x' (year) and 'y' (temperature) keys. Temperatures should already be in the target unit.
        date: Reference date for the summary
        period: Data period ("daily", "weekly", "monthly", or "yearly")
        unit_group: Temperature unit ("celsius" or "fahrenheit")
        mean: Pre-calculated mean temperature. When provided, used directly so the
              summary is consistent with the `anomaly` and `average.mean` fields in
              the API response. If omitted, falls back to calculate_historical_average.
        local_today: The current date in the user's/location's local timezone. When
              provided, "today"/"yesterday" comparisons use this instead of the
              server's local date. Callers should compute this from the requested
              location's timezone so summaries are correct across timezones.

    Note: Time-sensitive summaries (e.g., "the past week/month/year") should have
    very short cache durations (minutes) as they become invalid quickly.
    """
    unit_symbol = "°F" if unit_group.lower() == "fahrenheit" else "°C"

    data = [d for d in data if d.get("y") is not None]
    if not data or len(data) < 2:
        return "Not enough data to generate summary."

    expected_year = date.year
    latest = data[-1]
    if latest.get("x") != expected_year:
        return f"Temperature data for {date.year} is not yet available."
    if latest.get("y") is None:
        return "No valid temperature data for the latest year."

    avg_temp = round(mean, 2) if mean is not None else calculate_historical_average(data)
    diff = latest["y"] - avg_temp
    is_above_average = latest["y"] > avg_temp
    is_fahrenheit = unit_group.lower() == "fahrenheit"
    rounded_diff = int(round(diff, 0)) if is_fahrenheit else round(diff, 1)
    latest_temp = int(round(latest["y"], 0)) if is_fahrenheit else round(latest["y"], 1)

    today = local_today if local_today is not None else datetime.now().date()
    target_date = date.date()
    tense = _get_tense(period, target_date, today)

    previous = [p for p in data[:-1] if p.get("y") is not None]
    friendly_date = get_friendly_date(date)
    warm_summary, cold_summary = _compare_against_years(latest, previous, tense, friendly_date, is_above_average)

    period_context, period_context_alt = _get_period_context(period, target_date, tense, today)

    temperature = f"{latest_temp}{unit_symbol}."
    avg_summary = _build_avg_summary(
        diff, rounded_diff, period, period_context, period_context_alt, tense, warm_summary, cold_summary, unit_symbol
    )

    return " ".join(filter(None, [temperature, warm_summary, cold_summary, avg_summary]))
