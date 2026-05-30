"""Temperature calculation and summary generation utilities."""
import math
from typing import List, Dict, Optional
from datetime import datetime, timedelta, date as dt_date


def calculate_standard_deviation(values: List[float]) -> Optional[float]:
    """Calculate population standard deviation of a list of temperature values."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(variance ** 0.5, 2)


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
    historical_data = [p for p in data[:-1] if p.get('y') is not None]
    if not historical_data:
        return 0.0
    avg_temp = sum(p['y'] for p in historical_data) / len(historical_data)
    return round(avg_temp, 2)


def calculate_trend_slope(data: List[Dict[str, float]]) -> tuple[float, Optional[float], Optional[float]]:
    """Calculate temperature trend slope, R², and slope standard error using linear regression.

    Args:
        data: List of dictionaries with 'x' (year) and 'y' (temperature) keys

    Returns:
        Tuple of (slope in °C/decade, R², slope_error in °C/decade), rounded to 2 decimal places.
        R² and slope_error are None when they cannot be computed.
    """
    data = [d for d in data if d.get('y') is not None]
    n = len(data)
    if n < 2:
        return 0.0, None, None

    data = sorted(data, key=lambda d: d['x'])

    sum_x = sum(p['x'] for p in data)
    sum_y = sum(p['y'] for p in data)
    sum_xy = sum(p['x'] * p['y'] for p in data)
    sum_xx = sum(p['x'] ** 2 for p in data)

    denominator = n * sum_xx - sum_x ** 2
    if denominator == 0:
        return 0.0, None, None

    slope_per_year = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope_per_year * sum_x) / n

    ss_res = sum((p['y'] - (slope_per_year * p['x'] + intercept)) ** 2 for p in data)
    mean_y = sum_y / n
    ss_tot = sum((p['y'] - mean_y) ** 2 for p in data)

    r_squared = round(1 - ss_res / ss_tot, 2) if ss_tot != 0 else None

    # SE of slope per year = sqrt(SS_res / ((n-2) * SS_xx)); SS_xx = denominator / n
    slope_error = None
    if n > 2:
        ss_xx = denominator / n
        slope_error = (ss_res / ((n - 2) * ss_xx)) ** 0.5 * 10.0
        # Inflate SE for missing years: sparse data over a long span has more
        # uncertainty than OLS on the observed points alone can express.
        year_span = data[-1]['x'] - data[0]['x'] + 1
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


def generate_summary(data: List[Dict[str, float]], date: datetime, period: str = "daily", unit_group: str = "celsius", mean: Optional[float] = None, local_today: Optional[dt_date] = None) -> str:
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
    # Determine unit symbol
    unit_symbol = "°F" if unit_group.lower() == "fahrenheit" else "°C"

    # Filter out data points with None temperature
    data = [d for d in data if d.get('y') is not None]
    if not data or len(data) < 2:
        return "Not enough data to generate summary."

    # Check if we have data for the expected year (from the date parameter)
    expected_year = date.year
    latest = data[-1]

    # Verify the latest data point is actually for the expected year
    if latest.get('x') != expected_year:
        # Current year data is missing - don't generate a misleading summary
        return f"Temperature data for {date.year} is not yet available."

    if latest.get('y') is None:
        return "No valid temperature data for the latest year."

    avg_temp = round(mean, 2) if mean is not None else calculate_historical_average(data)
    diff = latest['y'] - avg_temp
    is_above_average = latest['y'] > avg_temp
    is_fahrenheit = unit_group.lower() == "fahrenheit"
    rounded_diff = int(round(diff, 0)) if is_fahrenheit else round(diff, 1)
    latest_temp = int(round(latest['y'], 0)) if is_fahrenheit else round(latest['y'], 1)

    friendly_date = get_friendly_date(date)
    warm_summary = ''
    cold_summary = ''
    temperature = f"{latest_temp}{unit_symbol}."

    # Determine tense based on date and period.
    # Use the caller-supplied local_today (location's timezone) when available so
    # "today"/"yesterday" reflect the user's local date, not the server's.
    today = local_today if local_today is not None else datetime.now().date()
    yesterday = today - timedelta(days=1)
    target_date = date.date()
    
    # For non-daily periods, determine if the period includes today or ended recently
    if period == "daily":
        if target_date == today:
            # Today - use present tense
            tense_context = "is"
            tense_context_alt = "is"
            tense_warm_cold = "is"
        elif target_date == yesterday:
            # Yesterday - use past tense but keep the actual date
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"
        else:
            # Past date - use past tense
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"
    else:
        # For weekly, monthly, yearly periods
        if period == "weekly":
            # Check if the week ending on target_date includes today
            week_start = target_date - timedelta(days=6)
            period_includes_today = week_start <= today <= target_date
            period_ended_recently = target_date == yesterday or target_date == today - timedelta(days=2)
        elif period == "monthly":
            # Check if the month containing target_date includes today
            month_start = target_date.replace(day=1)
            next_month = (month_start + timedelta(days=32)).replace(day=1)
            month_end = next_month - timedelta(days=1)
            period_includes_today = month_start <= today <= month_end
            # Check if the month ended recently
            period_ended_recently = (target_date.month == yesterday.month and target_date.year == yesterday.year) or \
                                  (target_date.month == (yesterday - timedelta(days=1)).month and target_date.year == (yesterday - timedelta(days=1)).year)
        elif period == "yearly":
            # Check if the target year is the current year
            period_includes_today = target_date.year == today.year
            period_ended_recently = target_date.year == yesterday.year
        else:
            # Default for other periods
            period_includes_today = target_date == today
            period_ended_recently = target_date == yesterday
        
        if period_includes_today:
            # Period includes today - use present perfect for consistency
            tense_context = "has been"
            tense_context_alt = "has been"
            tense_warm_cold = "has been"
        else:
            # Period ended recently or is in the past - use simple past tense
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"

    previous = [p for p in data[:-1] if p.get('y') is not None]
    is_warmest = bool(previous) and all(latest['y'] > p['y'] for p in previous)
    is_coldest = bool(previous) and all(latest['y'] < p['y'] for p in previous)

    # Check against last year first for consistency
    last_year_temp = next((p['y'] for p in reversed(previous) if p['x'] == latest['x'] - 1), None)
    
    # Generate mutually exclusive summaries to avoid contradictions
    if is_warmest:
        warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} on record."
    elif is_coldest:
        cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} on record."
    elif last_year_temp is not None:
        # Compare against last year first
        if latest['y'] > last_year_temp:
            # Warmer than last year - find last warmer year
            last_warmer = next((p['x'] for p in reversed(previous) if p['y'] > latest['y']), None)
            if last_warmer:
                years_since = int(latest['x'] - last_warmer)
                if years_since == 2:
                    if is_above_average:
                        warm_summary = f"It's warmer than last year but not as warm as {last_warmer}."
                    else:
                        cold_summary = f"It's not as cold as last year but cooler than {last_warmer}."
                elif years_since <= 10:
                    warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} since {last_warmer}."
                else:
                    warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} in {years_since} years."
            else:
                if is_above_average:
                    warm_summary = f"It's warmer than last year."
                else:
                    cold_summary = f"It's not as cold as last year."
        elif latest['y'] < last_year_temp:
            # Colder than last year - find last colder year
            last_colder = next((p['x'] for p in reversed(previous) if p['y'] < latest['y']), None)
            if last_colder:
                years_since = int(latest['x'] - last_colder)
                if years_since == 2:
                    if is_above_average:
                        warm_summary = f"It's not as warm as last year but warmer than {last_colder}."
                    else:
                        cold_summary = f"It's colder than last year but not as cold as {last_colder}."
                elif years_since <= 10:
                    cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} since {last_colder}."
                else:
                    cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} in {years_since} years."
            else:
                if is_above_average:
                    warm_summary = f"It's not as warm as last year."
                else:
                    cold_summary = f"It's colder than last year."

    # Generate period-appropriate language with correct tense and context
    if period == "daily":
        if target_date == today:
            period_context = "today"
            period_context_alt = "the time of year"
        elif target_date == yesterday:
            period_context = "yesterday"
            period_context_alt = "the time of year"
        else:
            period_context = "it"
            period_context_alt = "the time of year"
    elif period == "weekly":
        if tense_context == "has been":
            period_context = "this week"
            period_context_alt = "this week"
        else:
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the week ending {day}{suffix} {target_date.strftime('%B')}"
            period_context_alt = period_context
    elif period == "monthly":
        if tense_context == "has been":
            period_context = "this month"
            period_context_alt = "this month"
        else:
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the month ending {day}{suffix} {target_date.strftime('%B')}"
            period_context_alt = period_context
    elif period == "yearly":
        if tense_context == "has been":
            period_context = "this year"
            period_context_alt = "this year"
        else:
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the year ending {day}{suffix} {target_date.strftime('%B %Y')}"
            period_context_alt = period_context
    else:
        if tense_context == "has been":
            period_context = "this period"
            period_context_alt = "this period"
        else:
            period_context = "that period"
            period_context_alt = "that period"

    if rounded_diff == 0:
        # Special case for yearly summaries to sound more natural
        if period == "yearly":
            avg_summary = f"It {tense_context_alt} an average year."
        else:
            avg_summary = f"It {tense_context_alt} about average for {period_context_alt}."
    elif diff > 0:
        # For weekly/monthly/yearly periods, use "It was" to avoid repetition with "has been"
        if period in ["weekly", "monthly", "yearly"]:
            avg_summary = "However, " if cold_summary else ""
            avg_summary += f"{'it' if cold_summary else 'It'} was {rounded_diff}{unit_symbol} warmer than average."
        else:
            # For daily periods, use the period context
            avg_summary = "However, " if cold_summary else ""
            if cold_summary:
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {rounded_diff}{unit_symbol} warmer than average for the time of year."
            else:
                period_capitalised = period_context.capitalize()
                avg_summary += f"{period_capitalised} {tense_context} {rounded_diff}{unit_symbol} warmer than average for the time of year."
    else:
        # For weekly/monthly/yearly periods, use "It was" to avoid repetition with "has been"
        if period in ["weekly", "monthly", "yearly"]:
            avg_summary = "However, " if warm_summary else ""
            avg_summary += f"{'it' if warm_summary else 'It'} was {abs(rounded_diff)}{unit_symbol} cooler than average."
        else:
            # For daily periods, use the period context
            avg_summary = "However, " if warm_summary else ""
            if warm_summary:
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {abs(rounded_diff)}{unit_symbol} cooler than average for the time of year."
            else:
                period_capitalised = period_context.capitalize()
                avg_summary += f"{period_capitalised} {tense_context} {abs(rounded_diff)}{unit_symbol} cooler than average for the time of year."

    return " ".join(filter(None, [temperature, warm_summary, cold_summary, avg_summary]))
