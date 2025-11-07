"""Temperature calculation and summary generation utilities."""
from typing import List, Dict
from datetime import datetime, timedelta


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
    return round(avg_temp, 1)


def calculate_trend_slope(data: List[Dict[str, float]]) -> float:
    """Calculate temperature trend slope using linear regression.
    
    This function handles missing years correctly by using the actual years
    in the linear regression calculation. The slope represents the rate of
    temperature change per year, which is then converted to per decade.
    
    Args:
        data: List of dictionaries with 'x' (year) and 'y' (temperature) keys
        
    Returns:
        Slope in °C/decade, rounded to 2 decimal places
    """
    # Filter out None values
    data = [d for d in data if d.get('y') is not None]
    n = len(data)
    if n < 2:
        return 0.0

    # Sort by year to ensure proper ordering
    data = sorted(data, key=lambda d: d['x'])
    
    # Use actual years for the calculation - this is mathematically correct
    # Linear regression works with any x-values, not just consecutive integers
    sum_x = sum(p['x'] for p in data)
    sum_y = sum(p['y'] for p in data)
    sum_xy = sum(p['x'] * p['y'] for p in data)
    sum_xx = sum(p['x'] ** 2 for p in data)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x ** 2
    
    if denominator == 0:
        return 0.0
    
    # Calculate slope in °C per year
    slope_per_year = numerator / denominator
    
    # Convert to °C per decade
    slope_per_decade = slope_per_year * 10.0
    
    return round(slope_per_decade, 2)


def get_friendly_date(date: datetime) -> str:
    """Get a friendly date string with ordinal suffix."""
    day = date.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {date.strftime('%B')}"


def generate_summary(data: List[Dict[str, float]], date: datetime, period: str = "daily") -> str:
    """Generate a summary text for temperature data.
    
    Note: Time-sensitive summaries (e.g., "the past week/month/year") should have
    very short cache durations (minutes) as they become invalid quickly.
    """
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

    avg_temp = calculate_historical_average(data)
    diff = latest['y'] - avg_temp
    rounded_diff = round(diff, 1)

    friendly_date = get_friendly_date(date)
    warm_summary = ''
    cold_summary = ''
    temperature = f"{latest['y']}°C."

    # Determine tense based on date and period
    today = datetime.now().date()
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
        elif period_ended_recently:
            # Period ended recently - use past perfect for consistency
            tense_context = "had been"
            tense_context_alt = "had been"
            tense_warm_cold = "had been"
        else:
            # Period is in the past - use past tense
            tense_context = "was"
            tense_context_alt = "was"
            tense_warm_cold = "was"

    previous = [p for p in data[:-1] if p.get('y') is not None]
    is_warmest = all(latest['y'] >= p['y'] for p in previous)
    is_coldest = all(latest['y'] <= p['y'] for p in previous)

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
                    warm_summary = f"It's warmer than last year but not as warm as {last_warmer}."
                elif years_since <= 10:
                    warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} since {last_warmer}."
                else:
                    warm_summary = f"This {tense_warm_cold} the warmest {friendly_date} in {years_since} years."
            else:
                warm_summary = f"It's warmer than last year."
        elif latest['y'] < last_year_temp:
            # Colder than last year - find last colder year
            last_colder = next((p['x'] for p in reversed(previous) if p['y'] < latest['y']), None)
            if last_colder:
                years_since = int(latest['x'] - last_colder)
                if years_since == 2:
                    cold_summary = f"It's colder than last year but not as cold as {last_colder}."
                elif years_since <= 10:
                    cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} since {last_colder}."
                else:
                    cold_summary = f"This {tense_warm_cold} the coldest {friendly_date} in {years_since} years."
            else:
                cold_summary = f"It's colder than last year."

    # Generate period-appropriate language with correct tense and context
    if period == "daily":
        if target_date == today:
            period_context = "today"
            period_context_alt = "this date"
        elif target_date == yesterday:
            period_context = "yesterday"
            period_context_alt = "yesterday"
        else:
            period_context = "that day"
            period_context_alt = "that date"
    elif period == "weekly":
        if tense_context == "has been":
            period_context = "this week"
            period_context_alt = "this week"
        elif tense_context == "had been":
            period_context = "the past week"
            period_context_alt = "the past week"
        else:
            # For distant past weeks, use more specific language
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the week ending {day}{suffix} {target_date.strftime('%B')}"
            period_context_alt = period_context
    elif period == "monthly":
        if tense_context == "has been":
            period_context = "this month"
            period_context_alt = "this month"
        elif tense_context == "had been":
            period_context = "the past month"
            period_context_alt = "the past month"
        else:
            # For distant past months, use more specific language
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the month ending {day}{suffix} {target_date.strftime('%B')}"
            period_context_alt = period_context
    elif period == "yearly":
        if tense_context == "has been":
            period_context = "this year"
            period_context_alt = "this year"
        elif tense_context == "had been":
            period_context = "the past year"
            period_context_alt = "the past year"
        else:
            # For distant past years, use more specific language
            day = target_date.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            period_context = f"the year ending {day}{suffix} {target_date.strftime('%B %Y')}"
            period_context_alt = period_context
    else:
        if tense_context == "has been":
            period_context = "this period"
            period_context_alt = "this period"
        elif tense_context == "had been":
            period_context = "the past period"
            period_context_alt = "the past period"
        else:
            period_context = "that period"
            period_context_alt = "that period"

    if abs(diff) < 0.05:
        # Special case for yearly summaries to sound more natural
        if period == "yearly":
            avg_summary = f"It {tense_context_alt} an average year."
        else:
            avg_summary = f"It {tense_context_alt} about average for {period_context_alt}."
    elif diff > 0:
        # For weekly/monthly/yearly periods, use "It was" to avoid repetition with "has been"
        if period in ["weekly", "monthly", "yearly"]:
            avg_summary = "However, " if cold_summary else ""
            avg_summary += f"{'it' if cold_summary else 'It'} was {rounded_diff}°C warmer than average."
        else:
            # For daily periods, use the period context
            avg_summary = "However, " if cold_summary else ""
            if cold_summary:
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {rounded_diff}°C warmer than average."
            else:
                period_capitalised = period_context.capitalize()
                avg_summary += f"{period_capitalised} {tense_context} {rounded_diff}°C warmer than average."
    else:
        # For weekly/monthly/yearly periods, use "It was" to avoid repetition with "has been"
        if period in ["weekly", "monthly", "yearly"]:
            avg_summary = "However, " if warm_summary else ""
            avg_summary += f"{'it' if warm_summary else 'It'} was {abs(rounded_diff)}°C cooler than average."
        else:
            # For daily periods, use the period context
            avg_summary = "However, " if warm_summary else ""
            if warm_summary:
                period_lower = period_context.lower()
                avg_summary += f"{period_lower} {tense_context} {abs(rounded_diff)}°C cooler than average."
            else:
                period_capitalised = period_context.capitalize()
                avg_summary += f"{period_capitalised} {tense_context} {abs(rounded_diff)}°C cooler than average."

    return " ".join(filter(None, [temperature, warm_summary, cold_summary, avg_summary]))
