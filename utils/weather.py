"""Weather data utility functions."""
from typing import List, Dict
from datetime import date as dt_date, timedelta
from config import (
    FORECAST_DAY_CACHE_DURATION_SECONDS, FORECAST_NIGHT_CACHE_DURATION_SECONDS
)


def get_year_range(current_year: int, years_back: int = 50) -> List[int]:
    """Get a list of years for historical data analysis."""
    return list(range(current_year - years_back, current_year + 1))


def create_metadata(total_years: int, available_years: int, missing_years: List[Dict], 
                   additional_metadata: Dict = None) -> Dict:
    """Create standardized metadata for temperature data responses."""
    metadata = {
        "total_years": total_years,
        "available_years": available_years,
        "missing_years": missing_years,
        "completeness": round(available_years / total_years * 100, 1) if total_years > 0 else 0.0
    }
    if additional_metadata:
        metadata.update(additional_metadata)
    return metadata


def track_missing_year(missing_years: List[Dict], year: int, reason: str):
    """Add a missing year entry to the missing_years list."""
    missing_years.append({"year": year, "reason": reason})


def is_today(year: int, month: int, day: int) -> bool:
    """Check if the given date is today."""
    today = dt_date.today()
    return year == today.year and month == today.month and day == today.day


def is_today_or_future(year: int, month: int, day: int) -> bool:
    """Check if the given date is today or in the future."""
    today = dt_date.today()
    date = dt_date(year, month, day)
    return date >= today


def get_forecast_cache_duration() -> timedelta:
    """Get appropriate cache duration for forecast data based on time of day.
    
    Returns:
        timedelta: Short duration during active forecast hours (30 min), longer when stable (2 hours)
    """
    from datetime import datetime
    current_hour = datetime.now().hour
    
    # Stable hours (6 PM to Midnight) - forecast is more stable
    if current_hour >= 18:
        return timedelta(seconds=FORECAST_NIGHT_CACHE_DURATION_SECONDS)
    else:
        # Active hours (Midnight to 6 PM) - forecast can change frequently
        return timedelta(seconds=FORECAST_DAY_CACHE_DURATION_SECONDS)
