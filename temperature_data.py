"""
Core temperature data processing functions.
Extracted from main.py to avoid loading the entire FastAPI application when only these functions are needed.
"""

import os
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import Pydantic models directly to avoid loading main.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
import math

# Define the models locally to avoid importing from main.py
class TemperatureValue(BaseModel):
    """Individual temperature data point."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    year: int = Field(..., description="Year")
    temperature: float = Field(..., description="Temperature value")

class DateRange(BaseModel):
    """Date range for the data."""
    start: str = Field(..., description="Start date in YYYY-MM-DD format")
    end: str = Field(..., description="End date in YYYY-MM-DD format")
    years: int = Field(..., description="Number of years in range")

class AverageData(BaseModel):
    """Average temperature statistics."""
    mean: float = Field(..., description="Mean temperature")
    unit: str = Field("celsius", description="Temperature unit (celsius or fahrenheit)")
    data_points: int = Field(..., description="Number of data points used")

class TrendData(BaseModel):
    """Temperature trend analysis."""
    slope: float = Field(..., description="Temperature change per decade")
    unit: str = Field("°C/decade", description="Trend unit (changes based on temperature unit)")
    data_points: int = Field(..., description="Number of data points used")
    r_squared: Optional[float] = Field(None, description="R-squared value for trend fit")

# Import the necessary functions from main.py (this will still load main.py, but it's better than importing everything)
async def get_weather_for_date(location: str, date_str: str):
    """Get weather data for a specific location and date."""
    from constants import VC_BASE_URL
    import os
    import aiohttp
    
    API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
    if not API_KEY:
        raise RuntimeError("VISUAL_CROSSING_API_KEY is not configured")
    
    url = f"{VC_BASE_URL}/weatherdata/history"
    params = {
        "aggregateHours": 24,
        "contentType": "json",
        "unitGroup": "metric",
        "locations": location,
        "startDateTime": f"{date_str}T00:00:00",
        "endDateTime": f"{date_str}T23:59:59",
        "key": API_KEY,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers={"Accept-Encoding": "gzip"}) as r:
            r.raise_for_status()
            data = await r.json()
    
    # Extract temperature data
    if 'locations' in data and location in data['locations']:
        location_data = data['locations'][location]
        if 'values' in location_data and location_data['values']:
            # Get the first (and only) day's data
            day_data = location_data['values'][0]
            return {
                'average_temperature': day_data.get('temp'),
                'date': date_str
            }
    
    return None

async def _fetch_yearly_summary(location: str, start_year: int, end_year: int, unit_group: str = "metric"):
    """Fetch yearly summary data from Visual Crossing historysummary endpoint."""
    from constants import VC_BASE_URL
    import os
    import aiohttp
    
    API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
    if not API_KEY:
        raise RuntimeError("VISUAL_CROSSING_API_KEY is not configured")
    
    url = f"{VC_BASE_URL}/weatherdata/historysummary"
    params = {
        "aggregateHours": 24,
        "minYear": start_year,
        "maxYear": end_year,
        "chronoUnit": "years",
        "breakBy": "years",
        "dailySummaries": "false",
        "contentType": "json",
        "unitGroup": _vc_unit_group(unit_group),
        "locations": location,
        "key": API_KEY,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers={"Accept-Encoding": "gzip"}) as r:
            r.raise_for_status()
            data = await r.json()
    
    # Parse the response to extract yearly temperature data
    yearly_data = []
    if 'locations' in data and location in data['locations']:
        location_data = data['locations'][location]
        if 'values' in location_data:
            for value in location_data['values']:
                year = value.get('year')
                temp = value.get('temp')
                if year and temp is not None:
                    yearly_data.append((year, temp))
    
    return yearly_data

def _vc_unit_group(u: str) -> str:
    """Map our unit groups to Visual Crossing's expected values."""
    u = (u or "").lower()
    if u in ("c", "celsius", "metric", "si"):
        return "metric"
    if u in ("f", "fahrenheit", "us"):
        return "us"
    return "metric"

def track_missing_year(missing_years: List[Dict], year: int, reason: str):
    """Add a missing year entry to the missing_years list."""
    missing_years.append({"year": year, "reason": reason})

def calculate_historical_average(data: List[Dict[str, float]]) -> float:
    """Calculate historical average temperature."""
    if not data:
        return 0.0
    return sum(d.get('y', 0) for d in data) / len(data)

def calculate_trend_slope(data: List[Dict[str, float]]) -> float:
    """Calculate temperature trend slope."""
    if len(data) < 2:
        return 0.0
    
    n = len(data)
    x_values = list(range(n))
    y_values = [d.get('y', 0) for d in data]
    
    # Calculate slope using least squares
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    
    if denominator == 0:
        return 0.0
    
    slope = numerator / denominator
    return slope * 10  # Convert to per-decade

def generate_summary(data: List[Dict[str, float]], date: datetime, period: str = "daily") -> str:
    """Generate a summary text for temperature data."""
    # Filter out data points with None temperature
    data = [d for d in data if d.get('y') is not None]
    if not data or len(data) < 2:
        return "Not enough data to generate summary."
    
    # Get the most recent temperature
    latest_temp = data[-1].get('y', 0)
    
    # Calculate average
    avg_temp = sum(d.get('y', 0) for d in data) / len(data)
    
    # Generate summary
    if latest_temp > avg_temp:
        return f"{latest_temp:.1f}°C. It's warmer than average."
    else:
        return f"{latest_temp:.1f}°C. It's colder than average."

# Helper functions for parsing historysummary data
def _historysummary_values(payload):
    """Extract values from historysummary payload."""
    if 'locations' in payload:
        for location_data in payload['locations'].values():
            if 'values' in location_data:
                return location_data['values']
    return []

def _row_week_start(row):
    """Extract week start date from historysummary row."""
    return row.get('datetimeStr')

def _row_year(row):
    """Extract year from historysummary row."""
    return row.get('year')

def _row_mean_temp(row):
    """Extract mean temperature from historysummary row."""
    return row.get('temp')

def _row_month(row):
    """Extract month from historysummary row."""
    datetime_str = row.get('datetimeStr')
    if datetime_str:
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d").month
        except ValueError:
            return None
    return None

# Environment variables
USE_TIMELINE_APPROACH = os.getenv("USE_TIMELINE_APPROACH", "false").lower() == "true"

async def get_temperature_data_v1(location: str, period: str, identifier: str, unit_group: str = "celsius") -> Dict:
    """
    Get temperature data for a specific location, period, and identifier.
    This is the core function extracted from main.py to avoid loading the entire FastAPI app.
    """
    # Parse the identifier to get month and day
    month, day, _ = parse_identifier(period, identifier)
    
    # Get current year and calculate start year
    current_year = datetime.now().year
    start_year = current_year - 50  # 50 years of historical data
    
    # Initialize data structures
    all_temps = []
    values = []
    missing_years = []
    
    if period == "daily":
        # For daily data, fetch historical data for each year
        # This is simpler than the weekly/monthly/yearly logic since we don't need timeline approach
        for year in range(start_year, current_year + 1):
            try:
                # Use the Visual Crossing API to get historical data for the specific date
                date_str = f"{year}-{month:02d}-{day:02d}"
                weather_data = await get_weather_for_date(location, date_str)
                
                if weather_data and 'average_temperature' in weather_data:
                    temp = weather_data['average_temperature']
                    if temp is not None:
                        all_temps.append(temp)
                        values.append(TemperatureValue(
                            date=date_str,
                            year=year,
                            temperature=temp
                        ))
                else:
                    track_missing_year(missing_years, year, "No temperature data available")
            except Exception as e:
                track_missing_year(missing_years, year, f"API error: {str(e)}")
    
    elif period in ["weekly", "monthly"]:
        timeline_success = False
        if USE_TIMELINE_APPROACH:
            # Use timeline-based approach for more reliable data
            try:
                from routers.records_agg import _rolling_week_per_year_via_timeline, _rolling_30d_per_year_via_timeline, _historysummary_values, _row_week_start, _row_year, _row_mean_temp, _row_month
                
                if period == "weekly":
                    year_means = await _rolling_week_per_year_via_timeline(
                        location=location,
                        min_year=start_year,
                        max_year=current_year,
                        mm=month,
                        dd=day,
                        unit_group=unit_group
                    )
                else:  # monthly
                    year_means = await _rolling_30d_per_year_via_timeline(
                        location=location,
                        min_year=start_year,
                        max_year=current_year,
                        mm=month,
                        dd=day,
                        unit_group=unit_group
                    )
                
                # Convert to TemperatureValue objects and track missing years
                for year in range(start_year, current_year + 1):
                    temp = year_means.get(year)
                    if temp is not None:
                        all_temps.append(temp)
                        values.append(TemperatureValue(
                            date=f"{year}-{month:02d}-{day:02d}",
                            year=year,
                            temperature=round(temp, 1),
                        ))
                    else:
                        track_missing_year(missing_years, year, "insufficient_data_timeline")
                
                # Timeline approach completed successfully
                timeline_success = True
                        
            except Exception as e:
                # Fallback to old method if timeline fails
                timeline_success = False
        
        if not USE_TIMELINE_APPROACH or not timeline_success:
            # Use historysummary for weekly/monthly data (fallback or default)
            payload = await _fetch_yearly_summary(location, start_year, current_year, unit_group)
            rows = _historysummary_values(payload)
            
            if period == "weekly":
                # For weekly, we need to find the week containing the target date
                target_week = datetime.strptime(f"{current_year}-{month:02d}-{day:02d}", "%Y-%m-%d").isocalendar().week
                for r in rows:
                    if _row_week_start(r):
                        try:
                            week_start = datetime.strptime(_row_week_start(r), "%Y-%m-%d")
                            if week_start.isocalendar().week == target_week:
                                y = _row_year(r)
                                t = _row_mean_temp(r)
                                if y and t is not None:
                                    all_temps.append(t)
                                    values.append(TemperatureValue(
                                        date=f"{y}-{month:02d}-{day:02d}",
                                        year=y,
                                        temperature=round(t, 1),
                                    ))
                                else:
                                    track_missing_year(missing_years, y, "insufficient_data_historysummary")
                        except ValueError:
                            continue
            else:  # monthly
                for r in rows:
                    m = _row_month(r)
                    if m == month:
                        y = _row_year(r)
                        t = _row_mean_temp(r)
                        if y and t is not None:
                            all_temps.append(t)
                            values.append(TemperatureValue(
                                date=f"{y}-{month:02d}-{day:02d}",
                                year=y,
                                temperature=round(t, 1),
                            ))
                        else:
                            track_missing_year(missing_years, y, "insufficient_data_historysummary")
    
    elif period == "yearly":
        timeline_success = False
        if USE_TIMELINE_APPROACH:
            # Use timeline-based approach for yearly data
            try:
                from routers.records_agg import rolling_year_per_year_via_timeline
                
                year_means = await rolling_year_per_year_via_timeline(
                    location=location,
                    min_year=start_year,
                    max_year=current_year,
                    end_month=month,
                    end_day=day,
                    unit_group=unit_group
                )
                
                # Convert to TemperatureValue objects and track missing years
                for year in range(start_year, current_year + 1):
                    temp = year_means.get(year)
                    if temp is not None:
                        all_temps.append(temp)
                        values.append(TemperatureValue(
                            date=f"{year}-{month:02d}-{day:02d}",
                            year=year,
                            temperature=round(temp, 1),
                        ))
                    else:
                        track_missing_year(missing_years, year, "insufficient_data_timeline")
                
                # Timeline approach completed successfully
                timeline_success = True
                        
            except Exception as e:
                # Fallback to old method if timeline fails
                timeline_success = False
        
        if not USE_TIMELINE_APPROACH or not timeline_success:
            # Use historysummary for yearly data (fallback or default)
            payload = await _fetch_yearly_summary(location, start_year, current_year, unit_group)
            rows = _historysummary_values(payload)
            
            for r in rows:
                y = _row_year(r)
                t = _row_mean_temp(r)
                if y and t is not None:
                    all_temps.append(t)
                    values.append(TemperatureValue(
                        date=f"{y}-{month:02d}-{day:02d}",
                        year=y,
                        temperature=round(t, 1),
                    ))
                else:
                    track_missing_year(missing_years, y, "insufficient_data_historysummary")
    
    # Calculate date range
    if values:
        start_year = min(v.year for v in values)
        end_year = max(v.year for v in values)
        range_data = DateRange(
            start=f"{start_year}-{month:02d}-{day:02d}",
            end=f"{end_year}-{month:02d}-{day:02d}",
            years=end_year - start_year + 1
        )
    else:
        range_data = DateRange(start="", end="", years=0)
    
    # Calculate average
    if all_temps:
        avg_data = AverageData(
            mean=round(sum(all_temps) / len(all_temps), 1),
            unit=unit_group,
            data_points=len(all_temps)
        )
    else:
        avg_data = AverageData(mean=0.0, unit=unit_group, data_points=0)
    
    # Calculate trend
    if len(all_temps) >= 2:
        trend_slope = calculate_trend_slope([{"y": temp} for temp in all_temps])
        trend_data = TrendData(
            slope=round(trend_slope, 2),
            unit="°C/decade" if unit_group == "celsius" else "°F/decade",
            data_points=len(all_temps)
        )
    else:
        trend_data = TrendData(slope=0.0, unit="°C/decade" if unit_group == "celsius" else "°F/decade", data_points=0)
    
    # Generate summary
    summary_data = [{"y": temp} for temp in all_temps]
    summary = generate_summary(summary_data, datetime.now(), period)
    
    # Create metadata
    metadata = create_metadata(
        total_years=current_year - start_year + 1,
        available_years=len(values),
        missing_years=missing_years,
        additional_metadata={
            "period_days": 7 if period == "weekly" else 30 if period == "monthly" else 365 if period == "yearly" else 1,
            "end_date": f"{current_year}-{month:02d}-{day:02d}"
        }
    )
    
    return {
        "period": period,
        "location": location,
        "identifier": identifier,
        "range": range_data.dict() if hasattr(range_data, 'dict') else range_data,
        "unit_group": unit_group,
        "values": [v.dict() if hasattr(v, 'dict') else v for v in values],
        "average": avg_data.dict() if hasattr(avg_data, 'dict') else avg_data,
        "trend": trend_data.dict() if hasattr(trend_data, 'dict') else trend_data,
        "summary": summary,
        "metadata": metadata
    }

def parse_identifier(period: str, identifier: str) -> tuple:
    """Parse the identifier string to extract month and day."""
    if period == "daily":
        # identifier is in MM-DD format
        month, day = map(int, identifier.split("-"))
        return month, day, None
    elif period in ["weekly", "monthly", "yearly"]:
        # identifier is in MM-DD format
        month, day = map(int, identifier.split("-"))
        return month, day, None
    else:
        raise ValueError(f"Invalid period: {period}")

def create_metadata(total_years: int, available_years: int, missing_years: List[Dict], 
                   additional_metadata: Dict = None) -> Dict:
    """Create metadata dictionary with completeness information."""
    completeness = (available_years / total_years * 100) if total_years > 0 else 0.0
    
    metadata = {
        "total_years": total_years,
        "available_years": available_years,
        "missing_years": missing_years,
        "completeness": round(completeness, 1)
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    return metadata
