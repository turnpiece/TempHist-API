from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
import redis
import json
from datetime import datetime, timedelta
from typing import List, Dict

load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

app = FastAPI()
redis_client = redis.from_url(REDIS_URL)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite default port
        "https://temphist.onrender.com",  # Render frontend
        "https://*.onrender.com",  # Any Render subdomain
        "https://temphist.com",  # Main domain
        "https://www.temphist.com",  # www subdomain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

@app.api_route("/test-cors", methods=["GET", "OPTIONS"])
async def test_cors():
    """Test endpoint for CORS"""
    return {"message": "CORS is working"}

@app.api_route("/", methods=["GET", "OPTIONS"])
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "Temperature History API",
        "version": "1.0.0",
        "endpoints": [
            "/average/{location}/{month_day}",
            "/trend/{location}/{month_day}",
            "/weather/{location}/{date}",
            "/summary/{location}/{month_day}",
            "/forecast/{location}"
        ]
    }

def fetch_weather_from_api(location: str, date: str):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date}?unitGroup=metric&include=days&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
        return response.json()
    return {"error": response.text, "status": response.status_code}

def get_forecast_data(location: str, date: datetime.date) -> Dict:
    """
    Get forecast data for a specific location and date.
    Returns cached data if available, otherwise fetches from OpenWeatherMap API.
    """
    if not OPENWEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenWeatherMap API key not configured")
    
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"forecast_{location.lower()}_{date_str}"
    
    # Check cache first
    cached_data = redis_client.get(cache_key)
    if cached_data:
        print(f"Cache hit: {cache_key}")
        return json.loads(cached_data)
    
    print(f"Cache miss: {cache_key} — fetching from API")
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Filter forecasts for the specified date
        date_forecasts = [
            item for item in data['list']
            if datetime.fromtimestamp(item['dt']).date() == date
        ]
        
        if not date_forecasts:
            raise HTTPException(status_code=404, detail=f"No forecast data available for {date_str}")
        
        # Calculate maximum temperature for the day
        max_temp = max(item['main']['temp'] for item in date_forecasts)
        
        result = {
            "location": location,
            "date": date_str,
            "average_temperature": round(max_temp, 1),  # Using max_temp instead of average
            "unit": "celsius"
        }
        
        # Cache the result for 24 hours
        redis_client.setex(cache_key, timedelta(hours=24), json.dumps(result))
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching forecast data: {str(e)}")

def get_temperature_series(location: str, month: int, day: int) -> List[Dict[str, float]]:
    today = datetime.now()
    current_year = today.year
    years = list(range(current_year - 50, current_year + 1))

    data = []
    for year in years:
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = f"{location.lower()}_{date_str}"
        
        # If this is today's date, use the forecast data
        if year == current_year and month == today.month and day == today.day:
            try:
                forecast_data = get_forecast_data(location, today.date())
                # Use the forecast temperature for the current day
                data.append({"x": year, "y": forecast_data["average_temperature"]})
                continue
            except Exception as e:
                print(f"Error fetching forecast: {str(e)}")
                # Fall back to historical data if forecast fails
        
        # Use historical data for all other dates
        cached_data = redis_client.get(cache_key)
        if cached_data:
            weather = json.loads(cached_data)
        else:
            weather = fetch_weather_from_api(location, date_str)
            if "error" not in weather:
                redis_client.setex(cache_key, timedelta(hours=24), json.dumps(weather))
            else:
                continue

        try:
            # For historical data, use the maximum temperature of the day
            # This gives us a better comparison with the forecast temperature
            temp = weather["days"][0]["tempmax"]
            print(f"Year {year}: temp={weather['days'][0]['temp']}, tempmax={temp}, tempmin={weather['days'][0]['tempmin']}")
            data.append({"x": year, "y": temp})
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing data for {date_str}: {str(e)}")
            continue

    # Print summary of collected data
    print("\nData summary:")
    print(f"Total data points: {len(data)}")
    if data:
        temps = [d['y'] for d in data]
        print(f"Temperature range: {min(temps):.1f}°C to {max(temps):.1f}°C")
        print(f"Average temperature: {sum(temps)/len(temps):.1f}°C")
    
    return data

@app.get("/weather/{location}/{date}")
def get_weather(location: str, date: str):
    cache_key = f"{location.lower()}_{date}"

    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"Cache hit: {cache_key}")
            return json.loads(cached_data)
        print(f"Cache miss: {cache_key} — fetching from API")

    result = fetch_weather_from_api(location, date)

    # Only cache successful results if caching is enabled
    if CACHE_ENABLED and "error" not in result:
        redis_client.setex(cache_key, timedelta(hours=24), json.dumps(result))

    return result

# get a text summary
@app.get("/summary/{location}/{month_day}")
async def summary(location: str, month_day: str, request: Request):
    try:
        month, day = map(int, month_day.split("-"))
        today = datetime.now()
        current_year = today.year
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month-day format. Use MM-DD.")

    data = get_temperature_series(location, month, day)
    print(f"Summary data for {location} on {month_day}:")
    print(f"Number of data points: {len(data)}")
    print(f"Date range: {data[0]['x']} to {data[-1]['x']}")
    print(f"Current temperature: {data[-1]['y']}°C")
    
    if len(data) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    avg_temp = calculate_historical_average(data)
    print(f"Calculated average: {avg_temp}°C")
    print(f"Temperature difference: {round(data[-1]['y'] - avg_temp, 1)}°C")

    summary_text = get_summary(data, f"{current_year}-{month:02d}-{day:02d}")
    return {"summary": summary_text}

def calculate_historical_average(data: List[Dict[str, float]]) -> float:
    """
    Calculate the average temperature using only historical data (excluding current year).
    Returns the average temperature rounded to 1 decimal place.
    """
    if not data or len(data) < 2:
        return 0.0
    
    # Use all data except the most recent (current year)
    historical_data = data[:-1]
    avg_temp = sum(p['y'] for p in historical_data) / len(historical_data)
    return round(avg_temp, 1)

def get_summary(data: List[Dict[str, float]], date_str: str) -> str:
    def get_friendly_date(date: datetime) -> str:
        day = date.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{day}{suffix} {date.strftime('%B')}"

    def generate_summary(data: List[Dict[str, float]], date: datetime) -> str:
        if not data or len(data) < 2:
            return "Not enough data to generate summary."

        latest = data[-1]
        # Use the new function to calculate average
        avg_temp = calculate_historical_average(data)
        diff = latest['y'] - avg_temp
        rounded_diff = round(diff, 1)

        friendly_date = get_friendly_date(date)
        warm_summary = ''
        cold_summary = ''

        # For historical comparisons, use all previous years
        previous = data[:-1]
        is_warmest = all(latest['y'] > p['y'] for p in previous)
        is_coldest = all(latest['y'] < p['y'] for p in previous)

        if is_warmest:
            warm_summary = f"This is the warmest {friendly_date} on record."
        else:
            last_warmer = next((p['x'] for p in reversed(previous) if p['y'] > latest['y']), None)
            if last_warmer:
                years_since = int(latest['x'] - last_warmer)
                if years_since > 1:
                    if years_since <= 2:
                        warm_summary = f"It's not as warm as {last_warmer}."
                    elif years_since <= 10:
                        warm_summary = f"This is the warmest {friendly_date} since {last_warmer}."
                    else:
                        warm_summary = f"This is the warmest {friendly_date} in {years_since} years."

        if is_coldest:
            cold_summary = f"This is the coldest {friendly_date} on record."
        else:
            last_colder = next((p['x'] for p in reversed(previous) if p['y'] < latest['y']), None)
            if last_colder:
                years_since = int(latest['x'] - last_colder)
                if years_since > 1:
                    if years_since <= 2:
                        cold_summary = f"It's not as cold as {last_colder}."
                    if years_since <= 10:
                        cold_summary = f"This is the coldest {friendly_date} since {last_colder}."
                    else:
                        cold_summary = f"This is the coldest {friendly_date} in {years_since} years."

        if abs(diff) < 0.05:
            avg_summary = "It is about average for this date."
        elif diff > 0:
            avg_summary = f"It is {rounded_diff}°C warmer than average today."
        else:
            avg_summary = f"It is {abs(rounded_diff)}°C cooler than average today."

        return " ".join(filter(None, [warm_summary, cold_summary, avg_summary]))

    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."

    return generate_summary(data, date)

# get the warming/cooling trend
@app.get("/trend/{location}/{month_day}")
async def trend(location: str, month_day: str):
    try:
        month, day = map(int, month_day.split("-"))
        # Validate month and day
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        # Additional validation for specific months
        if month in [4, 6, 9, 11] and day > 30:
            raise ValueError(f"Month {month} has only 30 days")
        if month == 2 and day > 29:
            raise ValueError("February has only 29 days")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create cache key for the trend
    cache_key = f"trend_{location.lower()}_{month:02d}_{day:02d}"
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"Cache hit: {cache_key}")
            return json.loads(cached_data)
        print(f"Cache miss: {cache_key} — calculating trend")

    data = get_temperature_series(location, month, day)

    if len(data) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    slope = calculate_trend_slope(data)
    result = {"slope": slope, "units": "°C/year"}
    
    # Cache the result if caching is enabled
    if CACHE_ENABLED:
        redis_client.setex(cache_key, timedelta(hours=24), json.dumps(result))
    
    return result

# get the average temperature
@app.api_route("/average/{location}/{month_day}", methods=["GET", "OPTIONS"])
async def average(location: str, month_day: str):
    try:
        month, day = map(int, month_day.split("-"))
        # Validate month and day
        if not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31")
        # Additional validation for specific months
        if month in [4, 6, 9, 11] and day > 30:
            raise ValueError(f"Month {month} has only 30 days")
        if month == 2 and day > 29:
            raise ValueError("February has only 29 days")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create cache key for the average
    cache_key = f"average_{location.lower()}_{month:02d}_{day:02d}"
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"Cache hit: {cache_key}")
            return json.loads(cached_data)
        print(f"Cache miss: {cache_key} — calculating average")

    data = get_temperature_series(location, month, day)

    if len(data) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    # Calculate average temperature using the new function
    avg_temp = calculate_historical_average(data)
    
    result = {
        "average": avg_temp,
        "unit": "celsius",
        "data_points": len(data),
        "year_range": {
            "start": data[0]['x'],
            "end": data[-1]['x']
        }
    }
    
    # Cache the result if caching is enabled
    if CACHE_ENABLED:
        redis_client.setex(cache_key, timedelta(hours=24), json.dumps(result))
    
    return result

def calculate_trend_slope(data: List[Dict[str, float]]) -> float:
    n = len(data)
    if n < 2:
        return 0.0

    sum_x = sum(p['x'] for p in data)
    sum_y = sum(p['y'] for p in data)
    sum_xy = sum(p['x'] * p['y'] for p in data)
    sum_xx = sum(p['x'] ** 2 for p in data)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x ** 2
    return round(numerator / denominator, 3) if denominator != 0 else 0.0

@app.get("/forecast/{location}")
async def get_forecast(location: str):
    today = datetime.now().date()
    return get_forecast_data(location, today)

@app.get("/test-redis")
async def test_redis():
    try:
        # Try to set a test value
        redis_client.setex("test_key", timedelta(minutes=5), "test_value")
        # Try to get the test value
        test_value = redis_client.get("test_key")
        if test_value and test_value.decode('utf-8') == "test_value":
            return {"status": "success", "message": "Redis connection is working"}
        else:
            return {"status": "error", "message": "Redis connection test failed"}
    except Exception as e:
        return {"status": "error", "message": f"Redis connection error: {str(e)}"}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
