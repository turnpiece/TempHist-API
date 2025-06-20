from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from dotenv import load_dotenv
import os
import requests
import redis
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import aiohttp

load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

app = FastAPI()
redis_client = redis.from_url(REDIS_URL)

@app.middleware("http")
async def verify_token_middleware(request: Request, call_next):
    # Allow OPTIONS requests for CORS preflight
    if request.method == "OPTIONS":
        return await call_next(request)

    # Public paths that don't require a token
    public_paths = ["/", "/docs", "/openapi.json", "/redoc", "/test-cors", "/test-redis"]
    if request.url.path in public_paths or any(request.url.path.startswith(p) for p in ["/static"]):
        return await call_next(request)
    
    # All other paths require a token
    if not API_ACCESS_TOKEN:
        return JSONResponse(
            status_code=500, 
            content={"detail": "API access token is not configured on the server."}
        )

    token = request.headers.get("X-API-Token")
    if token != API_ACCESS_TOKEN:
        return JSONResponse(
            status_code=401, 
            content={"detail": "Invalid or missing API token."}
        )
        
    response = await call_next(request)
    return response

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

async def fetch_weather_batch(location: str, date_strs: list, max_concurrent: int = 3) -> dict:
    """
    Fetch weather data for multiple dates in parallel using aiohttp, with limited concurrency.
    Returns a dict mapping date_str to weather data.
    """
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(date_str):
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date_str}?unitGroup=metric&include=days&key={API_KEY}"
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200 and 'application/json' in resp.headers.get('Content-Type', ''):
                            data = await resp.json()
                            return date_str, data
                        else:
                            text = await resp.text()
                            return date_str, {"error": text, "status": resp.status}
                except Exception as e:
                    return date_str, {"error": str(e)}

    tasks = [fetch_one(date_str) for date_str in date_strs]
    for fut in asyncio.as_completed(tasks):
        date_str, result = await fut
        results[date_str] = result
    return results

def get_forecast_data(location: str, date: str) -> Dict:
    """
    Get forecast data for a specific location and date using Visual Crossing API.
    Returns cached data if available, otherwise fetches from Visual Crossing API.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Visual Crossing API key not configured")
    
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"forecast_{location.lower()}_{date_str}"
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            print(f"Cache hit: {cache_key}")
            return json.loads(cached_data)
        print(f"Cache miss: {cache_key} — fetching from API")
    
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date_str}?unitGroup=metric&include=days&key={API_KEY}"
    print(f"Fetching forecast from URL: {url}")
    
    try:
        response = requests.get(url)
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Headers: {response.headers}")
        response.raise_for_status()
        data = response.json()
        print(f"API Response Data: {json.dumps(data, indent=2)}")
        
        if not data.get('days'):
            print(f"No days data found in response for {date_str}")
            raise HTTPException(status_code=404, detail=f"No forecast data available for {date_str}")
        
        # Get the first day's data (since we're requesting a specific date)
        day_data = data['days'][0]
        print(f"Day data: {json.dumps(day_data, indent=2)}")
        
        result = {
            "location": location,
            "date": date_str,
            "average_temperature": round(day_data['temp'], 1),
            "unit": "celsius"
        }

        # Cache the result if caching is enabled
        if CACHE_ENABLED:
            redis_client.setex(cache_key, timedelta(hours=24), json.dumps(result))
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching forecast data: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def get_temperature_series(location: str, month: int, day: int) -> Dict:
    today = datetime.now()
    current_year = today.year
    years = list(range(current_year - 50, current_year + 1))

    data = []
    missing_years = []
    uncached_years = []
    uncached_date_strs = []
    year_to_date_str = {}
    for year in years:
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = f"{location.lower()}_{date_str}"
        year_to_date_str[year] = date_str
        # If this is today's date, use the forecast data
        if year == current_year and month == today.month and day == today.day:
            try:
                forecast_data = get_forecast_data(location, datetime(year, month, day).date())
                data.append({"x": year, "y": forecast_data["average_temperature"]})
                continue
            except Exception as e:
                print(f"Error fetching forecast: {str(e)}")
                missing_years.append({"year": year, "reason": "forecast_failed"})
                continue
        # Use historical data for all other dates
        if CACHE_ENABLED:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                weather = json.loads(cached_data)
                try:
                    temp = weather["days"][0]["temp"]
                    data.append({"x": year, "y": temp})
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Error processing cached data for {date_str}: {str(e)}")
                    missing_years.append({"year": year, "reason": "data_processing_error"})
                continue
            else:
                uncached_years.append(year)
                uncached_date_strs.append(date_str)
        else:
            uncached_years.append(year)
            uncached_date_strs.append(date_str)

    # Batch fetch for all uncached years
    if uncached_date_strs:
        batch_results = await fetch_weather_batch(location, uncached_date_strs)
        for year, date_str in zip(uncached_years, uncached_date_strs):
            weather = batch_results.get(date_str)
            if weather and "error" not in weather:
                try:
                    temp = weather["days"][0]["temp"]
                    data.append({"x": year, "y": temp})
                    # Cache the result if caching is enabled
                    if CACHE_ENABLED:
                        cache_key = f"{location.lower()}_{date_str}"
                        redis_client.setex(cache_key, timedelta(hours=24), json.dumps(weather))
                except (KeyError, IndexError, TypeError) as e:
                    print(f"Error processing batch data for {date_str}: {str(e)}")
                    missing_years.append({"year": year, "reason": "data_processing_error"})
            else:
                missing_years.append({"year": year, "reason": weather.get("error", "api_error") if weather else "api_error"})

    # Print summary of collected data
    print("\nData summary:")
    print(f"Total data points: {len(data)}")
    if data:
        temps = [d['y'] for d in data]
        if temps:  # Add check to ensure temps list is not empty
            print(f"Temperature range: {min(temps):.1f}°C to {max(temps):.1f}°C")
            print(f"Average temperature: {sum(temps)/len(temps):.1f}°C")

    data_list = sorted(data, key=lambda d: d['x'])

    return {
        "data": data_list,
        "metadata": {
            "total_years": len(years),
            "available_years": len(data),
            "missing_years": missing_years,
            "completeness": round(len(data) / len(years) * 100, 1) if years else 0
        }
    }

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

    data = await get_temperature_series(location, month, day)
    print(f"Summary data for {location} on {month_day}:")
    print(f"Number of data points: {len(data['data'])}")
    print(f"Date range: {data['data'][0]['x']} to {data['data'][-1]['x']}")
    print(f"Current temperature: {data['data'][-1]['y']}°C")
    
    if len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    data_list = sorted(data['data'], key=lambda d: d['x'])
    avg_temp = calculate_historical_average(data_list)
    print(f"Calculated average: {avg_temp}°C")
    print(f"Temperature difference: {round(data_list[-1]['y'] - avg_temp, 1)}°C")

    summary_text = await get_summary(data_list, f"{current_year}-{month:02d}-{day:02d}")
    return JSONResponse(content={"summary": summary_text}, headers={"Cache-Control": "public, max-age=3600"})

def calculate_historical_average(data: List[Dict[str, float]]) -> float:
    """
    Calculate the average temperature using only historical data (excluding current year).
    Returns the average temperature rounded to 1 decimal place.
    """
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not data or len(data) < 2:
        return 0.0
    historical_data = data[:-1]
    avg_temp = sum(p['y'] for p in historical_data) / len(historical_data)
    return round(avg_temp, 1)

async def get_summary(location: str, month_day: str, weather_data: Optional[List[Dict]] = None) -> str:
    def get_friendly_date(date: datetime) -> str:
        day = date.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{day}{suffix} {date.strftime('%B')}"

    def generate_summary(data: List[Dict[str, float]], date: datetime) -> str:
        if not data or len(data) < 2:
            return "Not enough data to generate summary."

        latest = data[-1]
        avg_temp = calculate_historical_average(data)
        diff = latest['y'] - avg_temp
        rounded_diff = round(diff, 1)

        friendly_date = get_friendly_date(date)
        warm_summary = ''
        cold_summary = ''
        temperature = f"{latest['y']}°C."

        previous = data[:-1]
        is_warmest = all(latest['y'] >= p['y'] for p in previous)
        is_coldest = all(latest['y'] <= p['y'] for p in previous)

        if is_warmest:
            warm_summary = f"This is the warmest {friendly_date} on record."
        else:
            last_warmer = next((p['x'] for p in reversed(previous) if p['y'] > latest['y']), None)
            if last_warmer:
                years_since = int(latest['x'] - last_warmer)
                if years_since > 1:
                    if years_since == 2:
                        warm_summary = f"It's warmer than last year but not as warm as {last_warmer}."
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
                    if years_since == 2:
                        cold_summary = f"It's colder than last year but not as cold as {last_colder}."
                    elif years_since <= 10:
                        cold_summary = f"This is the coldest {friendly_date} since {last_colder}."
                    else:
                        cold_summary = f"This is the coldest {friendly_date} in {years_since} years."

        if abs(diff) < 0.05:
            avg_summary = "It is about average for this date."
        elif diff > 0:
            avg_summary = "However, it is " if cold_summary else "It is "
            avg_summary += f"{rounded_diff}°C warmer than average today."
        else:
            avg_summary = "However, it is " if warm_summary else "It is "
            avg_summary += f"{abs(rounded_diff)}°C cooler than average today."

        return " ".join(filter(None, [temperature, warm_summary, cold_summary, avg_summary]))

    try:
        if weather_data is None:
            month, day = map(int, month_day.split("-"))
            weather_data = await get_temperature_series(location, month, day)

        if isinstance(weather_data, dict) and "data" in weather_data:
            weather_data = weather_data["data"]

        if not weather_data or not isinstance(weather_data, list):
            return "No weather data available."

        # Use the year from the latest data point or fallback to current year
        latest_year = max(d.get("x") for d in weather_data if d.get("x"))
        date = datetime.strptime(f"{latest_year}-{month_day}", "%Y-%m-%d")
        return generate_summary(weather_data, date)

    except Exception as e:
        print(f"Error in get_summary: {e}")
        return "Error generating summary."

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

    data = await get_temperature_series(location, month, day)

    if len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    slope = calculate_trend_slope(data['data'])
    result = {
        "slope": slope, 
        "units": "°C/decade",
        "data_points": len(data['data']),
        "completeness": data['metadata']['completeness'],
        "missing_years": data['metadata']['missing_years']
    }
    
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

    data = await get_temperature_series(location, month, day)

    if len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    # Calculate average temperature using the new function
    data_list = sorted(data['data'], key=lambda d: d['x'])
    avg_temp = calculate_historical_average(data_list)
    
    result = {
        "average": avg_temp,
        "unit": "celsius",
        "data_points": len(data['data']),
        "year_range": {
            "start": data_list[0]['x'],
            "end": data_list[-1]['x']
        },
        "missing_years": data['metadata']['missing_years'],
        "completeness": data['metadata']['completeness']
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
    return round(numerator / denominator, 2) * 10 if denominator != 0 else 0.0

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
            return JSONResponse(content={"status": "success", "message": "Redis connection is working"}, headers={"Cache-Control": "public, max-age=3600"})
        else:
            return JSONResponse(content={"status": "error", "message": "Redis connection test failed"}, headers={"Cache-Control": "public, max-age=3600"})
    except Exception as e:
        return JSONResponse(
    content={
        "status": "error",
        "message": f"Redis connection error: {str(e)}"
    },
    headers={"Cache-Control": "public, max-age=3600"}
)

@app.get("/data/{location}/{month_day}")
async def get_all_data(location: str, month_day: str):
    try:
        month, day = map(int, month_day.split("-"))
        weather_data = await get_temperature_series(location, month, day)

        # Check cache for summary
        summary_cache_key = f"summary_{location.lower()}_{month:02d}_{day:02d}"
        summary_data = None
        if CACHE_ENABLED:
            cached_summary = redis_client.get(summary_cache_key)
            if cached_summary:
                print(f"Cache hit: {summary_cache_key}")
                summary_data = json.loads(cached_summary)
        if not summary_data:
            summary_data = await get_summary(location, month_day)
            if CACHE_ENABLED:
                redis_client.setex(summary_cache_key, timedelta(hours=24), json.dumps(summary_data))

        # Check cache for average
        average_cache_key = f"average_{location.lower()}_{month:02d}_{day:02d}"
        average_data = None
        if CACHE_ENABLED:
            cached_average = redis_client.get(average_cache_key)
            if cached_average:
                print(f"Cache hit: {average_cache_key}")
                average_data = json.loads(cached_average)
        if not average_data:
            average_data = get_average_dict(weather_data)
            if CACHE_ENABLED:
                redis_client.setex(average_cache_key, timedelta(hours=24), json.dumps(average_data))

        trend_data = await get_trend(location, month_day, weather_data)

        return JSONResponse(
            content={
                "weather": weather_data,
                "summary": summary_data,
                "trend": trend_data,
                "average": average_data
            },
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            },
            headers={"Cache-Control": "public, max-age=60"},
            status_code=500
        )
    
async def get_trend(location: str, month_day: str, weather_data: Optional[List[Dict]] = None) -> dict:
    if weather_data is None:
        month, day = map(int, month_day.split("-"))
        weather_data = await get_temperature_series(location, month, day)

    if isinstance(weather_data, dict) and "data" in weather_data:
        weather_data = weather_data["data"]
    trend_input = [{"x": d["x"], "y": d["y"]} for d in weather_data if d.get("y") is not None]
    slope = calculate_trend_slope(trend_input)
    return {
        "slope": slope,
        "units": "°C/decade"
    }

def validate_month_day(month_day: str):
    try:
        month, day = map(int, month_day.split("-"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid month-day format. Use MM-DD.")
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    if not (1 <= day <= 31):
        raise HTTPException(status_code=400, detail="Day must be between 1 and 31")
    if month in [4, 6, 9, 11] and day > 30:
        raise HTTPException(status_code=400, detail=f"Month {month} has only 30 days")
    if month == 2 and day > 29:
        raise HTTPException(status_code=400, detail="February has only 29 days")
    return month, day

def get_average_dict(weather_data):
    data_list = sorted(weather_data['data'], key=lambda d: d['x'])
    avg_temp = calculate_historical_average(data_list)
    return {
        "average": avg_temp,
        "unit": "celsius",
        "data_points": len(weather_data['data']),
        "year_range": {
            "start": data_list[0]['x'],
            "end": data_list[-1]['x']
        },
        "missing_years": weather_data['metadata']['missing_years'],
        "completeness": weather_data['metadata']['completeness']
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
