from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv
import os
import httpx
import redis
import json
from datetime import datetime, timedelta
from typing import List, Dict
import asyncio
from functools import lru_cache

load_dotenv()
VC_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
VC_API_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Token for this API
API_TOKEN = os.getenv("API_ACCESS_TOKEN")

app = FastAPI()
redis_client = redis.from_url(REDIS_URL)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

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

# Verify token for this API
def verify_token(request: Request):
    token = request.headers.get("X-API-Token")
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing API token")

@app.api_route("/", methods=["GET", "OPTIONS"])
async def root(_: None = Depends(verify_token)):
    """Root endpoint that returns API information"""
    return {
        "name": "Temperature History API",
        "version": "1.0.0",
        "endpoints": [
            "/data/{location}/{month_day}",
            "/average/{location}/{month_day}",
            "/trend/{location}/{month_day}",
            "/weather/{location}/{date}",
            "/summary/{location}/{month_day}",
            "/forecast/{location}"
        ]
    }

async def fetch_weather_batch(location: str, dates: List[str], max_retries: int = 3) -> Dict[str, Dict]:
    """
    Fetch weather data for specific dates by making individual requests.
    """
    if not VC_API_KEY:
        print("Error: Visual Crossing API key not configured")
        raise HTTPException(status_code=500, detail="Visual Crossing API key not configured")
    
    # Split dates into chunks of 10 to avoid overwhelming the API
    chunk_size = 10
    date_chunks = [dates[i:i + chunk_size] for i in range(0, len(dates), chunk_size)]
    print(f"Split {len(dates)} dates into {len(date_chunks)} chunks")
    
    result = {}
    for chunk in date_chunks:
        print(f"Processing chunk with dates: {chunk}")
        
        # Process each date in the chunk
        for date in chunk:
            url = f"{VC_API_BASE_URL}/{location}/{date}?unitGroup=metric&include=days&key={VC_API_KEY}"
            print(f"Fetching data for {location} on {date}")
            
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        print(f"Making request to: {url} (attempt {attempt + 1}/{max_retries})")
                        response = await client.get(url)
                        print(f"Response status: {response.status_code}")
                        
                        if response.status_code != 200:
                            print(f"Error response: {response.text}")
                            if response.status_code == 404:
                                print(f"No data found for {date}")
                                break  # Skip to next date
                            if attempt < max_retries - 1:
                                print(f"Retrying after error status {response.status_code}...")
                                await asyncio.sleep(1)
                                continue
                            raise HTTPException(status_code=response.status_code, detail=f"Error from weather service: {response.text}")
                        
                        try:
                            data = response.json()
                        except json.JSONDecodeError as e:
                            print(f"Failed to decode JSON response: {response.text}")
                            if attempt < max_retries - 1:
                                print("Retrying after JSON decode error...")
                                await asyncio.sleep(1)
                                continue
                            raise HTTPException(status_code=500, detail="Invalid response from weather service")
                        
                        if not data.get('days'):
                            print(f"No days data found in response for {date}")
                            print(f"Response content: {data}")
                            if attempt < max_retries - 1:
                                print("Retrying after no days data...")
                                await asyncio.sleep(1)
                                continue
                            break  # Skip to next date
                        
                        # Get the weather data for this date
                        day_data = data['days'][0]
                        result[date] = {
                            "temp": day_data['temp'],
                            "tempmax": day_data['tempmax'],
                            "tempmin": day_data['tempmin']
                        }
                        print(f"Successfully fetched data for {date}")
                        break  # Success, move to next date
                        
                except httpx.TimeoutException:
                    print(f"Request timed out for {date} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print("Retrying after timeout...")
                        await asyncio.sleep(1)
                        continue
                    print(f"Skipping {date} after timeout")
                    break  # Skip to next date
                except httpx.ConnectError as e:
                    print(f"Connection error for {date}: {str(e)} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print("Retrying after connection error...")
                        await asyncio.sleep(1)
                        continue
                    print(f"Skipping {date} after connection error")
                    break  # Skip to next date
                except httpx.RequestError as e:
                    print(f"Request error for {date}: {str(e)} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print("Retrying after request error...")
                        await asyncio.sleep(1)
                        continue
                    print(f"Skipping {date} after request error")
                    break  # Skip to next date
                except Exception as e:
                    print(f"Unexpected error for {date}: {str(e)} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        print("Retrying after unexpected error...")
                        await asyncio.sleep(1)
                        continue
                    print(f"Skipping {date} after unexpected error")
                    break  # Skip to next date
            
            # Add a small delay between requests to avoid rate limiting
            await asyncio.sleep(0.1)
    
    print(f"Successfully fetched data for {len(result)} total dates")
    return result

async def get_temperature_series(location: str, month: int, day: int) -> Dict:
    try:
        today = datetime.now()
        current_year = today.year
        years = list(range(current_year - 50, current_year + 1))
        
        # Prepare all dates for batch request
        dates = [f"{year}-{month:02d}-{day:02d}" for year in years]
        print(f"Getting temperature series for {location} on {month:02d}-{day:02d}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        
        # Check cache for all dates
        cache_keys = [f"{location.lower()}_{date}" for date in dates]
        cached_data = {}
        
        if CACHE_ENABLED:
            print("Checking cache for existing data...")
            try:
                # Use Redis pipeline for multiple gets
                pipe = redis_client.pipeline()
                for key in cache_keys:
                    pipe.get(key)
                cached_results = await asyncio.to_thread(pipe.execute)
                
                # Process cached results
                for date, cached_result in zip(dates, cached_results):
                    if cached_result:
                        try:
                            cached_data[date] = json.loads(cached_result)
                        except json.JSONDecodeError:
                            print(f"Failed to decode cached data for {date}")
                            continue
                
                print(f"Found {len(cached_data)} dates in cache")
            except Exception as e:
                print(f"Error accessing Redis cache: {str(e)}")
                print("Continuing without cache...")
        
        # Find dates that need to be fetched
        dates_to_fetch = [date for date in dates if date not in cached_data]
        print(f"Need to fetch {len(dates_to_fetch)} dates from API")
        
        # Fetch missing data in batch
        if dates_to_fetch:
            try:
                batch_data = await fetch_weather_batch(location, dates_to_fetch)
                
                # Cache new data
                if CACHE_ENABLED and batch_data:
                    print(f"Caching {len(batch_data)} new data points")
                    try:
                        pipe = redis_client.pipeline()
                        for date, data in batch_data.items():
                            cache_key = f"{location.lower()}_{date}"
                            pipe.setex(cache_key, timedelta(hours=24), json.dumps(data))
                        await asyncio.to_thread(pipe.execute)
                    except Exception as e:
                        print(f"Error caching data: {str(e)}")
                
                cached_data.update(batch_data)
            except HTTPException as e:
                print(f"Error fetching batch data: {str(e)}")
                # If we get a 404, return what we have from cache
                if e.status_code == 404 and cached_data:
                    print("Returning cached data after 404 error")
                    pass
                else:
                    raise
        
        # Process the data
        data = []
        missing_years = []
        
        for year, date in zip(years, dates):
            if date in cached_data:
                try:
                    temp = cached_data[date]['temp']
                    data.append({"x": year, "y": temp})
                except (KeyError, TypeError) as e:
                    print(f"Error processing data for {date}: {str(e)}")
                    missing_years.append({"year": year, "reason": "data_processing_error"})
            else:
                missing_years.append({"year": year, "reason": "no_data"})
        
        print(f"Processed {len(data)} data points")
        print(f"Missing {len(missing_years)} years")
        
        # Return data even if incomplete, but with metadata about missing data
        return {
            "data": data,
            "metadata": {
                "total_years": len(years),
                "available_years": len(data),
                "missing_years": missing_years,
                "completeness": round(len(data) / len(years) * 100, 1) if years else 0
            }
        }
    except Exception as e:
        print(f"Error in get_temperature_series: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e.__dict__}")
        raise

async def get_forecast_data(location: str, date: str) -> Dict:
    """
    Get forecast data for a specific location and date using Visual Crossing API.
    Returns cached data if available, otherwise fetches from Visual Crossing API.
    """
    if not VC_API_KEY:
        raise HTTPException(status_code=500, detail="Visual Crossing API key not configured")
    
    date_str = date.strftime("%Y-%m-%d")
    cache_key = f"forecast_{location.lower()}_{date_str}"
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    
    url = f"{VC_API_BASE_URL}/{location}/{date_str}?unitGroup=metric&include=days&key={VC_API_KEY}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('days'):
                raise HTTPException(status_code=404, detail=f"No forecast data available for {date_str}")
            
            day_data = data['days'][0]
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
            
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Error fetching forecast data: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/weather/{location}/{date}")
async def get_weather(location: str, date: str, _: None = Depends(verify_token)):
    cache_key = f"{location.lower()}_{date}"

    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)

    # Fetch single date using batch endpoint
    batch_data = await fetch_weather_batch(location, [date])
    if date in batch_data:
        result = {"days": [{"temp": batch_data[date]["temp"]}]}
        if CACHE_ENABLED:
            redis_client.setex(cache_key, timedelta(hours=24), json.dumps(result))
        return result
    
    raise HTTPException(status_code=404, detail="Weather data not found")

@app.get("/summary/{location}/{month_day}")
async def summary(location: str, month_day: str, _: None = Depends(verify_token)):
    try:
        month, day = map(int, month_day.split("-"))
        today = datetime.now()
        current_year = today.year
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month-day format. Use MM-DD.")

    data = await get_temperature_series(location, month, day)
    
    if not data or not data.get('data') or len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    summary_text = get_summary(data['data'], f"{current_year}-{month:02d}-{day:02d}")
    return {"summary": summary_text}

def calculate_historical_average(data: List[Dict[str, float]]) -> float:
    """
    Calculate the average temperature using only historical data (excluding current year).
    Returns the average temperature rounded to 1 decimal place.
    """
    if not data or len(data) < 2:
        return 0.0
    
    # Filter out any None values and use all data except the most recent (current year)
    historical_data = [p for p in data[:-1] if p.get('y') is not None]
    
    if not historical_data:
        return 0.0
        
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

        # Filter out None values
        valid_data = [p for p in data if p.get('y') is not None]
        if not valid_data:
            return "No valid temperature data available."

        latest = valid_data[-1]
        # Use the new function to calculate average
        avg_temp = calculate_historical_average(valid_data)
        diff = latest['y'] - avg_temp
        rounded_diff = round(diff, 1)

        friendly_date = get_friendly_date(date)
        warm_summary = ''
        cold_summary = ''

        # Today's temperature
        temperature = f"{latest['y']}°C."

        # For historical comparisons, use all previous years
        previous = valid_data[:-1]
        if not previous:
            return f"{temperature} No historical data available for comparison."

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
                    if years_since <= 10:
                        cold_summary = f"This is the coldest {friendly_date} since {last_colder}."
                    else:
                        cold_summary = f"This is the coldest {friendly_date} in {years_since} years."

        if abs(diff) < 0.05:
            avg_summary = "It is about average for this date."
        elif diff > 0:
            if cold_summary == '':
                avg_summary = "It is"
            else:
                avg_summary = "However, it is"

            avg_summary += f" {rounded_diff}°C warmer than average today."
        else:
            if warm_summary == '':
                avg_summary = "It is"
            else:
                avg_summary = "However, it is"

            avg_summary += f" {abs(rounded_diff)}°C cooler than average today."

        return " ".join(filter(None, [temperature, warm_summary, cold_summary, avg_summary]))

    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."

    return generate_summary(data, date)

def calculate_trend_slope(data: List[Dict[str, float]]) -> float:
    """
    Calculate the trend slope using linear regression.
    Returns the slope in °C/decade.
    """
    # Filter out None values
    valid_data = [p for p in data if p.get('y') is not None]
    
    n = len(valid_data)
    if n < 2:
        return 0.0

    sum_x = sum(p['x'] for p in valid_data)
    sum_y = sum(p['y'] for p in valid_data)
    sum_xy = sum(p['x'] * p['y'] for p in valid_data)
    sum_xx = sum(p['x'] ** 2 for p in valid_data)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_xx - sum_x ** 2
    
    if denominator == 0:
        return 0.0
        
    return round(numerator / denominator, 2) * 10

@app.get("/trend/{location}/{month_day}")
async def trend(location: str, month_day: str, _: None = Depends(verify_token)):
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
        cached_data = await asyncio.to_thread(redis_client.get, cache_key)
        if cached_data:
            return json.loads(cached_data)

    try:
        data = await get_temperature_series(location, month, day)
        
        # Check if we have any data at all
        if not data or not data.get('data'):
            raise HTTPException(status_code=404, detail=f"No temperature data available for {location}")
        
        # Calculate trend
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
            await asyncio.to_thread(redis_client.setex, cache_key, timedelta(hours=24), json.dumps(result))
        
        return result
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in trend endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating trend: {str(e)}")

@app.api_route("/average/{location}/{month_day}", methods=["GET", "OPTIONS"])
async def average(location: str, month_day: str, _: None = Depends(verify_token)):
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
        cached_data = await asyncio.to_thread(redis_client.get, cache_key)
        if cached_data:
            return json.loads(cached_data)

    try:
        data = await get_temperature_series(location, month, day)
        
        # Check if we have any data at all
        if not data or not data.get('data'):
            raise HTTPException(status_code=404, detail=f"No temperature data available for {location}")
        
        # Calculate average temperature using the new function
        avg_temp = calculate_historical_average(data['data'])
        
        result = {
            "average": avg_temp,
            "unit": "celsius",
            "data_points": len(data['data']),
            "year_range": {
                "start": data['data'][0]['x'] if data['data'] else None,
                "end": data['data'][-1]['x'] if data['data'] else None
            },
            "missing_years": data['metadata']['missing_years'],
            "completeness": data['metadata']['completeness']
        }
        
        # Cache the result if caching is enabled
        if CACHE_ENABLED:
            await asyncio.to_thread(redis_client.setex, cache_key, timedelta(hours=24), json.dumps(result))
        
        return result
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in average endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating average: {str(e)}")

@app.get("/forecast/{location}")
async def get_forecast(location: str, _: None = Depends(verify_token)):
    today = datetime.now().date()
    return await get_forecast_data(location, today)

@app.get("/test-redis")
async def test_redis(_: None = Depends(verify_token)):
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

@app.get("/data/{location}/{month_day}")
async def get_all_data(location: str, month_day: str, _: None = Depends(verify_token)):
    """
    Get all data needed for a specific date in a single request.
    Returns average, trend, summary, and weather data.
    """
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

    # Create cache key for the combined data
    cache_key = f"all_data_{location.lower()}_{month:02d}_{day:02d}"
    
    # Check cache first if caching is enabled
    if CACHE_ENABLED:
        cached_data = await asyncio.to_thread(redis_client.get, cache_key)
        if cached_data:
            return json.loads(cached_data)

    # Get temperature series data (used by average, trend, and summary)
    data = await get_temperature_series(location, month, day)
    
    if not data or not data.get('data') or len(data['data']) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    # Calculate all the required data
    today = datetime.now()
    current_year = today.year
    current_date = f"{current_year}-{month:02d}-{day:02d}"
    
    # Get average temperature
    avg_temp = calculate_historical_average(data['data'])
    
    # Calculate trend
    slope = calculate_trend_slope(data['data'])
    
    # Generate summary
    summary_text = get_summary(data['data'], current_date)
    
    # Get current year's weather
    current_weather = None
    if current_date in data['data']:
        current_weather = {
            "temp": data['data'][-1]['y'],
            "year": current_year
        }
    
    # Combine all data
    result = {
        "average": {
            "temperature": avg_temp,
            "unit": "celsius",
            "data_points": len(data['data']),
            "year_range": {
                "start": data['data'][0]['x'],
                "end": data['data'][-1]['x']
            },
            "missing_years": data['metadata']['missing_years'],
            "completeness": data['metadata']['completeness']
        },
        "trend": {
            "slope": slope,
            "units": "°C/decade"
        },
        "summary": summary_text,
        "current_weather": current_weather,
        "series": {
            "data": data['data'],  # Include the full temperature series
            "metadata": {
                "location": location,
                "date": current_date,
                "total_years": data['metadata']['total_years'],
                "available_years": data['metadata']['available_years']
            }
        }
    }
    
    # Cache the result if caching is enabled
    if CACHE_ENABLED:
        await asyncio.to_thread(
            redis_client.setex,
            cache_key,
            timedelta(hours=24),
            json.dumps(result)
        )
    
    return result

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
