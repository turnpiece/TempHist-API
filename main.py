from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
from diskcache import Cache
from datetime import datetime
from typing import List, Dict

app = FastAPI()

load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
print(f"Loaded API key: {API_KEY}")

app = FastAPI()
cache = Cache("weather_cache")  # directory to store cache files

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_weather_from_api(location: str, date: str):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date}?unitGroup=metric&include=days&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
        return response.json()
    return {"error": response.text, "status": response.status_code}

def get_temperature_series(location: str, month: int, day: int) -> List[Dict[str, float]]:
    today = datetime.now()
    current_year = today.year
    years = list(range(current_year - 50, current_year + 1))

    data = []
    for year in years:
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = f"{location.lower()}_{date_str}"
        if cache_key in cache:
            weather = cache[cache_key]
        else:
            weather = fetch_weather_from_api(location, date_str)
            if "error" not in weather:
                cache.set(cache_key, weather, expire=60 * 60 * 24)
            else:
                continue

        try:
            temp = weather["days"][0]["temp"]
            data.append({"x": year, "y": temp})
        except (KeyError, IndexError, TypeError):
            continue

    return data

@app.get("/weather/{location}/{date}")
def get_weather(location: str, date: str):
    cache_key = f"{location.lower()}_{date}"

    # Check cache first
    if cache_key in cache:
        print(f"Cache hit: {cache_key}")
        return cache[cache_key]

    print(f"Cache miss: {cache_key} — fetching from API")
    result = fetch_weather_from_api(location, date)

    # Only cache successful results
    if "error" not in result:
        cache.set(cache_key, result, expire=60 * 60 * 24)  # 24h expiration

    return result

# get a text summary
@app.get("/summary/{location}/{month_day}")
async def summary(location: str, month_day: str, request: Request):
    try:
        month, day = map(int, month_day.split("-"))
        today = datetime.now()
        current_year = today.year
        years = list(range(current_year - 50, current_year + 1))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month-day format. Use MM-DD.")

    data = get_temperature_series(location, month, day)

    if len(data) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    summary_text = get_summary(data, f"{current_year}-{month:02d}-{day:02d}")
    return {"summary": summary_text}

def get_summary(data: List[Dict[str, float]], date_str: str) -> str:
    def get_friendly_date(date: datetime) -> str:
        day = date.day
        suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{day}{suffix} {date.strftime('%B')}"

    def generate_summary(data: List[Dict[str, float]], date: datetime) -> str:
        if not data or len(data) < 2:
            return "Not enough data to generate summary."

        latest = data[-1]
        previous = data[:-1]
        avg_prev = sum(p['y'] for p in previous) / len(previous)
        diff = latest['y'] - avg_prev
        rounded_diff = round(diff, 1)

        friendly_date = get_friendly_date(date)
        warm_summary = ''
        cold_summary = ''

        is_warmest = all(latest['y'] > p['y'] for p in previous)
        is_coldest = all(latest['y'] < p['y'] for p in previous)

        if is_warmest:
            warm_summary = f"This is the warmest {friendly_date} on record."
        else:
            last_warmer = next((p['x'] for p in reversed(previous) if p['y'] > latest['y']), None)
            if last_warmer:
                years_since = int(latest['x'] - last_warmer)
                if years_since > 1:
                    if years_since <= 10:
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
        today = datetime.now()
        current_year = today.year
        years = list(range(current_year - 50, current_year + 1))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid month-day format. Use MM-DD.")

    data = get_temperature_series(location, month, day)

    if len(data) < 2:
        raise HTTPException(status_code=404, detail="Not enough temperature data available.")

    slope = calculate_trend_slope(data)
    return {"slope": slope, "units": "°C/year"}

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

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)