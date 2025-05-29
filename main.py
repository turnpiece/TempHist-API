from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
from diskcache import Cache

load_dotenv()
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

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
