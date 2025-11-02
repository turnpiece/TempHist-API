#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Visual Crossing API directly to debug empty data issue.
"""

import aiohttp
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

async def test_vc_api():
    """Test Visual Crossing API directly."""
    
    print("🔍 TESTING VISUAL CROSSING API DIRECTLY")
    print("=" * 60)
    
    if not API_KEY:
        print("❌ VISUAL_CROSSING_API_KEY not found")
        return
    
    print(f"API Key: {API_KEY[:10]}...")
    
    # Test the same location and date from your issue
    location = "Berlin, Berlin, Germany"
    date_str = "2024-10-26"  # Current year
    
    print(f"\n📍 Testing location: {location}")
    print(f"📅 Testing date: {date_str}")
    
    # Test 1: Basic historical data
    url = f"{BASE_URL}/{location}/{date_str}"
    params = {
        "key": API_KEY,
        "unitGroup": "metric",
        "include": "days"
    }
    
    print(f"\n🔗 URL: {url}")
    print(f"📋 Params: {params}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                print(f"\n📊 Response Status: {resp.status}")
                print(f"📊 Response Headers: {dict(resp.headers)}")
                
                if resp.status == 200:
                    data = await resp.json()
                    print(f"\n📄 Response Data:")
                    print(f"  Keys: {list(data.keys())}")
                    
                    if 'days' in data and data['days']:
                        day_data = data['days'][0]
                        print(f"  Days count: {len(data['days'])}")
                        print(f"  First day keys: {list(day_data.keys())}")
                        print(f"  Temperature: {day_data.get('temp', 'N/A')}")
                        print(f"  Date: {day_data.get('datetime', 'N/A')}")
                    else:
                        print("  ❌ No 'days' data found")
                        print(f"  Full response: {json.dumps(data, indent=2)}")
                else:
                    text = await resp.text()
                    print(f"❌ Error response: {text}")
                    
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test 2: Historical data for multiple years
    print(f"\n" + "=" * 60)
    print("🕰️ TESTING HISTORICAL DATA FOR MULTIPLE YEARS")
    print("=" * 60)
    
    years_to_test = [2024, 2023, 2022, 2021, 2020]
    
    for year in years_to_test:
        test_date = f"{year}-10-26"
        url = f"{BASE_URL}/{location}/{test_date}"
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'days' in data and data['days']:
                            day_data = data['days'][0]
                            temp = day_data.get('temp', 'N/A')
                            print(f"  {year}: {temp}°C")
                        else:
                            print(f"  {year}: No data")
                    else:
                        print(f"  {year}: Error {resp.status}")
        except Exception as e:
            print(f"  {year}: Failed - {e}")

if __name__ == "__main__":
    asyncio.run(test_vc_api())
