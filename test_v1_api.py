#!/usr/bin/env python3
"""
Test script for the new v1 API endpoints
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

async def test_v1_api():
    """Test the new v1 API endpoints"""
    print("🧪 Testing TempHist v1 API")
    print("=" * 50)
    
    # Headers with test token for authentication
    headers = {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test 1: API Info
            print("\n1. Testing API Info...")
            response = await client.get(f"{BASE_URL}/")
            if response.status_code == 200:
                api_info = response.json()
                print(f"✅ API Info: {api_info['name']} v{api_info['version']}")
                print(f"   V1 endpoints available: {len(api_info['v1_endpoints']['records'])}")
            else:
                print(f"❌ API Info failed: {response.status_code}")
                return
            
            # Test 2: Daily Record
            print("\n2. Testing Daily Record...")
            response = await client.get(f"{BASE_URL}/v1/records/daily/london/01-15", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Daily Record: {data['period']} | {data['location']} | {data['identifier']}")
                print(f"   Data points: {len(data['values'])}")
                print(f"   Average: {data['average']['mean']}°C")
                print(f"   Trend: {data['trend']['slope']}°C/decade")
                print(f"   Summary: {data['summary'][:100]}...")
            else:
                print(f"❌ Daily Record failed: {response.status_code} - {response.text}")
            
            # Test 3: Daily Average Subresource
            print("\n3. Testing Daily Average Subresource...")
            response = await client.get(f"{BASE_URL}/v1/records/daily/london/01-15/average", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Daily Average: {data['data']['mean']}°C ({data['data']['data_points']} points)")
            else:
                print(f"❌ Daily Average failed: {response.status_code} - {response.text}")
            
            # Test 4: Daily Trend Subresource
            print("\n4. Testing Daily Trend Subresource...")
            response = await client.get(f"{BASE_URL}/v1/records/daily/london/01-15/trend", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Daily Trend: {data['data']['slope']}°C/decade ({data['data']['data_points']} points)")
            else:
                print(f"❌ Daily Trend failed: {response.status_code} - {response.text}")
            
            # Test 5: Daily Summary Subresource
            print("\n5. Testing Daily Summary Subresource...")
            response = await client.get(f"{BASE_URL}/v1/records/daily/london/01-15/summary", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Daily Summary: {data['data'][:100]}...")
            else:
                print(f"❌ Daily Summary failed: {response.status_code} - {response.text}")
            
            # Test 6: Legacy Endpoint (should work but show deprecation warning)
            print("\n6. Testing Legacy Endpoint...")
            response = await client.get(f"{BASE_URL}/data/london/01-15", headers=headers)
            if response.status_code == 200:
                data = response.json()
                deprecated_header = response.headers.get('X-Deprecated')
                new_endpoint = response.headers.get('X-New-Endpoint')
                print(f"✅ Legacy Endpoint: Works but deprecated")
                print(f"   Deprecation header: {deprecated_header}")
                print(f"   New endpoint: {new_endpoint}")
                print(f"   Data points: {len(data['weather']['data'])}")
            else:
                print(f"❌ Legacy Endpoint failed: {response.status_code} - {response.text}")
            
            # Test 7: Error Handling
            print("\n7. Testing Error Handling...")
            response = await client.get(f"{BASE_URL}/v1/records/daily/invalid_location/01-15", headers=headers)
            if response.status_code == 400:
                print(f"✅ Error Handling: Correctly returned 400 for invalid location")
            else:
                print(f"❌ Error Handling: Expected 400, got {response.status_code}")
            
            print("\n" + "=" * 50)
            print("🎉 V1 API Testing Complete!")
            
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_v1_api())
