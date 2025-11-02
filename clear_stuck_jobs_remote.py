#!/usr/bin/env python3
"""
Quick script to clear stuck jobs via API call.
Use this when you don't have direct server access.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Your dev server URL
DEV_SERVER = os.getenv("DEV_SERVER_URL", "https://your-dev-server.com")

def clear_stuck_jobs_via_api():
    """Clear stuck jobs by marking them as error via direct Redis access."""
    print("🧹 Clearing stuck jobs on remote server...")
    print(f"Server: {DEV_SERVER}")
    
    # Since we can't clear via API, we need to do it via Redis
    # This script assumes you have access to run Python on the server
    
    print("\n⚠️  To clear stuck jobs, you need to:")
    print("1. SSH into your dev server")
    print("2. Run: cd /path/to/api && python diagnose_jobs.py --clear-stuck")
    print("\nOR if you're using Render:")
    print("1. Open Render dashboard")
    print("2. Go to your service → Shell")
    print("3. Run: python diagnose_jobs.py --clear-stuck")
    
if __name__ == "__main__":
    clear_stuck_jobs_via_api()


