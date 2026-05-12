#!/usr/bin/env python3
"""
Verify that performance optimization indexes exist in the database.

Usage:
    python verify_indexes.py

Requirements:
    - TEMPHIST_PG_DSN or DATABASE_URL environment variable set
    - asyncpg installed
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

# Same as main.py: .env lives next to this file, not necessarily the process cwd.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


async def check_indexes() -> Tuple[bool, List[str]]:
    """Check if all required indexes exist."""
    try:
        import asyncpg
    except ImportError:
        print("❌ asyncpg not installed. Run: pip install asyncpg")
        return False, []

    dsn = os.getenv("TEMPHIST_PG_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        print("❌ TEMPHIST_PG_DSN or DATABASE_URL environment variable not set")
        return False, []

    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return False, []

    required_indexes = {
        "idx_locations_normalized": "locations",
        "idx_locations_coordinates": "locations",
        "idx_location_aliases_location_id": "location_aliases",
        "idx_daily_temperatures_location_day": "daily_temperatures",
    }

    missing = []
    found = []

    for index_name, table_name in required_indexes.items():
        result = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'public'
                  AND indexname = $1
                  AND tablename = $2
            )
            """,
            index_name,
            table_name,
        )

        if result:
            found.append(f"✅ {index_name} on {table_name}")
        else:
            missing.append(f"❌ {index_name} on {table_name}")

    await conn.close()

    return len(missing) == 0, found + missing


async def check_pool_connections() -> int:
    """Check current connection pool size."""
    try:
        import asyncpg
    except ImportError:
        return -1

    dsn = os.getenv("TEMPHIST_PG_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        return -1

    try:
        conn = await asyncpg.connect(dsn)
        count = await conn.fetchval(
            """
            SELECT count(*)
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND state = 'active'
            """
        )
        await conn.close()
        return count
    except Exception:
        return -1


async def main():
    """Main verification routine."""
    print("=" * 60)
    print("PostgreSQL Performance Optimization Verification")
    print("=" * 60)
    print()

    # Check indexes
    print("Checking Indexes...")
    print("-" * 60)
    all_present, results = await check_indexes()

    for result in results:
        print(result)

    print()

    if all_present:
        print("✅ All performance indexes are present!")
    else:
        print("⚠️  Some indexes are missing. They will be created on first API startup.")

    print()

    # Check connections
    print("Checking Database Connections...")
    print("-" * 60)
    active_connections = await check_pool_connections()

    if active_connections >= 0:
        print(f"📊 Active connections: {active_connections}")

        if active_connections >= 5:
            print("✅ Connection pool is active (expected: 5-20)")
        elif active_connections > 0:
            print("ℹ️  Connection pool size is lower than expected")
        else:
            print("ℹ️  No active connections (expected when API is idle)")
    else:
        print("⚠️  Could not check connection count")

    print()
    print("=" * 60)
    print("Verification Complete!")
    print("=" * 60)

    return 0 if all_present else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
