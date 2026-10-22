#!/usr/bin/env python3
"""One-off backfill of shares.country_code (API#151).

Shares made before country_code existed have it NULL, so clients can't show their
flag. This resolves the country of each such share the same way new shares are
resolved (preapproved location, country named in the location text, then the
geocoder using the share's coordinates when it has them) and stores it. Shares
whose country can't be determined confidently are left NULL.

GET /v1/shares/{id} caches each share in Redis for 30 days under `share:{id}`, so
the cache entry of every updated share is deleted; otherwise the new country
would not show until it expired.

Usage:
    python -m scripts.backfill_share_country            # dry run, changes nothing
    python -m scripts.backfill_share_country --execute  # apply

Needs TEMPHIST_PG_DSN (or DATABASE_URL), and REDIS_URL for the cache entries.
"""

import argparse
import asyncio
import os
import sys
from typing import Dict, Optional

import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import routers.locations as locations_module  # noqa: E402
from routers._share_country import resolve_share_country  # noqa: E402
from utils.share_store import ShareStore  # noqa: E402


async def prepare_resolver(r: Optional[redis.Redis]) -> None:
    """Set up what the app sets up at startup and the resolver relies on.

    - the preapproved-locations list (empty until loaded, so name matches would otherwise find nothing)
    - the Redis client the geocoder caches its results in (without it every lookup logs a
      "Redis client not initialized" warning and nothing is cached)
    """
    locations_module.locations_data, _etag, _modified = await locations_module.load_locations_data()
    if r is not None:
        locations_module.redis_client = r


async def backfill(store: ShareStore, r: Optional[redis.Redis], execute: bool) -> Dict[str, int]:
    await prepare_resolver(r)
    shares = await store.list_shares_missing_country()
    if shares is None:
        raise SystemExit("No Postgres DSN configured (TEMPHIST_PG_DSN or DATABASE_URL).")

    summary = {"missing": len(shares), "resolved": 0, "unresolved": 0, "updated": 0}
    for share in shares:
        code, detail = await resolve_share_country(
            share["location"], None, latitude=share["latitude"], longitude=share["longitude"]
        )
        if code is None:
            summary["unresolved"] += 1
            print(f"  ?  {share['id']}  {share['location']!r}  (left NULL: {detail})")
            continue
        summary["resolved"] += 1
        print(f"  {code} {share['id']}  {share['location']!r}  ({detail})")
        if execute and await store.set_country_code(share["id"], code):
            summary["updated"] += 1
            if r is not None:
                r.delete(f"share:{share['id']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Apply the changes (default is a dry run)")
    parser.add_argument("--redis-url", default=None, help="Redis URL (default: REDIS_URL env var)")
    args = parser.parse_args()

    redis_url = args.redis_url or os.getenv("REDIS_URL")
    r = redis.from_url(redis_url, decode_responses=True) if redis_url else None
    if args.execute and r is None:
        print("Warning: no REDIS_URL, so cached share records will keep the old value until they expire.")

    summary = asyncio.run(backfill(ShareStore(), r, args.execute))
    mode = "applied" if args.execute else "dry run, nothing changed"
    print(f"\n{summary['missing']} shares without a country: {summary['resolved']} resolved, "
          f"{summary['unresolved']} unresolved, {summary['updated']} updated ({mode})")


if __name__ == "__main__":
    main()
