#!/usr/bin/env python3
"""One-off cleanup for duplicate popular-location entries (API#103).

The web splash auto-detect flow used to POST /v1/locations/selections with
only a raw display-name string (no coordinates / country_code), so
_resolve_canonical_id (routers/locations.py) fell through to slugifying the
whole composite string instead of converging onto the existing catalog
entry. That minted a second, metadata-less ID for the same physical place,
e.g. "takayama_gifu_prefecture_japan" alongside the real "takayama".

Both the web client (splash.ts) and the API resolver have since been fixed
so new submissions converge correctly. This script merges the accumulated
Redis state for already-minted duplicate IDs into their canonical IDs:

  - selections:{date} sorted sets — bad_id's score is added to canonical_id's
    score (ZINCRBY) for every daily key in the popularity window, then bad_id
    is removed from that key.
  - recent_selections:24h — bad_id's recency score is merged into
    canonical_id's via MAX (a duplicate should not un-recency the real entry
    if the duplicate was selected more recently), then bad_id is removed.
  - loc_geo:v1 — bad_id is removed (canonical_id's position, if any, is left
    untouched; bad_id never had useful anchor value since it was minted
    without coordinates in the first place).
  - loc_meta:{bad_id} / loc_display:{bad_id} — deleted (canonical_id already
    carries correct catalog metadata/display string).

Usage:
    python -m scripts.merge_duplicate_locations              # dry run
    python -m scripts.merge_duplicate_locations --execute     # apply changes
    python -m scripts.merge_duplicate_locations --execute --days 120

Run after the web + API fixes for API#103 are deployed. Extend
KNOWN_DUPLICATE_MAPPINGS below with any additional bad_id -> canonical_id
pairs found via `GET /v1/locations/popular` before running with --execute.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.usage import RECENT_24H_KEY, LocationUsageTracker  # noqa: E402

# (bad_id, canonical_id) pairs identified from GET /v1/locations/popular.
# Add more here (without removing the existing ones) before re-running.
KNOWN_DUPLICATE_MAPPINGS: List[Tuple[str, str]] = [
    ("takayama_gifu_prefecture_japan", "takayama"),
    ("kanazawa_ishikawa_prefecture_japan", "kanazawa"),
    ("greater_london_england_united_kingdom", "london"),
]


def _daily_keys(days: int) -> List[str]:
    now = datetime.now(timezone.utc)
    return [f"selections:{(now - timedelta(days=i)).strftime('%Y%m%d')}" for i in range(days)]


def merge_location(
    r: redis.Redis,
    bad_id: str,
    canonical_id: str,
    days: int,
    dry_run: bool,
) -> dict:
    """Merge all Redis state for bad_id into canonical_id. Returns a summary dict."""
    summary = {
        "bad_id": bad_id,
        "canonical_id": canonical_id,
        "selections_merged": {},
        "recent_24h_merged": False,
        "geo_index_removed": False,
        "meta_removed": False,
        "display_removed": False,
    }

    for key in _daily_keys(days):
        score = r.zscore(key, bad_id)
        if score is None:
            continue
        summary["selections_merged"][key] = score
        if not dry_run:
            r.zincrby(key, score, canonical_id)
            r.zrem(key, bad_id)

    bad_recent = r.zscore(RECENT_24H_KEY, bad_id)
    if bad_recent is not None:
        summary["recent_24h_merged"] = bad_recent
        if not dry_run:
            canonical_recent = r.zscore(RECENT_24H_KEY, canonical_id)
            if canonical_recent is None or bad_recent > canonical_recent:
                r.zadd(RECENT_24H_KEY, {canonical_id: bad_recent})
            r.zrem(RECENT_24H_KEY, bad_id)

    if r.zscore(LocationUsageTracker.geo_index_key, bad_id) is not None:
        summary["geo_index_removed"] = True
        if not dry_run:
            r.zrem(LocationUsageTracker.geo_index_key, bad_id)

    if r.exists(f"loc_meta:{bad_id}"):
        summary["meta_removed"] = True
        if not dry_run:
            r.delete(f"loc_meta:{bad_id}")

    if r.exists(f"loc_display:{bad_id}"):
        summary["display_removed"] = True
        if not dry_run:
            r.delete(f"loc_display:{bad_id}")

    return summary


def _print_summary(summary: dict, dry_run: bool) -> None:
    verb = "Would merge" if dry_run else "Merged"
    total_score = sum(summary["selections_merged"].values())
    print(f"\n{summary['bad_id']} -> {summary['canonical_id']}")
    if not any(
        [
            summary["selections_merged"],
            summary["recent_24h_merged"],
            summary["geo_index_removed"],
            summary["meta_removed"],
            summary["display_removed"],
        ]
    ):
        print("  nothing found — already clean or never existed")
        return
    if summary["selections_merged"]:
        print(f"  {verb} {total_score:.0f} selection(s) across {len(summary['selections_merged'])} daily key(s)")
    if summary["recent_24h_merged"] is not False:
        print(f"  {verb} 24h recency score ({summary['recent_24h_merged']:.0f})")
    if summary["geo_index_removed"]:
        print(f"  {'Would remove' if dry_run else 'Removed'} from geo index")
    if summary["meta_removed"]:
        print(f"  {'Would delete' if dry_run else 'Deleted'} loc_meta:{summary['bad_id']}")
    if summary["display_removed"]:
        print(f"  {'Would delete' if dry_run else 'Deleted'} loc_display:{summary['bad_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Apply changes (default is dry-run)")
    parser.add_argument(
        "--days", type=int, default=120, help="How many daily selections:{date} keys to scan (default 120)"
    )
    parser.add_argument("--redis-url", default=None, help="Redis URL (default: REDIS_URL env var)")
    args = parser.parse_args()

    redis_url = args.redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
    r = redis.from_url(redis_url, decode_responses=True)
    r.ping()

    dry_run = not args.execute
    print(f"{'DRY RUN — no changes will be made' if dry_run else 'EXECUTING — changes will be applied'}")
    print(f"Scanning {args.days} daily selections:{{date}} keys for {len(KNOWN_DUPLICATE_MAPPINGS)} mapping(s)")

    for bad_id, canonical_id in KNOWN_DUPLICATE_MAPPINGS:
        summary = merge_location(r, bad_id, canonical_id, days=args.days, dry_run=dry_run)
        _print_summary(summary, dry_run)

    if dry_run:
        print("\nRe-run with --execute to apply the changes above.")
    else:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
