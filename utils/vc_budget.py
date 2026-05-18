"""Visual Crossing API daily record budget.

Tracks the number of weather-data records (days of data) fetched from the
Visual Crossing API each calendar day and refuses further fetches once the
configured budget is exhausted.

Budget is stored as a Redis counter under  vc:budget:YYYY-MM-DD  with a
slightly-longer-than-24h TTL so it survives midnight rollovers naturally.

Configuration (environment variables):
    VC_DAILY_RECORD_BUDGET  Maximum records per UTC calendar day (default: 100000).
                            Set to 0 to disable the limit.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import redis as _redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_BUDGET = 0  # disabled by default — set VC_DAILY_RECORD_BUDGET to a positive integer to enable


def _daily_budget() -> int:
    """Return the configured daily VC record budget (0 = unlimited)."""
    try:
        return int(os.getenv("VC_DAILY_RECORD_BUDGET", str(_DEFAULT_BUDGET)))
    except (TypeError, ValueError):
        return _DEFAULT_BUDGET


def _budget_key() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"vc:budget:{today}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_and_consume(
    r: _redis.Redis,
    records: int,
) -> bool:
    """Check whether *records* can be fetched without exceeding the daily budget.

    Atomically increments the counter by *records* and returns True if the
    resulting total is still within budget, False if the budget would be (or
    already is) exceeded.

    When budget is 0 (disabled) always returns True.

    Args:
        r:        Redis client.
        records:  Number of weather-data records about to be fetched.

    Returns:
        True → fetch is allowed; False → fetch should be skipped.
    """
    budget = _daily_budget()
    if budget <= 0:
        return True  # budget disabled

    key = _budget_key()
    try:
        # Increment first, then check.  If we're already over, the extra
        # increment is harmless (the key expires tomorrow anyway).
        new_total = r.incrby(key, records)

        # Set 25h TTL on first write (key may already have a TTL — only set once)
        if new_total == records:
            r.expire(key, 90_000)  # 25 hours

        if new_total > budget:
            logger.warning(
                "🚫 VC daily record budget exhausted: %d/%d records used today "
                "(attempted to add %d). Skipping VC fetch.",
                new_total,
                budget,
                records,
            )
            # Roll back so the overshoot doesn't inflate tomorrow's reading
            r.decrby(key, records)
            return False

        # Warn at 80 % and 95 %
        for threshold in (0.95, 0.80):
            if (new_total - records) / budget < threshold <= new_total / budget:
                logger.warning(
                    "⚠️  VC daily record budget at %.0f%%: %d/%d",
                    threshold * 100,
                    new_total,
                    budget,
                )
                break

        return True

    except Exception as exc:
        # Redis failure must not block the fetch — log and allow.
        logger.warning("VC budget check failed (Redis error): %s — allowing fetch", exc)
        return True


def current_usage(r: _redis.Redis) -> dict:
    """Return current budget usage stats for monitoring/admin endpoints."""
    budget = _daily_budget()
    key = _budget_key()
    try:
        used_raw = r.get(key)
        used = int(used_raw) if used_raw else 0
    except Exception:
        used = -1

    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "used": used,
        "budget": budget,
        "remaining": max(0, budget - used) if budget > 0 else None,
        "pct_used": round(used / budget * 100, 1) if budget > 0 and used >= 0 else None,
        "budget_enabled": budget > 0,
    }
