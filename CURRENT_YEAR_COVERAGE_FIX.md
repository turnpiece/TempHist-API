# Current Year Coverage Fix

## Problem

When requesting yearly/monthly/weekly data (e.g., `/v1/records/yearly/Berlin/10-30`), the **current year was frequently missing** from responses despite having partial data available.

### Root Cause

The coverage tolerance logic treated the current year the same as historical years:

1. For yearly period, required **90% coverage** (330 of 365 days)
2. For current year queries, many future dates don't exist yet in cache
3. When coverage fell below 90%, the code would:
   - Try to fetch from Visual Crossing API
   - If still below 90%, **skip the entire year** with `continue`
4. Result: Current year was excluded even when 50-80% of data was available

**The critical issue**: Lines 323-325 in `_collect_rolling_window_values`:
```python
if final_missing_dates and not coverage_ok:
    track_missing_year(missing_years, year, "coverage_below_threshold")
    continue  # <- SKIPS ENTIRE YEAR including current year!
```

## Solution

Implemented **relaxed coverage requirements for the current year** since it's incomplete by nature.

### Changes Made

#### 1. Added Current Year Coverage Tolerance (`routers/v1_records.py:186-192`)

```python
# Lower coverage requirements for the current year (incomplete by nature)
COVERAGE_TOLERANCE_CURRENT_YEAR = {
    # For current year, accept more sparse data since year is incomplete
    "weekly": {"min_ratio": 0.5, "min_days": 4},    # At least 4 of 7 days (57%)
    "monthly": {"min_ratio": 0.6, "min_days": 19},  # At least 19 of 31 days (61%)
    "yearly": {"min_ratio": 0.5, "min_days": 183},  # At least half the year (50%)
}
```

**Rationale**:
- Current year is always incomplete (we're still in it)
- More valuable to users than historical data
- Users expect to see current year even with partial data
- 50% coverage for yearly = ~6 months of data (reasonable for averages)

#### 2. Updated `_evaluate_coverage` Function (`routers/v1_records.py:195-223`)

Added `year` parameter to detect current year and apply appropriate thresholds:

```python
def _evaluate_coverage(
    period: str,
    available_days: int,
    expected_days: int,
    year: Optional[int] = None  # NEW PARAMETER
) -> Tuple[bool, float]:
    """Determine whether available data meets tolerance for requested period.

    Uses relaxed thresholds for current year since it's incomplete by nature.
    """
    if expected_days == 0:
        return False, 0.0

    # Use relaxed coverage requirements for current year
    current_year = datetime.now().year
    is_current_year = year == current_year

    if is_current_year:
        tolerance = COVERAGE_TOLERANCE_CURRENT_YEAR.get(period, {...})
    else:
        tolerance = COVERAGE_TOLERANCE.get(period, {...})

    # ... rest of function
```

#### 3. Updated All Calls to Pass Year (`routers/v1_records.py:276, 343`)

```python
# Before:
coverage_ok, _ = _evaluate_coverage(period, available_count, expected_days)

# After:
coverage_ok, _ = _evaluate_coverage(period, available_count, expected_days, year=year)
```

#### 4. Added Missing Import (`routers/v1_records.py:6`)

```python
from typing import Literal, Dict, List, Tuple, Optional
```

---

## Coverage Comparison

### Historical Years (e.g., 2024, 2023...)
| Period | Min Days | Min Ratio | Example |
|--------|----------|-----------|---------|
| Weekly | 6 of 7 | 86% | Missing 1 day is OK |
| Monthly | 28 of 31 | 90% | Missing 3 days is OK |
| Yearly | 330 of 365 | 90% | Missing 35 days is OK |

### Current Year (2025)
| Period | Min Days | Min Ratio | Example |
|--------|----------|-----------|---------|
| Weekly | 4 of 7 | 57% | Missing 3 days is OK |
| Monthly | 19 of 31 | 61% | Missing 12 days is OK |
| Yearly | 183 of 365 | 50% | Missing 182 days is OK |

---

## Example Scenarios

### Before Fix:
```
Request: /v1/records/yearly/Berlin/10-30
Current date: November 15, 2025

Year 2025:
- Expected: 365 days (ending Oct 30)
- Available: 280 days (77% coverage)
- Coverage check: FAIL (needs 90%)
- Result: Current year EXCLUDED ❌

Response: {
  "values": [
    {"year": 2024, "temperature": 12.5},
    {"year": 2023, "temperature": 11.8},
    // 2025 is missing!
  ]
}
```

### After Fix:
```
Request: /v1/records/yearly/Berlin/10-30
Current date: November 15, 2025

Year 2025:
- Expected: 365 days (ending Oct 30)
- Available: 280 days (77% coverage)
- Coverage check: PASS (needs 50% for current year)
- Result: Current year INCLUDED ✅

Response: {
  "values": [
    {"year": 2025, "temperature": 13.2},  // ← NOW INCLUDED!
    {"year": 2024, "temperature": 12.5},
    {"year": 2023, "temperature": 11.8},
  ]
}
```

---

## Edge Cases Handled

### 1. Very Sparse Current Year Data
If current year has < 50% coverage (< 183 days for yearly):
- Still excluded (prevents misleading averages)
- Logged as "coverage_below_threshold"

### 2. Future Dates
For dates in the future (e.g., requesting `/10-30` when it's only August):
- Many dates won't exist yet
- Relaxed threshold allows partial year aggregation
- Average computed from available data

### 3. Leap Years
- Feb 29 handling already exists in `_resolve_anchor_date`
- Coverage ratios adjust automatically (366 vs 365 days)

---

## Testing

### Manual Test:
```bash
# Test current year is included
curl "http://localhost:8000/api/v1/records/yearly/Berlin/10-30" | jq '.values[] | select(.year == 2025)'

# Should return current year data if >= 50% coverage available
```

### Expected Behavior:
- ✅ Current year appears in response with >= 50% coverage
- ✅ Historical years still require 90% coverage
- ✅ Averages are accurate (computed from available days only)
- ✅ Metadata shows `approximate: true` when coverage < 100%

---

## Migration Notes

- **No database changes required** (pure logic change)
- **Backward compatible** (historical years unchanged)
- **Immediate effect** (no restart needed, applies per-request)
- **Safe rollback** (revert changes to restore old behavior)

---

## Future Improvements

### Potential Enhancements:
1. **Gradual threshold decay**: Lower requirements as year progresses
   - January: Need 90% of available days
   - December: Need 50% of full year

2. **Adaptive thresholds**: Based on Visual Crossing API availability
   - If API call fails, accept lower coverage
   - If API succeeds, require higher coverage

3. **User preference**: Allow clients to specify minimum coverage
   - Query param: `?min_coverage=0.8`
   - Trade-off between completeness and availability

4. **Metadata enhancement**: Include coverage info in response
   ```json
   {
     "year": 2025,
     "temperature": 13.2,
     "coverage": {
       "available": 280,
       "expected": 365,
       "ratio": 0.77,
       "is_current_year": true
     }
   }
   ```

---

## Related Files

- `routers/v1_records.py` - Main implementation
- `cache_utils.py` - No changes (coverage is evaluated before caching)
- `models.py` - No changes (response schema unchanged)

---

## Impact

- ✅ Current year no longer missing from responses
- ✅ Better user experience (more complete data)
- ✅ Maintains data quality (still requires 50% minimum)
- ✅ No performance impact (same logic, different thresholds)
- ✅ Backward compatible (historical data unchanged)
