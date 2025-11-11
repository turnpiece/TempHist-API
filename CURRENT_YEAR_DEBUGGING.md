# Current Year Debugging Guide

## Issue
Current year (2025) is frequently missing from `/v1/records/yearly/Berlin/10-30` responses, even though all requested dates (Oct 31, 2024 → Oct 30, 2025) are in the past and should be available.

## Debug Logging Added

Comprehensive logging has been added to track the current year through the entire data collection pipeline.

### 1. Initial Coverage Check (`routers/v1_records.py:280-294`)

**When**: After fetching from PostgreSQL cache, before Visual Crossing API call

```
🔍 CURRENT YEAR DEBUG [2025]: Berlin | Period: yearly |
Anchor: 2025-10-30 |
Date range: 2024-10-31 to 2025-10-30 |
Expected: 365 days | Available: 280 days |
Missing: 85 days | Coverage: 76.7% |
Coverage OK: False
```

**What it tells you:**
- Which dates are requested (should all be in the past)
- How many days are in the cache vs missing
- Whether coverage passes the threshold (90% for yearly)
- Lists missing dates if < 10 dates missing

### 2. Visual Crossing API Fetch (`routers/v1_records.py:300-333`)

**When**: Coverage is below threshold, attempting to fetch missing data

```
🔍 CURRENT YEAR [2025]: Coverage below threshold, fetching from Visual Crossing API...
🔍 CURRENT YEAR [2025]: Fetching timeline 2024-11-01 to 2025-10-30
🔍 CURRENT YEAR [2025]: Timeline fetch succeeded, got 365 days
```

**OR if it fails:**
```
❌ timeline fetch failed for Berlin (2024-11-01 to 2025-10-30): HTTP 401 Unauthorized
❌ CURRENT YEAR [2025]: Timeline fetch FAILED - this is likely why current year is missing!
```

**What it tells you:**
- Whether API fetch was attempted
- Which date range was fetched
- Whether fetch succeeded or failed
- If failed, the exact error (rate limit, auth, network, etc.)

### 3. Post-Fetch Re-evaluation (`routers/v1_records.py:378-385`)

**When**: After storing fetched data, re-checking coverage

```
🔍 CURRENT YEAR [2025]: After timeline fetch -
Available: 365/365 days |
Missing: 0 |
Coverage: 100.0% |
Coverage OK: True
```

**What it tells you:**
- Whether fetching fixed the coverage issue
- How many days are now available
- Whether year will be included or skipped

### 4. Year Skipped (`routers/v1_records.py:393-401`)

**When**: Current year is excluded due to insufficient coverage

```
❌ CURRENT YEAR [2025] SKIPPED: Coverage below threshold! |
Final missing: 35 days |
Available: 330/365 |
Required: 330 days min
❌ Final missing dates: ['2025-01-15', '2025-02-20', ...]
```

**What it tells you:**
- Exactly why the year was skipped
- How many days are missing
- Which specific dates are missing (up to 20 shown)
- What the coverage threshold is

### 5. Year Included (`routers/v1_records.py:426-430`)

**When**: Current year successfully added to response

```
✅ CURRENT YEAR [2025] INCLUDED: 365/365 days |
Average temp: 13.2°C
```

**What it tells you:**
- Year was successfully included
- How many days were used in the average
- The calculated average temperature

---

## Common Failure Scenarios

### Scenario 1: Visual Crossing API Failure
```
🔍 CURRENT YEAR DEBUG [2025]: ... Coverage OK: False
🔍 CURRENT YEAR [2025]: Coverage below threshold, fetching from Visual Crossing API...
❌ timeline fetch failed: HTTP 429 Too Many Requests
❌ CURRENT YEAR [2025]: Timeline fetch FAILED
❌ CURRENT YEAR [2025] SKIPPED: Coverage below threshold!
```

**Cause**: API rate limit, auth failure, or network issue
**Solution**: Check API quota, verify API key, check network connectivity

### Scenario 2: Partial Cache + Failed API Fetch
```
🔍 CURRENT YEAR DEBUG [2025]: Available: 280 days | Missing: 85 days | Coverage: 76.7%
🔍 CURRENT YEAR [2025]: Fetching timeline 2024-11-01 to 2025-10-30
❌ timeline fetch failed: Invalid location
❌ CURRENT YEAR [2025] SKIPPED: Coverage below threshold!
```

**Cause**: Location name not recognized by Visual Crossing
**Solution**: Check location normalization, use preapproved locations

### Scenario 3: Coverage Just Below Threshold
```
🔍 CURRENT YEAR DEBUG [2025]: Available: 328 days | Missing: 37 days | Coverage: 89.9%
🔍 CURRENT YEAR [2025]: After timeline fetch - Available: 328/365 | Coverage: 89.9%
❌ CURRENT YEAR [2025] SKIPPED: Coverage below threshold! | Required: 330 days min
```

**Cause**: Coverage is 89.9% but threshold is 90% (330 days)
**Solution**:
- This is where the relaxed current year tolerance helps (now 50%)
- Or investigate why 37 days are missing from Visual Crossing

### Scenario 4: Success After API Fetch
```
🔍 CURRENT YEAR DEBUG [2025]: Available: 0 days | Missing: 365 days | Coverage: 0.0%
🔍 CURRENT YEAR [2025]: Fetching timeline 2024-10-31 to 2025-10-30
🔍 CURRENT YEAR [2025]: Timeline fetch succeeded, got 365 days
🔍 CURRENT YEAR [2025]: After timeline fetch - Available: 365/365 | Coverage: 100.0%
✅ CURRENT YEAR [2025] INCLUDED: 365/365 days | Average temp: 13.2°C
```

**Cause**: Fresh location, no cache, successful API fetch
**Solution**: Working as intended!

---

## How to Use This Debugging

### 1. Reproduce the Issue
```bash
curl "http://localhost:8000/api/v1/records/yearly/Berlin/10-30"
```

### 2. Check Logs
Look for the emoji-prefixed log lines:
- `🔍` = Diagnostic info
- `❌` = Error/failure
- `✅` = Success

### 3. Follow the Flow
Track the current year through all 5 stages:
1. Initial cache check
2. API fetch attempt (if needed)
3. Post-fetch re-evaluation
4. Skip decision OR Include decision
5. Final result

### 4. Identify Root Cause

| Symptom | Root Cause |
|---------|------------|
| No API fetch attempt | Coverage was OK from cache |
| API fetch fails | API issue (rate limit, auth, network) |
| API fetch succeeds but still skipped | Not enough days fetched, or bad data quality |
| Missing dates are in the future | Date calculation bug (should all be in past) |
| All 365 days missing from cache | New location, cold start |

---

## Monitoring in Production

### Watch for Patterns
```bash
# Count current year failures
grep "CURRENT YEAR.*SKIPPED" temphist.log | wc -l

# Most common failure reasons
grep "timeline fetch failed" temphist.log | cut -d: -f5 | sort | uniq -c

# Check if coverage is consistently close to threshold
grep "CURRENT YEAR.*Coverage: " temphist.log | grep -oP 'Coverage: \K[0-9.]+%'
```

### Alert Thresholds
- **High**: >10% of current year requests result in SKIPPED
- **Medium**: >5% timeline fetch failures
- **Low**: Coverage consistently 85-95% (suggests threshold tuning needed)

---

## Next Steps Based on Findings

### If API is failing frequently:
1. Check Visual Crossing API status page
2. Verify API key and rate limits
3. Add retry logic with exponential backoff
4. Consider fallback data source

### If coverage is consistently just below threshold:
1. Review if 90% threshold is too strict for current year
2. Consider implementing the relaxed threshold (already added)
3. Investigate why specific dates are consistently missing

### If missing dates are in the future:
1. Bug in date calculation - anchor date is wrong
2. Check `_resolve_anchor_date` logic
3. Verify timezone handling

### If all data is missing:
1. PostgreSQL connection issue
2. Location name normalization problem
3. Cache eviction too aggressive

---

## Temporary Debugging Increase

To get even more detail temporarily:

```python
# In routers/v1_records.py, add at top of _collect_rolling_window_values:
logger.warning(f"🔍 Processing year {year} for {location}")

# Add after cache fetch:
logger.warning(f"🔍 Cache returned {len(cache)} days for {year}")
```

Remove these after debugging to avoid log spam.

---

## Files Modified

- `routers/v1_records.py` - Added 5 debug logging points for current year tracking

## Revert Instructions

If debug logging is too verbose, remove lines:
- 280-294 (Initial coverage check)
- 300-303 (API fetch notification)
- 307-318 (API fetch success/failure)
- 378-385 (Post-fetch re-evaluation)
- 393-401 (Skip decision)
- 426-430 (Include success)

Or simply comment out with `if False:` wrapper.
