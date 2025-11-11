# Debug Current Year Failure - Quick Start

## What Changed

Added `DEBUG_CURRENT_YEAR_FAILURES = True` flag in `routers/v1_records.py:45`

When enabled, the API will **immediately raise an HTTP 500 exception** with full diagnostic details whenever the current year (2025) is skipped from a response.

## How to Use

### 1. Start Your API
```bash
# Run your API normally
uvicorn main:app --reload
```

### 2. Test the Endpoint
```bash
curl "http://localhost:8000/api/v1/records/yearly/Berlin/10-30"
```

### 3. See the Failure Details in Console

If the current year is being skipped, you'll immediately see a **clear error** in your console:

```json
{
  "detail": "❌ CURRENT YEAR [2025] SKIPPED FOR BERLIN! | Period: yearly | Anchor Date: 2025-10-30 | Date Range: 2024-10-31 to 2025-10-30 | Expected Days: 365 | Available Days: 328 | Missing Days: 37 | Coverage: 89.9% | Required: 90% (330 days) | Timeline Failed: False | Missing Dates: 2024-11-05, 2024-11-12, 2024-11-19, ..."
}
```

OR if Visual Crossing API failed:

```json
{
  "detail": "❌ CURRENT YEAR [2025] TIMELINE FETCH FAILED FOR BERLIN! | Period: yearly | Requested Range: 2024-10-31 to 2025-10-30 | Error: HTTPStatusError: 429 Too Many Requests | Initial Missing Days: 365 | Initial Coverage: 0.0% | This Visual Crossing API failure is preventing current year data from being included. | Check: API key validity, rate limits, network connectivity, location name."
}
```

## What You'll Learn

The error message tells you **exactly** why the current year was excluded:

### Scenario 1: Coverage Below Threshold
```
Expected Days: 365
Available Days: 328
Missing Days: 37
Coverage: 89.9%
Required: 90% (330 days)
Timeline Failed: False
```

**Meaning**:
- Cache had 328 days
- Needed 330 days (90% threshold)
- Just 2 days short!
- Visual Crossing API was called and succeeded
- But still didn't meet threshold

**Action**: Lower the threshold or investigate why those specific dates are missing

### Scenario 2: Visual Crossing API Failed
```
Error: HTTPStatusError: 429 Too Many Requests
Initial Missing Days: 365
Initial Coverage: 0.0%
Timeline Failed: True
```

**Meaning**:
- No data in cache (cold start)
- Visual Crossing API call failed
- Can't fetch the data needed

**Action**: Check API key, rate limits, or network issues

### Scenario 3: Location Invalid
```
Error: HTTPStatusError: 404 Not Found
Missing Dates: (all 365 days)
```

**Meaning**:
- Location name not recognized by Visual Crossing
- Or location doesn't exist

**Action**: Check location normalization, use preapproved locations

## Disable Debug Mode

Once you've identified the issue, disable the debug flag:

```python
# In routers/v1_records.py line 45:
DEBUG_CURRENT_YEAR_FAILURES = False  # Change to False
```

This will restore normal behavior (current year silently skipped, logged only).

## Common Issues & Solutions

| Error Message Contains | Problem | Solution |
|------------------------|---------|----------|
| `Coverage: 89.9% | Required: 90%` | Just below threshold | Lower threshold or investigate missing dates |
| `429 Too Many Requests` | API rate limit | Wait or upgrade API plan |
| `401 Unauthorized` | Invalid API key | Check VISUAL_CROSSING_API_KEY env var |
| `404 Not Found` | Invalid location | Use different location name or preapproved location |
| `Timeline Failed: True` | API call failed completely | Check network, API status, logs |
| `Missing Days: 365` | Cold start, no cache | Normal for first request, should succeed after API fetch |

## Next Steps After Diagnosis

### If it's an API issue:
1. Check Visual Crossing API status page
2. Verify your API key is valid and has quota
3. Check network connectivity
4. Try a different location to see if it's location-specific

### If it's a coverage issue:
1. Note which specific dates are missing
2. Check if those dates exist on Visual Crossing for that location
3. Consider adjusting the coverage threshold
4. Investigate if there's a pattern (e.g., always missing recent dates)

### If it's a location issue:
1. Try a preapproved location like "London, England"
2. Check location normalization logic
3. Verify the location exists in Visual Crossing

## Files Modified

- `routers/v1_records.py:45` - Added DEBUG_CURRENT_YEAR_FAILURES flag
- `routers/v1_records.py:330-347` - Timeline fetch failure exception
- `routers/v1_records.py:397-427` - Coverage threshold failure exception

## Restore Normal Behavior

Set `DEBUG_CURRENT_YEAR_FAILURES = False` when you're done debugging to restore normal API operation without exceptions.
