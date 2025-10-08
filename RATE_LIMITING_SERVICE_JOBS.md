# Rate Limiting Bypass for Service Jobs

## Summary

Modified the API to exclude service jobs (authenticated with `API_ACCESS_TOKEN`) from rate limiting. Rate limiting is now only applied to actual users connecting with Firebase authentication tokens.

## Changes Made

### 1. Main Middleware Changes (`main.py`)

#### Lines 875-883: Early Detection of Service Jobs

- Added early detection of `API_ACCESS_TOKEN` in the authorization header
- Set `is_service_job = True` when the token matches
- Added debug logging to indicate when service jobs are detected

#### Lines 889-890: Updated Rate Limiting Condition

- Modified the rate limiting check to exclude service jobs
- Rate limiting is now skipped for:
  - Whitelisted IPs
  - Service jobs (using `API_ACCESS_TOKEN`)

#### Lines 969-972: Optimized Token Verification

- Reused the already-extracted `auth_header` and `is_service_job` flag
- Avoided redundant token parsing and comparison
- Added comment explaining the optimization

### 2. Rate Limit Status Endpoint (`main.py`, lines 1783-1822)

Updated `/rate-limit-status` endpoint to:

- Detect and report service job status
- Include `service_job` field in the `ip_status` response
- Exclude service jobs from rate limit statistics
- Update `rate_limited` calculation to consider service jobs

### 3. Test Coverage (`test_main.py`)

Added comprehensive test coverage:

#### `test_service_job_bypasses_rate_limits` (lines 761-790)

- Verifies that requests with `API_ACCESS_TOKEN` bypass rate limiting
- Tests the `/rate-limit-status` endpoint with service token
- Confirms service jobs can access protected endpoints without rate limits

#### Updated `test_rate_limit_status_shows_ip_status` (lines 792-811)

- Added assertion for `service_job` field
- Verifies the field is a boolean value

## API Response Changes

### `/rate-limit-status` Endpoint

**Before:**

```json
{
  "client_ip": "192.168.1.1",
  "ip_status": {
    "whitelisted": false,
    "blacklisted": false,
    "rate_limited": true
  },
  "location_monitor": {...},
  "request_monitor": {...},
  "rate_limits": {...}
}
```

**After:**

```json
{
  "client_ip": "192.168.1.1",
  "ip_status": {
    "whitelisted": false,
    "blacklisted": false,
    "service_job": true,
    "rate_limited": false
  },
  "location_monitor": {},
  "request_monitor": {},
  "rate_limits": {...}
}
```

## Usage

### Service Jobs (No Rate Limiting)

```bash
curl -H "Authorization: Bearer $API_ACCESS_TOKEN" \
     https://api.temphist.com/v1/records/daily/New%20York/01-15
```

### Regular Users (Rate Limited)

```bash
curl -H "Authorization: Bearer $FIREBASE_ID_TOKEN" \
     https://api.temphist.com/v1/records/daily/New%20York/01-15
```

## Testing

All tests pass successfully:

```bash
# Test service job bypass
pytest test_main.py::TestIPWhitelistBlacklistIntegration::test_service_job_bypasses_rate_limits -v

# Test rate limit status endpoint
pytest test_main.py::TestIPWhitelistBlacklistIntegration::test_rate_limit_status_shows_ip_status -v

# Run all IP whitelist/blacklist integration tests
pytest test_main.py::TestIPWhitelistBlacklistIntegration -v

# Run all rate limiting integration tests
pytest test_main.py::TestRateLimitingIntegration -v
```

## Debug Logging

When `DEBUG=true`, service job requests are logged:

```
🔧 SERVICE JOB DETECTED: 192.168.1.1 | GET /v1/records/daily/london/01-15 | Rate limiting bypassed
```

## Security Considerations

- Service job detection happens early in the middleware, before rate limiting checks
- The `API_ACCESS_TOKEN` is still validated for authentication
- Service jobs are still subject to IP blacklisting (if configured)
- Service jobs cannot bypass authentication, only rate limiting
- The implementation follows the same pattern as whitelisted IPs

## Backward Compatibility

This change is fully backward compatible:

- Existing Firebase authentication continues to work
- Whitelisted IPs still bypass rate limiting as before
- Rate limiting behavior for regular users is unchanged
- Only adds new bypass logic for service jobs

## Environment Variables

Required for service job functionality:

```bash
API_ACCESS_TOKEN=your_secure_token_here
```

Optional rate limiting configuration:

```bash
RATE_LIMIT_ENABLED=true
MAX_LOCATIONS_PER_HOUR=10
MAX_REQUESTS_PER_HOUR=100
RATE_LIMIT_WINDOW_HOURS=1
```

## Benefits

1. **Efficient Automated Systems**: Cron jobs and background tasks can query the API without hitting rate limits
2. **Better User Experience**: Rate limits are focused on protecting against user abuse, not internal services
3. **Clear Distinction**: Service jobs are clearly identified in logs and monitoring
4. **Flexible Management**: Service jobs can be independently configured from user rate limits
5. **Cost Optimization**: Automated cache warming and prefetching can operate efficiently

## Future Enhancements

Potential improvements for consideration:

- Add separate rate limiting for service jobs (if needed)
- Track service job usage separately in analytics
- Add admin endpoint to view service job request statistics
- Support multiple API access tokens for different services
