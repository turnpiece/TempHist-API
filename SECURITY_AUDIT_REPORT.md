# Security & Code Quality Audit Report

## TempHist API - Comprehensive Security Analysis

**Date**: 2025-01-28  
**Auditor**: Security Analysis Tool  
**Scope**: Complete codebase security and quality review

---

## Executive Summary

This audit identified **25 security vulnerabilities and code quality issues** across the codebase, categorized by severity:

- **Critical**: 3 issues
- **High**: 7 issues
- **Medium**: 10 issues
- **Low**: 5 issues

### Key Findings

1. **Authentication vulnerabilities** with hardcoded test tokens and insufficient token validation
2. **Input validation gaps** allowing potential injection attacks
3. **Information disclosure** through error messages and logging
4. **Dependency vulnerabilities** in outdated packages
5. **Missing CSRF protection** for state-changing operations
6. **Performance bottlenecks** in caching and rate limiting
7. **Code duplication** across multiple modules

---

## Critical Severity Issues

### CRI-001: Hardcoded Test Token Bypass in Production Code

**File**: `main.py`  
**Line**: 1039-1042  
**Severity**: Critical

**Issue**: Test token bypass exists in production middleware without proper environment checks.

```python
# Special bypass for testing
if id_token == TEST_TOKEN:
    if DEBUG:
        logger.debug(f"[DEBUG] Middleware: Using test token bypass")
    request.state.user = {"uid": "testuser"}
```

**Vulnerability**: If `TEST_TOKEN` is set in production, attackers can bypass Firebase authentication entirely.

**Fix**:

```python
# Only allow test token in development
if DEBUG and id_token == TEST_TOKEN:
    logger.debug(f"[DEBUG] Middleware: Using test token bypass")
    request.state.user = {"uid": "testuser"}
else:
    # Proceed with Firebase verification
```

**Recommendation**: Add environment check to ensure test token is only valid when `DEBUG=true` and in non-production environments.

---

### CRI-002: API Key Exposure in URL Parameters and Logs

**File**: `main.py`, `routers/records_agg.py`  
**Line**: 422, 441  
**Severity**: Critical

**Issue**: API keys are embedded in URL query parameters and may be logged.

```python
base_params = f"unitGroup={VISUAL_CROSSING_UNIT_GROUP}&include={VISUAL_CROSSING_INCLUDE_PARAMS}&key={API_KEY}"
```

**Vulnerability**:

- API keys in URLs may appear in server logs, browser history, referrer headers
- Logging might expose keys in error messages

**Fix**:

```python
# Use headers instead of query params when possible
params = {
    "unitGroup": VISUAL_CROSSING_UNIT_GROUP,
    "include": VISUAL_CROSSING_INCLUDE_PARAMS
}
headers = {"X-API-Key": API_KEY}  # If API supports headers
# Otherwise, ensure URLs are never logged
```

**Recommendation**:

- Sanitize API keys from all log messages
- Use environment variables that are never logged
- Consider using header-based authentication where API supports it

---

### CRI-003: Unsafe Location Input in URL Construction

**File**: `main.py`, `routers/records_agg.py`  
**Line**: 410-426, 435  
**Severity**: Critical

**Issue**: Location parameter is URL-encoded but not fully validated before use in external API calls.

```python
def build_visual_crossing_url(location: str, date: str, remote: bool = True) -> str:
    cleaned_location = clean_location_string(location)
    encoded_location = quote(cleaned_location, safe='')
    return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{date}?{base_params}"
```

**Vulnerability**:

- Path injection possible if location contains special characters
- SSRF risk if location can be crafted to point to internal services
- No length validation on location string

**Fix**:

```python
def build_visual_crossing_url(location: str, date: str, remote: bool = True) -> str:
    # Validate length
    if len(location) > 200:  # Reasonable limit
        raise ValueError("Location string too long")

    # Whitelist allowed characters (letters, numbers, spaces, commas, hyphens)
    import re
    if not re.match(r'^[a-zA-Z0-9\s,\-\.]+$', location):
        raise ValueError("Invalid characters in location")

    # Validate date format
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        raise ValueError("Invalid date format")

    cleaned_location = clean_location_string(location)
    encoded_location = quote(cleaned_location, safe='')

    # Additional validation: prevent SSRF
    if any(char in encoded_location for char in ['//', '@', ':', '?', '#']):
        raise ValueError("Invalid location format")

    return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{date}?{base_params}"
```

**Recommendation**:

- Implement strict input validation whitelist
- Add maximum length limits
- Prevent SSRF by blocking internal IP ranges and special characters

---

## High Severity Issues

### HIGH-001: Missing CSRF Protection

**File**: `main.py`  
**Line**: 3549, 2035, 2098  
**Severity**: High

**Issue**: POST endpoints that modify state lack CSRF token validation.

**Affected Endpoints**:

- `POST /analytics`
- `POST /cache-warm`
- `POST /cache-warm/job`
- `POST /v1/records/{period}/{location}/{identifier}/async`

**Vulnerability**: Cross-Site Request Forgery attacks can be executed against authenticated users.

**Fix**:

```python
from fastapi_csrf_protect import CsrfProtect

@app.post("/analytics", response_model=AnalyticsResponse)
async def submit_analytics(
    request: Request,
    analytics_data: AnalyticsData,
    csrf_protect: CsrfProtect = Depends()
):
    await csrf_protect.validate_csrf(request)
    # ... rest of handler
```

**Recommendation**: Implement CSRF tokens for all state-changing operations, especially POST/PUT/DELETE requests.

---

### HIGH-002: Information Disclosure in Error Messages

**File**: `main.py`, `routers/records_agg.py`  
**Line**: 842, 182, 449  
**Severity**: High

**Issue**: Detailed error messages expose internal implementation details.

```python
except Exception as e:
    raise HTTPException(status_code=403, detail="Invalid token")
    # Should not expose: detail=f"Invalid Firebase token: {str(e)}"
```

**Examples**:

- Firebase token errors expose token format details
- API errors expose internal structure
- Stack traces in error responses

**Fix**:

```python
# Generic error messages in production
except Exception as e:
    logger.error(f"Firebase token verification failed: {e}", exc_info=True)
    if DEBUG:
        raise HTTPException(status_code=403, detail=f"Invalid token: {str(e)}")
    else:
        raise HTTPException(status_code=403, detail="Authentication failed")
```

**Recommendation**:

- Log detailed errors server-side only
- Return generic error messages to clients
- Use DEBUG flag to control error verbosity

---

### HIGH-003: Weak Rate Limiting Implementation

**File**: `main.py`  
**Line**: 200-288  
**Severity**: High

**Issue**: Rate limiting uses in-memory storage, making it ineffective across multiple instances.

```python
class LocationDiversityMonitor:
    def __init__(self, max_locations: int, window_hours: int):
        self.ip_locations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
```

**Vulnerability**:

- Rate limits can be bypassed by distributed attacks
- Limits reset on server restart
- Not shared across multiple server instances

**Fix**:

```python
class LocationDiversityMonitor:
    def __init__(self, max_locations: int, window_hours: int, redis_client: redis.Redis):
        self.redis = redis_client
        self.max_locations = max_locations
        self.window_seconds = window_hours * 3600

    def check_location_diversity(self, ip: str, location: str) -> tuple[bool, str]:
        key = f"rate_limit:locations:{ip}"
        # Use Redis sorted set with TTL
        now = time.time()
        self.redis.zremrangebyscore(key, 0, now - self.window_seconds)
        count = self.redis.zcard(key)

        if count >= self.max_locations:
            return False, f"Maximum {self.max_locations} unique locations per {self.window_seconds//3600} hours"

        self.redis.zadd(key, {location: now})
        self.redis.expire(key, self.window_seconds)
        return True, "OK"
```

**Recommendation**:

- Implement Redis-based rate limiting for distributed systems
- Use sliding window algorithm
- Store rate limit data with appropriate TTL

---

### HIGH-004: Insufficient Input Validation on CSV Parameters

**File**: `routers/records_agg.py`  
**Line**: 537-538, 583-584  
**Severity**: High

**Issue**: CSV parsing for include/exclude parameters lacks proper validation.

```python
include: str | None = Query(None, description="CSV of sections to include"),
exclude: str | None = Query(None, description="CSV of sections to exclude"),

def _parse_csv(s: str | None) -> Set[str]:
    return {p.strip() for p in s.split(",")} if s else set()
```

**Vulnerability**:

- No length validation on CSV string
- No validation that parsed values are in allowed set
- Potential for DoS with very long CSV strings

**Fix**:

```python
def _parse_csv(s: str | None, allowed: Set[str], max_length: int = 100) -> Set[str]:
    if not s:
        return set()

    if len(s) > max_length:
        raise HTTPException(status_code=400, detail=f"CSV parameter too long (max {max_length} chars)")

    parsed = {p.strip() for p in s.split(",") if p.strip()}
    invalid = parsed - allowed

    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid values: {', '.join(invalid)}")

    return parsed
```

**Recommendation**:

- Add maximum length validation
- Validate against whitelist of allowed values
- Limit number of items in CSV

---

### HIGH-005: Missing Authentication on Some Endpoints

**File**: `main.py`  
**Line**: 941-945  
**Severity**: High

**Issue**: Several endpoints are marked as public that may expose sensitive information.

**Vulnerability**:

- `/rate-limit-status` exposes rate limiting configuration
- `/rate-limit-stats` may expose usage patterns
- `/cache-stats/*` endpoints expose internal metrics

**Fix**:

```python
# Make these endpoints require authentication
public_paths = ["/", "/docs", "/openapi.json", "/redoc", "/health", "/v1/records/rolling-bundle/test-cors"]

# Add authentication to stats endpoints
@app.get("/rate-limit-stats")
async def get_rate_limit_stats(user=Depends(verify_firebase_token)):
    # ... handler
```

**Recommendation**:

- Review all "public" endpoints
- Require authentication for any endpoint exposing system metrics
- Consider admin-only access for statistics

---

### HIGH-006: Redis KEYS Command Usage

**File**: `cache_utils.py`  
**Line**: 1170, 1305, 1377  
**Severity**: High

**Issue**: Redis `KEYS` command is used which can block the server on large datasets.

```python
matching_keys = self.redis_client.keys(pattern)
```

**Vulnerability**:

- `KEYS` command is blocking and O(N) complexity
- Can cause Redis to become unresponsive
- Not suitable for production environments

**Fix**:

```python
# Use SCAN instead of KEYS
def invalidate_by_pattern(self, pattern: str, dry_run: bool = False) -> Dict:
    matching_keys = []
    cursor = 0

    while True:
        cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
        matching_keys.extend(keys)
        if cursor == 0:
            break

    # Process matching_keys...
```

**Recommendation**:

- Replace all `KEYS` commands with `SCAN`
- Use cursor-based iteration
- Set appropriate COUNT parameter

---

### HIGH-007: Insecure Direct Object References (IDOR)

**File**: `main.py`, `routers/records_agg.py`  
**Line**: Multiple locations  
**Severity**: High

**Issue**: No access control checks on resources - users can access any location data.

**Vulnerability**: While this API appears designed for public access, there's no validation that:

- Users shouldn't access certain locations (if business logic requires)
- Rate limits apply consistently
- Resource exhaustion attacks are prevented

**Fix**: Add resource access validation if business rules require it:

```python
def validate_location_access(location: str, user: dict) -> bool:
    # Implement business logic for location access
    # For example, premium locations only for premium users
    if location in PREMIUM_LOCATIONS and not user.get("premium"):
        return False
    return True
```

**Recommendation**:

- Review business requirements for access control
- Add validation if specific locations should be restricted
- Implement proper authorization checks

---

## Medium Severity Issues

### MED-001: Missing HTTPS Enforcement

**File**: `main.py`  
**Line**: 1070-1080  
**Severity**: Medium

**Issue**: No explicit HTTPS enforcement or HSTS headers.

**Fix**:

```python
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)

    # Force HTTPS in production
    if not DEBUG and request.url.scheme != "https":
        return JSONResponse(
            status_code=400,
            content={"error": "HTTPS required"}
        )

    # Add security headers
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    return response
```

---

### MED-002: Weak ETag Generation

**File**: `cache_utils.py`, `routers/locations_preapproved.py`  
**Line**: 273, 92  
**Severity**: Medium

**Issue**: MD5 used for ETag generation (MD5 is cryptographically broken).

```python
return f'"{hashlib.md5(json_str.encode()).hexdigest()}"'
```

**Fix**:

```python
# Use SHA256 for ETags
return f'"{hashlib.sha256(json_str.encode()).hexdigest()[:16]}"'
```

---

### MED-003: Missing Request Timeout on External APIs

**File**: `main.py`, `routers/records_agg.py`  
**Line**: Multiple locations  
**Severity**: Medium

**Issue**: Some external API calls lack explicit timeouts, risking resource exhaustion.

**Fix**: Ensure all external calls have timeouts:

```python
async with aiohttp.ClientSession(
    timeout=aiohttp.ClientTimeout(total=60, connect=10)
) as session:
    # ... API calls
```

---

### MED-004: Code Duplication in Cache Key Generation

**File**: `main.py`, `cache_utils.py`, `routers/records_agg.py`  
**Severity**: Medium

**Issue**: Cache key generation logic is duplicated across multiple files.

**Recommendation**: Centralize cache key generation in `cache_utils.py` and reuse everywhere.

---

### MED-005: Missing Input Sanitization in Logging

**File**: `main.py`  
**Line**: Multiple locations  
**Severity**: Medium

**Issue**: User inputs are logged without sanitization, potentially exposing sensitive data.

**Fix**:

```python
def sanitize_for_logging(data: str, max_length: int = 100) -> str:
    """Sanitize data before logging."""
    if len(data) > max_length:
        data = data[:max_length] + "..."
    # Remove potential sensitive patterns
    data = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', data)
    data = re.sub(r'key=[^&]+', 'key=[REDACTED]', data)
    return data
```

---

### MED-006: Insufficient Error Handling in Async Jobs

**File**: `job_worker.py`, `cache_utils.py`  
**Line**: 203-209  
**Severity**: Medium

**Issue**: Job processing errors may not be properly logged or handled.

**Fix**: Improve error handling:

```python
except Exception as e:
    logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
    # Store detailed error for diagnostics
    job_manager.update_job_status(
        job_id,
        JobStatus.ERROR,
        error=str(e),
        error_details=traceback.format_exc()  # Store in separate field
    )
```

---

### MED-007: Missing Content Security Policy Headers

**File**: `main.py`  
**Severity**: Medium

**Issue**: No CSP headers to prevent XSS attacks.

**Fix**: Add CSP headers in security middleware (see MED-001).

---

### MED-008: Inconsistent Error Response Format

**File**: `main.py`, `routers/records_agg.py`  
**Severity**: Medium

**Issue**: Error responses use inconsistent formats across endpoints.

**Recommendation**: Standardize error response format:

```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    code: str
    details: Optional[Dict] = None
```

---

### MED-009: Missing Input Length Limits

**File**: `main.py`, `routers/records_agg.py`  
**Line**: 3091  
**Severity**: Medium

**Issue**: Location parameter has no maximum length validation.

**Fix**:

```python
location: str = Path(..., description="Location name", max_length=200)
```

---

### MED-010: Potential Denial of Service in Cache Invalidation

**File**: `cache_utils.py`  
**Line**: 1162-1211  
**Severity**: Medium

**Issue**: Cache invalidation by pattern can be resource-intensive and cause DoS.

**Fix**: Add rate limiting and size limits:

```python
def invalidate_by_pattern(self, pattern: str, dry_run: bool = False, max_keys: int = 10000) -> Dict:
    # Limit number of keys to prevent DoS
    # ... existing code with max_keys check
    if len(matching_keys) > max_keys:
        raise HTTPException(status_code=400, detail=f"Pattern matches too many keys (>{max_keys})")
```

---

## Low Severity Issues

### LOW-001: Outdated Dependencies

**File**: `requirements.txt`  
**Severity**: Low

**Issue**: Several packages may have known vulnerabilities.

**Fix**: Run security audit:

```bash
pip install safety
safety check
```

**Recommendation**: Regularly update dependencies and monitor for security advisories.

---

### LOW-002: Missing API Versioning Documentation

**File**: README.md  
**Severity**: Low

**Issue**: API versioning strategy is not clearly documented.

**Recommendation**: Document versioning policy and deprecation strategy.

---

### LOW-003: Inefficient Rate Limiting Cleanup

**File**: `main.py`  
**Line**: 222-233  
**Severity**: Low

**Issue**: Rate limiting cleanup runs on every request, can be optimized.

**Fix**: Run cleanup in background task periodically.

---

### LOW-004: Missing Request ID for Tracing

**File**: `main.py`  
**Severity**: Low

**Issue**: No request ID for distributed tracing.

**Fix**: Add request ID middleware:

```python
import uuid

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

---

### LOW-005: Hardcoded Magic Numbers

**File**: Multiple files  
**Severity**: Low

**Issue**: Magic numbers throughout code should be constants.

**Recommendation**: Extract magic numbers to named constants at module level.

---

## Performance Bottlenecks

### PERF-001: Synchronous Redis Operations in Async Context

**File**: `cache_utils.py`, `routers/locations_preapproved.py`  
**Line**: Multiple locations

**Issue**: Using synchronous `redis.Redis` client instead of async client.

**Fix**: Use `aioredis` or `redis.asyncio`:

```python
import redis.asyncio as aioredis

redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
```

---

### PERF-002: Missing Connection Pooling

**File**: `routers/records_agg.py`  
**Line**: 118-132

**Issue**: HTTP client sessions may not be properly pooled.

**Recommendation**: Ensure proper connection pooling and reuse.

---

## Code Quality Issues

### QUAL-001: Code Duplication

**Files**: Multiple  
**Severity**: Medium

**Issues**:

- Cache key generation duplicated
- Error handling patterns repeated
- Validation logic scattered

**Recommendation**: Create shared utility modules for common operations.

---

### QUAL-002: Missing Type Hints

**File**: Multiple files  
**Severity**: Low

**Issue**: Some functions lack complete type hints.

**Recommendation**: Add comprehensive type hints for better maintainability.

---

### QUAL-003: Inconsistent Logging

**File**: Multiple files  
**Severity**: Low

**Issue**: Logging levels and formats inconsistent.

**Recommendation**: Standardize logging format and levels.

---

## Recommendations Summary

### Immediate Actions (Critical & High)

1. Remove or secure test token bypass
2. Move API keys out of URL parameters
3. Add strict input validation for location parameter
4. Implement CSRF protection
5. Move rate limiting to Redis
6. Replace Redis KEYS with SCAN

### Short Term (Medium)

1. Add security headers (HSTS, CSP, etc.)
2. Sanitize error messages
3. Add request timeouts everywhere
4. Implement HTTPS enforcement
5. Centralize cache key generation

### Long Term (Low & Quality)

1. Update dependencies regularly
2. Improve code organization
3. Add comprehensive tests
4. Implement request tracing
5. Document API versioning strategy

---

## Testing Recommendations

1. **Penetration Testing**: Conduct professional penetration testing
2. **Dependency Scanning**: Integrate automated dependency scanning in CI/CD
3. **Security Linting**: Use tools like `bandit` for Python security linting
4. **Load Testing**: Test rate limiting and DoS protections
5. **Input Fuzzing**: Test all input parameters with malicious inputs

---

## Compliance Considerations

- **OWASP Top 10**: Address injection, authentication, and sensitive data exposure
- **CWE**: Address common weaknesses identified
- **PCI DSS**: If handling payment data, ensure compliance
- **GDPR**: Review data retention and logging practices

---

## Conclusion

This codebase has several security vulnerabilities that need immediate attention, particularly around authentication bypass, API key exposure, and input validation. The good news is that most issues have straightforward fixes. Prioritize addressing Critical and High severity issues first, then work through Medium severity items.

**Estimated Effort**:

- Critical/High: 2-3 days
- Medium: 1 week
- Low/Quality: Ongoing

**Priority**: Address all Critical and High severity issues before next production deployment.
