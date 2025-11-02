# Security Audit Report - TempHist API
**Date:** 2025-11-02
**Auditor:** Claude Code Security Analysis
**Version Audited:** 1.0.3

---

## Executive Summary

This security audit identified **25 security and code quality issues** across the TempHist API codebase:
- **3 Critical** severity issues requiring immediate attention
- **8 High** severity issues requiring prompt remediation
- **9 Medium** severity issues that should be addressed
- **5 Low** severity issues for improvement

The most critical findings include outdated dependencies with known CVEs, authentication bypass mechanisms, and insufficient input validation.

---

## Critical Severity Issues

### 1. Critical Dependency Vulnerabilities
**File:** requirements.txt
**Severity:** CRITICAL
**CWE:** CWE-1035 (Using Components with Known Vulnerabilities)

**Description:**
Multiple dependencies have known security vulnerabilities:

1. **FastAPI 0.104.1** → CVE-2024-24762 (ReDoS vulnerability)
   - Impact: Denial of Service through malicious Content-Type headers
   - Fix: Upgrade to ≥0.109.1

2. **Gunicorn 21.2.0** → CVE-2024-1135, CVE-2024-6827
   - Impact: HTTP Request Smuggling, cache poisoning, SSRF, XSS
   - Fix: Upgrade to ≥22.0.0

3. **aiohttp 3.9.3** → CVE-2024-27306, CVE-2024-30251, CVE-2024-52304, CVE-2025-53643
   - Impact: XSS, Denial of Service, Request Smuggling
   - Fix: Upgrade to ≥3.12.14

4. **requests 2.31.0** → CVE-2024-35195, CVE-2024-47081
   - Impact: Certificate verification bypass, .netrc credential leakage
   - Fix: Upgrade to ≥2.32.4

5. **starlette 0.27.0** → CVE-2024-47874, CVE-2025-54121, CVE-2025-62727
   - Impact: DoS through malicious form uploads, CPU exhaustion
   - Fix: Upgrade to ≥0.49.1

**Recommendation:**
```bash
# Update requirements.txt:
fastapi>=0.109.1
gunicorn>=22.0.0
aiohttp>=3.12.14
requests>=2.32.4
# Note: Starlette will be updated automatically with FastAPI
```

---

### 2. Hardcoded Test Token Authentication Bypass
**File:** main.py:1039-1042
**Line:** 1039
**Severity:** CRITICAL
**CWE:** CWE-798 (Use of Hard-coded Credentials)

**Description:**
The application uses a `TEST_TOKEN` environment variable that bypasses Firebase authentication entirely:

```python
if id_token == TEST_TOKEN:
    if DEBUG:
        logger.debug(f"[DEBUG] Middleware: Using test token bypass")
    request.state.user = {"uid": "testuser"}
```

**Impact:**
- Anyone with knowledge of the TEST_TOKEN can bypass authentication
- No audit trail for test token usage in production
- Token likely stored in version control or configuration files

**Recommendation:**
1. Remove TEST_TOKEN bypass from production code
2. Use proper test environments with separate authentication
3. Implement service accounts with proper audit logging for automated systems
4. Add monitoring/alerting for API_ACCESS_TOKEN usage

**Code Fix:**
```python
# Remove TEST_TOKEN entirely and use proper service authentication
# Keep only API_ACCESS_TOKEN for documented, monitored service jobs
if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
    logger.warning(f"🔧 SERVICE ACCESS: {client_ip} | {request.url.path}")
    request.state.user = {"uid": "service", "system": True}
    # Send alert/metric for monitoring
else:
    # Always require Firebase token validation
    decoded_token = auth.verify_id_token(id_token)
    request.state.user = decoded_token
```

---

### 3. Sensitive Information Logging
**File:** main.py:59, main.py:422
**Lines:** 59, 422
**Severity:** CRITICAL
**CWE:** CWE-532 (Insertion of Sensitive Information into Log File)

**Description:**
API keys and sensitive URLs are logged:

```python
# Line 59 - Redis URL may contain credentials
_temp_logger.info(f"🔍 DEBUG: REDIS_URL environment variable = {REDIS_URL}")

# Line 422 - API key in URL construction
base_params = f"unitGroup={VISUAL_CROSSING_UNIT_GROUP}&include={VISUAL_CROSSING_INCLUDE_PARAMS}&key={API_KEY}"
```

**Impact:**
- API keys exposed in logs
- Redis credentials exposed if URL contains password
- Log aggregation systems may capture sensitive data
- Compliance violations (PCI-DSS, GDPR)

**Recommendation:**
```python
# Sanitize REDIS_URL before logging
def sanitize_url(url: str) -> str:
    """Remove credentials from URL for logging."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    if parsed.password:
        netloc = f"{parsed.username}:***@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        parsed = parsed._replace(netloc=netloc)
    return urlunparse(parsed)

_temp_logger.info(f"🔍 DEBUG: REDIS_URL = {sanitize_url(REDIS_URL)}")

# Never log API keys
# Remove line 422 logging or sanitize it
```

---

## High Severity Issues

### 4. Missing CSRF Protection
**File:** main.py (entire application)
**Severity:** HIGH
**CWE:** CWE-352 (Cross-Site Request Forgery)

**Description:**
The API has no CSRF protection for state-changing operations. Endpoints like `/cache-warm`, `/cache/invalidate/*`, `/cache/clear`, and `/analytics` accept POST/DELETE requests without CSRF tokens.

**Impact:**
- Malicious websites can trigger cache invalidation
- Unauthorized cache warming operations
- Analytics data pollution
- Resource exhaustion attacks

**Recommendation:**
```python
from fastapi import Header
from secrets import token_urlsafe

# Add CSRF token validation
async def verify_csrf_token(
    x_csrf_token: str = Header(None),
    request: Request = None
):
    # For API-only apps, validate Origin/Referer
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")

    allowed_origins = ["https://yourdomain.com"]

    if origin not in allowed_origins and not referer.startswith(tuple(allowed_origins)):
        raise HTTPException(403, "Invalid origin")
    return True

# Apply to state-changing endpoints
@app.post("/cache-warm", dependencies=[Depends(verify_csrf_token)])
async def trigger_cache_warming():
    # ...
```

---

### 5. Overly Broad Exception Handling
**File:** Multiple files (141 instances)
**Severity:** HIGH
**CWE:** CWE-396 (Declaration of Catch-All Exception Handler)

**Description:**
The codebase uses bare `except Exception as e` handlers extensively, catching all exceptions including critical system errors:

**Examples:**
- main.py: 51 instances
- cache_utils.py: 35 instances
- routers/records_agg.py: 7 instances

**Impact:**
- Security exceptions silently caught
- Difficult to debug production issues
- May hide authentication/authorization failures
- Resource leaks from unhandled errors

**Recommendation:**
```python
# Bad - catches everything
try:
    data = await fetch_data()
except Exception as e:
    logger.error(f"Error: {e}")
    return {"error": "failed"}

# Good - catch specific exceptions
try:
    data = await fetch_data()
except (aiohttp.ClientError, asyncio.TimeoutError) as e:
    logger.error(f"Network error fetching data: {e}")
    raise HTTPException(503, "Service temporarily unavailable")
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    raise HTTPException(400, "Invalid response from upstream")
# Let critical exceptions propagate
```

---

### 6. Redis Connection Without Authentication Validation
**File:** main.py:750
**Line:** 750
**Severity:** HIGH
**CWE:** CWE-306 (Missing Authentication for Critical Function)

**Description:**
Redis connection established without validating if authentication is required:

```python
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
```

**Impact:**
- No validation that Redis requires password
- Potential connection to wrong Redis instance
- No SSL/TLS enforcement for Redis connections
- Credentials may be transmitted in plaintext

**Recommendation:**
```python
import ssl

# Validate Redis connection and enforce security
def create_redis_client(url: str):
    """Create Redis client with security validation."""
    from urllib.parse import urlparse

    parsed = urlparse(url)

    # Enforce password in production
    if not parsed.password and os.getenv("ENVIRONMENT") == "production":
        raise ValueError("Redis password required in production")

    # Enforce SSL in production
    ssl_context = None
    if parsed.scheme == "rediss":
        ssl_context = ssl.create_default_context()
    elif os.getenv("ENVIRONMENT") == "production":
        logger.warning("⚠️  Redis not using SSL in production!")

    return redis.from_url(
        url,
        decode_responses=True,
        ssl_cert_reqs=ssl.CERT_REQUIRED if ssl_context else None,
        ssl_ca_certs=None if ssl_context else None
    )

redis_client = create_redis_client(REDIS_URL)
```

---

### 7. Insufficient Input Validation on Location Parameter
**File:** main.py:2461-2482, routers/records_agg.py
**Lines:** 2461-2482
**Severity:** HIGH
**CWE:** CWE-20 (Improper Input Validation)

**Description:**
Location parameter validation is minimal and doesn't prevent injection attacks:

```python
def is_location_likely_invalid(location: str) -> bool:
    # Only checks for common JS errors, not security issues
    invalid_patterns = [
        "[object Object]",
        "undefined",
        "null"
    ]
    return any(pattern.lower() in location_lower for pattern in invalid_patterns)
```

**Impact:**
- Potential for log injection attacks
- Cache poisoning with malicious location strings
- May pass through to external APIs causing errors
- No length limits enforced

**Recommendation:**
```python
import re
from pydantic import validator

def validate_location(location: str) -> str:
    """Comprehensive location validation."""
    # Length check
    if not location or len(location) > 200:
        raise HTTPException(400, "Invalid location length")

    # Whitelist allowed characters
    if not re.match(r'^[a-zA-Z0-9\s,.\-\']+$', location):
        raise HTTPException(400, "Location contains invalid characters")

    # Prevent path traversal attempts
    if '..' in location or '/' in location or '\\' in location:
        raise HTTPException(400, "Invalid location format")

    # Check for control characters
    if any(ord(c) < 32 for c in location):
        raise HTTPException(400, "Location contains control characters")

    return location.strip()

# Use Pydantic for validation
class LocationRequest(BaseModel):
    location: str

    @validator('location')
    def validate_location_field(cls, v):
        return validate_location(v)
```

---

### 8. Missing Security Headers
**File:** main.py (application configuration)
**Severity:** HIGH
**CWE:** CWE-1021 (Improper Restriction of Rendered UI Layers)

**Description:**
Application doesn't set critical security headers:
- No Content-Security-Policy
- No X-Frame-Options
- No Strict-Transport-Security (HSTS)
- No X-Content-Type-Options

**Impact:**
- Vulnerable to clickjacking attacks
- No CSP protection against XSS
- Browsers may not enforce HTTPS
- MIME-type sniffing vulnerabilities

**Recommendation:**
```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Prevent MIME sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # XSS protection (legacy browsers)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )

    # HSTS (only if using HTTPS)
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Permissions policy
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=()"
    )

    return response
```

---

### 9. Potential CORS Misconfiguration
**File:** main.py:71-72, 1088-1110
**Lines:** 71-72
**Severity:** HIGH
**CWE:** CWE-942 (Overly Permissive CORS Policy)

**Description:**
CORS configuration loaded from environment variables without validation:

```python
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", "").strip()
```

**Impact:**
- Misconfigured CORS_ORIGIN_REGEX could allow all origins
- No validation of allowed origins
- Potential for credential-bearing cross-origin requests
- May expose sensitive data to unauthorized domains

**Recommendation:**
```python
def validate_cors_config():
    """Validate CORS configuration."""
    origins = os.getenv("CORS_ORIGINS", "").strip()
    regex = os.getenv("CORS_ORIGIN_REGEX", "").strip()

    # Warn about permissive configurations
    if not origins and not regex:
        logger.warning("⚠️  No CORS origins configured - API may be inaccessible")

    if regex:
        # Test regex is valid and not too permissive
        import re
        try:
            pattern = re.compile(regex)
            # Warn if regex looks too permissive
            if regex in [".*", ".+", ".*\\.*"] or ".*" in regex:
                logger.warning(f"⚠️  CORS regex very permissive: {regex}")
        except re.error as e:
            raise ValueError(f"Invalid CORS_ORIGIN_REGEX: {e}")

    if origins == "*":
        logger.error("❌ CORS_ORIGINS set to '*' - this is insecure!")
        raise ValueError("Wildcard CORS not allowed")

    return origins, regex

CORS_ORIGINS, CORS_ORIGIN_REGEX = validate_cors_config()
```

---

### 10. Rate Limiting Bypass via Service Tokens
**File:** main.py:952-956, 1039-1048
**Lines:** 952-956, 1039-1048
**Severity:** HIGH
**CWE:** CWE-307 (Improper Restriction of Excessive Authentication Attempts)

**Description:**
Both `TEST_TOKEN` and `API_ACCESS_TOKEN` bypass rate limiting entirely:

```python
if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
    is_service_job = True
    # Rate limiting bypassed
```

**Impact:**
- Service tokens can be used for DoS attacks
- No rate limiting on automated systems
- Single compromised token = unlimited access
- No distinction between different service callers

**Recommendation:**
```python
# Implement tiered rate limiting for service tokens
SERVICE_RATE_LIMITS = {
    "default": {"requests_per_hour": 10000, "locations_per_hour": 1000},
    "cache_warmer": {"requests_per_hour": 50000, "locations_per_hour": 5000}
}

def get_service_tier(token: str) -> str:
    """Get service tier from token (could use JWT claims)."""
    # Implement token-to-tier mapping
    # Could use JWT with 'tier' claim
    return "default"

# Apply service-specific rate limits
if API_ACCESS_TOKEN and token == API_ACCESS_TOKEN:
    tier = get_service_tier(token)
    limits = SERVICE_RATE_LIMITS.get(tier, SERVICE_RATE_LIMITS["default"])
    # Apply tier-specific rate limiting
    if not check_service_rate_limit(client_ip, limits):
        raise HTTPException(429, "Service rate limit exceeded")
```

---

### 11. Client IP Logging Privacy Concern
**File:** main.py:479, 853, 861, etc.
**Severity:** HIGH
**CWE:** CWE-359 (Exposure of Private Personal Information)

**Description:**
Client IP addresses are extensively logged throughout the application:

```python
"client_ip": client_ip,  # Line 479 - stored in analytics
logger.debug(f"🌐 REQUEST: {request.method} {request.url.path} | IP: {client_ip}")  # Line 853
```

**Impact:**
- GDPR/CCPA compliance issues (IP is PII)
- No user consent for IP logging
- Long retention in analytics (7 days)
- Logs may be aggregated/exported

**Recommendation:**
```python
import hashlib

def anonymize_ip(ip: str) -> str:
    """Anonymize IP address for logging/storage."""
    # Hash IP with daily salt for privacy
    salt = datetime.now().strftime("%Y-%m-%d")
    return hashlib.sha256(f"{ip}{salt}".encode()).hexdigest()[:16]

# Use anonymized IP for non-security logging
anonymized_ip = anonymize_ip(client_ip)
logger.debug(f"🌐 REQUEST: {request.method} {request.url.path} | IP: {anonymized_ip}")

# Store anonymized IP in analytics
analytics_record = {
    "client_ip_hash": anonymize_ip(client_ip),  # Not raw IP
    # ... other fields
}

# Keep raw IP only for rate limiting (in-memory, short TTL)
# Add privacy policy disclosure for IP collection
```

---

## Medium Severity Issues

### 12. No Request Body Size Limit for All Endpoints
**File:** main.py:869-909
**Lines:** 891
**Severity:** MEDIUM
**CWE:** CWE-770 (Allocation of Resources Without Limits)

**Description:**
Request size validation only applies to POST/PUT/PATCH, and only checks Content-Length header:

```python
max_size = 1024 * 1024  # 1MB default limit
if content_length != "unknown":
    # Only validates if Content-Length header present
```

**Impact:**
- Attackers can omit Content-Length header
- GET requests with bodies not validated
- Chunked encoding bypasses size check
- Memory exhaustion possible

**Recommendation:**
```python
from starlette.requests import Request
from starlette.datastructures import UploadFile

# Configure Starlette/FastAPI max body size
app = FastAPI(
    # ...
    # Limit request body size globally
)

# Add body size validator middleware
@app.middleware("http")
async def limit_request_body_size(request: Request, call_next):
    """Enforce request body size limit."""
    max_size = 1024 * 1024  # 1MB

    if request.method in ["POST", "PUT", "PATCH"]:
        # Read body with size limit
        body_size = 0
        async for chunk in request.stream():
            body_size += len(chunk)
            if body_size > max_size:
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request body too large"}
                )

    return await call_next(request)
```

---

### 13. Firebase Service Account from Environment Variable
**File:** main.py:783-797
**Lines:** 783-797
**Severity:** MEDIUM
**CWE:** CWE-522 (Insufficiently Protected Credentials)

**Description:**
Firebase service account JSON loaded from environment variable:

```python
firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT") or os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
firebase_creds = json.loads(firebase_creds_json)
```

**Impact:**
- Entire private key in environment variable
- May be exposed in process listings
- Logged in deployment systems
- Difficult to rotate credentials

**Recommendation:**
```python
# Use secret management service
import boto3  # AWS Secrets Manager
# Or Google Secret Manager, HashiCorp Vault, etc.

def load_firebase_credentials():
    """Load Firebase credentials from secure secret store."""
    env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        # Use secret manager in production
        secret_name = os.getenv("FIREBASE_SECRET_NAME")
        if not secret_name:
            raise ValueError("FIREBASE_SECRET_NAME required in production")

        # Example with AWS Secrets Manager
        client = boto3.client('secretsmanager')
        response = client.get_secret_value(SecretId=secret_name)
        creds = json.loads(response['SecretString'])
    else:
        # File-based for development
        with open("firebase-service-account.json") as f:
            creds = json.load(f)

    return credentials.Certificate(creds)

cred = load_firebase_credentials()
firebase_admin.initialize_app(cred)
```

---

### 14. Synchronous Operations in Async Context
**File:** main.py:2384-2404
**Lines:** 2384
**Severity:** MEDIUM
**CWE:** CWE-834 (Excessive Iteration)

**Description:**
Using synchronous `requests.get()` in async context blocks the event loop:

```python
response = requests.get(url)  # Blocks event loop!
```

**Impact:**
- Blocks event loop during HTTP requests
- Reduces application throughput
- Potential cascading delays
- Poor user experience

**Recommendation:**
```python
# Use async HTTP client
async def fetch_data(url: str):
    """Async HTTP request."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Or use run_in_threadpool for legacy sync code
from starlette.concurrency import run_in_threadpool

async def legacy_fetch(url: str):
    """Wrap sync call in thread pool."""
    return await run_in_threadpool(requests.get, url)
```

---

### 15. No Rate Limiting on Analytics Endpoint
**File:** main.py:3547-3674
**Lines:** 3547
**Severity:** MEDIUM
**CWE:** CWE-770 (Allocation of Resources Without Limits)

**Description:**
Analytics endpoint `/analytics` is in public_paths list and has no rate limiting:

```python
# Line 941 - analytics in public paths
public_paths = [..., "/analytics", ...]
```

**Impact:**
- Analytics data pollution
- Storage exhaustion (Redis)
- Processing overhead
- No protection against spam

**Recommendation:**
```python
# Remove analytics from public_paths
# Add dedicated rate limit for analytics
ANALYTICS_RATE_LIMIT = 100  # per hour

@app.middleware("http")
async def analytics_rate_limit(request: Request, call_next):
    """Rate limit analytics submissions."""
    if request.url.path.startswith("/analytics") and request.method == "POST":
        client_ip = get_client_ip(request)

        # Check analytics-specific rate limit
        key = f"analytics_limit:{client_ip}"
        count = redis_client.get(key) or 0

        if int(count) >= ANALYTICS_RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"error": "Analytics rate limit exceeded"}
            )

        redis_client.incr(key)
        redis_client.expire(key, 3600)

    return await call_next(request)
```

---

### 16. Weak ETag Generation
**File:** main.py:454, routers/locations_preapproved.py:91
**Lines:** 454, 91
**Severity:** MEDIUM
**CWE:** CWE-326 (Inadequate Encryption Strength)

**Description:**
ETag uses SHA256 but only first 16 characters:

```python
etag = hashlib.sha256(key_parts.encode("utf-8")).hexdigest()[:16]
```

**Impact:**
- Increased collision probability
- Cache confusion possible
- Weak cache validation

**Recommendation:**
```python
# Use full hash or at least 32 characters
etag = hashlib.sha256(key_parts.encode("utf-8")).hexdigest()  # Full 64 chars
# Or minimum 32 chars for 128-bit security
etag = hashlib.sha256(key_parts.encode("utf-8")).hexdigest()[:32]

response.headers["ETag"] = f'W/"{etag}"'
```

---

### 17. Missing Timeout on External HTTP Requests
**File:** routers/records_agg.py:123, 342
**Lines:** 123, 342
**Severity:** MEDIUM
**CWE:** CWE-400 (Uncontrolled Resource Consumption)

**Description:**
HTTP client has timeout but some requests may hang:

```python
_client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
```

**Impact:**
- Requests may hang indefinitely on connection issues
- Resource exhaustion
- Cascading failures

**Recommendation:**
```python
# Use comprehensive timeout configuration
timeout = aiohttp.ClientTimeout(
    total=30,      # Total request timeout
    connect=5,     # Connection timeout
    sock_read=10,  # Socket read timeout
    sock_connect=5 # Socket connection timeout
)

_client = aiohttp.ClientSession(timeout=timeout)

# Also add retry logic with exponential backoff
from aiohttp_retry import RetryClient, ExponentialRetry

retry_options = ExponentialRetry(
    attempts=3,
    start_timeout=0.5,
    max_timeout=3.0,
    factor=2.0
)

_client = RetryClient(
    client_session=aiohttp.ClientSession(timeout=timeout),
    retry_options=retry_options
)
```

---

### 18. Cache Key Collision Risk
**File:** cache_utils.py, main.py:3118
**Lines:** 3118
**Severity:** MEDIUM
**CWE:** CWE-404 (Improper Resource Shutdown)

**Description:**
Cache key generation may have collisions due to insufficient separation:

```python
cache_key = f"records:{period}:{normalized_location}:{identifier}:celsius:v1:values,average,trend,summary"
```

**Impact:**
- Different requests may get wrong cached data
- Data leakage between similar locations
- Inconsistent responses

**Recommendation:**
```python
import hashlib

def generate_cache_key(components: dict) -> str:
    """Generate collision-resistant cache key."""
    # Sort and hash components
    sorted_items = sorted(components.items())
    key_string = "|".join(f"{k}={v}" for k, v in sorted_items)

    # Use hash to ensure consistent length and no collision
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()

    # Use prefix + hash for debugging
    prefix = components.get("type", "cache")
    return f"{prefix}:{key_hash}"

# Usage
cache_key = generate_cache_key({
    "type": "records",
    "period": period,
    "location": normalized_location,
    "identifier": identifier,
    "unit": "celsius",
    "version": "v1",
    "sections": "values,average,trend,summary"
})
```

---

### 19. No Validation of Redis Key TTL
**File:** cache_utils.py, routers/locations_preapproved.py:28
**Lines:** 28
**Severity:** MEDIUM
**CWE:** CWE-1333 (Inefficient Regular Expression Complexity)

**Description:**
Cache TTL values are hardcoded without validation:

```python
CACHE_TTL = 86400  # 24 hours - no validation
```

**Impact:**
- TTL misconfiguration could cause cache exhaustion
- Very large TTL = memory issues
- No bounds checking

**Recommendation:**
```python
# Validate TTL values
MIN_TTL = 60  # 1 minute
MAX_TTL = 604800  # 7 days

def validate_ttl(ttl: int, name: str) -> int:
    """Validate cache TTL value."""
    if not isinstance(ttl, int):
        raise ValueError(f"{name} must be integer")
    if ttl < MIN_TTL:
        logger.warning(f"{name} TTL {ttl}s too low, using {MIN_TTL}s")
        return MIN_TTL
    if ttl > MAX_TTL:
        logger.warning(f"{name} TTL {ttl}s too high, using {MAX_TTL}s")
        return MAX_TTL
    return ttl

CACHE_TTL = validate_ttl(
    int(os.getenv("CACHE_TTL", "86400")),
    "CACHE_TTL"
)
```

---

### 20. Debug Mode Enabled via Environment Variable
**File:** main.py:62
**Line:** 62
**Severity:** MEDIUM
**CWE:** CWE-489 (Active Debug Code)

**Description:**
Debug mode controlled only by environment variable:

```python
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
```

**Impact:**
- Verbose logging in production
- Sensitive data in logs
- Performance degradation
- Security information disclosure

**Recommendation:**
```python
# Enforce debug mode restrictions
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Never allow DEBUG in production
if ENVIRONMENT == "production" and DEBUG:
    logger.error("❌ DEBUG mode not allowed in production")
    raise ValueError("DEBUG=true forbidden in production environment")

# Add application-level debug endpoint protection
@app.get("/debug/jobs")
async def debug_jobs_endpoint():
    if not DEBUG or ENVIRONMENT == "production":
        raise HTTPException(404, "Not found")
    # ... debug logic
```

---

## Low Severity Issues

### 21. Code Duplication in Error Handling
**File:** Multiple files
**Severity:** LOW
**Category:** Code Quality

**Description:**
Extensive code duplication in error handling patterns across the codebase. 141 instances of identical exception handling.

**Impact:**
- Harder to maintain
- Inconsistent error messages
- Difficult to add logging/monitoring

**Recommendation:**
```python
# Create reusable error handlers
def handle_api_error(e: Exception, context: str) -> HTTPException:
    """Standardized API error handler."""
    logger.error(f"API error in {context}: {e}", exc_info=True)

    if isinstance(e, aiohttp.ClientError):
        return HTTPException(503, "External service unavailable")
    elif isinstance(e, asyncio.TimeoutError):
        return HTTPException(504, "Request timeout")
    elif isinstance(e, ValueError):
        return HTTPException(400, f"Invalid input: {str(e)}")
    else:
        return HTTPException(500, "Internal server error")

# Usage
try:
    data = await fetch_external_data()
except Exception as e:
    raise handle_api_error(e, "fetch_external_data")
```

---

### 22. Missing API Versioning Strategy
**File:** main.py (v1 endpoints mixed with legacy)
**Severity:** LOW
**Category:** Code Quality

**Description:**
Mix of versioned (`/v1/records`) and unversioned (`/weather`, `/forecast`) endpoints:

**Impact:**
- Difficult to deprecate old endpoints
- Breaking changes affect all clients
- No migration path

**Recommendation:**
```python
# Deprecate legacy endpoints with warnings
@app.get("/weather/{location}/{date}")
async def get_weather_legacy(location: str, date: str, response: Response):
    """DEPRECATED: Use /v1/records/daily instead."""
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Mon, 01 Jun 2026 00:00:00 GMT"
    response.headers["Link"] = '</v1/records/daily>; rel="successor-version"'

    logger.warning(f"Legacy endpoint used: /weather/{location}/{date}")
    # Redirect to new endpoint or return with deprecation notice

# Add API version header to all responses
@app.middleware("http")
async def add_api_version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["API-Version"] = "1.0.3"
    return response
```

---

### 23. Large Main File (4067 lines)
**File:** main.py
**Severity:** LOW
**Category:** Code Quality

**Description:**
Main application file is 4067 lines - very difficult to maintain.

**Impact:**
- Hard to navigate
- Difficult code reviews
- Higher bug potential
- Poor separation of concerns

**Recommendation:**
```python
# Split into logical modules:
# - main.py: Application setup, middleware (< 500 lines)
# - routers/weather.py: Weather endpoints
# - routers/cache.py: Cache management endpoints
# - routers/analytics.py: Analytics endpoints
# - services/auth.py: Authentication logic
# - services/rate_limit.py: Rate limiting
# - services/validation.py: Input validation
# - models/: Pydantic models
```

---

### 24. Inconsistent Error Response Format
**File:** Multiple files
**Severity:** LOW
**Category:** Code Quality

**Description:**
Error responses have inconsistent formats:

```python
{"detail": "error message"}  # HTTPException
{"error": "error message"}   # Custom responses
{"message": "error message"} # Some endpoints
```

**Impact:**
- Difficult for clients to parse errors
- Inconsistent API experience
- Harder to document

**Recommendation:**
```python
# Standardize error response format
class ErrorResponse(BaseModel):
    error: str
    message: str
    code: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.status_code,
            message=exc.detail,
            code=ERROR_CODES.get(exc.status_code, "UNKNOWN_ERROR"),
            timestamp=datetime.now()
        ).model_dump()
    )
```

---

### 25. Missing Health Check Dependencies
**File:** main.py:1787-1876
**Lines:** 1787-1876
**Severity:** LOW
**Category:** Code Quality

**Description:**
Detailed health check doesn't verify all critical dependencies:

```python
# Missing checks:
# - Firebase auth service connectivity
# - External API (Visual Crossing) availability
# - Worker service health
```

**Impact:**
- False healthy status
- Cascading failures not detected
- Poor observability

**Recommendation:**
```python
@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }

    # Check Redis
    try:
        redis_client.ping()
        health["checks"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # Check Firebase
    try:
        # Attempt token verification with test token
        auth.verify_id_token("test", check_revoked=False)
    except Exception as e:
        if "invalid" in str(e).lower():
            health["checks"]["firebase"] = {"status": "healthy"}
        else:
            health["checks"]["firebase"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"

    # Check external API
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/",
                timeout=5.0
            )
            health["checks"]["visual_crossing_api"] = {
                "status": "healthy" if resp.status_code < 500 else "degraded"
            }
    except Exception as e:
        health["checks"]["visual_crossing_api"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # Check worker service
    try:
        heartbeat = redis_client.get("worker:heartbeat")
        if heartbeat:
            health["checks"]["worker"] = {"status": "healthy"}
        else:
            health["checks"]["worker"] = {"status": "unhealthy", "error": "No heartbeat"}
            health["status"] = "degraded"
    except Exception as e:
        health["checks"]["worker"] = {"status": "unknown", "error": str(e)}

    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)
```

---

## Summary Table

| Severity | Count | Description |
|----------|-------|-------------|
| Critical | 3 | Outdated dependencies, authentication bypass, credential logging |
| High | 8 | CSRF, exception handling, validation, security headers, CORS |
| Medium | 9 | Rate limits, privacy, timeouts, caching, debug mode |
| Low | 5 | Code quality, duplication, health checks |
| **Total** | **25** | **Issues identified** |

---

## Priority Recommendations

### Immediate Actions (This Week)
1. **Update all dependencies** to patched versions
2. **Remove TEST_TOKEN** authentication bypass
3. **Remove or sanitize** sensitive data logging
4. **Add CSRF protection** to state-changing endpoints
5. **Implement security headers** middleware

### Short-term Actions (This Month)
1. Replace broad exception handlers with specific ones
2. Add comprehensive input validation for location parameters
3. Implement proper secret management for Firebase credentials
4. Add rate limiting to analytics endpoint
5. Fix synchronous operations in async context

### Long-term Improvements (This Quarter)
1. Refactor main.py into smaller modules
2. Implement standardized error response format
3. Add comprehensive monitoring and alerting
4. Conduct penetration testing
5. Create security documentation and runbooks

---

## Testing Recommendations

1. **Dependency Scanning**: Integrate `pip-audit` or `safety` into CI/CD
2. **SAST**: Use Bandit or Semgrep for static analysis
3. **Penetration Testing**: Test authentication bypass, rate limiting, input validation
4. **Load Testing**: Verify DoS protections work under load
5. **Security Headers**: Test with securityheaders.com
6. **CORS**: Test cross-origin requests from various domains

---

## Compliance Considerations

- **GDPR**: IP address logging requires privacy policy and consent
- **PCI-DSS**: If processing payments, additional controls needed
- **SOC 2**: Logging, access controls, and encryption gaps identified
- **OWASP Top 10**: Addressed broken authentication, security misconfiguration, insufficient logging

---

**End of Report**
