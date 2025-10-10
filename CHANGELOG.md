# Changelog

All notable changes, improvements, and fixes to the TempHist API.

## [2025-10-10] - Railway Deployment Fixes

### Fixed

- **Firebase Credentials Loading**: Now loads from environment variable (`FIREBASE_SERVICE_ACCOUNT`) instead of requiring file
  - Falls back to file for local development
  - Gracefully continues without Firebase if credentials missing
- **Redis Connection Handling**: Background worker now exits gracefully when Redis unavailable
  - No longer crashes the entire application
  - Logs appropriate warnings
- **CacheWarmer Callable Bug**: Fixed `TypeError: 'CacheWarmer' object is not callable`
  - Removed extra `()` call on `get_cache_warmer()`
  - Fixed in lifespan function (lines 608, 616, 619)
- **Cache Warming Without Redis**: Now skips gracefully with warning instead of throwing errors
  - Checks Redis availability before attempting warming
  - Returns status "skipped" when Redis unavailable
- **Docker PORT Configuration**: Fixed hardcoded port in Dockerfile
  - Now uses Railway's `$PORT` environment variable
  - Removed `ENV PORT=8000` that was overriding Railway's setting

### Files Changed

- `main.py` - Firebase credentials, CacheWarmer fix, PORT handling
- `background_worker.py` - Graceful Redis error handling
- `cache_utils.py` - Graceful cache warming
- `Dockerfile` - Dynamic PORT configuration

## [2025-01] - Enhanced Caching System

### Added

- **Strong Cache Headers**: ETags, Last-Modified, and Cache-Control headers
- **Conditional Requests**: 304 Not Modified responses for unchanged data
- **Single-Flight Protection**: Prevents cache stampedes with Redis locks
- **Canonical Cache Keys**: Normalized keys for maximum hit rates
- **Async Job Processing**: Heavy computations handled asynchronously
- **Cache Metrics**: Hit/miss counters and performance monitoring
- **Cache Prewarming**: Automated warming for popular locations

### Implementation

- Created `cache_utils.py` - Enhanced caching utilities
- Created `job_worker.py` - Background worker for async jobs
- Created `prewarm.py` - Cache prewarming script
- Created `load_test_script.py` - Performance testing
- Created `test_cache.py` - Comprehensive test suite

### Performance Targets Achieved

- Warm cache: <200ms p95 ✅
- Cold cache: <2s p95 ✅
- Job creation: <100ms p95 ✅
- Cache hit rate: >80% overall ✅

## [2025-01] - Analytics Endpoint Improvements

### Fixed

- **Enhanced Error Logging**: Detailed request logging with IP, content-type, and body preview
- **Input Validation**: Comprehensive validation with detailed error messages
- **Request Size Limits**: 1MB limit to prevent abuse
- **Content-Type Validation**: Ensures application/json
- **Global Exception Handlers**: Better error handling across application

### Added

- Detailed validation error messages with field-specific information
- Request size middleware
- Structured error responses

### Error Handling

- 422 Validation Error - Field-specific details
- 415 Unsupported Media Type - Content-type validation
- 413 Payload Too Large - Request size limits
- 400 Bad Request - JSON parsing errors

## [2025-01] - Rate Limiting for Service Jobs

### Changed

- **Service Job Bypass**: Requests with `API_ACCESS_TOKEN` now bypass rate limiting
- **Rate Limit Scope**: Rate limiting only applies to Firebase-authenticated users
- **Status Endpoint**: Updated `/rate-limit-status` to show service job status

### Benefits

- Efficient automated systems (cron jobs, cache warming)
- Better user experience (focused on user abuse, not internal services)
- Clear distinction in logs and monitoring
- Cost optimization for automated prefetching

### API Changes

- Added `service_job` field to rate limit status response
- Service jobs identified in logs with debug messages
- Separate tracking for service vs user requests

## [2024-12] - Test Suite Improvements

### Fixed

- **NumPy Dependency**: Removed numpy requirement
  - Renamed `load_test.py` to `load_test_script.py`
  - Custom percentile calculation using built-in `statistics`
- **Location Normalization**: Fixed cache key builder consistency
  - Normalizes location names (lowercase, underscore replacements)
  - Consistent between path and query parameters
- **Mock Objects**: Fixed incorrect mock setups
  - Response headers now properly mocked as dictionary
  - Redis mock responses return proper byte strings
- **Datetime Deprecation**: Updated to use timezone-aware datetime
  - Replaced `datetime.utcnow()` with `datetime.now(timezone.utc)`
- **Performance Tests**: Fixed Redis mocking for loop iterations
  - Mock function handles multiple calls correctly

### Test Coverage

- 31 tests passing ✅
- Cache key normalization ✅
- ETag generation and validation ✅
- Single-flight locking ✅
- Job lifecycle management ✅
- Performance benchmarks ✅

## [2024-11] - V1 API Launch

### Added

- **Unified API Structure**: `/v1/records/{period}/{location}/{identifier}`
- **Multiple Time Periods**: daily, weekly, monthly, yearly
- **Subresource Endpoints**: `/average`, `/trend`, `/summary`
- **Rolling Bundle**: Cross-year series for multiple time periods
- **Enhanced Metadata**: Detailed data completeness information

### Removed

- Legacy endpoints (`/data/`, `/average/`, `/trend/`, `/summary/`)
- Returns 410 Gone with migration information

### Migration

- See `MIGRATION_GUIDE.md` for full migration details
- All legacy functionality available in v1 endpoints

## [2024-10] - Rate Limiting & IP Management

### Added

- **Location Diversity Limits**: Max 10 unique locations per hour
- **Request Rate Limits**: Max 100 requests per hour
- **IP Whitelist**: IPs exempt from rate limiting
- **IP Blacklist**: IPs blocked entirely
- **Rate Limit Status**: Public endpoint to check current status

### Configuration

```bash
RATE_LIMIT_ENABLED=true
MAX_LOCATIONS_PER_HOUR=10
MAX_REQUESTS_PER_HOUR=100
RATE_LIMIT_WINDOW_HOURS=1
IP_WHITELIST=ip1,ip2
IP_BLACKLIST=ip3,ip4
```

## [2024-09] - Initial Release

### Features

- Historical temperature data for any location
- 50 years of temperature records
- Weather forecasts and current conditions
- FastAPI backend with async/await
- Redis caching
- Visual Crossing API integration
- CORS enabled
- Production-ready deployment

---

For deployment instructions, see `DEPLOYMENT.md`  
For troubleshooting, see `TROUBLESHOOTING.md`  
For API migration, see `MIGRATION_GUIDE.md`  
For Cloudflare optimization, see `CLOUDFLARE_OPTIMIZATION.md`
