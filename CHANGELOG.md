# Changelog

All notable changes, improvements, and fixes to the TempHist API.

## [2026-10-09] - Pin ERA5 Rather Than ERA5-Land (unreleased)

Follow-up to the P1-173 model pin (#123).

### Changed

- **`OPEN_METEO_ARCHIVE_MODEL` now defaults to `era5`** (was `era5_land`). After backfilling 24 locations on ERA5-Land, comparing the same locations on ERA5 showed the two agree where there is no reason to expect a difference (Manchester 0.32 v 0.34 °C/decade, Hong Kong 0.25 v 0.28) but not in Singapore, where ERA5-Land gave 0.12 (r² 0.45) against 0.27 (r² 0.82) for ERA5 and the Meteorological Service Singapore's 0.25 since 1948. ERA5 was not clearly worse anywhere tested (Cape Town 0.10 v 0.13, San Francisco 0.11 v 0.16, Chicago 0.20 v 0.25, Toronto 0.31 v 0.30). ERA5's cells are coarser (0.25°, about 25 km, against 0.1°, about 11 km for ERA5-Land). The README now documents where the data comes from and how values and trends are built.
- `scripts/backfill_open_meteo.py` now names its target source from the configured model in report mode too, so the report counts locations still on another model as pending.

### Note

- Locations already backfilled on ERA5-Land are tagged `open-meteo:era5_land`, so the script treats them as pending and re-fetches them on ERA5 without `--force`. Rows written by the normal request path are tagged `timeline`, so they also show as pending.

---

## [2026-10-07] - Pin Open-Meteo Archive Model (unreleased)

Part of P1-173: unexpectedly low 50-year trends for Manchester, Hong Kong and Dublin (#123).

### Fixed

- **Historical requests now pin `models=era5_land`** (`OPEN_METEO_ARCHIVE_MODEL`, default `era5_land`; empty restores the old behaviour). Without it, Open-Meteo's default `best_match` silently changes model on 2017-01-01, putting a location-dependent step into the long-term series: about -0.9 °C in Hong Kong (turning a 0.28 °C/decade trend into 0.09) and about +0.4 °C in Manchester. Before 2017 `best_match` returned ERA5-Land, so values already stored for those years are unaffected.
- The forecast endpoint, which supplies the most recent ~7 days, is not pinned because it returns no ERA5-Land data. Those days can still carry `best_match`'s offset.

### Added

- **`scripts/backfill_open_meteo.py`**: re-fetches each location's stored history from the pinned model and upserts it, tagging rows `open-meteo:<model>`. Read-only by default (a database report with an estimated Open-Meteo call cost); `--compare` previews each location's trend before and after; `--execute --backup-dir DIR` backs up the rows it replaces to CSV, then writes. It stops at the first incomplete fetch, skips locations already on the pinned source so a run can resume, and leaves the newest 8 days to the forecast endpoint.
- Why it is needed: rows loaded before the 2026-06-03 move to Open-Meteo came from Visual Crossing (about 982k of 2.43M rows across 56 of 132 locations), and rows loaded since then use the unpinned `best_match`. Both put steps into the series that skew 50-year trends. A 51-year request costs roughly 2,600 weighted Open-Meteo calls against free-tier limits of 5,000/hour and 10,000/day, so a full run needs a paid plan.

### Note

- This only affects days fetched from now on. Days already stored in `daily_temperatures` are never re-fetched, so 2017+ rows for existing locations keep the old values until they are purged and backfilled.

---

## [2026-10-06] - Climate Descriptions for Preapproved Locations (unreleased)

Each curated location now carries a short climate description, so the website's location pages can say something specific about each place instead of the same generic text (#122).

### Added

- **`description` on every item of `GET /v1/locations/preapproved`**: one or two paragraphs of plain text (separated by a blank line), at most 100 words, focused on temperature: climate type, typical temperatures in °C and seasonal variation. Precipitation and wind appear only where they affect temperature, with no rainfall figures. Figures are 1991–2020 normals, and descriptions also give the 50-year warming trend from this API's yearly records endpoint (as of October 2026) where it is clear (r² ≥ 0.4), since the website charts about 50 years of temperatures. That covers London, Birmingham, Edinburgh, Glasgow, Cardiff, Belfast, Sydney, Singapore and Cape Town. Manchester's trend is not significant and looks oddly low, so it is left out. The text lives in `data/preapproved_locations.json`. Locations that are close together, such as the UK cities, each name a genuine local difference.
- `LocationItem` requires `description` and rejects an empty one or one over 100 words, so the app fails to load a location data file that breaks the rule. A test checks the shipped file, including that descriptions are distinct.

### Changed

- The preapproved response cache prefix is now `preapproved:v3` (was `v2`), so responses cached before this change, which lack `description`, are not served. The old keys expire on their own TTL.
- `description` is a required field in the `LocationItem` schema in the OpenAPI spec.

---

## [2026-10-04] - Remove Stale Render References (unreleased)

The API is hosted on Railway. These leftovers from an earlier Render deployment were misleading.

### Removed

- **`render.yaml`**: unused since the move to Railway (last changed 2025-10-03). The deployment configuration is `railway.json` and `DEPLOYMENT.md`.
- A code comment in `routers/health.py` and two `README.md` lines that pointed at `render.yaml` or described a Render deployment. The file tree in the README now lists `railway.json`.

---

## [2026-10-04] - Public-Facing OpenAPI Descriptions and Server URL (unreleased)

Follow-up to the spec cleanup. Spec-only: no endpoint's behaviour changes.

### Changed

- **Operation descriptions rewritten for a public audience.** They were the raw docstrings, which exposed implementation detail: `/health` mentioned "Render load balancers", `/weather` the `validate_location_for_ssrf()` function, `/v1/locations/search` the `MAPBOX_TOKEN` variable and "dev / CI", and `/v1/locations/popular` carried an ad-hoc response-shape blurb that the response schema now covers. The docstrings now describe what each endpoint returns and how to use it (including `ETag` / `If-None-Match` on records and the polling flow for async jobs). The internal notes moved to code comments next to the code they describe.
- `GET /v1/og/{share_id}.png` is summarised as "Get share preview image" instead of the generated "Og Image".

### Added

- **`servers` in the spec**, taken from the existing per-environment `BASE_URL` setting (the same one that builds absolute image URLs). Production declares `https://api.temphist.com`, and the dev deployment declares its own host, so docs hosted on another origin call the right API and "Try it out" on the dev docs never reaches production. With `BASE_URL` unset or pointing at localhost (the default) no server is declared and tools resolve paths against the host that served the spec. **Requires `BASE_URL` to be set on each hosted service.**
- Tests that fail if internal terms reappear in a description and that cover how `servers` is derived.
## [2026-10-03] - Retry-After on 429 and 503 Responses (unreleased)

### Fixed

- **Headers set on an `HTTPException` were silently dropped.** The shared exception handler in `exceptions.py` built its response without `exc.headers`, so a `Retry-After` (or any other header) never reached the client. `POST /analytics` already tried to send `Retry-After: 3600` on its 429 and lost it; it now arrives. A side effect: `405 Method Not Allowed` responses now include the `Allow` header, as HTTP requires.
- **`429 Too Many Requests` from the Locations endpoints now sends `Retry-After: 60`** (the length of the limiter's window, the longest a client can have to wait). Covers `/v1/locations/preapproved`, `/search`, `/popular`, `/popular/display-strings` and `POST /v1/locations/selections`.
- **`503 Service Unavailable` now sends `Retry-After`**: 5 seconds from `/v1/locations/search` while the locations data is still loading, and 30 seconds from `GET /v1/shares` and `POST /v1/shares` when the share store is unreachable. These two values are suggestions rather than measured recovery times.

### Changed

- The OpenAPI spec documents the `Retry-After` header on those 429 and 503 responses, so every documented 429 and 503 now declares it. The rate limiter on `/weather`, `/forecast` and `/v1/records` and the async job queue's 503 already sent it.

---

## [2026-10-03] - OpenAPI Spec Cleanup for Public Docs (unreleased)

The spec at `/openapi.json` is now fit to back public developer documentation (Scalar, Redoc, Swagger UI at `/docs` and `/redoc`). No endpoint's runtime behaviour changed: everything below is about what the spec says. Routes dropped from the spec are still served.

### Changed

- **Spec metadata**: title `TempHist API`, version taken from `version.py`, a description (authentication, periods and identifiers, units, rate limits, data source), and tags with descriptions: Records, Locations, Shares, Jobs, Weather, Health.
- **Only public endpoints are documented**: 68 paths became 20. Excluded from the spec (but still routable): `/cache*`, `/cache-warm*`, `/cache-stats*`, `/usage-stats*`, `/rate-limit-stats`, `/rate-limit-status`, `/admin/*`, `/debug/*`, `/test-*`, `/analytics*`, `/protected-endpoint`, `/health/detailed`, `/`, `/v1/jobs/diagnostics/*`, `/v1/locations/popular/stats` and `/v1/locations/popular/display-strings`. `/health` stays public.
- **Authentication is declared per operation**: a `FirebaseBearer` security scheme (HTTP bearer, Firebase ID token) is applied to every operation the auth middleware protects, which is every documented operation except `/health`, `GET /v1/shares`, `GET /v1/shares/{share_id}` and `/v1/og/{share_id}.png`. It is derived from the middleware's own public-path rule, so the two cannot drift. Protected operations document the middleware's `401` and `403` bodies (`{"detail": ...}`).
- **Response models for every public route** (many `200`s were an empty `schema: {}`): `/weather`, `/forecast`, `/health`, `/v1/locations/search`, `/popular`, `/preapproved/status`, `/popular/status`, `POST /v1/records/.../async`, `GET /v1/jobs/{job_id}`, `/v1/shares` (list, create, get). `/v1/og/{share_id}.png` now declares `image/png`. `/weather` and `/forecast` document that they can answer `200` with an `{"error": ...}` body.
- **Parameters**: `identifier` is documented as `MM-DD` (same for all periods; the rolling window ends on that date) with a pattern and example, `date` on `/weather` as `YYYY-MM-DD`, and `unit_group` as `celsius|fahrenheit` on `/weather` and `/forecast`. These are documentation only: validation is unchanged, so inputs that were accepted before (including the legacy `unit_group=us`) still work.
- **Examples** added to `RecordResponse`, `MetaResponse`, `ErrorResponse`, the share request body and the path and query parameters.
- **`/v1/locations/popular`** has its own item model, because its results omit image fields unless `include_images=true` and entries for user-selected, non-curated locations have no `continent`, `tier` or images. `LocationItem` (used by `/preapproved`, which always includes images) is unchanged.
- **Error documentation**: `429` on `/weather`, `/forecast` and `/v1/records/*` is documented with the limiter's real body (`detail`, `reason`, `retry_after`) and a `Retry-After` header; `422` is documented as `ErrorResponse`, which is what the validation handler returns, instead of FastAPI's default `HTTPValidationError`.

### Fixed

- **Duplicate `operationId`s** (`root__options` and the `test-cors` GET/OPTIONS pairs): those routes are no longer in the spec.
- **Mojibake in the spec** (`â€”`, `Â°C`, `Ã—`, `âˆ’`): the source files and the served bytes were correct UTF-8, but `application/json` carries no charset and some consumers decode it as Latin-1. `/openapi.json` is now served with every non-ASCII character escaped (`°`), which reads identically in any encoding. The spec genuinely contains non-ASCII text (the `°C/decade` trend unit), so rewording descriptions would not have been enough.
- **`POST /v1/records/{period}/{location}/{identifier}/async`** is documented as `202 Accepted` (it always returned 202 but was declared `200`), with the `503` queue-full response and its `Retry-After`.
- **`RecordResponse.identifier`** described `YYYY-MM` for monthly records, which the API never accepted; every period uses `MM-DD`.

### Removed from the docs: legacy endpoints

These already answered `410 Gone` and are no longer listed in the spec. Their `410` responses carry `X-New-Endpoint` and a migration message.

| Removed | Use instead |
|---|---|
| `GET /data/{location}/{month_day}` | `GET /v1/records/daily/{location}/{month_day}` |
| `GET /average/{location}/{month_day}` | `GET /v1/records/daily/{location}/{month_day}/average` |
| `GET /trend/{location}/{month_day}` | `GET /v1/records/daily/{location}/{month_day}/trend` |
| `GET /summary/{location}/{month_day}` | `GET /v1/records/daily/{location}/{month_day}/summary` |
| `POST /v1/records/rolling-bundle/{location}/{anchor}/async` | `GET /v1/records/{period}/{location}/{identifier}` for each of `daily`, `weekly`, `monthly`, `yearly` |

### Added

- `tests/test_openapi.py` locks the contract: an allow-list of documented operations (a new route fails the test until it is added or hidden on purpose), unique operation IDs, no empty response schemas, security matching the middleware, an ASCII-only served spec, and that hidden routes are still routable.

---

## [2026-08-27] - Open-Meteo Paid Tier & Rate Limits for Store Launch (unreleased)

### Added

- **Open-Meteo paid plan support**: `OPEN_METEO_API_KEY` is appended as `?apikey=` to every outbound Open-Meteo request when set, and `OPEN_METEO_ARCHIVE_URL` / `OPEN_METEO_FORECAST_URL` are now environment-configurable so they can point at `customer-archive-api.open-meteo.com` / `customer-api.open-meteo.com`. All three default to the existing free hosts with no key, so local dev, tests and CI are unchanged. The free tier is capped at 10k calls/day and licensed non-commercial — a paid plan is required for public store listings. Attribution stays mandatory either way: the data remains CC-BY-4.0.
  - Set all three on **both** the API service and the worker service — the worker fetches from Open-Meteo in-process while building records.
  - `/health/detailed`'s Open-Meteo probe now sends the key too, so it reports `healthy` against a customer endpoint.

### Changed

- **Rate limit defaults raised**: `MAX_LOCATIONS_PER_HOUR` 10 → 60 and `MAX_REQUESTS_PER_HOUR` 100 → 400. Each period view calls `/v1/records/` once, so browsing ~10 cities across all four periods was enough to trigger `429 Location diversity limit exceeded` from a single IP.

### Security

- **API key never reaches logs or responses**: the key is added at request time only, so the URLs the Open-Meteo client classifies and logs stay unkeyed; URL and exception log arguments are additionally passed through `sanitize_url` / `sanitize_for_logging`. `/health/detailed` is a public endpoint and previously returned raw probe exception text, which could embed the request URL — it is now redacted.

### Removed

- **`SHARE_BASE_URL` documentation**: the variable has not been read since share and OG URLs became relative (`/s/{id}`, `/v1/og/{id}.png`). Consumers prepend their own origin. The stale README configuration block has been removed.

---

## [2026-05-18] - Search Results Include Canonical Location ID (v1.2.15)

### Added

- **`location_id` field in `/v1/locations/search` results**: Each result now includes a `location_id` field — the canonical preapproved location ID when the result matches a preapproved location, or `null` otherwise. Clients should pass this value to `POST /v1/locations/selections` when non-null. This removes the need for clients to know anything about canonical IDs; they simply reflect back what the search endpoint provides.
  - Mapbox path: matched against preapproved list by name + country code after geocoding
  - Preapproved fallback path: always populated (every fallback result is a preapproved location)

---

## [2026-05-18] - Popular Locations Ordering, Image Opt-in & Stats (v1.2.14)

### Fixed

- **Popular locations ordering**: `GET /v1/locations/popular` was silently discarding the popularity rank because the shared `filter_locations()` helper sorted results alphabetically. Popularity order is now preserved when filtering.

### Added

- **`GET /v1/locations/popular/stats`**: Debug/ops endpoint returning each location ID ranked by total selections in the rolling window, along with `total_selections`, `min_selections_threshold`, `using_signal` (whether live signal or preapproved fallback is currently active), and an `in_preapproved` flag per entry.
- **`include_images` query parameter** on `GET /v1/locations/popular` (default `false`): image fields (`imageUrl`, `imageAlt`, `imageAttribution`) are omitted by default to reduce response size for app clients that don't display location images. Pass `include_images=true` to restore them. The cache always stores full data; stripping is applied at response time.

---

## [2026-05-18] - Location Selections & Dynamic Popular Locations (v1.2.13)

### Added

- **`POST /v1/locations/selections`**: New authenticated endpoint for clients to record a canonical location ID selected by a user. Requires Firebase auth (anonymous users accepted). Returns 204 on success; silently no-ops when the usage tracker is unavailable. Used to build usage-derived popularity signal over time.
- **`LocationUsageTracker.record_selection(location_id, user_uid)`**: Records one selection per user per location per day into a daily Redis sorted set (`selections:YYYYMMDD`). Per-user deduplication prevents a single user inflating a location's count; dedup key has a 25-hour TTL to span day boundaries safely.
- **`LocationUsageTracker.get_popular_from_selections(limit, days)`**: Aggregates daily sorted sets over a rolling window using `ZUNIONSTORE` and returns location IDs ranked by total selections.
- **`LocationUsageTracker.get_total_selections(days)`**: Returns the sum of all selection scores in the rolling window; used to check against the minimum-signal threshold before switching away from the preapproved fallback.
- **Dynamic popular locations**: `GET /v1/locations/popular` now returns usage-derived rankings once the rolling window accumulates at least `POPULARITY_MIN_SELECTIONS` total selections. Falls back to the preapproved list until sufficient signal exists.

### Configuration

Three new environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `POPULARITY_WINDOW_DAYS` | `30` | Rolling window (days) for selection aggregation |
| `POPULARITY_MIN_SELECTIONS` | `100` | Minimum total selections before switching from preapproved fallback |
| `POPULARITY_MAX_LOCATIONS` | `20` | Maximum locations returned by `/v1/locations/popular` when no `limit` is specified |

---

## [2026-05-18] - Popular Locations Endpoint (v1.2.12)

### Added

- **`GET /v1/locations/popular`**: New endpoint returning popular locations. Currently falls back to the preapproved list; will surface usage-derived results once popularity tracking is implemented. Supports the same `country_code`, `tier`, and `limit` query parameters as `/v1/locations/preapproved`.
- **`GET /v1/locations/popular/status`**: Status/health endpoint for the popular locations service. Includes a `fallback` field indicating the current data source.

---

## [2026-05-17] - Firebase App Check & CORS Fix (v1.2.10)

### Added

- **Firebase App Check support**: New `APP_CHECK_ENFORCEMENT` environment variable (`off` / `monitor` / `enforce`). When enabled, the API verifies the `X-Firebase-AppCheck` token sent by web and mobile clients. `off` is the default; `monitor` logs failures without blocking; `enforce` rejects unauthenticated requests with 403.

### Fixed

- **CORS preflight 400 with App Check enabled**: Added `x-firebase-appcheck` to the CORS `allow_headers` list. Previously, any deployment where the web frontend had `VITE_RECAPTCHA_SITE_KEY` set would fail all OPTIONS preflight requests with 400 because the browser's `Access-Control-Request-Headers` included `x-firebase-appcheck`.

### Changed

- **Yearly coverage tolerance**: Relaxed from 50% to 80% for current-year data, so the yearly chart fills in earlier in the calendar year.
- **Removed anonymous-user location guard**: Anonymous Firebase users are no longer restricted to the preapproved location list — rate limiting provides sufficient abuse protection.

---

## [2026-05-15] - App Check Middleware & Cache Key Fix (v1.2.9)

### Added

- **Firebase App Check verification in middleware**: Token verification runs inside `verify_token_middleware` after the Firebase ID token is validated. All three enforcement modes (`off`, `monitor`, `enforce`) are supported.

### Fixed

- **Cache key import path**: Corrected `from cache_utils import …` → `from cache.keys import …` after the cache module was reorganised.

---

## [2026-04-11] - Social Sharing & OG Images (v1.1.5)

### Added

- **Social Share Endpoints**: New endpoints for creating and retrieving shareable temperature snapshots
  - `POST /v1/shares` — creates a share record (Firebase auth required), returns a short ID and URL
  - `GET /v1/shares/{share_id}` — retrieves share metadata by ID (public, no auth)
  - Share records are persisted in PostgreSQL and cached in Redis for 30 days
- **OG Preview Image**: `GET /v1/og/{share_id}.png` — generates a horizontal bar chart PNG for use as an `og:image` in social previews
  - Reference year bar highlighted in green; historical bars in red
  - Chart title shows city name and period; supports both Celsius and Fahrenheit
  - Placeholder image returned gracefully if data is unavailable

### Fixed

- **OG image Fahrenheit label**: Bar annotation now displays Fahrenheit temperatures as integers (e.g. `49°F`) instead of one decimal place (e.g. `49.3°F`); Celsius labels remain one decimal place
- **Fahrenheit summary text**: Fixed "0°F warmer than average" showing instead of "about average" when the difference rounds to zero

### Configuration

- `SHARE_BASE_URL` — base URL prepended to generated share URLs (default: `https://temphist.com`)
- Requires PostgreSQL (`TEMPHIST_PG_DSN` / `DATABASE_URL`); `shares` table is auto-created

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
