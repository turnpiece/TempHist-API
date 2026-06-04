# Deployment Guide

Complete deployment instructions for the TempHist API.

## Table of Contents

- [Quick Start](#quick-start)
- [Railway Deployment](#railway-deployment)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- GitHub account
- Railway account (https://railway.app/)
- Firebase service account JSON (optional)

### Basic Deployment Steps

1. Create Railway project
2. Add Redis database
3. Deploy API service from GitHub
4. Deploy worker service from same repo
5. Set environment variables
6. Deploy and verify

---

## Railway Deployment

### Architecture

Railway deployment uses a single project with multiple services:

```
┌──────────────────────────────────┐
│       TempHist Project           │
│                                  │
│  ┌────────┐  ┌────────┐          │
│  │  API   │  │ Redis  │          │
│  │Service │  │Database│          │
│  └────────┘  └────────┘          │
│                                  │
│  All services communicate        │
│  via private networking          │
└──────────────────────────────────┘
```

### Step 1: Create Railway Project

1. Go to https://railway.app/
2. Click **"New Project"** → **"Empty Project"**
3. Name it: `temphist-api`

### Step 2: Add Redis Database

1. In your project, click **"+ New"**
2. Select **"Database" → "Redis"**
3. Railway will:
   - Deploy Redis instance
   - Generate `REDIS_URL` variable
   - Make it available to your services

### Step 3: Deploy API Service

1. Click **"+ New" → "GitHub Repo"**
2. Select your `TempHist-api` repository
3. Railway will detect Python and configure automatically

### Step 4: Deploy Worker Service

1. Click **"+ New" → "GitHub Repo"**
2. Select the **same repository** as the API service
3. Name the service: `worker`
4. Set custom start command: `python worker_service.py`
5. Leave port empty (the worker doesn't expose HTTP)

The worker processes background jobs from the Redis queue. Without it, jobs will queue up but never complete.

### Step 5: Configure Environment Variables

In your API service **Variables** tab, add a **reference variable** to Redis:

```bash
# Reference Redis URL from the Redis service
REDIS_URL=${{Redis.REDIS_URL}}
```

**Important**: Replace `Redis` with your actual Redis service name if different.

Then add your other environment variables:

```bash
# Required
API_ACCESS_TOKEN=your_token_here

# Weather provider (defaults to open_meteo — no key needed)
# Set to visual_crossing and supply the key to use Visual Crossing instead:
# WEATHER_PROVIDER=visual_crossing
# VISUAL_CROSSING_API_KEY=your_key_here

# Firebase (optional - see Firebase section below)
FIREBASE_SERVICE_ACCOUNT={"type":"service_account",...}

# Configuration
CACHE_ENABLED=true
DEBUG=false
RATE_LIMIT_ENABLED=true
MAX_LOCATIONS_PER_HOUR=10
MAX_REQUESTS_PER_HOUR=100
```

### Step 6: Deploy

1. Railway auto-deploys when you push to GitHub
2. Or click **"Deploy"** in the Railway dashboard
3. Railway will generate a public URL: `https://your-app.up.railway.app`

### Step 7: Verify Deployment

```bash
# Check health
curl https://your-app.up.railway.app/health

# Check worker status (requires REDIS_URL in local .env)
python check_worker_status.py

# Test API (with your token)
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.up.railway.app/v1/records/daily/London/01-15
```

---

## Environment Variables

### Required Variables

| Variable                  | Description                                                      | Example                                           |
| ------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| `API_ACCESS_TOKEN`        | API access token for automated systems                           | `ghi789...`                                       |
| `REDIS_URL`               | Redis connection URL (auto-provided by Railway)                  | `redis://default:...@redis.railway.internal:6379` |
| `MAPBOX_TOKEN`            | Mapbox public token for location geocoding                       | `pk.eyJ1...`                                      |

### Weather Provider Variables

| Variable                  | Default       | Description                                                                  |
| ------------------------- | ------------- | ---------------------------------------------------------------------------- |
| `WEATHER_PROVIDER`        | `open_meteo`  | `open_meteo` (free, no key) or `visual_crossing` (requires API key below)    |
| `VISUAL_CROSSING_API_KEY` | *(none)*      | Required only when `WEATHER_PROVIDER=visual_crossing`                        |

### Optional Variables

| Variable                   | Default | Description                           |
| -------------------------- | ------- | ------------------------------------- |
| `FIREBASE_SERVICE_ACCOUNT`          | None          | Firebase credentials JSON (see below)                              |
| `APP_CHECK_ENFORCEMENT`             | `off`         | Firebase App Check mode: `off`, `monitor`, or `enforce`            |
| `CACHE_ENABLED`                     | `true`        | Enable/disable caching                                             |
| `DEBUG`                             | `false`       | Enable debug logging (forbidden when `ENVIRONMENT=production`)     |
| `ENVIRONMENT`                       | `development` | Set to `production` to enable safety checks                        |
| `LOG_VERBOSITY`                     | `normal`      | Logging detail: `minimal`, `normal`, or `verbose`                  |
| `BASE_URL`                          | *(localhost)* | Public API URL used for job callbacks                              |
| `RATE_LIMIT_ENABLED`                | `true`        | Enable rate limiting                                               |
| `MAX_LOCATIONS_PER_HOUR`            | `10`          | Max unique locations per hour (standard tokens)                    |
| `MAX_REQUESTS_PER_HOUR`             | `100`         | Max requests per hour (standard tokens)                            |
| `RATE_LIMIT_WINDOW_HOURS`           | `1`           | Rate limit time window                                             |
| `SERVICE_TOKEN_REQUESTS_PER_HOUR`   | `5000`        | Rate limit for service token requests                              |
| `SERVICE_TOKEN_LOCATIONS_PER_HOUR`  | `500`         | Rate limit for service token locations                             |
| `SERVICE_TOKEN_WINDOW_HOURS`        | `1`           | Service token rate limit window                                    |
| `IP_WHITELIST`                      | Empty         | Comma-separated whitelisted IPs                                    |
| `IP_BLACKLIST`                      | Empty         | Comma-separated blacklisted IPs                                    |
| `FILTER_WEATHER_DATA`               | `true`        | Filter to essential temperature data                               |
| `USAGE_TRACKING_ENABLED`            | `true`        | Enable usage tracking                                              |
| `USAGE_RETENTION_DAYS`              | `7`           | Days to retain usage data                                          |
| `ANALYTICS_RATE_LIMIT`              | `100`         | Max analytics requests per hour per IP                             |
| `CACHE_WARMING_ENABLED`             | `true`        | Enable cache warming on startup and interval                       |
| `CACHE_WARMING_INTERVAL_HOURS`      | `4`           | How often cache warming runs (hours)                               |
| `CACHE_WARMING_DAYS_BACK`           | `7`           | How many days back to warm                                         |
| `CACHE_WARMING_CONCURRENT_REQUESTS` | `3`           | Concurrent requests during warming                                 |
| `CACHE_WARMING_MAX_LOCATIONS`       | `15`          | Max locations to warm per cycle                                    |
| `CACHE_STATS_ENABLED`               | `true`        | Enable cache hit/miss statistics                                   |
| `CACHE_STATS_RETENTION_HOURS`       | `24`          | How long to retain cache stats                                     |
| `CACHE_HEALTH_THRESHOLD`            | `0.7`         | Cache hit-rate threshold below which health is degraded            |
| `CACHE_INVALIDATION_ENABLED`        | `true`        | Enable cache invalidation                                          |
| `CACHE_INVALIDATION_DRY_RUN`        | `false`       | Log invalidations without deleting (for debugging)                 |
| `CACHE_INVALIDATION_BATCH_SIZE`     | `100`         | Keys to process per invalidation batch                             |
| `CORS_ORIGINS`                      | Default       | Comma-separated allowed origins                                    |
| `CORS_ORIGIN_REGEX`                 | Default       | Regex pattern for allowed origins                                  |

### Firebase Configuration

Firebase is **optional**. The API works without it if you only use `API_ACCESS_TOKEN` authentication.

To enable Firebase authentication:

1. Download your Firebase service account JSON from Firebase Console
2. Copy the **entire JSON content**
3. Set it as the `FIREBASE_SERVICE_ACCOUNT` variable:

```bash
FIREBASE_SERVICE_ACCOUNT={"type":"service_account","project_id":"...","private_key":"..."}
```

**Note**: Railway handles JSON formatting automatically - just paste the content.

### Firebase App Check Configuration

Firebase App Check adds an extra layer of protection by verifying that requests come from your genuine app (using reCAPTCHA v3 on the web). It is **optional** and requires Firebase credentials to be configured first.

| `APP_CHECK_ENFORCEMENT` value | Behaviour |
|---|---|
| `off` (default) | App Check tokens are ignored; all authenticated requests pass |
| `monitor` | Tokens are verified and logged, but invalid/missing tokens are not blocked |
| `enforce` | Requests without a valid `X-Firebase-AppCheck` token are rejected with `403` |

**Web client setup**: set `VITE_RECAPTCHA_SITE_KEY` on the frontend service to your reCAPTCHA v3 site key. When set, the web app automatically obtains and attaches App Check tokens to every API request via the `X-Firebase-AppCheck` header. Register the domain in the [Google reCAPTCHA console](https://console.cloud.google.com/security/recaptcha) and enable App Check in the Firebase console.

**CORS note**: the `X-Firebase-AppCheck` header is included in the API's CORS `allow_headers`, so browser preflight requests succeed when App Check is enabled on the frontend.

### Redis Configuration

Railway automatically provides `REDIS_URL` when you add a Redis database to your project.

**Use variable reference** (recommended):

```bash
REDIS_URL=${{Redis.REDIS_URL}}
```

This automatically updates if Redis credentials change.

**Or use direct URL**:

```bash
REDIS_URL=redis://default:password@redis.railway.internal:6379
```

---

## Troubleshooting

### Common Issues

#### 1. App Returns 502 Error

**Symptom**: `{"status": "error", "code": 502, "message": "Application failed to respond"}`

**Causes**:

- App not listening on correct port
- App crashed during startup
- Dockerfile has wrong PORT configuration

**Solution**:

- Check that Dockerfile doesn't hardcode PORT
- Verify logs show `Uvicorn running on http://0.0.0.0:XXXX`
- Check deployment logs for Python errors

#### 2. Firebase Credentials Not Found

**Symptom**: `FileNotFoundError: firebase-service-account.json`

**Solution**:

- Set `FIREBASE_SERVICE_ACCOUNT` environment variable with JSON content
- Or remove Firebase requirement if not using Firebase auth

#### 3. Redis Connection Refused

**Symptom**: `Error 111 connecting to localhost:6379`

**Causes**:

- `REDIS_URL` not set correctly
- Using `localhost` instead of Railway's Redis service
- Redis and API in different projects

**Solutions**:

- Use variable reference: `REDIS_URL=${{Redis.REDIS_URL}}`
- Ensure Redis service is in the **same Railway project** as API
- Check Redis service name matches the reference

#### 4. DNS Resolution Failed for Redis

**Symptom**: `Error -2 connecting to redis.railway.internal:6379. Name or service not known`

**Cause**: API and Redis services are in **different Railway projects**

**Solution**: Both services must be in the **same Railway project** for private networking to work.

#### 5. Worker Service Not Processing Jobs

**Symptom**: Jobs created but not completed; `check_worker_status.py` reports no heartbeat

**Causes**:

- Worker service not created in Railway
- Worker service crashed at startup

**Solutions**:

- Check Railway dashboard — you should see both `api` and `worker` services
- If worker service is missing, create it manually (see Step 4 above)
- Check worker service logs for Python errors

#### 6. Cache Warming Errors

**Symptom**: `❌ CACHE WARMING FAILED: Error connecting to Redis`

**Solution**: This is now handled gracefully. The app continues running. To fix:

- Ensure Redis is properly configured
- Check `REDIS_URL` variable
- Verify Redis service is running

### Expected Log Output

#### Successful Startup (with Firebase + Redis)

```
✅ Firebase initialized successfully
✅ Background worker started successfully
✅ Worker Redis connection verified
💓 Worker heartbeat initialized
INFO: Uvicorn running on http://0.0.0.0:8080
```

#### Successful Startup (without Firebase)

```
⚠️ No Firebase credentials found - Firebase features will be disabled
✅ Background worker started successfully
✅ Worker Redis connection verified
INFO: Uvicorn running on http://0.0.0.0:8080
```

#### Startup without Redis (degraded mode)

```
⚠️ No Firebase credentials found - Firebase features will be disabled
✅ Background worker started successfully
❌ Worker cannot connect to Redis
⚠️ Background worker will exit - Redis not available
⚠️ Cache warming skipped - Redis not available
INFO: Uvicorn running on http://0.0.0.0:8080
```

### Checking Deployment Health

```bash
# Health check
curl https://your-app.railway.app/health

# Redis connectivity
curl https://your-app.railway.app/test-redis

# Rate limit status
curl https://your-app.railway.app/rate-limit-status
```

---

## Post-Deployment Checklist

- [ ] Health check passes (`/health`)
- [ ] Redis connectivity works (`/test-redis`)
- [ ] API endpoints return data
- [ ] Both `api` and `worker` services are deployed and healthy in Railway
- [ ] Worker heartbeat present (`python check_worker_status.py`)
- [ ] Rate limiting is functioning
- [ ] Cache warming is working (check logs)
- [ ] Environment variables are set correctly
- [ ] Custom domain configured (if applicable)
- [ ] Client applications updated with new URL
- [ ] Firebase allowed domains updated (if using Firebase)
- [ ] Monitoring/alerts configured
- [ ] 24-hour stability monitoring complete

---

## Support Resources

- **Railway Docs**: https://docs.railway.app/
- **Railway Discord**: https://discord.gg/railway
- **API Documentation**: `/docs` endpoint on your deployed app
- **Health Checks**: `/health`, `/test-redis`, `/rate-limit-status`

---

## Advanced Configuration

### Custom Start Command

If you need to customize the start command in Railway:

**Settings → Deploy → Custom Start Command**:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
```

### Dockerfile Deployment

If using the Dockerfile (already configured):

Railway will automatically detect and use it. The Dockerfile:

- Uses Python 3.12-slim
- Installs dependencies
- Uses dynamic `$PORT` from Railway
- Runs with 2 workers for better performance

### Health Check Configuration

Railway can monitor your app's health:

**Settings → Deploy → Health Check Path**: `/health`

### Multiple Regions

Railway supports multi-region deployments. Configure in:

**Settings → Deploy → Regions**

---

For additional troubleshooting, see `TROUBLESHOOTING.md`  
For API usage, see `README.md`  
For recent changes, see `CHANGELOG.md`
