# Railway Deployment Guide

Complete deployment and management guide for the TempHist API on Railway.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Initial Deployment](#initial-deployment)
- [Environment Variables Setup](#environment-variables-setup)
- [Worker Service Setup](#worker-service-setup)
- [Environment Migration](#environment-migration)
- [Troubleshooting](#troubleshooting)
- [Migration from Render](#migration-from-render)
- [Performance Optimization](#performance-optimization)

---

## Quick Start

### Prerequisites

- GitHub account
- Railway account (https://railway.app/)
- Visual Crossing API key
- OpenWeather API key
- Firebase service account JSON (optional)

### Basic Deployment Steps

1. Create Railway project
2. Add Redis database
3. Deploy API service from GitHub
4. Set environment variables
5. Create worker service
6. Deploy and verify

---

## Architecture Overview

Railway deployment uses a single project with multiple services:

```
┌──────────────────────────────────┐
│       TempHist Project           │
│                                  │
│  ┌────────┐  ┌────────┐  ┌────┐  │
│  │  API   │  │ Worker │  │Redis│  │
│  │Service │  │Service │  │ DB  │  │
│  └────────┘  └────────┘  └────┘  │
│                                  │
│  All services communicate        │
│  via private networking          │
└──────────────────────────────────┘
```

**Service Responsibilities:**

- **API Service**: Handles HTTP requests, creates jobs
- **Worker Service**: Processes background jobs from Redis queue
- **Redis Database**: Job queue, caching, inter-service communication

---

## Initial Deployment

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

### Step 4: Create Worker Service

1. Click **"+ New" → "GitHub Repo"**
2. Select the **same repository** as your API service
3. Name the service: `worker`
4. Set custom start command: `python worker_service.py`
5. Leave port empty (worker doesn't need a port)

### Step 5: Configure Environment Variables

See [Environment Variables Setup](#environment-variables-setup) section below.

### Step 6: Deploy and Verify

1. Railway auto-deploys when you push to GitHub
2. Or click **"Deploy"** in the Railway dashboard
3. Railway will generate public URLs for your services

**Verification Commands:**

```bash
# Check API health
curl https://your-api.up.railway.app/health

# Check worker status
python check_worker_status.py

# Test API endpoint
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-api.up.railway.app/v1/records/daily/London/01-15
```

---

## Environment Variables Setup

### Project-Level Variables (Shared)

Set these in Railway project dashboard → Variables tab for both services:

```
# Required API Keys
VISUAL_CROSSING_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
API_ACCESS_TOKEN=your_token_here

# Redis Configuration
REDIS_URL=${{Redis.REDIS_URL}}

# Firebase (optional)
FIREBASE_SERVICE_ACCOUNT={"type":"service_account",...}

# Core Configuration
CACHE_ENABLED=true
DEBUG=false
RATE_LIMIT_ENABLED=true
MAX_LOCATIONS_PER_HOUR=10
MAX_REQUESTS_PER_HOUR=100
FILTER_WEATHER_DATA=true

# Cache Warming Configuration
CACHE_WARMING_ENABLED=true

# CORS Configuration (optional - uses defaults if not set)
CORS_ORIGINS=https://yourdomain.com,https://staging.yourdomain.com
CORS_ORIGIN_REGEX=^https://.*\.yourdomain\.com$
CACHE_WARMING_INTERVAL_HOURS=4
CACHE_WARMING_DAYS_BACK=7
CACHE_WARMING_CONCURRENT_REQUESTS=3
CACHE_WARMING_MAX_LOCATIONS=15

# Cache Statistics
CACHE_STATS_ENABLED=true
CACHE_STATS_RETENTION_HOURS=24
CACHE_HEALTH_THRESHOLD=0.7

# Cache Invalidation
CACHE_INVALIDATION_ENABLED=true
CACHE_INVALIDATION_DRY_RUN=false
CACHE_INVALIDATION_BATCH_SIZE=100

# Usage Tracking
USAGE_TRACKING_ENABLED=true
USAGE_RETENTION_DAYS=7
```

### Service-Specific Variables

#### API Service Variables

Set these in the API service → Variables tab:

```
PORT=${{PORT}}  # Railway automatically sets this
SERVICE_NAME=api
```

#### Worker Service Variables

Set these in the Worker service → Variables tab:

```
SERVICE_NAME=worker
# PORT not needed for worker service
```

### Environment-Specific Values

Adjust these values based on your environment:

| Variable                            | Develop | Staging | Production |
| ----------------------------------- | ------- | ------- | ---------- |
| `DEBUG`                             | `true`  | `false` | `false`    |
| `CACHE_WARMING_ENABLED`             | `true`  | `false` | `true`     |
| `CACHE_WARMING_CONCURRENT_REQUESTS` | `3`     | `2`     | `5`        |
| `CACHE_WARMING_MAX_LOCATIONS`       | `15`    | `10`    | `25`       |
| `MAX_CONCURRENT_REQUESTS`           | `2`     | `2`     | `5`        |

### Firebase Configuration

Firebase is **optional**. The API works without it if you only use `API_ACCESS_TOKEN` authentication.

To enable Firebase authentication:

1. Download your Firebase service account JSON from Firebase Console
2. Copy the **entire JSON content**
3. Set it as the `FIREBASE_SERVICE_ACCOUNT` variable

**Note**: Railway handles JSON formatting automatically - just paste the content.

---

## Worker Service Setup

### Current Issue

Railway should create both API and Worker services, but sometimes only the API service is created. This causes jobs to be created but not processed.

### Verification Steps

1. **Check Railway Dashboard:**

   - You should see 2 services: `api` and `worker`
   - Both should show as "Deployed" and "Healthy"

2. **Run Diagnostic Script:**

   ```bash
   python check_worker_status.py
   ```

   Should show:

   ```
   ✅ Worker heartbeat: Xs ago
   ✅ Worker appears to be running
   ```

### Manual Worker Service Creation

If the worker service is missing:

1. **In Railway Dashboard:**

   - Click "New Service"
   - Select "GitHub Repo"
   - Choose the same repository as your API service

2. **Configure the Worker Service:**

   - **Service Name:** `worker`
   - **Command:** `python worker_service.py`
   - **Port:** Leave empty (worker doesn't need a port)
   - **Environment Variables:** Copy all from the main service

3. **Deploy:**
   - Railway will build and deploy the worker service
   - Both services will share the same environment variables

### Troubleshooting Worker Issues

#### Worker Service Won't Start

- Check logs for startup errors
- Verify `python worker_service.py` command works locally
- Ensure all environment variables are set

#### Jobs Still Not Processing

- Check worker service logs
- Verify Redis connection in worker service
- Run diagnostic script to check job queue status

#### Redis Connection Issues

- Ensure both services use the same `REDIS_URL`
- Check Redis service is running and accessible

---

## Environment Migration

### Export/Import Variables Between Environments

Use the provided scripts to migrate environment variables:

#### Method 1: Railway CLI (Recommended)

```bash
# Export from develop environment
railway link                    # Link to develop project
railway variables --json > develop_vars.json

# Import to staging environment
railway link                    # Link to staging project
cat develop_vars.json | jq -r '.[] | "\(.name)=\(.value)"' | while read line; do
  railway variables --set "$line"
done

# Import to production environment
railway link                    # Link to production project
cat develop_vars.json | jq -r '.[] | "\(.name)=\(.value)"' | while read line; do
  railway variables --set "$line"
done
```

#### Method 2: Using Scripts

```bash
# Run the interactive script
./railway/railway_export_simple.sh

# Or use the Python script
python railway/export_env_vars.py
```

#### Method 3: Manual Dashboard

1. Go to develop project → Variables tab
2. Select all variables (Ctrl+A) and copy (Ctrl+C)
3. Go to staging/production project → Variables tab
4. Paste variables (Ctrl+V)
5. Adjust environment-specific values

### Environment-Specific Adjustments

After importing, adjust these variables:

#### Staging Adjustments

```bash
railway variables --set "DEBUG=false"
railway variables --set "CACHE_WARMING_ENABLED=false"
railway variables --set "CACHE_WARMING_CONCURRENT_REQUESTS=2"
railway variables --set "CACHE_WARMING_MAX_LOCATIONS=10"
```

#### Production Adjustments

```bash
railway variables --set "DEBUG=false"
railway variables --set "CACHE_WARMING_ENABLED=true"
railway variables --set "CACHE_WARMING_CONCURRENT_REQUESTS=5"
railway variables --set "CACHE_WARMING_MAX_LOCATIONS=25"
railway variables --set "MAX_CONCURRENT_REQUESTS=5"
```

---

## Troubleshooting

### Common Issues

#### 1. App Returns 502 Error

**Symptom**: `{"status": "error", "code": 502, "message": "Application failed to respond"}`

**Causes:**

- App not listening on correct port
- App crashed during startup
- Missing environment variables

**Solution:**

- Check that app is listening on `$PORT`
- Verify logs show `Uvicorn running on http://0.0.0.0:XXXX`
- Check deployment logs for Python errors

#### 2. Redis Connection Refused

**Symptom**: `Error 111 connecting to localhost:6379`

**Causes:**

- `REDIS_URL` not set correctly
- Using `localhost` instead of Railway's Redis service
- Redis and API in different projects

**Solutions:**

- Use variable reference: `REDIS_URL=${{Redis.REDIS_URL}}`
- Ensure Redis service is in the **same Railway project** as API
- Check Redis service name matches the reference

#### 3. Worker Service Not Processing Jobs

**Symptom**: Jobs created but not processed, "No weekly data found" errors

**Causes:**

- Worker service not running
- Worker service not created
- Redis connection issues in worker

**Solutions:**

- Check Railway dashboard for worker service
- Create worker service manually if missing
- Verify worker service logs
- Run diagnostic script: `python check_worker_status.py`

#### 4. Environment Variables Not Loading

**Symptom**: App uses default values instead of configured variables

**Solutions:**

1. Check variable names are spelled correctly
2. Ensure variables are set at project level (not service level)
3. Redeploy after adding/changing variables
4. Check for typos in variable references: `${{Service.VARIABLE}}`

### Expected Log Output

#### Successful Startup (API Service)

```
✅ Firebase initialized successfully
✅ Redis connection verified
🚀 STARTUP CACHE WARMING: Triggering initial warming cycle
INFO: Uvicorn running on http://0.0.0.0:8080
```

#### Successful Startup (Worker Service)

```
✅ Worker Redis connection verified
💓 Worker heartbeat initialized
🔄 Worker main loop started
📋 Found X pending jobs
```

### Health Check Commands

```bash
# API health
curl https://your-app.railway.app/health

# Worker status
python check_worker_status.py

# Redis connectivity
curl https://your-app.railway.app/test-redis

# Rate limit status
curl https://your-app.railway.app/rate-limit-status
```

---

## Migration from Render

### Why Migrate to Railway?

#### Current Issues on Render Free Tier

- ❌ **Spin down when idle** → Background worker stops
- ❌ **Background thread crashes** → Hard to debug
- ❌ **Requires paid plan** for multiple services

#### Benefits of Railway

- ✅ **No spin down** on hobby plan
- ✅ **Multiple services** supported affordably
- ✅ **Better logging** → Easier debugging
- ✅ **Private networking** between services
- ✅ **Simpler configuration**

### Migration Steps

1. **Prepare Railway Project** - Follow initial deployment steps
2. **Migrate Environment Variables** - Use migration scripts
3. **Test Railway Deployment** - Verify all functionality works
4. **Update Client Applications** - Point to new Railway URLs
5. **DNS/Domain Update** - If using custom domain
6. **Monitor and Verify** - Run both for 24-48 hours
7. **Decommission Render** - After verification period

### Cost Comparison

| Platform          | Configuration      | Monthly Cost               |
| ----------------- | ------------------ | -------------------------- |
| **Render Free**   | API only, no Redis | $0 (with spin-down)        |
| **Render Paid**   | API + Redis        | $14/month ($7 × 2)         |
| **Railway Hobby** | API + Redis        | ~$6/month (with $5 credit) |

Railway is more cost-effective for this use case.

---

## Performance Optimization

### Cloudflare Integration

The API is optimized for Cloudflare edge caching:

#### Cache Headers

```
Cache-Control: public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800
ETag: "abc123def456"
Last-Modified: Mon, 15 Jan 2024 12:00:00 GMT
```

#### Cache Rules Configuration

Configure these rules in Cloudflare Dashboard:

1. **Cache API Endpoints:**

   ```
   Expression: (http.request.method eq "GET") and (http.host contains "api.temphist.com")
   Action: Cache Level = Cache Everything
   Edge TTL: 1 day
   ```

2. **Bypass Cache for Jobs:**
   ```
   Expression: (http.request.uri contains "/v1/jobs/") or (http.request.uri contains "/async")
   Action: Cache Level = Bypass
   ```

### Performance Targets

- **Warm cache**: <200ms p95
- **Cold cache**: <2s p95
- **Cache hit rate**: >80% overall
- **Origin load**: <10% of total requests

### Monitoring

Key metrics to monitor:

1. **Cache Hit Rate** - Target: >80%
2. **Response Times** - Target: <500ms p95
3. **Origin Load** - Target: <10% of requests
4. **Error Rates** - Target: <1% 5xx errors

---

## Post-Deployment Checklist

- [ ] Health check passes (`/health`)
- [ ] Redis connectivity works (`/test-redis`)
- [ ] API endpoints return data
- [ ] Worker service is running and processing jobs
- [ ] Rate limiting is functioning
- [ ] Cache warming is working (check logs)
- [ ] Environment variables are set correctly
- [ ] Both API and Worker services are deployed
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

### Custom Start Commands

**API Service:**

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
```

**Worker Service:**

```bash
python worker_service.py
```

### Health Check Configuration

**Settings → Deploy → Health Check Path**: `/health`

### Multiple Regions

Railway supports multi-region deployments. Configure in:
**Settings → Deploy → Regions**

---

For additional troubleshooting, see `TROUBLESHOOTING.md`  
For API usage, see `README.md`  
For recent changes, see `CHANGELOG.md`
