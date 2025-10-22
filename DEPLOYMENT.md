# Deployment Guide

Complete deployment instructions for the TempHist API.

## Table of Contents

- [Quick Start](#quick-start)
- [Railway Deployment](#railway-deployment)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Migration from Render](#migration-from-render)

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
5. Deploy and verify

---

## Railway Deployment

### Architecture

Railway deployment uses a single project with multiple services:

```
┌──────────────────────────────────┐
│       TempHist Project           │
│                                  │
│  ┌────────┐  ┌────────┐         │
│  │  API   │  │ Redis  │         │
│  │Service │  │Database│         │
│  └────────┘  └────────┘         │
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

### Step 4: Configure Environment Variables

In your API service **Variables** tab, add a **reference variable** to Redis:

```bash
# Reference Redis URL from the Redis service
REDIS_URL=${{Redis.REDIS_URL}}
```

**Important**: Replace `Redis` with your actual Redis service name if different.

Then add your other environment variables:

```bash
# Required API Keys
VISUAL_CROSSING_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
API_ACCESS_TOKEN=your_token_here

# Firebase (optional - see Firebase section below)
FIREBASE_SERVICE_ACCOUNT={"type":"service_account",...}

# Configuration
CACHE_ENABLED=true
DEBUG=false
RATE_LIMIT_ENABLED=true
MAX_LOCATIONS_PER_HOUR=10
MAX_REQUESTS_PER_HOUR=100
```

### Step 5: Deploy

1. Railway auto-deploys when you push to GitHub
2. Or click **"Deploy"** in the Railway dashboard
3. Railway will generate a public URL: `https://your-app.up.railway.app`

### Step 6: Verify Deployment

```bash
# Check health
curl https://your-app.up.railway.app/health

# Test API (with your token)
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.up.railway.app/v1/records/daily/London/01-15
```

---

## Environment Variables

### Required Variables

| Variable                  | Description                                     | Example                                           |
| ------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| `VISUAL_CROSSING_API_KEY` | Visual Crossing weather API key                 | `abc123...`                                       |
| `OPENWEATHER_API_KEY`     | OpenWeather API key                             | `def456...`                                       |
| `API_ACCESS_TOKEN`        | API access token for automated systems          | `ghi789...`                                       |
| `REDIS_URL`               | Redis connection URL (auto-provided by Railway) | `redis://default:...@redis.railway.internal:6379` |

### Optional Variables

| Variable                   | Default | Description                           |
| -------------------------- | ------- | ------------------------------------- |
| `FIREBASE_SERVICE_ACCOUNT` | None    | Firebase credentials JSON (see below) |
| `CACHE_ENABLED`            | `true`  | Enable/disable caching                |
| `DEBUG`                    | `false` | Enable debug logging                  |
| `RATE_LIMIT_ENABLED`       | `true`  | Enable rate limiting                  |
| `MAX_LOCATIONS_PER_HOUR`   | `10`    | Max unique locations per hour         |
| `MAX_REQUESTS_PER_HOUR`    | `100`   | Max requests per hour                 |
| `RATE_LIMIT_WINDOW_HOURS`  | `1`     | Rate limit time window                |
| `IP_WHITELIST`             | Empty   | Comma-separated whitelisted IPs       |
| `IP_BLACKLIST`             | Empty   | Comma-separated blacklisted IPs       |
| `FILTER_WEATHER_DATA`      | `true`  | Filter to essential temperature data  |
| `CORS_ORIGINS`             | Default | Comma-separated allowed origins       |
| `CORS_ORIGIN_REGEX`        | Default | Regex pattern for allowed origins     |

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

#### 5. Cache Warming Errors

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

#### 1. Prepare Railway Project

Follow the [Railway Deployment](#railway-deployment) steps above to set up your Railway project.

#### 2. Migrate Environment Variables

Export from Render and import to Railway:

```bash
# Render environment variables to migrate:
VISUAL_CROSSING_API_KEY
OPENWEATHER_API_KEY
FIREBASE_SERVICE_ACCOUNT
API_ACCESS_TOKEN
CACHE_ENABLED
DEBUG
RATE_LIMIT_ENABLED
MAX_LOCATIONS_PER_HOUR
MAX_REQUESTS_PER_HOUR
```

#### 3. Test Railway Deployment

Before switching over:

```bash
# Test health
curl https://your-railway-app.up.railway.app/health

# Test API endpoint
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-railway-app.up.railway.app/v1/records/daily/London/01-15

# Compare responses with Render
diff <(curl https://your-render-app.onrender.com/health) \
     <(curl https://your-railway-app.up.railway.app/health)
```

#### 4. Update Client Applications

Update your client apps to use the new Railway URL:

```javascript
// Old
const API_URL = "https://your-app.onrender.com";

// New
const API_URL = "https://your-app.up.railway.app";
```

#### 5. DNS/Domain Update (if applicable)

If using a custom domain:

1. Update DNS records to point to Railway
2. Configure custom domain in Railway dashboard
3. Wait for DNS propagation (up to 48 hours)

#### 6. Monitor and Verify

Keep both deployments running for 24-48 hours:

- Monitor Railway logs for errors
- Compare response times
- Verify cache warming is working
- Check background worker status

#### 7. Decommission Render

Once Railway is stable:

1. Disable auto-deploy on Render
2. Keep Render running for 1 week as backup
3. Delete Render service after verification period

### Cost Comparison

| Platform          | Configuration      | Monthly Cost               |
| ----------------- | ------------------ | -------------------------- |
| **Render Free**   | API only, no Redis | $0 (with spin-down)        |
| **Render Paid**   | API + Redis        | $14/month ($7 × 2)         |
| **Railway Hobby** | API + Redis        | ~$6/month (with $5 credit) |

Railway is more cost-effective for this use case.

### Rollback Plan

If issues arise:

1. Keep Render deployment active during migration
2. DNS/domain can be quickly switched back
3. Client apps can revert to old URL
4. Data is not migrated (API is stateless except Redis cache)

---

## Post-Deployment Checklist

- [ ] Health check passes (`/health`)
- [ ] Redis connectivity works (`/test-redis`)
- [ ] API endpoints return data
- [ ] Rate limiting is functioning
- [ ] Cache warming is working (check logs)
- [ ] Background worker is running (if using Redis)
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
