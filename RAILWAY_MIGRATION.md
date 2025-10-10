# Migration Guide: Render → Railway with Separate Worker

## Why Migrate?

### Current Issues on Render Free Tier

- ❌ **Spin down when idle** → Worker stops processing jobs
- ❌ **Background thread crashes** → Hard to debug
- ❌ **Requires paid plan** for separate worker service

### Benefits of Railway

- ✅ **No spin down** on hobby plan
- ✅ **Multiple services** supported affordably
- ✅ **Separate worker service** → More robust
- ✅ **Better logging** → Easier debugging
- ✅ **Independent scaling**

## Architecture Change

### Before (Render):

```
┌─────────────────────┐
│   FastAPI App       │
│  ┌──────────────┐   │
│  │ Main API     │   │
│  │ +            │   │
│  │ Background   │   │← Crashes together
│  │ Worker Thread│   │← Stops when app sleeps
│  └──────────────┘   │
└─────────────────────┘
```

### After (Railway):

```
┌──────────────┐        ┌──────────────┐
│ FastAPI API  │        │ Worker       │
│              │        │ Service      │
│              │        │              │
└───────┬──────┘        └──────┬───────┘
        │                      │
        └──────┬───────────────┘
               │
        ┌──────▼──────┐
        │   Redis     │
        └─────────────┘
```

## Migration Steps

### 1. Prepare Code Changes

#### A. Remove Background Worker from Main App

Edit `main.py` and remove/comment out:

```python
# Remove this section:
# try:
#     from background_worker import start_background_worker
#     start_background_worker(redis_client)
#     logger.info("✅ Background worker started successfully")
# except Exception as e:
#     logger.error(f"❌ Failed to start background worker: {e}")
#     logger.warning("Async job processing will not be available")
```

#### B. Add Procfile for Railway

Create `Procfile` in project root:

```procfile
# Main API service
web: uvicorn main:app --host 0.0.0.0 --port $PORT

# Worker service
worker: python worker_service.py
```

#### C. Create railway.json

Create `railway.json` for service configuration:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "numReplicas": 1,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### 2. Set Up Railway Project

1. **Create Railway account**: https://railway.app/
2. **Create new project**: "New Project" → "Empty Project"
3. **Name it**: `temphist-api`

### 3. Add Redis Service

1. In Railway project, click **"+ New"**
2. Select **"Database" → "Redis"**
3. Railway will:
   - Deploy Redis instance
   - Generate `REDIS_URL` environment variable
   - Make it available to your services

### 4. Deploy API Service

1. Click **"+ New" → "GitHub Repo"**
2. Select your `TempHist-api` repository
3. Railway will detect Python and install dependencies
4. **Configure service:**

   - **Name**: `api`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory**: `/` (or wherever your code is)

5. **Add environment variables:**

   - `VISUAL_CROSSING_API_KEY`: Your API key
   - `OPENWEATHER_API_KEY`: Your API key
   - `FIREBASE_SERVICE_ACCOUNT`: Your Firebase credentials JSON
   - `API_ACCESS_TOKEN`: Your access token
   - `CACHE_ENABLED`: `true`
   - `DEBUG`: `false` (or `true` for testing)
   - `RATE_LIMIT_ENABLED`: `true`
   - `MAX_LOCATIONS_PER_HOUR`: `10`
   - `MAX_REQUESTS_PER_HOUR`: `100`
   - (Redis URL is auto-provided by Railway)

6. **Generate domain**: Railway will create a public URL
7. **Deploy**: Railway auto-deploys on git push

### 5. Deploy Worker Service

1. In same Railway project, click **"+ New" → "GitHub Repo"**
2. Select **same repository**
3. **Configure service:**

   - **Name**: `worker`
   - **Start Command**: `python worker_service.py`
   - **Root Directory**: `/`

4. **Add environment variables** (same as API):

   - `VISUAL_CROSSING_API_KEY`
   - `OPENWEATHER_API_KEY`
   - `CACHE_ENABLED`: `true`
   - `DEBUG`: `true` (to see worker logs)
   - (Redis URL is auto-shared from Redis service)

5. **Deploy**: Worker starts automatically

### 6. Verify Deployment

#### Check API Health

```bash
curl https://your-railway-app.up.railway.app/health
```

#### Check Worker Status

```bash
curl https://your-railway-app.up.railway.app/v1/jobs/diagnostics/worker-status
```

Should show:

```json
{
  "worker": {
    "alive": true,
    "status": "healthy"
  }
}
```

#### Test Async Job

```bash
# Create a job
curl -X POST "https://your-railway-app.up.railway.app/v1/records/daily/London/10-09/async" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Check job status (use job_id from response)
curl "https://your-railway-app.up.railway.app/v1/jobs/{job_id}"
```

### 7. Monitor Logs

**API Logs:**

- Railway Dashboard → `api` service → "Logs" tab

**Worker Logs:**

- Railway Dashboard → `worker` service → "Logs" tab
- Should see: "🚀 ASYNC JOB WORKER SERVICE STARTING"

**Redis Logs:**

- Railway Dashboard → Redis service → "Logs" tab

## Cost Comparison

### Render (Current)

- **Free tier**: API only, no worker (worker in same process)
- **Paid tier**: $7/month per service
  - API: $7/month
  - Worker: $7/month
  - **Total**: $14/month

### Railway

- **Hobby Plan**: $5/month (includes $5 credit)
- **Usage-based** beyond free tier
- Estimated cost for TempHist:
  - API: ~$3/month
  - Worker: ~$2/month
  - Redis: $1/month (512MB)
  - **Total**: ~$6/month (after free $5 credit)

**Railway is cheaper and better!**

## Rollback Plan

If something goes wrong:

1. **Keep Render running** during migration
2. **Test Railway thoroughly** before switching
3. **Update DNS/domain** only when Railway is proven working
4. **Keep both running** for 1 week before shutting down Render

## Post-Migration Checklist

- [ ] API health check passes
- [ ] Worker status shows "healthy"
- [ ] Create test async job
- [ ] Job completes successfully
- [ ] Check worker logs for errors
- [ ] Update client apps with new Railway URL
- [ ] Update Firebase allowed domains
- [ ] Update environment variable documentation
- [ ] Monitor for 24 hours
- [ ] Shut down Render services

## Troubleshooting

### Worker Not Starting

```bash
# Check worker logs in Railway dashboard
# Look for:
# - "Missing required environment variables"
# - "Failed to connect to Redis"
# - Import errors
```

### Jobs Not Processing

```bash
# Check diagnostics
curl https://your-app.up.railway.app/v1/jobs/diagnostics/worker-status

# Check worker logs for processing errors
```

### Redis Connection Issues

- Verify `REDIS_URL` is set correctly
- Check Redis service is running in Railway
- Verify services are in same project (can access each other)

## Need Help?

- Railway Docs: https://docs.railway.app/
- Railway Discord: https://discord.gg/railway
- This project's issues: Create GitHub issue
