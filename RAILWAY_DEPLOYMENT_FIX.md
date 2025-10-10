# Railway Deployment Fix - October 9, 2025

## Issues Found and Fixed

### 1. ❌ Firebase Credentials File Not Found

**Problem:** The app tried to load `firebase-service-account.json` from the filesystem, which doesn't exist in Railway deployments.

**Fix Applied:** Updated `main.py` (line 733-758) to:

- Load Firebase credentials from environment variable (`FIREBASE_SERVICE_ACCOUNT` or `FIREBASE_SERVICE_ACCOUNT_JSON`)
- Fall back to file for local development
- Gracefully continue without Firebase if credentials are missing
- Log appropriate warnings/errors

### 2. ❌ Redis Connection Refused Causing Crash

**Problem:** Background worker crashed the entire app when Redis wasn't available.

**Fix Applied:** Updated `background_worker.py` (line 99-101) to:

- Exit gracefully when Redis is unavailable
- Log warning instead of crashing
- Allow the main app to continue running

### 3. ❌ CacheWarmer Object Not Callable

**Problem:** Application startup failed with `TypeError: 'CacheWarmer' object is not callable` in the lifespan function.

**Error:**

```
File "/app/main.py", line 608, in lifespan
    if CACHE_WARMING_ENABLED and get_cache_warmer()():
                                 ^^^^^^^^^^^^^^^^^^^^
TypeError: 'CacheWarmer' object is not callable
```

**Fix Applied:** Updated `main.py` (lines 608, 616, 619) to:

- Removed extra `()` call on `get_cache_warmer()` - changed from `get_cache_warmer()()` to `get_cache_warmer()`
- Fixed all three instances in the lifespan function

### 4. ⚠️ Cache Warming Failing Without Redis

**Problem:** Cache warming throws errors when Redis is not available, cluttering logs.

**Error:**

```
2025-10-09 19:56:52,388 - cache_utils - ERROR - ❌ CACHE WARMING FAILED: Error 111 connecting to localhost:6379. Connection refused.
```

**Fix Applied:** Updated `cache_utils.py` (lines 682-687) to:

- Check Redis availability before attempting cache warming
- Skip warming gracefully with a warning instead of throwing errors
- Return status "skipped" when Redis is unavailable

## What You Need to Do on Railway

### Required Environment Variables

Set these in your Railway project:

1. **FIREBASE_SERVICE_ACCOUNT** (or FIREBASE_SERVICE_ACCOUNT_JSON)

   - Copy the entire contents of your `firebase-service-account.json` file
   - Paste it as a single-line JSON string in Railway
   - Example: `{"type":"service_account","project_id":"...","private_key":"..."}`

2. **REDIS_URL**

   - Add a Redis database to your Railway project
   - Railway will automatically provide this variable
   - Format: `redis://default:password@host:port`

3. **Other Required Variables:**
   - `VISUAL_CROSSING_API_KEY` - Your weather API key
   - `OPENWEATHER_API_KEY` - Your OpenWeather API key
   - `API_ACCESS_TOKEN` - Your API access token
   - `CACHE_ENABLED` - Set to `true`
   - `DEBUG` - Set to `false` for production
   - `RATE_LIMIT_ENABLED` - Set to `true`
   - `MAX_LOCATIONS_PER_HOUR` - Set to `10`
   - `MAX_REQUESTS_PER_HOUR` - Set to `100`

### Deployment Steps

1. **Add Redis to Railway:**

   - In your Railway project, click "+ New"
   - Select "Database" → "Redis"
   - Railway will auto-configure REDIS_URL

2. **Set Environment Variables:**

   - Go to your service settings
   - Add all the environment variables listed above
   - Make sure FIREBASE_SERVICE_ACCOUNT is valid JSON (test with `jq` or online JSON validator)

3. **Redeploy:**

   - Push your code changes to GitHub
   - Railway will automatically redeploy
   - OR click "Redeploy" in Railway dashboard

4. **Verify:**

   ```bash
   # Check if app is running
   curl https://your-app.railway.app/health

   # Check logs in Railway dashboard
   # Should see:
   # - "✅ Firebase initialized successfully"
   # - "✅ Background worker started successfully"
   # - "⚠️ Background worker will exit - Redis not available" (if no Redis)
   ```

## Expected Behavior After Fix

### With Firebase + Redis Configured:

```
✅ Firebase initialized successfully
✅ Background worker started successfully
✅ Worker Redis connection verified
💓 Worker heartbeat initialized
```

### With Firebase, No Redis:

```
✅ Firebase initialized successfully
✅ Background worker started successfully
❌ Worker cannot connect to Redis
⚠️ Background worker will exit - Redis not available
⚠️ Cache warming skipped - Redis not available
(App continues running normally)
```

### No Firebase, No Redis:

```
⚠️ No Firebase credentials found - Firebase features will be disabled
✅ Background worker started successfully
❌ Worker cannot connect to Redis
⚠️ Background worker will exit - Redis not available
⚠️ Cache warming skipped - Redis not available
(App continues running with limited features)
```

## Files Changed

- ✅ `main.py` - Firebase credentials loading (lines 733-758) + CacheWarmer fix (lines 608, 616, 619)
- ✅ `background_worker.py` - Graceful Redis error handling (lines 99-101)
- ✅ `cache_utils.py` - Graceful cache warming when Redis unavailable (lines 682-687)

## Next Steps

1. Add Redis database in Railway
2. Set FIREBASE_SERVICE_ACCOUNT environment variable
3. Set other required environment variables
4. Redeploy
5. Monitor logs for successful startup

## Troubleshooting

### "FileNotFoundError: firebase-service-account.json"

- FIREBASE_SERVICE_ACCOUNT environment variable is not set or is invalid JSON
- Check that the JSON is properly escaped/formatted

### "Worker cannot connect to Redis"

- Redis database not added to Railway project
- REDIS_URL environment variable not set
- This is now non-fatal - app will continue without background jobs

### "Firebase features will be disabled"

- No Firebase credentials found
- If you need Firebase auth, set the FIREBASE_SERVICE_ACCOUNT variable
- If you don't use Firebase auth, this warning is safe to ignore

## Testing

After deploying, test these endpoints:

```bash
# Health check
curl https://your-app.railway.app/health

# API test (if you have auth)
curl https://your-app.railway.app/v1/records/daily/London/10-09 \
  -H "Authorization: Bearer YOUR_TOKEN"
```
