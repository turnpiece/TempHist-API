# Troubleshooting Guide

Common issues and solutions for the TempHist API.

## Table of Contents

- [Deployment Issues](#deployment-issues)
- [Async Jobs & Background Worker](#async-jobs--background-worker)
- [Redis & Caching](#redis--caching)
- [Rate Limiting](#rate-limiting)
- [API Errors](#api-errors)
- [Performance Issues](#performance-issues)

---

## Deployment Issues

### 502 Bad Gateway

**Symptoms**:

```json
{
  "status": "error",
  "code": 502,
  "message": "Application failed to respond"
}
```

**Common Causes**:

1. App not listening on correct PORT
2. App crashed during startup
3. Hardcoded port in Dockerfile

**Solutions**:

```bash
# Check deployment logs for:
INFO: Uvicorn running on http://0.0.0.0:XXXX

# Verify Dockerfile uses dynamic PORT
# Should NOT have: ENV PORT=8000
# Should have: CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
```

### Firebase Credentials Error

**Symptom**:

```
FileNotFoundError: [Errno 2] No such file or directory: 'firebase-service-account.json'
```

**Solution**:

```bash
# Set environment variable with JSON content
FIREBASE_SERVICE_ACCOUNT='{"type":"service_account","project_id":"...",...}'

# Or if Firebase not needed, the app will continue with warning:
# ⚠️ No Firebase credentials found - Firebase features will be disabled
```

### Environment Variables Not Loading

**Symptom**: App uses default values instead of configured environment variables

**Solutions**:

1. Check variable names are spelled correctly
2. Ensure variables are set in the correct environment
3. Redeploy after adding/changing variables
4. Check for typos in variable references: `${{Service.VARIABLE}}`

---

## Async Jobs & Background Worker

### Jobs Timing Out After 100 Polls

**Symptom**:

```
Failed to fetch week data for London: Job timed out after 100 polls
```

This means the background worker is not processing jobs.

#### Quick Diagnosis

**Check worker status** (if you had this endpoint):

```bash
# Note: This endpoint was removed in recent updates
# Check logs instead for worker status
```

**Check deployment logs**:

```bash
# Look for these messages:
✅ Background worker started successfully
✅ Worker Redis connection verified
💓 Worker heartbeat initialized
```

#### Common Causes & Solutions

##### 1. Worker Not Running

**Symptoms**:

- Jobs stuck in PENDING state
- No worker heartbeat in logs
- No job processing messages

**Solutions**:

1. **Check Redis connection**:

```bash
curl https://your-app.com/test-redis
```

2. **Check deployment logs** for startup errors:

```bash
# Look for:
❌ Failed to start background worker
❌ Worker cannot connect to Redis
```

3. **Restart the application**:

- Push a new commit, or
- Click "Redeploy" in Railway dashboard

##### 2. Redis Not Available

**Symptoms**:

```
❌ Worker cannot connect to Redis: Error 111 connecting to localhost:6379
⚠️ Background worker will exit - Redis not available
```

**Solutions**:

1. **Verify Redis is in the same Railway project**:

   - API and Redis must be in the **same project**
   - Check "Services" tab shows both

2. **Check REDIS_URL variable**:

```bash
# Should use reference:
REDIS_URL=${{Redis.REDIS_URL}}

# Or direct connection:
REDIS_URL=redis://default:password@redis.railway.internal:6379
```

3. **Verify Redis service is running**:
   - Check Redis service logs in Railway
   - Look for "Ready to accept connections"

##### 3. Worker Stuck or Frozen

**Symptoms**:

- Worker heartbeat is stale (> 60 seconds old)
- Jobs stuck in PROCESSING state
- No worker activity in logs

**Solutions**:

1. **Restart the application** to restart the worker

2. **Check for long-running jobs** in logs:

```bash
# Look for:
🔄 Processing job: job_id
# But no corresponding:
✅ Job completed: job_id
```

3. **Monitor recent logs**:

```bash
# Search for:
- "Processing job"
- "Job completed"
- "Job worker error"
```

##### 4. Jobs Failing with Errors

**Symptoms**:

- Jobs move to ERROR status
- Error messages in job status

**Common Causes**:

1. Visual Crossing API errors
2. Invalid location parameters
3. Timeout errors
4. Rate limiting

**Solutions**:

1. **Check individual job error** (if job status endpoint available)

2. **Common error causes**:

   - **API errors**: Verify API key is valid and has quota
   - **Invalid location**: Check location name format
   - **Timeout**: Network or API response time issues
   - **Redis errors**: Check Redis connectivity

3. **Check deployment logs**:

```bash
# Look for:
❌ Error processing job
❌ Computation error
```

---

## Redis & Caching

### Redis Connection Refused

**Symptom**:

```
Error 111 connecting to localhost:6379. Connection refused.
```

**Causes**:

1. Using `localhost` instead of Railway's Redis service
2. Redis not started
3. Wrong REDIS_URL

**Solutions**:

1. **Use Railway's Redis service**:

```bash
# Correct (Railway):
REDIS_URL=${{Redis.REDIS_URL}}

# Wrong:
REDIS_URL=redis://localhost:6379
```

2. **Verify services are in same project**:

   - Private networking only works within a project
   - Check Railway dashboard shows both services

3. **Check Redis service logs**:
   - Look for "Ready to accept connections"
   - Check for error messages

### DNS Resolution Failed

**Symptom**:

```
Error -2 connecting to redis.railway.internal:6379. Name or service not known
```

**Cause**: API and Redis are in **different Railway projects**

**Solution**: Move both services to the **same Railway project**

### Cache Not Working

**Symptoms**:

- Slow response times
- High API usage
- Cache miss rate very high

**Solutions**:

1. **Verify Redis is running**:

```bash
curl https://your-app.com/test-redis
```

2. **Check cache configuration**:

```bash
CACHE_ENABLED=true  # Must be set
```

3. **Monitor cache stats** (if endpoint available):

```bash
curl https://your-app.com/cache-stats
```

4. **Check logs for cache errors**:

```bash
# Look for:
❌ Cache error
⚠️ Cache warning
```

### Cache Warming Errors

**Symptom**:

```
❌ CACHE WARMING FAILED: Error connecting to Redis
```

**Solution**: This is now handled gracefully. The app continues running. If you want cache warming:

1. Ensure Redis is properly configured
2. Check `REDIS_URL` variable
3. Verify Redis service is running
4. The warning `⚠️ Cache warming skipped - Redis not available` is normal without Redis

---

## Rate Limiting

### Getting 429 Too Many Requests

**Symptoms**:

```json
{
  "detail": "Rate limit exceeded: Too many unique locations. Maximum 10 unique locations per 1 hour(s).",
  "status_code": 429
}
```

**Solutions**:

1. **Use API_ACCESS_TOKEN for automated systems**:

```bash
# Service jobs bypass rate limiting
curl -H "Authorization: Bearer $API_ACCESS_TOKEN" \
     https://your-app.com/v1/records/daily/London/01-15
```

2. **Check your current status**:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.com/rate-limit-status
```

3. **Wait for rate limit window to reset**:

   - Default window: 1 hour
   - Check `Retry-After` header in 429 response

4. **Request IP whitelist** (if legitimate high-volume use):
   - Contact administrator to add your IP to whitelist
   - Set via `IP_WHITELIST` environment variable

### Rate Limiting Not Working

**Symptoms**:

- No rate limits being enforced
- Can make unlimited requests

**Solutions**:

1. **Check configuration**:

```bash
RATE_LIMIT_ENABLED=true  # Must be set
```

2. **Verify not using service token**:

   - `API_ACCESS_TOKEN` bypasses rate limiting (by design)
   - Use Firebase token to test rate limiting

3. **Check IP not whitelisted**:

```bash
# Check whitelist:
echo $IP_WHITELIST
```

---

## API Errors

### 401 Unauthorized

**Symptoms**:

```json
{
  "detail": "Unauthorized: Missing or invalid authorization token"
}
```

**Solutions**:

1. **Include Authorization header**:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.com/v1/records/daily/London/01-15
```

2. **Verify token is valid**:

   - Firebase ID token (for users)
   - API_ACCESS_TOKEN (for automated systems)

3. **Check token hasn't expired** (Firebase tokens expire after 1 hour)

### 422 Validation Error

**Symptoms**:

```json
{
  "error": "Validation Error",
  "message": "Validation failed: field required",
  "details": [...]
}
```

**Solutions**:

1. **Check request format** matches API documentation

2. **Verify required fields** are included

3. **Check data types** match expected formats

4. **Review error details** for specific field issues

### 404 Not Found

**Solutions**:

1. **Check endpoint URL** is correct

2. **Verify location name** format:

```bash
# Correct:
/v1/records/daily/New%20York/01-15  # URL-encoded
/v1/records/daily/London/01-15

# Wrong:
/v1/records/daily/New York/01-15  # Not URL-encoded
```

3. **Check identifier format**:

```bash
# Daily: MM-DD
/v1/records/daily/London/01-15  # Correct

# Not:
/v1/records/daily/London/2024-01-15  # Wrong format
```

### 500 Internal Server Error

**Symptoms**:

```json
{
  "detail": "Internal server error"
}
```

**Solutions**:

1. **Check deployment logs** for Python errors

2. **Common causes**:

   - Missing environment variables
   - API key issues (Visual Crossing, OpenWeather)
   - Redis connection problems
   - Invalid data from external API

3. **Verify configuration**:

```bash
# Check all required variables are set:
VISUAL_CROSSING_API_KEY
OPENWEATHER_API_KEY
API_ACCESS_TOKEN
REDIS_URL
```

---

## Performance Issues

### Slow Response Times

**Symptoms**:

- Responses take > 3 seconds
- Timeouts

**Solutions**:

1. **Check cache status**:

```bash
# Should see cache hits in logs
✅ Cache hit for key: ...
```

2. **Warm cache for popular locations**:

```bash
# If you have prewarm script:
python prewarm.py --locations 20 --days 7
```

3. **Check external API response times**:

   - Visual Crossing API may be slow
   - Network latency issues

4. **Use async job endpoints** for heavy computations:

```bash
# Instead of:
GET /v1/records/rolling-bundle/London/2024-01-15

# Use:
POST /v1/records/rolling-bundle/London/2024-01-15/async
# Then poll: GET /v1/jobs/{job_id}
```

### High Memory Usage

**Solutions**:

1. **Check for memory leaks** in logs

2. **Restart the application** to clear memory

3. **Monitor Redis memory usage**:

   - Large cache can consume memory
   - Set appropriate TTLs

4. **Reduce worker count** if needed:

```bash
# In start command:
uvicorn main:app --workers 1  # Instead of 2
```

---

## Debug Commands

### Health Checks

```bash
# Application health
curl https://your-app.com/health

# Redis connectivity
curl https://your-app.com/test-redis

# Rate limit status
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.com/rate-limit-status
```

### Testing Endpoints

```bash
# Simple GET
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.com/v1/records/daily/London/01-15

# Test caching (look for ETag header)
curl -v -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.com/v1/records/daily/London/01-15

# Test async jobs
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.com/v1/records/daily/London/01-15/async
```

### Log Monitoring

**Railway Dashboard**:

1. Go to your service
2. Click "Deployments" tab
3. Click on active deployment
4. View logs in real-time

**Search logs** for:

- `ERROR` - Error messages
- `WARNING` - Warning messages
- `Cache` - Cache-related events
- `Worker` - Background worker events
- `Job` - Job processing events

---

## Getting Help

If you've tried everything and still have issues:

### 1. Gather Information

```bash
# Health check
curl https://your-app.com/health > health.json

# Test Redis
curl https://your-app.com/test-redis > redis.json

# Rate limit status
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://your-app.com/rate-limit-status > ratelimit.json
```

### 2. Collect Logs

- Copy recent deployment logs from Railway
- Include error messages and stack traces
- Note the timestamp when issue occurred

### 3. Prepare Report

Include in your report:

- Exact error message
- Steps to reproduce
- Expected vs actual behavior
- Health check outputs
- Relevant log excerpts
- Environment (Railway, local, etc.)
- Time/date of occurrence

---

For deployment instructions, see `DEPLOYMENT.md`  
For API usage, see `README.md`  
For recent changes, see `CHANGELOG.md`
