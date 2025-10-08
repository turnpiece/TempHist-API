# Async Jobs Troubleshooting Guide

## Problem: Jobs Timing Out After 100 Polls

If you're seeing errors like:

```
Failed to fetch week data for London, England, United Kingdom:
Job record_computation_1759943112733_5c768768 timed out after 100 polls
```

This means the background worker is not processing jobs. Follow this guide to diagnose and fix the issue.

---

## Quick Diagnosis

### 1. Check Worker Status (API Endpoint)

```bash
curl https://your-api.com/v1/jobs/diagnostics/worker-status
```

This will show you:

- **Worker status**: Is the background worker alive?
- **Heartbeat**: When did the worker last update its heartbeat?
- **Queue length**: How many jobs are waiting?
- **Jobs by status**: How many jobs are pending, processing, ready, or errored?
- **Stuck jobs**: Jobs that have been pending/processing for > 5 minutes
- **Recommendations**: Specific actions to take based on the diagnosis

**Example healthy response:**

```json
{
  "worker": {
    "alive": true,
    "heartbeat": "2024-01-15T10:30:45.123456+00:00",
    "heartbeat_age_seconds": 5,
    "status": "healthy"
  },
  "queue": {
    "length": 0,
    "jobs_found": 0
  },
  "jobs": {
    "by_status": {
      "pending": 0,
      "processing": 0,
      "ready": 0,
      "error": 0
    },
    "stuck_count": 0,
    "stuck_jobs": []
  },
  "recommendations": [
    {
      "severity": "success",
      "issue": "System is healthy",
      "actions": ["No action needed"]
    }
  ]
}
```

**Example unhealthy response:**

```json
{
  "worker": {
    "alive": false,
    "heartbeat": null,
    "heartbeat_age_seconds": null,
    "status": "unhealthy"
  },
  "queue": {
    "length": 5,
    "jobs_found": 5
  },
  "jobs": {
    "by_status": {
      "pending": 5,
      "processing": 0,
      "ready": 0,
      "error": 0
    },
    "stuck_count": 0,
    "stuck_jobs": []
  },
  "recommendations": [
    {
      "severity": "critical",
      "issue": "Background worker is not running",
      "actions": [
        "Check server logs for worker startup errors",
        "Restart the API server",
        "Verify Redis connection is available"
      ]
    }
  ]
}
```

### 2. Check Worker Status (Command Line)

If you have access to the server, run:

```bash
cd /Users/paul/Sites/TempHist-api
python diagnose_jobs.py
```

This will show comprehensive diagnostics including:

- Redis connection status
- Job queue length
- All jobs with their details
- Jobs by status
- Stuck jobs
- Recommendations

---

## Common Issues and Solutions

### Issue 1: Worker Not Running

**Symptoms:**

- `worker.alive: false`
- No heartbeat
- Jobs stuck in PENDING state

**Causes:**

1. Background worker failed to start
2. Background worker crashed
3. Redis connection issue

**Solutions:**

1. **Check server logs for startup errors:**

   ```bash
   # Look for background worker startup messages
   grep -i "background worker" /path/to/logs
   grep -i "job worker" /path/to/logs

   # Look for specific error messages
   grep -i "failed to start background worker" /path/to/logs
   ```

2. **Check for Redis connection issues:**

   ```bash
   # Test Redis connection
   curl https://your-api.com/test-redis
   ```

3. **Restart the API server:**

   ```bash
   # This will restart the background worker
   # Method depends on your deployment platform

   # Render:
   # - Go to dashboard → Manual Deploy → Deploy Latest Commit

   # Local development:
   pkill -f "uvicorn main:app"
   uvicorn main:app --reload
   ```

### Issue 2: Worker Stuck or Frozen

**Symptoms:**

- `worker.alive: true` but heartbeat is stale (> 60 seconds old)
- Jobs stuck in PROCESSING state
- Worker logs show no activity

**Causes:**

1. Worker thread crashed but didn't exit
2. Worker is deadlocked
3. Long-running job blocking the worker

**Solutions:**

1. **Restart the API server** to restart the worker

2. **Check for stuck jobs:**

   ```bash
   python diagnose_jobs.py --clear-stuck
   ```

3. **Check server logs for the last job processed:**
   ```bash
   grep -i "processing job" /path/to/logs | tail -20
   grep -i "job completed" /path/to/logs | tail -20
   grep -i "job worker error" /path/to/logs | tail -20
   ```

### Issue 3: Jobs Failing with Errors

**Symptoms:**

- Jobs move to ERROR status
- `jobs.by_status.error > 0`

**Causes:**

1. API errors (Visual Crossing API issues)
2. Invalid parameters
3. Timeout errors
4. Rate limiting

**Solutions:**

1. **Check individual job errors:**

   ```bash
   # Get specific job status
   curl https://your-api.com/v1/jobs/{job_id}
   ```

2. **Common error causes:**

   - **Visual Crossing API errors**: Check if API key is valid, check rate limits
   - **Invalid location**: Verify location name format
   - **Timeout**: Increase timeout in `main.py` (HTTP_TIMEOUT)
   - **Redis errors**: Check Redis connection

3. **View recent errors in logs:**
   ```bash
   grep -i "error processing job" /path/to/logs | tail -20
   grep -i "computation error" /path/to/logs | tail -20
   ```

### Issue 4: Jobs Stuck for Extended Period

**Symptoms:**

- Jobs in PENDING or PROCESSING for > 5 minutes
- `stuck_count > 0`

**Causes:**

1. Worker not processing the queue
2. Job computation is very slow
3. Worker crashed mid-processing

**Solutions:**

1. **Clear stuck jobs:**

   ```bash
   python diagnose_jobs.py --clear-stuck
   ```

   This will mark jobs stuck for > 5 minutes as ERROR.

2. **Check if jobs are actually processing:**

   ```bash
   # Watch diagnostics in real-time
   watch -n 2 'python diagnose_jobs.py'
   ```

3. **If worker is healthy but jobs are slow**, check:
   - Visual Crossing API response times
   - Redis latency
   - Server CPU/memory

---

## Diagnostic Tools

### 1. API Endpoint: `/v1/jobs/diagnostics/worker-status`

**Purpose**: Quick health check from anywhere  
**Authentication**: Public (no auth required)  
**Usage**:

```bash
curl https://your-api.com/v1/jobs/diagnostics/worker-status
```

### 2. Command-Line Tool: `diagnose_jobs.py`

**Purpose**: Comprehensive diagnostics with server access  
**Location**: `/Users/paul/Sites/TempHist-api/diagnose_jobs.py`

**Commands:**

```bash
# Run full diagnostics
python diagnose_jobs.py

# Clear stuck jobs (>5 minutes old)
python diagnose_jobs.py --clear-stuck

# Clear ALL jobs (use with caution!)
python diagnose_jobs.py --clear-all

# Watch diagnostics in real-time
watch -n 2 'python diagnose_jobs.py'
```

---

## Enhanced Logging

The background worker now includes enhanced logging to help track down issues:

### Startup Logs

```
🚀 Background worker thread started
📊 Thread info: Thread-1, daemon=True, alive=True
🚀 Background worker main loop started
🔄 Recreated Redis client with decode_responses=True
✅ Worker Redis connection verified
💓 Worker heartbeat initialized
✅ JobWorker instance created
🚀 Job worker started
```

### Job Processing Logs

```
📋 Found 1 pending jobs
🔄 Processing job: record_computation_1234567890_abc123
📋 Job type: record_computation, Params: {...}
🔢 Starting record computation...
✅ Job completed: record_computation_1234567890_abc123 (took 2.34s)
```

### Heartbeat Logs

```
💓 Heartbeat updated (poll #10)
💓 Heartbeat updated (poll #20)
```

### Error Logs

```
❌ Job worker error: [error message]
❌ Traceback: [full stack trace]
❌ Computation error for job [...]: [error]
❌ Error processing job [...]: [error]
```

---

## Server Log Locations

### Local Development

```bash
# Console output or
./temphist.log
```

### Render (Production)

1. Go to Render Dashboard
2. Select your service
3. Click "Logs" tab
4. Search for keywords: "background worker", "job worker", "error"

---

## Prevention / Best Practices

1. **Monitor worker health**: Set up alerts for `/v1/jobs/diagnostics/worker-status`
2. **Regular log checks**: Review logs weekly for worker errors
3. **Clear old jobs**: Run `diagnose_jobs.py --clear-stuck` periodically
4. **Test after deploys**: Always check worker status after deploying
5. **Redis monitoring**: Ensure Redis is healthy and accessible

---

## What Happens When Jobs Time Out

When a client times out after 100 polls:

1. The job **remains in the queue** (unless processed)
2. The worker **will still process it** when it gets to it
3. The client gives up waiting, but the computation continues
4. Result is cached when computation completes

**Action**: Even if the client times out, check if the job eventually completes. The data may be cached for future requests.

---

## Debugging Checklist

When jobs timeout, check in this order:

- [ ] Check worker status endpoint
- [ ] Check worker heartbeat age
- [ ] Check queue length
- [ ] Check jobs by status (pending, processing, error)
- [ ] Check for stuck jobs
- [ ] Review server logs for errors
- [ ] Test Redis connection
- [ ] Check Visual Crossing API status
- [ ] Verify environment variables (API keys, Redis URL)
- [ ] Check server resources (CPU, memory)

---

## Getting Help

If you've tried everything and jobs are still timing out:

1. **Gather diagnostics:**

   ```bash
   python diagnose_jobs.py > diagnostics.txt
   curl https://your-api.com/v1/jobs/diagnostics/worker-status > worker_status.json
   ```

2. **Collect logs:**

   ```bash
   grep -i "background worker\|job worker\|error" /path/to/logs > worker_logs.txt
   ```

3. **Check specific job:**

   ```bash
   curl https://your-api.com/v1/jobs/{job_id} > job_status.json
   ```

4. **Include in your report:**
   - Diagnostics output
   - Worker status JSON
   - Relevant logs
   - Job ID that timed out
   - Timestamp when issue occurred
