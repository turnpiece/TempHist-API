# Cloudflare Optimization Guide for TempHist API

This document provides comprehensive guidance for optimizing the TempHist API for Cloudflare edge caching and performance.

## Overview

The TempHist API has been enhanced with Cloudflare-friendly caching to achieve:

- **Fast response times**: <500ms for warm cache, <3s for cold cache
- **High cache hit rates**: 80%+ for popular locations
- **Reduced origin load**: 90%+ requests served from edge cache
- **Resilient performance**: Graceful degradation under load

## Cache Configuration

### Cache Headers

The API now emits strong cache headers optimized for Cloudflare:

```
Cache-Control: public, max-age=3600, s-maxage=86400, stale-while-revalidate=604800
ETag: "abc123def456"
Last-Modified: Mon, 15 Jan 2024 12:00:00 GMT
Vary: Accept-Encoding
```

**Header Breakdown:**

- `max-age=3600`: Browser cache for 1 hour
- `s-maxage=86400`: Cloudflare cache for 24 hours
- `stale-while-revalidate=604800`: Serve stale content for 7 days while revalidating
- `ETag`: Strong validation for 304 Not Modified responses
- `Last-Modified`: HTTP/1.1 conditional request support

### Cache Key Normalization

Cache keys are canonicalized to maximize hit rates:

- Query parameters sorted alphabetically
- Coordinates rounded to 4 decimal places (~11m precision)
- Default values stripped (unit_group=celsius, month_mode=rolling1m)
- Location names normalized to lowercase

Example:

```
Input: /v1/records/daily/New%20York%2C%20NY/01-15?unit_group=celsius&month_mode=rolling1m
Cache Key: temphist:v1/records:location=new_york,_ny:period=daily:identifier=01-15
```

## Cloudflare Rules Configuration

### Cache Rules

Configure these rules in Cloudflare Dashboard > Rules > Cache Rules:

#### 1. Cache API Endpoints

```
Rule: Cache GET requests to API endpoints
Expression: (http.request.method eq "GET") and (http.host contains "api.temphist.com")
Action: Cache Level = Cache Everything
Edge TTL: 1 day
Browser TTL: 1 hour
```

#### 2. Bypass Cache for Jobs

```
Rule: Bypass cache for job endpoints
Expression: (http.request.uri contains "/v1/jobs/") or (http.request.uri contains "/async")
Action: Cache Level = Bypass
```

#### 3. Bypass Cache for Admin

```
Rule: Bypass cache for admin endpoints
Expression: (http.request.uri contains "/admin/") or (http.request.uri contains "/analytics")
Action: Cache Level = Bypass
```

#### 4. Respect Origin Cache Headers

```
Rule: Respect origin cache headers
Expression: (http.request.method eq "GET") and (http.host contains "api.temphist.com")
Action: Cache Level = Respect Existing Headers
```

### Page Rules (Alternative to Cache Rules)

If using Page Rules instead of Cache Rules:

```
Rule 1: Cache API endpoints
URL: api.temphist.com/v1/records/*
Settings: Cache Level = Cache Everything, Edge Cache TTL = 1 day

Rule 2: Bypass job endpoints
URL: api.temphist.com/v1/jobs/*
Settings: Cache Level = Bypass

Rule 3: Bypass async endpoints
URL: api.temphist.com/*/async
Settings: Cache Level = Bypass
```

## Performance Optimizations

### 1. Enable Brotli Compression

In Cloudflare Dashboard > Speed > Optimization:

- Enable Brotli compression
- Enable Gzip compression (fallback)

### 2. HTTP/2 and HTTP/3

Enable in Cloudflare Dashboard > Network:

- HTTP/2: Enabled by default
- HTTP/3 (QUIC): Enable for faster connections

### 3. Browser Cache TTL

Set appropriate browser cache TTLs:

- Static endpoints: 1 hour
- Dynamic endpoints: 15 minutes
- Job endpoints: No cache

### 4. Edge Cache TTL

Configure edge cache TTLs based on data freshness:

- Historical data: 24 hours (rarely changes)
- Recent data: 1 hour (may update)
- Forecast data: 15 minutes (changes frequently)

## Security and Rate Limiting

### Rate Limiting Rules

Configure in Cloudflare Dashboard > Security > WAF > Rate Limiting Rules:

#### 1. API Rate Limiting

```
Rule: Limit API requests per IP
Expression: (http.request.uri contains "/v1/records/")
Rate: 60 requests per minute
Action: Block for 10 minutes
```

#### 2. Job Endpoint Limiting

```
Rule: Limit job creation requests
Expression: (http.request.uri contains "/async")
Rate: 10 requests per minute
Action: Block for 5 minutes
```

#### 3. Burst Protection

```
Rule: Burst protection
Expression: (http.request.uri contains "/v1/")
Rate: 200 requests per minute
Action: Block for 1 hour
```

### IP Whitelisting

For high-volume clients, configure IP whitelisting:

```
Rule: Whitelist trusted IPs
Expression: (ip.src in {"1.2.3.4" "5.6.7.8"})
Action: Skip all rate limiting
```

## Monitoring and Analytics

### Key Metrics to Monitor

1. **Cache Hit Rate**

   - Target: >80% for popular endpoints
   - Monitor: Cloudflare Analytics > Caching

2. **Response Times**

   - Target: <500ms p95 for cached responses
   - Monitor: Cloudflare Analytics > Performance

3. **Origin Load**

   - Target: <10% of total requests hit origin
   - Monitor: Origin request rate vs total requests

4. **Error Rates**
   - Target: <1% 5xx errors
   - Monitor: Cloudflare Analytics > Security

### Cache Warming Strategy

Run the prewarm script regularly:

```bash
# Daily prewarming for popular locations
python prewarm.py --locations 20 --days 7

# Hourly prewarming for today's data
python prewarm.py --locations 10 --days 1 --endpoints v1/records/daily
```

Schedule with cron:

```bash
# Daily at 6 AM UTC
0 6 * * * cd /path/to/api && python prewarm.py --locations 20 --days 7

# Hourly for recent data
0 * * * * cd /path/to/api && python prewarm.py --locations 10 --days 1
```

## Async Job Processing

For heavy computations, use the async job endpoints:

### 1. Create Job

```bash
POST /v1/records/daily/New%20York/01-15/async
```

Response:

```json
{
  "job_id": "record_computation_1642234567890_abc12345",
  "status": "pending",
  "retry_after": 3,
  "status_url": "/v1/jobs/record_computation_1642234567890_abc12345"
}
```

### 2. Check Job Status

```bash
GET /v1/jobs/record_computation_1642234567890_abc12345
```

Response:

```json
{
  "id": "record_computation_1642234567890_abc12345",
  "status": "ready",
  "result": {
    "cache_key": "temphist:v1/records:...",
    "etag": "\"def456\"",
    "computed_at": "2024-01-15T12:00:00Z"
  }
}
```

### 3. Fetch Cached Result

Once job is ready, the result is cached and can be fetched via the regular GET endpoint.

## Troubleshooting

### Common Issues

#### 1. Low Cache Hit Rate

- Check cache rules configuration
- Verify cache headers are being sent
- Monitor cache key normalization
- Review prewarming script execution

#### 2. High Origin Load

- Increase cache TTL for stable data
- Improve prewarming coverage
- Check for cache-busting parameters
- Monitor cache invalidation patterns

#### 3. Slow Response Times

- Enable Brotli compression
- Check origin server performance
- Monitor cache miss rates
- Review job queue processing

#### 4. Cache Invalidation Issues

- Use cache invalidation API endpoints
- Monitor cache key patterns
- Check TTL settings
- Review data freshness requirements

### Debug Commands

```bash
# Check cache status
curl -I "https://api.temphist.com/v1/records/daily/New%20York/01-15"

# Test ETag validation
curl -H "If-None-Match: \"abc123\"" "https://api.temphist.com/v1/records/daily/New%20York/01-15"

# Check cache statistics
curl "https://api.temphist.com/cache-stats"

# Test job creation
curl -X POST "https://api.temphist.com/v1/records/daily/New%20York/01-15/async"
```

## Performance Targets

### Response Time Goals

- **Warm cache**: <200ms p95
- **Cold cache**: <2s p95
- **Job creation**: <100ms p95
- **Job status check**: <50ms p95

### Cache Efficiency Goals

- **Hit rate**: >80% overall
- **Popular locations**: >95% hit rate
- **Origin load**: <10% of total requests
- **Cache size**: <1GB total

### Reliability Goals

- **Uptime**: >99.9%
- **Error rate**: <0.1%
- **Job success rate**: >99%
- **Cache consistency**: 100%

## Best Practices

1. **Always use HTTPS** for API requests
2. **Implement retry logic** with exponential backoff
3. **Use job endpoints** for heavy computations
4. **Monitor cache metrics** regularly
5. **Prefer GET requests** for data retrieval
6. **Use conditional requests** with ETags
7. **Implement circuit breakers** for resilience
8. **Cache at multiple levels** (browser, CDN, application)

## Migration Guide

### From Existing API

1. **Update client code** to handle 304 responses
2. **Implement job polling** for heavy operations
3. **Add retry logic** with appropriate delays
4. **Monitor cache hit rates** and adjust TTLs
5. **Update documentation** with new endpoints

### Testing Strategy

1. **Load testing** with realistic traffic patterns
2. **Cache warming** before production deployment
3. **A/B testing** for cache configuration
4. **Monitoring** during and after deployment
5. **Rollback plan** if issues arise

This optimization guide ensures the TempHist API delivers fast, reliable, and cost-effective performance through Cloudflare's global network.
