# Visual Crossing API Cost Optimization

## Problem

Visual Crossing API charges per **record/day retrieved**, not per API call. When fetching historical data using the `/timeline` endpoint with date ranges:

- **50 years of data** = ~18,250 records per yearly query
- **Monthly queries** = 50 years × 30 days = ~1,500 records
- Cost accumulates quickly with frequent queries

The cost increased significantly after switching from `historysummary` to `timeline` endpoints because:

- Timeline endpoints return raw daily data (one record per day)
- Historysummary likely returned aggregated data (fewer records)

## Solutions Implemented

### 1. Reduced Default Years Back (50 → 20)

**Impact**: ~60% reduction in records fetched per query

- **Before**: 50 years = ~18,250 records per yearly query
- **After**: 20 years = ~7,300 records per yearly query
- **Savings**: ~11,000 records per yearly query

**Configuration**:

```bash
# Reduce years back (default: 20)
YEARS_BACK=10  # Even more conservative (10 years)

# Or increase if needed (not recommended due to cost)
YEARS_BACK=30
```

### 2. Extended Cache TTL for Historical Data

Historical data doesn't change, so we extended cache from 1 day to **7 days**:

- Reduces redundant API calls for the same data
- Expensive queries are cached and reused
- Cache logging now highlights expensive queries

### 3. Cost-Aware Logging

Added warnings for expensive queries:

- Queries fetching >1,000 days now log cost estimates
- Cache hits are highlighted for expensive queries
- Helps identify patterns causing high costs

### 4. Improved Cache Usage

The improved cache system now:

- Canonicalizes locations (reduces duplicate queries)
- Provides temporal tolerance (reuses nearby dates)
- Better cache hit rates reduce API calls

## Cost Estimation

Visual Crossing pricing (as of 2024):

- Each record/day retrieved = cost unit
- Example: 20 years = 7,300 records per yearly query

**Monthly Cost Example**:

- 100 yearly queries/day × 7,300 records = 730,000 records/day
- 730,000 × 30 days = 21.9M records/month

**With 20 years vs 50 years**:

- 50 years: ~10.0M records/month (your current usage)
- 20 years: ~4.0M records/month (60% reduction)
- Potential savings: **6M records/month**

## Configuration

Set these environment variables to control costs:

```bash
# Reduce years of historical data (default: 20)
YEARS_BACK=10           # Very conservative (10 years)
YEARS_BACK=20           # Balanced (default, recommended)
YEARS_BACK=30           # More data (higher cost)

# Request throttling
VC_MAX_CONCURRENT_REQUESTS=1    # Limit concurrent requests
VC_MIN_REQUEST_INTERVAL=1.0     # Minimum seconds between requests
```

## Best Practices

1. **Use caching effectively**: The improved cache system reduces duplicate API calls
2. **Monitor logs**: Look for "⚠️ EXPENSIVE QUERY" warnings
3. **Consider use cases**: Do you really need 50 years? 20 years covers most climate analysis
4. **Batch requests**: Use rolling-bundle endpoint for multiple periods in one call
5. **Cache warm-up**: Pre-fetch common queries during off-peak hours

## Monitoring

Check logs for:

- `⚠️ EXPENSIVE QUERY` - Queries fetching >1,000 days
- `💾 CACHED EXPENSIVE QUERY` - Successful cache saves for expensive data
- `Using cached data` - Cache hits (good, saves money!)

## Next Steps

If costs are still too high:

1. **Further reduce YEARS_BACK**: Set to 10 or 15 years
2. **Increase cache TTL**: Extend historical cache beyond 7 days (30+ days)
3. **Rate limiting**: Increase `VC_MIN_REQUEST_INTERVAL` to 2-3 seconds
4. **Consider Visual Crossing plan upgrade**: Higher tier plans may have better pricing
5. **Contact Visual Crossing support**: Ask about bulk/aggregated endpoints for historical data

## Migration Notes

After applying these changes:

1. **Existing cache**: Will expire and repopulate with new year ranges
2. **API responses**: Will contain fewer years (20 instead of 50) unless configured otherwise
3. **Cost reduction**: Immediate for new queries, gradual as cache expires

## Additional Optimizations

If still hitting cost limits:

- **Limit concurrent users**: Reduce server load during peak times
- **Cache warming**: Pre-populate cache for common queries
- **Query optimization**: Use more specific date ranges instead of full year spans
- **Consider alternatives**: Evaluate other weather data providers for specific use cases
