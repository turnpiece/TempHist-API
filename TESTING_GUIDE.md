# Testing Guide for Rate Limiting

This guide will help you thoroughly test your rate limiting system before deploying to production. Testing is crucial to ensure your API protection works correctly and doesn't interfere with legitimate users.

## 📁 **Test Organization**

All tests are consolidated in your existing `test_main.py` file:

- **`TestLocationDiversityMonitor`** - Tests location diversity limits
- **`TestRequestRateMonitor`** - Tests request rate limits
- **`TestRateLimitingIntegration`** - Tests FastAPI integration
- **`TestClientIPDetection`** - Tests IP address detection
- **`TestRateLimitingPerformance`** - Tests performance under load
- **`TestRateLimitingEdgeCases`** - Tests unusual scenarios
- **`TestRateLimitingManual`** - Interactive testing helpers

## 🚀 **Quick Start Testing**

### 1. **Run All Tests (Recommended)**

```bash
# Run all tests including rate limiting, weather data, and more
pytest test_main.py -v

# Run only rate limiting related tests
pytest test_main.py -v -k "rate"

# Run specific test classes
pytest test_main.py::TestLocationDiversityMonitor -v
pytest test_main.py::TestRequestRateMonitor -v
pytest test_main.py::TestRateLimitingIntegration -v
```

### 2. **Quick Test Categories**

```bash
# Unit tests only
pytest test_main.py::TestLocationDiversityMonitor test_main.py::TestRequestRateMonitor -v

# Integration tests only
pytest test_main.py::TestRateLimitingIntegration -v

# Performance tests only
pytest test_main.py::TestRateLimitingPerformance -v

# Edge case tests only
pytest test_main.py::TestRateLimitingEdgeCases -v
```

### 2. **Manual Testing with pytest**

```bash
# Start your app locally
uvicorn main:app --reload

# In another terminal, run manual tests
pytest test_main.py::TestRateLimitingManual -v -s
```

### 3. **Performance Testing**

```bash
# Run all tests including rate limiting
pytest test_main.py -v

# Run only rate limiting tests
pytest test_main.py -v -k "rate"
```

## 🧪 **Testing Strategy**

### **Phase 1: Unit Testing**

- Test individual rate limiting classes (`LocationDiversityMonitor`, `RequestRateMonitor`)
- Verify logic for location diversity and request counting
- Test edge cases and error handling
- Test client IP detection logic

### **Phase 2: Integration Testing**

- Test rate limiting with actual FastAPI endpoints
- Verify middleware integration and authentication
- Test rate limit responses (HTTP 429) and headers
- Test monitoring endpoints (`/rate-limit-status`, `/rate-limit-stats`)

### **Phase 3: Performance Testing**

- Test rate limiting performance under load
- Verify memory usage and cleanup mechanisms
- Test concurrent access safety
- Measure response time impact

### **Phase 4: Manual Testing**

- Use pytest with `-s` flag for interactive testing
- Test with different clients (browser, Postman, curl)
- Verify rate limit recovery after time windows
- Test configuration changes and edge cases

## 📋 **Test Scenarios**

### **Scenario 1: Location Diversity Limit**

**Goal**: Verify that users can't request data for too many different locations

**Test Steps**:

1. Make requests to 5 different locations (should succeed)
2. Make requests to 5 more locations (should succeed)
3. Make request to 11th location (should be rate limited with 429 status)

**Expected Result**:

- First 10 locations: HTTP 200
- 11th location: HTTP 429 with "Location diversity limit exceeded"

**Test Command**:

```bash
# Test location diversity limit
pytest test_main.py::TestLocationDiversityMonitor::test_check_location_diversity_over_limit -v

# Or run all location diversity tests
pytest test_main.py::TestLocationDiversityMonitor -v
```

### **Scenario 2: Request Rate Limit**

**Goal**: Verify that users can't make too many requests overall

**Test Steps**:

1. Make 50 requests rapidly (should succeed)
2. Make 50 more requests (should succeed)
3. Make 101st request (should be rate limited)

**Expected Result**:

- First 100 requests: HTTP 200
- 101st request: HTTP 429 with "Rate limit exceeded"

**Test Command**:

```bash
# Test request rate limit
pytest test_main.py::TestRequestRateMonitor::test_check_request_rate_over_limit -v

# Or run all request rate tests
pytest test_main.py::TestRequestRateMonitor -v
```

### **Scenario 3: Rate Limit Recovery**

**Goal**: Verify that rate limits reset after the time window

**Test Steps**:

1. Hit a rate limit (get 429 response)
2. Wait for the time window to expire
3. Make a new request (should succeed)

**Expected Result**:

- After waiting: HTTP 200
- Rate limit status should show reset counters

**Test Command**:

```bash
# Check current status
curl http://localhost:8000/rate-limit-status

# Wait for window to expire, then check again
curl http://localhost:8000/rate-limit-status
```

### **Scenario 4: Concurrent Requests**

**Goal**: Verify rate limiting works with multiple simultaneous requests

**Test Steps**:

1. Start 10 concurrent workers
2. Each worker makes requests at 5 req/sec
3. Monitor total rate and rate limiting

**Expected Result**:

- Total rate should be limited to configured maximum
- Some requests should get 429 responses
- System should remain stable

**Test Command**:

```bash
# Test performance under load
pytest test_main.py::TestRateLimitingPerformance::test_rapid_requests_performance -v

# Or run all performance tests
pytest test_main.py::TestRateLimitingPerformance -v
```

## 🔧 **Configuration Testing**

### **Test Different Rate Limit Settings**

```bash
# Test with very strict limits
export MAX_LOCATIONS_PER_HOUR=3
export MAX_REQUESTS_PER_HOUR=20
export RATE_LIMIT_WINDOW_HOURS=1

# Restart your app and test
python test_rate_limiting.py
```

### **Test Rate Limiting Disabled**

```bash
# Disable rate limiting
export RATE_LIMIT_ENABLED=false

# Restart your app and verify no rate limiting occurs
python test_rate_limiting.py
```

## 📊 **Monitoring During Tests**

### **Real-time Status Monitoring**

```bash
# Watch rate limiting status in real-time
watch -n 1 'curl -s http://localhost:8000/rate-limit-status | jq'

# Monitor rate limiting stats
watch -n 1 'curl -s http://localhost:8000/rate-limit-stats | jq'
```

### **Log Monitoring**

```bash
# Watch your app logs for rate limiting events
tail -f temphist.log | grep -i "rate\|limit\|429"

# Or if using uvicorn
uvicorn main:app --reload --log-level debug
```

## 🚨 **What to Look For**

### **✅ Success Indicators**

- Rate limiting blocks excessive requests (HTTP 429)
- Rate limiting allows legitimate requests (HTTP 200)
- Rate limit headers are present (`Retry-After`)
- Error messages are clear and helpful
- System performance remains stable under load

### **❌ Problem Indicators**

- Rate limiting doesn't work (all requests succeed)
- Rate limiting is too aggressive (legitimate requests blocked)
- System crashes under load
- Memory usage grows indefinitely
- Response times degrade significantly

### **⚠️ Warning Signs**

- High error rates (>10%)
- Memory usage growing over time
- Response times increasing with load
- Rate limiting not triggering when expected

## 🔍 **Debugging Common Issues**

### **Issue: Rate Limiting Not Working**

**Possible Causes**:

- Environment variables not set correctly
- Rate limiting disabled in configuration
- Middleware not properly integrated

**Debug Steps**:

1. Check `/rate-limit-status` endpoint
2. Verify environment variables
3. Check app logs for initialization messages
4. Ensure middleware is active

### **Issue: Too Many False Positives**

**Possible Causes**:

- Limits set too low
- Time window too short
- IP detection not working correctly

**Debug Steps**:

1. Check client IP detection
2. Adjust rate limit values
3. Verify time window settings
4. Test with different clients

### **Issue: Performance Degradation**

**Possible Causes**:

- Memory leaks in rate limiting
- Inefficient data structures
- Cleanup not working properly

**Debug Steps**:

1. Monitor memory usage during tests
2. Check cleanup intervals
3. Profile rate limiting functions
4. Test with smaller time windows

## 📈 **Performance Benchmarks**

### **Expected Performance**

- **Response Time**: <100ms for rate limit checks
- **Memory Usage**: <100MB for 1000+ IPs
- **Throughput**: Handle 1000+ req/sec without degradation
- **Cleanup**: <1 second for full cleanup cycle

### **Performance Testing Targets**

```bash
# Light Load (Development)
pytest test_main.py::TestRateLimitingPerformance::test_rapid_requests_performance -v

# Medium Load (Staging)
pytest test_main.py::TestRateLimitingPerformance::test_memory_usage_cleanup -v

# Heavy Load (Production Simulation)
pytest test_main.py::TestRateLimitingEdgeCases::test_concurrent_access_safety -v

# All Performance Tests
pytest test_main.py::TestRateLimitingPerformance -v
```

## 🚀 **Production Readiness Checklist**

Before deploying to production, ensure:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Load tests show acceptable performance
- [ ] Rate limiting triggers at expected thresholds
- [ ] Rate limiting recovers after time windows
- [ ] Error messages are user-friendly
- [ ] Monitoring endpoints work correctly
- [ ] Memory usage is stable under load
- [ ] Response times remain acceptable
- [ ] Rate limit headers are properly set

## 🆘 **Getting Help**

If you encounter issues:

1. **Check the logs**: Look for error messages and rate limiting events
2. **Verify configuration**: Ensure environment variables are set correctly
3. **Test incrementally**: Start with basic tests before moving to load tests
4. **Check dependencies**: Ensure Redis and other services are running
5. **Review the code**: Check the rate limiting implementation in `main.py`

## 📚 **Additional Resources**

- **API Documentation**: Visit `http://localhost:8000/docs` when running locally
- **Rate Limit Status**: `/rate-limit-status` for current client status
- **Rate Limit Stats**: `/rate-limit-stats` for admin monitoring
- **All Tests**: Everything is consolidated in your existing `test_main.py`
- **Configuration**: See the README.md for rate limiting environment variables
- **Test Categories**: Use pytest filters to run specific test types

---

**Remember**: Thorough testing now will save you from production issues later. Take the time to validate your rate limiting system works correctly under various conditions!
