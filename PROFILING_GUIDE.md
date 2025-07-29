# TempHist API Profiling Guide

This guide covers various approaches to profile your FastAPI application and identify performance bottlenecks.

## 🔍 **Profiling Results Summary**

Based on our profiling analysis:

### **Performance Metrics:**

- **URL Building**: ~2.5M operations/second (microseconds per call)
- **Historical Average**: ~58K operations/second (17 microseconds per call)
- **Trend Slope**: ~23K operations/second (44 microseconds per call)
- **Memory Usage**: Minimal impact (0 bytes difference)

### **Key Findings:**

✅ **URL building is extremely fast** - no optimization needed  
✅ **Calculation functions are efficient** for typical dataset sizes  
✅ **Memory usage is minimal** - no memory leaks detected  
✅ **Caching is already implemented** - good for performance

## 🛠️ **Profiling Tools & Methods**

### **1. Py-spy (System-level Profiling)**

**Installation:**

```bash
pip install py-spy
```

**Usage Options:**

#### **A. Real-time CPU Monitoring (Requires Root on macOS)**

```bash
# Terminal 1: Start your app
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: Monitor CPU usage (requires sudo on macOS)
sudo py-spy top -- python main.py
```

#### **B. Generate Flame Graph (Most Detailed)**

```bash
# Record profiling data
py-spy record --format flamegraph --output profile_flamegraph.svg -- python main.py

# View the flame graph in browser
open profile_flamegraph.svg
```

#### **C. Generate Speedscope Profile**

```bash
# Generate speedscope format
py-spy record --format speedscope --output profile.speedscope -- python main.py

# View online at https://www.speedscope.app/
```

### **2. Python Built-in Profiling**

#### **A. Built-in Performance Testing**

```bash
# Run the included performance test script
python performance_test.py
```

#### **B. cProfile (CPU Profiling)**

```bash
# Profile specific function
python -c "
import cProfile
import pstats
from main import calculate_historical_average

profiler = cProfile.Profile()
profiler.enable()
calculate_historical_average([{'x': 2020, 'y': 15.5}, {'x': 2021, 'y': 16.2}])
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
"
```

#### **B. Memory Profiling**

```bash
pip install memory-profiler

# Add @profile decorator to functions you want to profile
# Then run:
python -m memory_profiler main.py
```

### **3. FastAPI-specific Profiling**

#### **A. Request Timing Middleware**

```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

#### **B. Database Query Profiling**

```python
# Add to your Redis operations
import time

def get_cache(cache_key):
    start_time = time.time()
    result = redis_client.get(cache_key)
    process_time = time.time() - start_time
    print(f"Cache lookup took: {process_time:.4f}s")
    return result
```

### **4. Load Testing**

#### **A. Using Apache Bench (ab)**

```bash
# Install Apache Bench
brew install httpd

# Test your API endpoints
ab -n 1000 -c 10 http://localhost:8000/data/London/01-15
```

#### **B. Using wrk (High-performance HTTP benchmarking)**

```bash
# Install wrk
brew install wrk

# Test with multiple threads
wrk -t12 -c400 -d30s http://localhost:8000/data/London/01-15
```

## 📊 **Performance Monitoring**

### **1. Application Metrics**

```python
# Add to your FastAPI app
from prometheus_client import Counter, Histogram
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQUEST_COUNT.inc()
    start_time = time.time()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response
```

### **2. Redis Performance Monitoring**

```python
# Monitor Redis performance
def monitor_redis_performance():
    info = redis_client.info()
    print(f"Redis memory usage: {info['used_memory_human']}")
    print(f"Redis hit rate: {info['keyspace_hits']}/{info['keyspace_misses']}")
```

## 🚀 **Performance Optimization Strategies**

### **1. Caching Improvements**

```python
# Implement more aggressive caching
CACHE_DURATION_MAPPING = {
    'today': timedelta(minutes=30),
    'recent': timedelta(hours=2),
    'historical': timedelta(days=7)
}
```

### **2. Database Optimization**

```python
# Use connection pooling for Redis
import redis
from redis.connection import ConnectionPool

pool = ConnectionPool.from_url(REDIS_URL, max_connections=20)
redis_client = redis.Redis(connection_pool=pool)
```

### **3. Async Improvements**

```python
# Increase concurrency for batch operations
async def fetch_weather_batch(location: str, date_strs: list, max_concurrent: int = 10):
    # Increase from 1 to 10 for better performance
```

### **4. Response Optimization**

```python
# Compress responses
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## 📈 **Monitoring Dashboard**

### **1. Grafana Dashboard**

```yaml
# docker-compose.yml
version: "3.8"
services:
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
```

### **2. Health Check Endpoint**

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "redis": redis_client.ping(),
        "cache_hit_rate": get_cache_hit_rate(),
        "memory_usage": get_memory_usage()
    }
```

## 🔧 **Troubleshooting Common Issues**

### **1. High Memory Usage**

```python
# Monitor memory usage
import psutil
import gc

def check_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

    # Force garbage collection
    gc.collect()
```

### **2. Slow API Responses**

```python
# Add request timing
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    if process_time > 1.0:  # Log slow requests
        print(f"Slow request: {request.url.path} took {process_time:.2f}s")

    return response
```

### **3. Redis Connection Issues**

```python
# Implement Redis connection retry
import redis
from redis.exceptions import ConnectionError

def get_cache_with_retry(cache_key, max_retries=3):
    for attempt in range(max_retries):
        try:
            return redis_client.get(cache_key)
        except ConnectionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
```

## 📋 **Performance Checklist**

- [ ] **Caching**: Implement Redis caching for API responses
- [ ] **Async Operations**: Use async/await for I/O operations
- [ ] **Connection Pooling**: Configure Redis connection pooling
- [ ] **Monitoring**: Set up metrics collection
- [ ] **Load Testing**: Test with realistic load
- [ ] **Memory Profiling**: Check for memory leaks
- [ ] **Database Optimization**: Optimize Redis queries
- [ ] **Response Compression**: Enable gzip compression
- [ ] **Health Checks**: Implement health check endpoints
- [ ] **Error Handling**: Proper error handling and logging

## 🎯 **Next Steps**

1. **Set up continuous monitoring** with Prometheus/Grafana
2. **Implement load testing** in your CI/CD pipeline
3. **Add performance alerts** for slow responses
4. **Optimize based on real usage patterns**
5. **Consider horizontal scaling** if needed

---

**Note**: The profiling results show your application is already well-optimized. Focus on monitoring and scaling rather than premature optimization.
