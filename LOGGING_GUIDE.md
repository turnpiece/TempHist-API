# TempHist API Logging Guide

This guide explains the logging setup and best practices for the TempHist API.

## 🔧 **Logging Configuration**

### **Current Setup**

The API uses Python's built-in `logging` module with the following configuration:

```python
# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('temphist.log') if DEBUG else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)
```

### **Log Levels Used**

- **DEBUG**: Detailed information for debugging (cache hits/misses, API calls)
- **INFO**: General information about application flow
- **WARNING**: Issues that don't stop execution (invalid locations, missing data)
- **ERROR**: Serious problems that need attention (API failures, data processing errors)

## 📊 **Log Output Examples**

### **Debug Mode (DEBUG=true)**

```
2025-07-29 17:53:41,126 - main - DEBUG - get_weather_for_date for London on 2024-01-15
2025-07-29 17:53:41,127 - main - DEBUG - Cache miss: london_2024-01-15 — fetching from API
2025-07-29 17:53:41,128 - main - DEBUG - Cache hit: london_2024-01-15
2025-07-29 17:53:41,129 - main - INFO - Request processed successfully
```

### **Production Mode (DEBUG=false)**

```
2025-07-29 17:53:41,126 - main - INFO - Request processed successfully
2025-07-29 17:53:41,127 - main - WARNING - Invalid location: InvalidCity
2025-07-29 17:53:41,128 - main - ERROR - API request failed: Connection timeout
```

## 🎯 **Benefits of Using Logging Module**

### **1. Structured Output**

- Timestamps for all log entries
- Log levels for filtering
- Module name for context
- Consistent formatting

### **2. Performance Benefits**

- **Lazy evaluation**: Debug messages only processed when DEBUG is enabled
- **Configurable levels**: Can disable debug logs in production
- **File output**: Logs saved to `temphist.log` in debug mode

### **3. Production Ready**

- **No debug overhead** in production
- **Error tracking** for monitoring
- **Warning detection** for issues
- **Structured format** for log parsing

## 🔍 **Logging Usage Examples**

### **Cache Operations**

```python
logger.debug(f"Cache hit: {cache_key}")
logger.debug(f"Cache miss: {cache_key} — fetching from API")
```

### **API Operations**

```python
logger.debug(f"get_weather_for_date for {location} on {date_str}")
logger.error(f"Error decoding cached data for {cache_key}: {e}")
```

### **Data Processing**

```python
logger.debug(f"Calculated average: {avg_temp}°C")
logger.warning(f"Invalid location: {location}")
logger.error(f"Error in get_summary: {e}")
```

## 🛠️ **Environment Configuration**

### **Development (.env)**

```bash
DEBUG=true
CACHE_ENABLED=true
```

### **Production (.env)**

```bash
DEBUG=false
CACHE_ENABLED=true
```

## 📈 **Monitoring and Alerting**

### **1. Log File Monitoring**

```bash
# Watch logs in real-time
tail -f temphist.log

# Search for errors
grep "ERROR" temphist.log

# Count warnings
grep -c "WARNING" temphist.log
```

### **2. Log Rotation (Production)**

```python
# Add to main.py for production
from logging.handlers import RotatingFileHandler

if not DEBUG:
    handler = RotatingFileHandler(
        'temphist.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    logger.addHandler(handler)
```

### **3. Structured Logging (Advanced)**

```python
import json
from datetime import datetime

def log_structured(level, message, **kwargs):
    """Log structured data for better parsing."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "message": message,
        **kwargs
    }
    logger.log(level, json.dumps(log_entry))

# Usage
log_structured(logging.INFO, "API request",
              location="London",
              response_time=0.5,
              cache_hit=True)
```

## 🚀 **Best Practices**

### **1. Use Appropriate Log Levels**

- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARNING**: Issues that don't stop execution
- **ERROR**: Serious problems requiring attention

### **2. Include Context**

```python
# Good
logger.error(f"API request failed for {location}: {error}")

# Better
logger.error(f"API request failed", extra={
    "location": location,
    "error": str(error),
    "endpoint": "/weather"
})
```

### **3. Avoid Sensitive Data**

```python
# Don't log API keys or tokens
logger.debug(f"Making request to Visual Crossing API")
# NOT: logger.debug(f"API key: {API_KEY}")
```

### **4. Performance Considerations**

```python
# Use lazy evaluation for expensive operations
logger.debug(f"Processing {len(data)} data points")

# Avoid this in production
if DEBUG:
    logger.debug(f"Full data: {json.dumps(data)}")
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **1. No Logs Appearing**

```bash
# Check environment variable
echo $DEBUG

# Check log file exists
ls -la temphist.log
```

#### **2. Too Many Debug Messages**

```python
# Set specific logger levels
logging.getLogger('main').setLevel(logging.INFO)
```

#### **3. Log File Too Large**

```bash
# Implement log rotation
# See "Log Rotation" section above
```

## 📋 **Migration from debug_print**

### **Before (Old)**

```python
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

debug_print(f"Cache hit: {cache_key}")
```

### **After (New)**

```python
logger = logging.getLogger(__name__)

logger.debug(f"Cache hit: {cache_key}")
logger.error(f"API error: {error}")
logger.warning(f"Invalid input: {input}")
```

## 🎯 **Next Steps**

1. **Monitor logs** in development to understand application flow
2. **Set up log aggregation** for production (ELK stack, etc.)
3. **Add structured logging** for better analytics
4. **Implement log rotation** for production deployment
5. **Add log-based alerting** for critical errors

---

**Note**: The logging module provides much better performance and structure than the previous `debug_print` function. Debug messages are only processed when `DEBUG=true`, making production deployments more efficient.
