# TempHist API

A FastAPI backend for historical temperature data using Visual Crossing with comprehensive caching, rate limiting, and monitoring capabilities.

## 🚀 Features

- **Historical Temperature Data**: 50 years of temperature data for any location
- **Smart Caching**: Redis-based caching with intelligent cache durations
- **Rate Limiting**: Built-in protection against API abuse and misuse
- **Weather Forecasts**: Current weather data and forecasts
- **Performance Monitoring**: Built-in profiling and monitoring tools
- **CORS Enabled**: Ready for web applications
- **Production Ready**: Deploy on Render with one click

## 📋 Requirements

- Python 3.8+
- Redis server (local or cloud)
- Visual Crossing API key

## ⚙️ Configuration

Create a `.env` file with the following variables:

```bash
# Required API Keys
VISUAL_CROSSING_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
API_ACCESS_TOKEN=your_key_here

# Redis Configuration
REDIS_URL=redis://localhost:6379  # Optional, defaults to localhost
CACHE_ENABLED=true  # Optional, defaults to true

# Debugging
DEBUG=false  # Set to true for development

# Rate Limiting Configuration
RATE_LIMIT_ENABLED=true  # Defaults to true
MAX_LOCATIONS_PER_HOUR=10  # Defaults to 10
MAX_REQUESTS_PER_HOUR=100  # Defaults to 100
RATE_LIMIT_WINDOW_HOURS=1  # Defaults to 1

# IP Address Management
IP_WHITELIST=192.168.1.100,10.0.0.5  # IPs exempt from rate limiting
IP_BLACKLIST=192.168.1.200,10.0.0.99  # IPs blocked entirely

# Data Filtering
FILTER_WEATHER_DATA=true  # Filter to essential temperature data only
```

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd TempHist-api
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**

   ```bash
   cp .env.example .env  # Create from template
   # Edit .env with your API keys
   ```

5. **Start Redis** (if running locally)

   ```bash
   redis-server
   ```

6. **Run the development server**
   ```bash
   uvicorn main:app --reload
   ```

## 📡 API Endpoints

### V1 API (Recommended)

The new v1 API provides a unified structure for accessing temperature records across different time periods.

#### Main Record Endpoints

| Endpoint                                                   | Description                 | Format                       |
| ---------------------------------------------------------- | --------------------------- | ---------------------------- |
| `GET /v1/records/{period}/{location}/{identifier}`         | Complete temperature record | See identifier formats below |
| `GET /v1/records/{period}/{location}/{identifier}/average` | Average temperature data    | Subresource                  |
| `GET /v1/records/{period}/{location}/{identifier}/trend`   | Temperature trend data      | Subresource                  |
| `GET /v1/records/{period}/{location}/{identifier}/summary` | Text summary                | Subresource                  |

#### Rolling Bundle Endpoint

| Endpoint                                             | Description                                 | Format       |
| ---------------------------------------------------- | ------------------------------------------- | ------------ |
| `GET /v1/records/rolling-bundle/{location}/{anchor}` | Cross-year series for multiple time periods | `YYYY-MM-DD` |

Aggregates daily, weekly, monthly and yearly endpoint responses into one.

#### Period Types and Identifier Formats

All periods use the same `MM-DD` identifier format, representing the **end date** of the period:

| Period    | Identifier Format | Example | Description                                      |
| --------- | ----------------- | ------- | ------------------------------------------------ |
| `daily`   | `MM-DD`           | `01-15` | January 15th across all years                    |
| `weekly`  | `MM-DD`           | `01-15` | 7 days ending on January 15th across all years   |
| `monthly` | `MM-DD`           | `01-15` | 30 days ending on January 15th across all years  |
| `yearly`  | `MM-DD`           | `01-15` | 365 days ending on January 15th across all years |

#### Example V1 Requests

```bash
# Get daily record for January 15th
GET /v1/records/daily/london/01-15

# Get weekly record ending on January 15th
GET /v1/records/weekly/london/01-15

# Get monthly record ending on January 15th
GET /v1/records/monthly/london/01-15

# Get yearly record ending on January 15th
GET /v1/records/yearly/london/01-15

# Get just the average for a daily record
GET /v1/records/daily/london/01-15/average

# Get just the trend for a daily record
GET /v1/records/daily/london/01-15/trend
```

#### Rolling Bundle Endpoint Details

The rolling bundle endpoint provides cross-year temperature series for multiple time periods in a single request, computed efficiently from cached daily data.

**Parameters:**

- `location`: Location name (e.g., "london", "new_york")
- `anchor`: Anchor date in YYYY-MM-DD format (e.g., "2024-01-15")
- `unit_group`: Temperature unit group - "metric" (default) or "us"
- `month_mode`: Month calculation mode - "rolling1m" (default), "calendar", or "rolling30d"
- `days_back`: Number of previous days to include (0-10, default: 0)
- `include`: CSV of sections to include (valid: day, week, month, year). If present, exclude is ignored.
- `exclude`: CSV of sections to exclude (valid: day, week, month, year). Ignored if include is present.

**Example Rolling Bundle Requests:**

```bash
# Get rolling bundle for January 15th, 2024
GET /v1/records/rolling-bundle/london/2024-01-15

# With custom month mode (calendar month)
GET /v1/records/rolling-bundle/london/2024-01-15?month_mode=calendar

# With US units
GET /v1/records/rolling-bundle/london/2024-01-15?unit_group=us

# Include only weekly, monthly, and yearly data (exclude daily)
GET /v1/records/rolling-bundle/london/2024-01-15?include=week,month,year

# Exclude daily data (include everything else)
GET /v1/records/rolling-bundle/london/2024-01-15?exclude=day

# Include 3 previous days with only weekly and monthly data
GET /v1/records/rolling-bundle/london/2024-01-15?days_back=3&include=week,month
```

**Include/Exclude Parameters:**

The `include` and `exclude` parameters allow you to control which sections are returned in the response:

- **Valid sections**: `day`, `week`, `month`, `year`
- **Include parameter**: Returns only the specified sections. If present, `exclude` is ignored.
- **Exclude parameter**: Returns all sections except the specified ones. Ignored if `include` is present.
- **Previous days**: Controlled separately by `days_back` parameter and are not affected by include/exclude.
- **Performance**: Only requested sections are computed, improving response times.

**Examples:**

- `?include=week,month,year` - Returns only weekly, monthly, and yearly data
- `?exclude=day` - Returns everything except daily data
- `?days_back=3&exclude=day` - Returns 3 previous days + weekly/monthly/yearly data

**Month Mode Options:**

- `rolling1m`: Calendar-aware 1-month window ending on anchor (default)
- `calendar`: Full calendar month (1st to last day of anchor month)
- `rolling30d`: Fixed 30-day rolling window ending on anchor

**Benefits of Rolling Bundle:**

- **Efficiency**: Single API call returns multiple time series
- **Performance**: Uses cached daily data for fast computation
- **Selective Loading**: Include/exclude parameters allow fetching only needed sections
- **Consistency**: All series use the same anchor date for comparison
- **Flexibility**: Multiple month calculation modes for different use cases

**Use Cases:**

- **Weather Dashboards**: Display multiple time periods simultaneously
- **Climate Analysis**: Compare daily, weekly, monthly, and yearly trends
- **Data Visualization**: Create comprehensive temperature charts
- **Research**: Analyze temperature patterns across different time scales

### Legacy Endpoints (Deprecated)

⚠️ **These endpoints are deprecated and will be removed in a future version. Please migrate to v1 endpoints.**

| Endpoint                              | Description                      | Format  | V1 Equivalent                                      |
| ------------------------------------- | -------------------------------- | ------- | -------------------------------------------------- |
| `GET /data/{location}/{month_day}`    | Complete data package            | `MM-DD` | `/v1/records/daily/{location}/{month_day}`         |
| `GET /average/{location}/{month_day}` | Historical average temperature   | `MM-DD` | `/v1/records/daily/{location}/{month_day}/average` |
| `GET /trend/{location}/{month_day}`   | Temperature trend over time      | `MM-DD` | `/v1/records/daily/{location}/{month_day}/trend`   |
| `GET /summary/{location}/{month_day}` | Text summary of temperature data | `MM-DD` | `/v1/records/daily/{location}/{month_day}/summary` |

### Other Endpoints

| Endpoint                         | Description               | Format       |
| -------------------------------- | ------------------------- | ------------ |
| `GET /`                          | API information           | -            |
| `GET /weather/{location}/{date}` | Weather for specific date | `YYYY-MM-DD` |
| `GET /forecast/{location}`       | Current weather forecast  | -            |

### Monitoring Endpoints

| Endpoint                 | Description                      | Access |
| ------------------------ | -------------------------------- | ------ |
| `GET /rate-limit-status` | Current rate limiting status     | Public |
| `GET /rate-limit-stats`  | Overall rate limiting statistics | Admin  |
| `GET /test-redis`        | Redis connection test            | Public |
| `GET /health`            | Health check                     | Public |

### V1 API Response Format

**GET `/v1/records/daily/london/01-15`** returns:

```json
{
  "period": "daily",
  "location": "london",
  "identifier": "01-15",
  "range": {
    "start": "1975-01-15",
    "end": "2024-01-15",
    "years": 50
  },
  "unit_group": "metric",
  "values": [
    {
      "date": "1975-01-15",
      "year": 1975,
      "temperature": 15.0,
      "temp_min": null,
      "temp_max": null
    }
  ],
  "average": {
    "mean": 12.5,
    "temp_min": null,
    "temp_max": null,
    "unit": "celsius",
    "data_points": 50
  },
  "trend": {
    "slope": 0.25,
    "unit": "°C/decade",
    "data_points": 50,
    "r_squared": null
  },
  "summary": "15.0°C. It is 2.5°C warmer than average today.",
  "metadata": {
    "total_years": 50,
    "available_years": 50,
    "missing_years": [],
    "completeness": 100.0
  }
}
```

#### Rolling Bundle Response Format

**GET `/v1/records/rolling-bundle/london/2024-01-15`** returns:

```json
{
  "period": "rolling",
  "location": "london",
  "anchor": "2024-01-15",
  "unit_group": "metric",
  "day": {
    "values": [
      { "year": 1975, "temp": 7.8 },
      { "year": 1976, "temp": 8.2 },
      { "year": 2024, "temp": 9.1 }
    ],
    "count": 50
  },
  "day_minus_1": {
    "values": [
      { "year": 1975, "temp": 8.1 },
      { "year": 1976, "temp": 7.9 },
      { "year": 2024, "temp": 8.8 }
    ],
    "count": 50
  },
  "day_minus_2": {
    "values": [
      { "year": 1975, "temp": 7.7 },
      { "year": 1976, "temp": 8.0 },
      { "year": 2024, "temp": 8.5 }
    ],
    "count": 50
  },
  "day_minus_3": {
    "values": [
      { "year": 1975, "temp": 8.3 },
      { "year": 1976, "temp": 7.8 },
      { "year": 2024, "temp": 8.2 }
    ],
    "count": 50
  },
  "week": {
    "values": [
      { "year": 1975, "temp": 7.9 },
      { "year": 1976, "temp": 8.1 },
      { "year": 2024, "temp": 8.7 }
    ],
    "count": 50
  },
  "month": {
    "values": [
      { "year": 1975, "temp": 8.0 },
      { "year": 1976, "temp": 7.8 },
      { "year": 2024, "temp": 8.4 }
    ],
    "count": 50
  },
  "year": {
    "values": [
      { "year": 1975, "temp": 9.8 },
      { "year": 1976, "temp": 10.2 },
      { "year": 2024, "temp": 11.1 }
    ],
    "count": 50
  },
  "notes": "Month uses calendar-aware 1-month window ending on anchor (EOM-clipped)."
}
```

### Legacy Response Format (Deprecated)

**GET `/data/London/01-15`** returns:

```json
{
  "weather": {
    "data": [
      { "x": 1975, "y": 15.0 },
      { "x": 1976, "y": 15.5 },
      { "x": 1977, "y": 16.0 },
      { "x": 2024, "y": 17.0 }
    ],
    "metadata": {
      "total_years": 50,
      "available_years": 50,
      "missing_years": [],
      "completeness": 100.0
    }
  },
  "summary": "17.0°C. This is the warmest 15th January on record. It is 2.0°C warmer than average today.",
  "trend": {
    "slope": 0.3,
    "units": "°C/decade"
  },
  "average": {
    "average": 15.0,
    "unit": "celsius",
    "data_points": 50,
    "year_range": { "start": 1975, "end": 2024 },
    "missing_years": [],
    "completeness": 100.0
  }
}
```

## 🛡️ Rate Limiting

The API includes comprehensive rate limiting to prevent abuse:

### Location Diversity Limits

- **Max unique locations per hour**: 10 (configurable)
- **Purpose**: Prevents requesting data for too many different locations

### Request Rate Limits

- **Max requests per hour**: 100 (configurable)
- **Purpose**: Prevents excessive API calls overall

### IP Management

- **Whitelist**: IPs exempt from all rate limiting
- **Blacklist**: IPs blocked entirely (HTTP 403)

### Rate Limit Responses

When limits are exceeded:

- **HTTP 429** (Too Many Requests)
- **Retry-After header** with retry time
- **Detailed error message** explaining the limit

## 🚀 Caching System

### Cache Strategy

- **Today's data**: 1 hour cache duration
- **Historical data**: 1 week cache duration
- **Redis-based**: High-performance caching
- **Smart invalidation**: Automatic cleanup

### Cache Keys

- Weather data: `{location}_{date}`
- Series data: `series_{location}_{month}_{day}`
- Analysis data: `{type}_{location}_{month}_{day}`

### Cache Monitoring

```bash
# Check cache status
curl http://localhost:8000/test-redis

# Monitor cache performance
curl http://localhost:8000/rate-limit-stats
```

## 📊 Performance & Monitoring

### Built-in Profiling

The API includes comprehensive performance monitoring:

```bash
# Run performance tests
python performance_test.py

# Profile specific functions
python -c "
import cProfile
from main import calculate_historical_average
cProfile.run('calculate_historical_average([{\"x\": 2020, \"y\": 15.5}])')
"
```

### Performance Metrics

- **URL Building**: ~2.5M operations/second
- **Historical Average**: ~58K operations/second
- **Trend Calculation**: ~23K operations/second
- **Memory Usage**: Minimal impact

### Monitoring Tools

- **Real-time logs**: `tail -f temphist.log`
- **Error tracking**: `grep "ERROR" temphist.log`
- **Performance metrics**: Built-in timing middleware

## 🧪 Testing

### Quick Start

```bash
# Run all tests
pytest test_main.py -v

# Run specific test categories
pytest test_main.py -k "rate" -v  # Rate limiting tests
pytest test_main.py -k "cache" -v  # Caching tests
pytest test_main.py -k "performance" -v  # Performance tests
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Full API endpoint testing
- **Performance Tests**: Load and stress testing
- **Rate Limiting Tests**: Rate limit validation

### Manual Testing

```bash
# Test rate limiting
curl http://localhost:8000/rate-limit-status

# Test weather data
curl http://localhost:8000/data/London/01-15

# Test with authentication
curl -H "Authorization: Bearer test_token" http://localhost:8000/data/London/01-15
```

## 📝 Logging

### Log Levels

- **DEBUG**: Detailed debugging information (development only)
- **INFO**: General application flow
- **WARNING**: Issues that don't stop execution
- **ERROR**: Serious problems requiring attention

### Log Configuration

```python
# Development
DEBUG=true  # Enables debug logging and file output

# Production
DEBUG=false  # Disables debug overhead
```

### Log Monitoring

```bash
# Watch logs in real-time
tail -f temphist.log

# Search for specific events
grep "ERROR" temphist.log
grep "CACHE" temphist.log
grep "RATE" temphist.log
```

## 🚀 Deployment

### Render Deployment

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your repository
4. Add environment variables in Render dashboard
5. Deploy!

### Environment Variables for Production

```bash
DEBUG=false
CACHE_ENABLED=true
RATE_LIMIT_ENABLED=true
REDIS_URL=your_redis_url
VISUAL_CROSSING_API_KEY=your_key
```

### Health Checks

- **Health endpoint**: `/health`
- **Redis check**: `/test-redis`
- **Rate limit status**: `/rate-limit-status`

## 🔧 Troubleshooting

### Common Issues

#### Rate Limiting Not Working

```bash
# Check configuration
curl http://localhost:8000/rate-limit-status

# Verify environment variables
echo $RATE_LIMIT_ENABLED
```

#### Cache Issues

```bash
# Test Redis connection
curl http://localhost:8000/test-redis

# Check Redis logs
redis-cli monitor
```

#### Performance Issues

```bash
# Run performance tests
python performance_test.py

# Monitor memory usage
python -c "import psutil; print(psutil.Process().memory_info())"
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
DEBUG=true uvicorn main:app --reload
```

## 📈 Performance Optimization

### Caching Improvements

- Implemented Redis caching for all endpoints
- Smart cache durations based on data type
- Automatic cache invalidation

### Rate Limiting Optimization

- Memory-efficient data structures
- Automatic cleanup of old entries
- Configurable limits and time windows

### API Optimization

- Async/await for I/O operations
- Connection pooling for Redis
- Response compression

## 🛠️ Development

### Code Structure

```
main.py              # Main FastAPI application
test_main.py         # Comprehensive test suite
performance_test.py  # Performance testing utilities
requirements.txt     # Python dependencies
render.yaml         # Render deployment configuration
```

### Development Workflow

1. **Make changes** to the code
2. **Run tests** with `pytest`
3. **Check performance** with `python performance_test.py`
4. **Test rate limiting** with manual requests
5. **Deploy** to Render

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📚 Additional Resources

- **API Documentation**: `http://localhost:8000/docs` (when running locally)
- **Interactive API**: `http://localhost:8000/redoc`
- **Rate Limit Status**: `/rate-limit-status`
- **Performance Metrics**: Built-in profiling tools

## 📄 License

MIT License - see LICENSE file for details

---

**Note**: This API is designed for production use with comprehensive monitoring, rate limiting, and caching. The performance profiling shows excellent optimization with minimal memory usage and high throughput capabilities.
