# TempHist API

A FastAPI backend for historical temperature data using Visual Crossing with comprehensive caching, rate limiting, and monitoring capabilities.

## 🚀 Features

- **Historical Temperature Data**: 50 years of temperature data for any location
- **Enhanced Caching**: Cloudflare-optimized caching with ETags and conditional requests
- **Async Job Processing**: Heavy computations handled asynchronously with job tracking
- **Rate Limiting**: Built-in protection against API abuse and misuse
- **Weather Forecasts**: Current weather data and forecasts
- **Performance Monitoring**: Built-in profiling and monitoring tools
- **Cache Prewarming**: Automated cache warming for popular locations
- **CORS Enabled**: Ready for web applications
- **Production Ready**: Deploy on Railway or Render

## 📚 Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Release notes and version history
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment guide for Railway and other platforms
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Migrating from legacy to v1 API
- **[CLOUDFLARE_OPTIMIZATION.md](CLOUDFLARE_OPTIMIZATION.md)** - CDN optimization guide
- **[railway/](railway/)** - Railway-specific deployment tools and documentation

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

# Authentication Tokens
TEST_TOKEN=your_test_token_here  # Development/testing token
API_ACCESS_TOKEN=your_api_access_token_here  # API access token for automated systems
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

## 🚀 Enhanced Caching System

The API now includes a comprehensive caching system optimized for Cloudflare and high performance:

### Cache Features

- **Strong Cache Headers**: ETags, Last-Modified, and Cache-Control headers
- **Conditional Requests**: 304 Not Modified responses for unchanged data
- **Single-Flight Protection**: Prevents cache stampedes with Redis locks
- **Canonical Cache Keys**: Normalized keys for maximum hit rates
- **Async Job Processing**: Heavy computations handled asynchronously
- **Cache Metrics**: Hit/miss counters and performance monitoring
- **Cache Prewarming**: Automated warming for popular locations

### Cache Endpoints

#### Regular Endpoints (with enhanced caching)

```bash
# All existing endpoints now include enhanced caching
GET /v1/records/{period}/{location}/{identifier}
GET /v1/records/rolling-bundle/{location}/{anchor}
```

#### Async Job Endpoints (New)

```bash
# Create async job for heavy computations
POST /v1/records/{period}/{location}/{identifier}/async
POST /v1/records/rolling-bundle/{location}/{anchor}/async

# Check job status and retrieve results
GET /v1/jobs/{job_id}

# Job status values: pending, processing, ready, error
```

### Async Job Processing

The API now supports asynchronous processing for heavy computations:

#### Job Lifecycle

1. **Submit Job**: POST to async endpoint returns `202 Accepted` with `job_id`
2. **Monitor Progress**: GET `/v1/jobs/{job_id}` to check status
3. **Retrieve Results**: When status is `ready`, results are available
4. **Error Handling**: Failed jobs return status `error` with details

#### Example Async Usage

```bash
# Start async job for heavy computation
curl -X POST "https://api.temphist.com/v1/records/rolling-bundle/london/2024-01-15/async" \
     -H "Authorization: Bearer YOUR_TOKEN"

# Response: {"job_id": "job_abc123", "status": "pending", "message": "Job queued"}

# Check job status
curl "https://api.temphist.com/v1/jobs/job_abc123" \
     -H "Authorization: Bearer YOUR_TOKEN"

# Response when ready:
# {
#   "job_id": "job_abc123",
#   "status": "ready",
#   "result": { /* temperature data */ },
#   "created_at": "2024-01-15T10:00:00Z",
#   "completed_at": "2024-01-15T10:02:30Z"
# }
```

#### Job Worker Process

The system includes a background worker that processes async jobs:

```bash
# Start the job worker (typically run as a separate service)
python job_worker.py

# The worker will:
# - Poll Redis for pending jobs
# - Process heavy computations
# - Cache results automatically
# - Update job status to 'ready' or 'error'
```

### Cache Management

```bash
# View cache statistics and performance
GET /cache-stats

# Reset cache statistics
POST /cache-stats/reset

# Invalidate specific cache entries
DELETE /cache/invalidate/key/{cache_key}
DELETE /cache/invalidate/location/{location}
DELETE /cache/invalidate/pattern

# Cache health check
GET /cache-stats/health
```

### Cache Prewarming

```bash
# Prewarm popular locations for various endpoints
python prewarm.py --locations 20 --days 7

# Run comprehensive load tests
python load_test_script.py --requests 1000 --concurrent 10

# Test cache performance specifically
python load_test_script.py --endpoint cache --requests 500
```

### Cache Headers and ETag Support

All endpoints now support conditional requests:

```bash
# First request returns data with ETag
curl -H "Authorization: Bearer YOUR_TOKEN" \
     "https://api.temphist.com/v1/records/daily/london/01-15"

# Response includes: ETag: "abc123", Cache-Control: "public, max-age=3600"

# Subsequent requests with ETag return 304 if unchanged
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "If-None-Match: abc123" \
     "https://api.temphist.com/v1/records/daily/london/01-15"

# Response: 304 Not Modified (use cached data)
```

For detailed Cloudflare optimization guidance, see [CLOUDFLARE_OPTIMIZATION.md](CLOUDFLARE_OPTIMIZATION.md).

## 🔌 Client Integration Guide

### Authentication

The API supports multiple authentication methods depending on your use case:

#### 1. Firebase Authentication (Users)

For end-user applications, use Firebase authentication tokens:

```bash
# Include Firebase token in Authorization header
curl -H "Authorization: Bearer FIREBASE_ID_TOKEN" \
     https://api.temphist.com/v1/records/daily/New%20York/01-15
```

#### 2. API Access Token (Automated Systems)

For automated systems like cron jobs, server-side prefetching, or internal services, use the API access token:

```bash
# Include API access token in Authorization header
curl -H "Authorization: Bearer $API_ACCESS_TOKEN" \
     https://api.temphist.com/v1/records/daily/New%20York/01-15
```

**Benefits of API Access Token:**

- ✅ No Firebase authentication overhead
- ✅ **Bypasses rate limiting** for efficient automated operations
- ✅ Identified as system/admin usage in logs
- ✅ Efficient for automated prefetching and cache warming
- ✅ Perfect for cron jobs and background tasks

#### 3. Test Token (Development)

For development and testing:

```bash
# Include test token in Authorization header
curl -H "Authorization: Bearer $TEST_TOKEN" \
     http://localhost:8000/v1/records/daily/New%20York/01-15
```

### Base URLs

```bash
# Production
BASE_URL=https://api.temphist.com

# Development
BASE_URL=http://localhost:8000
```

### Client Implementation Examples

#### JavaScript/TypeScript Client

```typescript
class TempHistClient {
  private baseUrl: string;
  private apiToken: string;

  constructor(baseUrl: string, apiToken: string) {
    this.baseUrl = baseUrl;
    this.apiToken = apiToken;
  }

  // Basic request with caching support
  async getTemperatureData(
    period: string,
    location: string,
    identifier: string
  ) {
    const response = await fetch(
      `${this.baseUrl}/v1/records/${period}/${encodeURIComponent(
        location
      )}/${identifier}`,
      {
        headers: {
          Authorization: `Bearer ${this.apiToken}`,
          Accept: "application/json",
        },
      }
    );

    // Handle 304 Not Modified (cached response)
    if (response.status === 304) {
      return null; // Use your local cache
    }

    return await response.json();
  }

  // Async job processing
  async getTemperatureDataAsync(
    period: string,
    location: string,
    identifier: string
  ) {
    // Create job
    const jobResponse = await fetch(
      `${this.baseUrl}/v1/records/${period}/${encodeURIComponent(
        location
      )}/${identifier}/async`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiToken}`,
          "Content-Type": "application/json",
        },
      }
    );

    const job = await jobResponse.json();

    // Poll for completion
    return await this.pollJobStatus(job.job_id);
  }

  private async pollJobStatus(jobId: string): Promise<any> {
    while (true) {
      const response = await fetch(`${this.baseUrl}/v1/jobs/${jobId}`, {
        headers: {
          Authorization: `Bearer ${this.apiToken}`,
        },
      });

      const status = await response.json();

      if (status.status === "ready") {
        return status.result;
      } else if (status.status === "error") {
        throw new Error(`Job failed: ${status.error}`);
      }

      // Wait before polling again
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }
}
```

#### Python Client

```python
import requests
import time
from typing import Optional, Dict, Any

class TempHistClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Accept': 'application/json'
        })

    def get_temperature_data(self, period: str, location: str, identifier: str,
                           etag: Optional[str] = None) -> Dict[str, Any]:
        """Get temperature data with optional ETag support."""
        headers = {}
        if etag:
            headers['If-None-Match'] = etag

        response = self.session.get(
            f"{self.base_url}/v1/records/{period}/{location}/{identifier}",
            headers=headers
        )

        # Handle 304 Not Modified
        if response.status_code == 304:
            return None  # Use cached data

        response.raise_for_status()
        return response.json()

    def get_temperature_data_async(self, period: str, location: str, identifier: str) -> Dict[str, Any]:
        """Get temperature data using async job processing."""
        # Create job
        job_response = self.session.post(
            f"{self.base_url}/v1/records/{period}/{location}/{identifier}/async"
        )
        job_response.raise_for_status()
        job = job_response.json()

        # Poll for completion
        return self._poll_job_status(job['job_id'])

    def _poll_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll job status until completion."""
        while True:
            response = self.session.get(f"{self.base_url}/v1/jobs/{job_id}")
            response.raise_for_status()
            status = response.json()

            if status['status'] == 'ready':
                return status['result']
            elif status['status'] == 'error':
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")

            time.sleep(3)  # Wait 3 seconds before polling again

    def get_rolling_bundle(self, location: str, anchor: str, **params) -> Dict[str, Any]:
        """Get rolling bundle data."""
        response = self.session.get(
            f"{self.base_url}/v1/records/rolling-bundle/{location}/{anchor}",
            params=params
        )
        response.raise_for_status()
        return response.json()
```

### Error Handling

#### HTTP Status Codes

| Code  | Description      | Action                        |
| ----- | ---------------- | ----------------------------- |
| `200` | Success          | Process response data         |
| `202` | Accepted         | Job created (async endpoints) |
| `304` | Not Modified     | Use cached data               |
| `400` | Bad Request      | Check request parameters      |
| `401` | Unauthorized     | Verify API token              |
| `404` | Not Found        | Check endpoint URL            |
| `422` | Validation Error | Fix request format            |
| `429` | Rate Limited     | Wait and retry                |
| `500` | Server Error     | Retry with backoff            |

#### Error Response Format

```json
{
  "detail": "Error message",
  "status_code": 400,
  "error_type": "validation_error"
}
```

### Performance Best Practices

#### Use Appropriate Endpoints

```bash
# For real-time data - use regular endpoints
GET /v1/records/daily/New%20York/01-15

# For heavy computations - use async jobs
POST /v1/records/rolling-bundle/New%20York/2024-01-15/async

# For repeated requests - implement ETag caching
If-None-Match: "abc123def456"
```

#### Retry Logic with Exponential Backoff

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
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

#### Async Job Endpoints (New)

| Endpoint                                                    | Description                  | Response       |
| ----------------------------------------------------------- | ---------------------------- | -------------- |
| `POST /v1/records/{period}/{location}/{identifier}/async`   | Create async job for records | `202 Accepted` |
| `POST /v1/records/rolling-bundle/{location}/{anchor}/async` | Create async job for bundle  | `202 Accepted` |
| `GET /v1/jobs/{job_id}`                                     | Check job status and results | Job status     |

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
- `unit_group`: Temperature unit - "celsius" (default) or "fahrenheit"
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

### Removed Endpoints

❌ **These endpoints have been removed. Please use v1 endpoints instead.**

| Removed Endpoint                      | V1 Equivalent                                      |
| ------------------------------------- | -------------------------------------------------- |
| `GET /data/{location}/{month_day}`    | `/v1/records/daily/{location}/{month_day}`         |
| `GET /average/{location}/{month_day}` | `/v1/records/daily/{location}/{month_day}/average` |
| `GET /trend/{location}/{month_day}`   | `/v1/records/daily/{location}/{month_day}/trend`   |
| `GET /summary/{location}/{month_day}` | `/v1/records/daily/{location}/{month_day}/summary` |

**Note:** Removed endpoints return `410 Gone` with migration information.

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

#### Preapproved Locations Endpoint

The API provides access to a curated list of preapproved locations that are guaranteed to work with the weather endpoints.

**Endpoint:** `GET /v1/locations/preapproved`

**Query Parameters:**

- `country_code` (optional): Filter by ISO 3166-1 alpha-2 country code (e.g., "US", "GB")
- `tier` (optional): Filter by location tier (e.g., "global")
- `limit` (optional): Limit results (1-500, default: no limit)

**Response Headers:**

- `Cache-Control`: `public, max-age=3600, s-maxage=86400`
- `ETag`: Stable ETag for conditional requests
- `Last-Modified`: File modification timestamp

**Example Requests:**

```bash
# Get all preapproved locations
GET /v1/locations/preapproved

# Get locations in the United States
GET /v1/locations/preapproved?country_code=US

# Get global tier locations
GET /v1/locations/preapproved?tier=global

# Get first 10 locations
GET /v1/locations/preapproved?limit=10

# Combined filters
GET /v1/locations/preapproved?country_code=GB&tier=global&limit=5
```

**Response Format:**

```json
{
  "version": 1,
  "count": 20,
  "generated_at": "2024-01-15T10:30:00Z",
  "locations": [
    {
      "id": "london",
      "slug": "london",
      "name": "London",
      "admin1": "England",
      "country_name": "United Kingdom",
      "country_code": "GB",
      "latitude": 51.5074,
      "longitude": -0.1278,
      "timezone": "Europe/London",
      "tier": "global"
    }
  ]
}
```

**Caching and Performance:**

- Full response cached in Redis for 24 hours
- Filtered responses cached separately for optimal performance
- Supports conditional requests with ETag and Last-Modified headers
- Rate limited to 60 requests per minute per IP

**Status Endpoint:** `GET /v1/locations/preapproved/status`

Returns service health and configuration information.

## 🛡️ Rate Limiting

The API includes comprehensive rate limiting to prevent abuse:

### Exemptions

The following are **exempt from rate limiting**:

- **Service Jobs**: Requests using `API_ACCESS_TOKEN` (automated systems, cron jobs, cache warming)
- **Whitelisted IPs**: IPs configured in `IP_WHITELIST`

> **Note**: Rate limiting only applies to end users with Firebase authentication. Automated systems using `API_ACCESS_TOKEN` can operate without rate limits for efficient prefetching and cache warming.

### Location Diversity Limits

- **Max unique locations per hour**: 10 (configurable)
- **Purpose**: Prevents requesting data for too many different locations
- **Applied to**: Firebase-authenticated user requests only

### Request Rate Limits

- **Max requests per hour**: 100 (configurable)
- **Purpose**: Prevents excessive API calls overall
- **Applied to**: Firebase-authenticated user requests only

### IP Management

- **Whitelist**: IPs exempt from all rate limiting
- **Blacklist**: IPs blocked entirely (HTTP 403)

### Rate Limit Responses

When limits are exceeded:

- **HTTP 429** (Too Many Requests)
- **Retry-After header** with retry time
- **Detailed error message** explaining the limit

### Checking Rate Limit Status

```bash
# Check your current rate limit status
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.temphist.com/rate-limit-status

# Response includes:
# - service_job: true if using API_ACCESS_TOKEN
# - whitelisted: true if IP is whitelisted
# - rate_limited: false if exempt from rate limiting
```

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
# Run all tests (140+ tests)
pytest -v

# Run specific test categories
pytest test_main.py -k "rate" -v  # Rate limiting tests
pytest test_main.py -k "cache" -v  # Caching tests
pytest test_main.py -k "performance" -v  # Performance tests
pytest test_cache.py -v  # Enhanced caching system tests
pytest tests/routers/ -v  # Router-specific tests
```

### Test Categories

- **Unit Tests**: Individual component testing (test_main.py)
- **Enhanced Caching Tests**: Cache utilities, ETags, jobs (test_cache.py)
- **Integration Tests**: Full API endpoint testing
- **Performance Tests**: Load and stress testing
- **Rate Limiting Tests**: Rate limit validation
- **Router Tests**: Endpoint-specific functionality

### Enhanced Caching Tests

```bash
# Test cache key normalization
pytest test_cache.py::TestCacheKeyBuilder -v

# Test ETag generation and conditional requests
pytest test_cache.py::TestETagGenerator -v

# Test async job processing
pytest test_cache.py::TestJobWorker -v

# Test single-flight protection
pytest test_cache.py::TestSingleFlightLock -v

# Test cache performance
pytest test_cache.py::TestPerformance -v
```

### Load Testing

```bash
# Run comprehensive load tests
python load_test_script.py --requests 1000 --concurrent 10

# Test specific endpoints
python load_test_script.py --endpoint records --requests 500
python load_test_script.py --endpoint cache --requests 200

# Test async job performance
python load_test_script.py --endpoint async --requests 50
```

### Manual Testing

```bash
# Test rate limiting
curl http://localhost:8000/rate-limit-status

# Test weather data
curl http://localhost:8000/v1/records/daily/London/01-15

# Test with authentication
curl -H "Authorization: Bearer $TEST_TOKEN" \
     http://localhost:8000/v1/records/daily/London/01-15

# Test async job processing
curl -X POST -H "Authorization: Bearer $TEST_TOKEN" \
     http://localhost:8000/v1/records/daily/London/01-15/async

# Test cache headers
curl -v -H "Authorization: Bearer $TEST_TOKEN" \
     http://localhost:8000/v1/records/daily/London/01-15

# Test conditional requests (304 Not Modified)
curl -H "Authorization: Bearer $TEST_TOKEN" \
     -H "If-None-Match: your_etag_here" \
     http://localhost:8000/v1/records/daily/London/01-15
```

### Cache Prewarming

```bash
# Prewarm popular locations
python prewarm.py --locations 20 --days 7

# Prewarm with verbose output
python prewarm.py --locations 10 --days 3 --verbose
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

**For complete deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**

### Quick Start - Railway (Recommended)

1. Create Railway project with Redis database
2. Deploy from GitHub repository
3. Set environment variables
4. Deploy!

## 📋 Environment Variables

### Hosting Requirements

**Minimum Requirements:**

- **Python 3.8+** runtime environment
- **Redis database** (version 6.0+ recommended)
- **Memory**: 512MB RAM minimum, 1GB+ recommended
- **Storage**: 100MB+ for application and cache data
- **Network**: Outbound HTTPS access to weather APIs

**Supported Platforms:**

- Railway, Render, Heroku, DigitalOcean App Platform
- AWS Elastic Beanstalk, Google Cloud Run, Azure Container Instances
- VPS/Cloud servers (Ubuntu 20.04+, CentOS 8+)
- Docker containers

### Environment Variables

#### 🔑 **Required Variables**

**API Keys (Required for main service):**

```bash
VISUAL_CROSSING_API_KEY=your_visual_crossing_key    # Primary weather data source
OPENWEATHER_API_KEY=your_openweather_key            # Backup weather data source
```

**Database (Required):**

```bash
REDIS_URL=redis://localhost:6379                    # Redis connection string
# Examples:
# redis://username:password@host:port/db
# rediss://username:password@host:port/db  (SSL)
# redis://default:password@redis.internal:6379  (Railway)
```

#### 🔧 **Main API Service Variables**

**Core Configuration:**

```bash
# Server settings
PORT=8000                                           # Server port (default: 8000)
BASE_URL=https://your-api-domain.com               # Public API URL for job callbacks

# Caching
CACHE_ENABLED=true                                  # Enable/disable caching (default: true)

# Logging & Debug
DEBUG=false                                         # Enable debug logging (default: false)
LOG_VERBOSITY=normal                               # minimal|normal|verbose (default: normal)
```

**Rate Limiting & Security:**

```bash
# Rate limiting
RATE_LIMIT_ENABLED=true                            # Enable rate limiting (default: true)
MAX_LOCATIONS_PER_HOUR=10                          # Max unique locations per IP/hour (default: 10)
MAX_REQUESTS_PER_HOUR=100                          # Max requests per IP/hour (default: 100)
RATE_LIMIT_WINDOW_HOURS=1                          # Rate limit window in hours (default: 1)

# IP filtering (comma-separated lists)
IP_WHITELIST=192.168.1.100,10.0.0.5               # Bypass rate limits (optional)
IP_BLACKLIST=192.168.1.200,10.0.0.99              # Block specific IPs (optional)

# API access
API_ACCESS_TOKEN=your_secure_token                 # Token for automated access (optional)
TEST_TOKEN=your_test_token                         # Token for testing endpoints (optional)
```

**Data Processing:**

```bash
FILTER_WEATHER_DATA=true                           # Filter invalid weather data (default: true)
UNIT_GROUP=celsius                                 # Default temperature unit (default: celsius)
```

**Firebase Integration (Optional):**

```bash
FIREBASE_SERVICE_ACCOUNT={"type":"service_account",...}  # Firebase service account JSON
# OR
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}  # Alternative variable name
```

#### ⚙️ **Job Worker Service Variables**

**Worker Configuration:**

```bash
# All main API variables above, plus:
DEBUG=false                                         # Worker debug logging
BASE_URL=https://your-api-domain.com               # API URL for job callbacks
```

**Note:** The job worker service uses the same environment variables as the main API service, as it needs access to the same Redis instance and API keys.

#### 🗄️ **Cache Management Variables**

**Cache Warming:**

```bash
CACHE_WARMING_ENABLED=true                         # Enable automatic cache warming (default: true)
CACHE_WARMING_INTERVAL_HOURS=4                     # Hours between warming cycles (default: 4)
CACHE_WARMING_DAYS_BACK=7                          # Days of data to warm (default: 7)
CACHE_WARMING_CONCURRENT_REQUESTS=3                # Concurrent warming requests (default: 3)
CACHE_WARMING_MAX_LOCATIONS=15                     # Max locations to warm (default: 15)
CACHE_WARMING_POPULAR_LOCATIONS=london,new_york,paris,tokyo,sydney,berlin,madrid,rome,amsterdam,dublin
```

**Cache Statistics:**

```bash
CACHE_STATS_ENABLED=true                           # Enable cache statistics (default: true)
CACHE_STATS_RETENTION_HOURS=24                     # Hours to retain stats (default: 24)
CACHE_HEALTH_THRESHOLD=0.7                         # Hit rate threshold for health (default: 0.7)
```

**Cache Invalidation:**

```bash
CACHE_INVALIDATION_ENABLED=true                    # Enable cache invalidation (default: true)
CACHE_INVALIDATION_DRY_RUN=false                  # Test invalidation without executing (default: false)
CACHE_INVALIDATION_BATCH_SIZE=100                 # Batch size for invalidation (default: 100)
```

**Usage Tracking:**

```bash
USAGE_TRACKING_ENABLED=true                        # Enable usage tracking (default: true)
USAGE_RETENTION_DAYS=7                             # Days to retain usage data (default: 7)
```

### Platform-Specific Examples

**Railway:**

```bash
REDIS_URL=redis://default:password@redis.internal:6379
```

**Render:**

```bash
REDIS_URL=redis://username:password@host:port
```

**Heroku:**

```bash
REDIS_URL=redis://username:password@host:port
```

**Docker:**

```bash
REDIS_URL=redis://redis:6379
```

**Local Development:**

```bash
REDIS_URL=redis://localhost:6379
DEBUG=true
LOG_VERBOSITY=verbose
```

### Health Checks

- **Health endpoint**: `/health`
- **Redis check**: `/test-redis`
- **Rate limit status**: `/rate-limit-status`

See [DEPLOYMENT.md](DEPLOYMENT.md) for:

- Complete Railway setup guide
- Environment variable configuration
- Firebase credentials setup
- Multi-service deployment
- Migration from Render
- Troubleshooting deployment issues

## 🔧 Troubleshooting

**For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### Quick Diagnostics

```bash
# Health checks
curl https://your-app.com/health
curl https://your-app.com/test-redis
curl https://your-app.com/rate-limit-status

# Enable debug logging
DEBUG=true uvicorn main:app --reload
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for help with:

- Deployment issues (502 errors, environment variables)
- Async jobs and background worker problems
- Redis and caching issues
- Rate limiting problems
- API errors (401, 422, 429, 500)
- Performance issues

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

### Managing Preapproved Locations Data

The preapproved locations are stored in `data/preapproved_locations.json` and loaded at startup.

#### Adding New Locations

1. **Edit the data file:**

   ```bash
   # Add new location to data/preapproved_locations.json
   {
     "id": "new_city",
     "slug": "new-city",
     "name": "New City",
     "admin1": "State/Province",
     "country_name": "Country Name",
     "country_code": "CC",  # ISO 3166-1 alpha-2
     "latitude": 0.0,
     "longitude": 0.0,
     "timezone": "Continent/City",
     "tier": "global"
   }
   ```

2. **Restart the application** to load new data:

   ```bash
   # The cache will be automatically warmed on startup
   python main.py
   ```

3. **Verify the data loaded:**
   ```bash
   curl http://localhost:8000/v1/locations/preapproved/status
   ```

#### Cache Management

- **Warm cache manually:** The cache is automatically warmed on startup
- **Clear cache:** Use Redis commands or restart the application
- **Monitor cache:** Check Redis keys with pattern `preapproved:v1:*`

#### Data Validation

The application validates all location data against the `LocationItem` schema:

- Country codes must be valid ISO 3166-1 alpha-2 format
- Coordinates must be valid numbers
- All required fields must be present

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
