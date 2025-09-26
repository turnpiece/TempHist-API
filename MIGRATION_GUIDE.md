# Migration Guide: Legacy to V1 API

⚠️ **IMPORTANT: Legacy endpoints have been removed as of the latest version.**

This guide documents the migration from the removed legacy TempHist API endpoints to the new v1 API structure.

## Overview

The legacy endpoints have been completely removed and now return `410 Gone` responses. The new v1 API provides a unified structure for accessing temperature records across different time periods (daily, weekly, monthly, yearly) with consistent response formats and subresource endpoints.

## Key Changes

### 1. Endpoint Structure

**Removed (Legacy):**

```
/data/{location}/{month_day}          → 410 Gone
/average/{location}/{month_day}       → 410 Gone
/trend/{location}/{month_day}         → 410 Gone
/summary/{location}/{month_day}       → 410 Gone
```

**New (V1):**

```
/v1/records/{period}/{location}/{identifier}
/v1/records/{period}/{location}/{identifier}/average
/v1/records/{period}/{location}/{identifier}/trend
/v1/records/{period}/{location}/{identifier}/summary
```

### 2. Period Types and Identifiers

| Period    | Identifier Format | Example    | Description                   |
| --------- | ----------------- | ---------- | ----------------------------- |
| `daily`   | `MM-DD`           | `01-15`    | January 15th across all years |
| `monthly` | `YYYY-MM`         | `2024-01`  | January 2024                  |
| `yearly`  | `YYYY`            | `2024`     | Year 2024                     |
| `weekly`  | `YYYY-WW`         | `2024-W03` | Week 3 of 2024                |

## Migration Examples

### JavaScript/TypeScript

**Before:**

```javascript
// Old way
const data = await fetch("/data/london/01-15");
const average = await fetch("/average/london/01-15");
const trend = await fetch("/trend/london/01-15");
const summary = await fetch("/summary/london/01-15");
```

**After:**

```javascript
// New way
const data = await fetch("/v1/records/daily/london/01-15");
const average = await fetch("/v1/records/daily/london/01-15/average");
const trend = await fetch("/v1/records/daily/london/01-15/trend");
const summary = await fetch("/v1/records/daily/london/01-15/summary");
```

### Python

**Before:**

```python
import requests

# Old way
data = requests.get('http://localhost:8000/data/london/01-15').json()
average = requests.get('http://localhost:8000/average/london/01-15').json()
trend = requests.get('http://localhost:8000/trend/london/01-15').json()
summary = requests.get('http://localhost:8000/summary/london/01-15').json()
```

**After:**

```python
import httpx

# New way
async with httpx.AsyncClient() as client:
    data = await client.get('http://localhost:8000/v1/records/daily/london/01-15')
    average = await client.get('http://localhost:8000/v1/records/daily/london/01-15/average')
    trend = await client.get('http://localhost:8000/v1/records/daily/london/01-15/trend')
    summary = await client.get('http://localhost:8000/v1/records/daily/london/01-15/summary')
```

## Response Format Changes

### Legacy Response Format

```json
{
  "weather": {
    "data": [
      { "x": 1970, "y": 15.0 },
      { "x": 1971, "y": 15.5 }
    ],
    "metadata": { "completeness": 100 }
  },
  "summary": "15.0°C. It is 2.5°C warmer than average today.",
  "trend": {
    "slope": 0.25,
    "units": "°C/decade"
  },
  "average": {
    "average": 12.5,
    "unit": "celsius",
    "data_points": 55
  }
}
```

### V1 Response Format

```json
{
  "period": "daily",
  "location": "london",
  "identifier": "01-15",
  "range": {
    "start": "1970-01-15",
    "end": "2024-01-15",
    "years": 55
  },
  "unit_group": "metric",
  "values": [
    {
      "date": "1970-01-15",
      "year": 1970,
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
    "data_points": 55
  },
  "trend": {
    "slope": 0.25,
    "unit": "°C/decade",
    "data_points": 55,
    "r_squared": null
  },
  "summary": "15.0°C. It is 2.5°C warmer than average today.",
  "metadata": {
    "total_years": 55,
    "available_years": 55,
    "missing_years": [],
    "completeness": 100.0
  }
}
```

## Data Access Changes

### Legacy Format

```javascript
// Accessing data in legacy format
const temperatures = data.weather.data;
const avgTemp = data.average.average;
const trendSlope = data.trend.slope;
const summaryText = data.summary;
```

### V1 Format

```javascript
// Accessing data in v1 format
const temperatures = data.values;
const avgTemp = data.average.mean;
const trendSlope = data.trend.slope;
const summaryText = data.summary;
```

## New Features

### 1. Multiple Time Periods

The v1 API supports different time periods:

```javascript
// Daily data (same as legacy)
const daily = await fetch("/v1/records/daily/london/01-15");

// Monthly data (new)
const monthly = await fetch("/v1/records/monthly/london/2024-01");

// Yearly data (new)
const yearly = await fetch("/v1/records/yearly/london/2024");
```

### 2. Subresource Endpoints

Access specific parts of the data without fetching everything:

```javascript
// Get only the average
const average = await fetch("/v1/records/daily/london/01-15/average");

// Get only the trend
const trend = await fetch("/v1/records/daily/london/01-15/trend");

// Get only the summary
const summary = await fetch("/v1/records/daily/london/01-15/summary");
```

### 3. Enhanced Metadata

The v1 API provides more detailed metadata:

```javascript
const data = await fetch("/v1/records/daily/london/01-15");
const metadata = data.metadata;
console.log(`Data completeness: ${metadata.completeness}%`);
console.log(`Total years: ${metadata.total_years}`);
console.log(`Available years: ${metadata.available_years}`);
```

## Backward Compatibility

- **Legacy endpoints have been removed** and return `410 Gone` responses
- Removed endpoints include helpful migration information in the response
- Response includes `X-Removed` header and `X-New-Endpoint` header
- All functionality is now available through v1 endpoints only

## Timeline

- **Phase 1 (Completed)**: Legacy endpoints were deprecated
- **Phase 2 (Completed)**: Legacy endpoints showed deprecation warnings
- **Phase 3 (Current)**: Legacy endpoints have been removed (410 Gone)

## Getting Help

- Check the API documentation at `/` for current endpoint information
- Use the test script `test_v1_api.py` to verify your implementation
- Legacy endpoints include `X-New-Endpoint` headers showing the v1 equivalent

## Example Migration Script

```javascript
// Migration helper function
function migrateToV1(legacyEndpoint, location, monthDay) {
  const v1Endpoint = legacyEndpoint
    .replace("/data/", "/v1/records/daily/")
    .replace("/average/", "/v1/records/daily/")
    .replace("/trend/", "/v1/records/daily/")
    .replace("/summary/", "/v1/records/daily/");

  if (
    legacyEndpoint.includes("/average/") ||
    legacyEndpoint.includes("/trend/") ||
    legacyEndpoint.includes("/summary/")
  ) {
    return v1Endpoint + "/" + legacyEndpoint.split("/")[2];
  }

  return v1Endpoint;
}

// Usage
const oldUrl = "/data/london/01-15";
const newUrl = migrateToV1(oldUrl, "london", "01-15");
// Result: '/v1/records/daily/london/01-15'
```
