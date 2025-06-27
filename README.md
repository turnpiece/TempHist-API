# TempHist API

A FastAPI backend for historical temperature data using Visual Crossing.

## Features

- Caches results to reduce API calls
- Loads API key from `.env` file
- Ready to deploy on Render
- Provides historical temperature data and trends
- Includes weather forecasts
- CORS enabled for web applications

## Requirements

- Python 3.8+
- A `.env` file with:
  ```
  VISUAL_CROSSING_API_KEY=your_key_here
  OPENWEATHER_API_KEY=your_key_here
  REDIS_URL=redis://localhost:6379  # Optional, defaults to localhost
  CACHE_ENABLED=true  # Optional, defaults to true
  API_ACCESS_TOKEN=your_key_here
  ```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys
5. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

- `GET /` - API information
- `GET /weather/{location}/{date}` - Get weather for a specific date
- `GET /average/{location}/{month_day}` - Get historical average temperature (format: MM-DD)
- `GET /trend/{location}/{month_day}` - Get temperature trend over time
- `GET /summary/{location}/{month_day}` - Get a text summary of temperature data
- `GET /forecast/{location}` - Get current weather forecast
- `GET /data/{location}/{month_day}` - Get all temperature data, summary, trend, and average for a specific date (format: MM-DD)

### Example response for `/data/{location}/{month_day}`

```json
{
  "weather": {
    "data": [
      { "x": 1970, "y": 15.0 },
      { "x": 1971, "y": 15.5 },
      { "x": 1972, "y": 16.0 },
      // ... more years ...
      { "x": 2024, "y": 17.0 }
    ],
    "metadata": {
      "total_years": 55,
      "available_years": 55,
      "missing_years": [],
      "completeness": 100.0
    }
  },
  "summary": "17.0°C. This is the warmest 15th May on record. It is 2.0°C warmer than average today.",
  "trend": {
    "slope": 0.3,
    "units": "°C/decade"
  },
  "average": {
    "average": 15.0,
    "unit": "celsius",
    "data_points": 55,
    "year_range": { "start": 1970, "end": 2024 },
    "missing_years": [],
    "completeness": 100.0
  }
}
```

## Deployment

The API is configured for deployment on Render. The `render.yaml` file includes the necessary configuration.

1. Fork this repository
2. Create a new Web Service on Render
3. Connect your repository
4. Add your environment variables in the Render dashboard
5. Deploy!

## Development

- Run tests: `pytest`
- API documentation: Visit `http://localhost:8000/docs` when running locally

## License

MIT
