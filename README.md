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
