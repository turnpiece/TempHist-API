# AGENTS.md

## Cursor Cloud specific instructions

### Overview

TempHist API is a Python/FastAPI backend serving historical temperature data (50 years) via the Visual Crossing weather API. It uses Redis for caching/rate-limiting/queuing and optionally PostgreSQL for persistent cache with location aliasing.

### Services

| Service | Required | How to start |
|---------|----------|-------------|
| Redis | Yes | `redis-server --daemonize yes` |
| FastAPI dev server | Yes | `uvicorn main:app --reload --host 0.0.0.0 --port 8000` |
| Job Worker | Optional (async jobs) | `python3 worker_service.py` |
| PostgreSQL | Optional (persistent cache) | Not needed locally; app degrades gracefully |

### Required environment variables

Secrets are injected automatically. Key ones: `REDIS_URL`, `VISUAL_CROSSING_API_KEY`, `API_ACCESS_TOKEN`, `TEMPHIST_PG_DSN`. See `README.md` for the full list.

### Running tests

```bash
python3 -m pytest tests/ -v
```

All tests use mocks and do not require external services. Redis does not need to be running for tests.

### Running the dev server

1. Start Redis: `redis-server --daemonize yes`
2. Start the server: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
3. Verify: `curl http://localhost:8000/health`

### Gotchas

- `pip install` puts binaries in `~/.local/bin`; ensure this is on `PATH`.
- The `.env` file is loaded relative to `main.py`'s directory, not the working directory.
- Firebase auth is optional; `API_ACCESS_TOKEN` in the `Authorization: Bearer` header is sufficient for all endpoints during development.
- The Swagger UI at `/docs` requires external CDN access (cdn.jsdelivr.net) for its JS/CSS assets.
- `CACHE_WARMING_ENABLED` should be `false` for local dev to avoid background API calls to Visual Crossing.
- `RATE_LIMIT_ENABLED` can be `false` for local dev to simplify testing.
- The `package.json` in the repo root is vestigial (only a `cors` npm dependency) and irrelevant to the Python application.
