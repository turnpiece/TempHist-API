# AGENTS.md

## AI agent instructions

### Overview

TempHist API is a Python/FastAPI backend serving historical temperature data (50 years) via Open-Meteo (ERA5 reanalysis data). Open-Meteo is free for non-commercial use (no API key); a paid API key unlocks higher rate limits. It uses Redis for caching/rate-limiting/queuing and optionally PostgreSQL for persistent cache with location aliasing.

### Services

| Service | Required | How to start |
|---------|----------|-------------|
| Redis | Yes | `redis-server --daemonize yes` |
| FastAPI dev server | Yes | `uvicorn main:app --reload --host 0.0.0.0 --port 8000` |
| Job Worker | Optional (async jobs) | `python3 worker_service.py` |
| PostgreSQL | Optional (persistent cache) | Not needed locally; app degrades gracefully |

Or use `./start.sh` to start both the API server and worker together via uv.

### Required environment variables

Secrets are injected automatically. Key ones: `REDIS_URL`, `API_ACCESS_TOKEN`, `TEMPHIST_PG_DSN`, `MAPBOX_TOKEN`. See `README.md` for the full list.

`VISUAL_CROSSING_API_KEY` is not used — the weather source is Open-Meteo, which requires no API key.

### Running tests

```bash
python3 -m pytest tests/ -v
```

All tests use mocks and do not require external services. Redis does not need to be running for tests.

### Linting

[ruff](https://docs.astral.sh/ruff/) is configured in `pyproject.toml`. Run via `uvx` (no separate install needed):

```bash
uvx ruff check .            # show lint issues
uvx ruff check . --fix      # auto-fix safe lint issues
uvx ruff format .           # fix whitespace/blank-line issues (run this too — the linter skips these)
```

Rules enabled: `E/W` (style), `F` (unused imports/vars), `I` (import order), `ASYNC` (missing awaits), `S` (security). Line length is 120. `E501` (line-too-long) and `S101` (assert in tests) are ignored.

### Running the dev server

1. Start Redis: `redis-server --daemonize yes`
2. Start the server: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
3. Verify: `curl http://localhost:8000/health`

### Gotchas

- Use `uv` for dependency management (`uv pip install -r requirements.txt`). `uv.lock` is gitignored.
- The `.env` file is loaded from the project root via `config.DOTENV_PATH` (directory containing `config.py` / `main.py`), not from the process working directory.
- Firebase auth is optional; `API_ACCESS_TOKEN` in the `Authorization: Bearer` header is sufficient for all endpoints during development.
- Firebase App Check (`APP_CHECK_ENFORCEMENT`) defaults to `off`. When the frontend has `VITE_RECAPTCHA_SITE_KEY` set it sends an `X-Firebase-AppCheck` header; the API's CORS config allows this header so browser preflights don't fail.
- The Swagger UI at `/docs` requires external CDN access (cdn.jsdelivr.net) for its JS/CSS assets.
- `CACHE_WARMING_ENABLED` should be `false` for local dev to avoid spurious background requests.
- `RATE_LIMIT_ENABLED` can be `false` for local dev to simplify testing.
- The `package.json` in the repo root is vestigial (only a `cors` npm dependency) and irrelevant to the Python application.
