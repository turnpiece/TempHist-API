#!/usr/bin/env bash
set -euo pipefail

echo "Starting TempHist API local stack..."

# Ensure uv uses its own .venv project environment, not any activated virtualenv
unset VIRTUAL_ENV

# 1) Ensure local dependencies are synced using uv
uv sync

# 2) Optionally start Redis (best-effort)
if command -v redis-cli >/dev/null 2>&1; then
  if redis-cli ping >/dev/null 2>&1; then
    echo "Redis already running."
  elif command -v brew >/dev/null 2>&1; then
    echo "Attempting to start Redis via Homebrew..."
    brew services start redis || true
  elif command -v systemctl >/dev/null 2>&1; then
    echo "Attempting to start Redis via systemctl..."
    sudo systemctl start redis || sudo systemctl start redis-server || true
  else
    echo "Redis not running and no service manager detected. Start it manually if needed."
  fi
else
  echo "redis-cli not found; skipping Redis auto-start."
fi

# 3) Optionally start PostgreSQL (best-effort)
if command -v psql >/dev/null 2>&1; then
  if psql -d postgres -c "select 1" >/dev/null 2>&1; then
    echo "PostgreSQL already running."
  elif command -v brew >/dev/null 2>&1; then
    echo "Attempting to start PostgreSQL via Homebrew..."
    brew services start postgresql || true
  elif command -v systemctl >/dev/null 2>&1; then
    echo "Attempting to start PostgreSQL via systemctl..."
    sudo systemctl start postgresql || true
  else
    echo "PostgreSQL not running and no service manager detected. Start it manually if needed."
  fi
else
  echo "psql not found; skipping PostgreSQL auto-start."
fi

# 4) Start the API server in background using the synced .venv
echo "Starting API server..."
.venv/bin/python -m uvicorn main:app --reload &

# 5) Start the worker in foreground using the synced .venv
echo "Starting worker..."
.venv/bin/python job_worker.py
