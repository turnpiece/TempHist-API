#!/bin/bash
# clear_queue.sh — Emergency script to clear the Redis job queue.
#
# Usage:
#   REDIS_URL=redis://... ./clear_queue.sh
#   ./clear_queue.sh redis://...
#
# The script connects to Redis using the REDIS_URL environment variable
# (or a URL passed as the first argument), deletes the "job_queue" key,
# reports how many jobs were cleared, and verifies the queue is empty.

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

echo -e "${BLUE}=== TempHist-API — Job Queue Emergency Clear ===${NC}"
echo ""

# ---------------------------------------------------------------------------
# Resolve Redis URL
# ---------------------------------------------------------------------------
REDIS_URL="${REDIS_URL:-}"
if [ -z "$REDIS_URL" ] && [ $# -ge 1 ]; then
    REDIS_URL="$1"
fi

if [ -z "$REDIS_URL" ]; then
    echo -e "${RED}❌  REDIS_URL is not set.${NC}"
    echo "    Export it before running this script:"
    echo "      export REDIS_URL=redis://<host>:<port>/<db>"
    echo "    Or pass it as the first argument:"
    echo "      ./clear_queue.sh redis://<host>:<port>/<db>"
    exit 1
fi

echo -e "${BLUE}ℹ️   Redis URL: ${REDIS_URL}${NC}"
echo ""

# ---------------------------------------------------------------------------
# Delegate to admin_clear_queue.py (preferred — uses the same logic as the
# API's /admin/clear-job-queue endpoint and handles auth/decode correctly).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/admin_clear_queue.py"

if [ -f "$PYTHON_SCRIPT" ]; then
    # Ensure the Python script is executable
    chmod +x "$PYTHON_SCRIPT"

    echo -e "${YELLOW}⚡  Running admin_clear_queue.py …${NC}"
    echo ""

    if REDIS_URL="$REDIS_URL" python3 "$PYTHON_SCRIPT"; then
        echo ""
        echo -e "${GREEN}✅  Queue cleared successfully. The API should now be able to start.${NC}"
    else
        echo ""
        echo -e "${RED}❌  admin_clear_queue.py reported an error (see output above).${NC}"
        exit 1
    fi

# ---------------------------------------------------------------------------
# Fallback: use redis-cli directly if Python / the script is unavailable.
# ---------------------------------------------------------------------------
elif command -v redis-cli &>/dev/null; then
    echo -e "${YELLOW}⚡  Python script not found — falling back to redis-cli …${NC}"
    echo ""

    # Parse host, port, and db from REDIS_URL
    # Supports: redis://[user:pass@]host:port[/db]
    REDIS_HOST=$(echo "$REDIS_URL" | sed -E 's|redis://([^:@/]+@)?([^:/]+).*|\2|')
    REDIS_PORT=$(echo "$REDIS_URL" | sed -E 's|redis://[^:@/]*(:[^@/]+)?@?([^:/]+):([0-9]+).*|\3|')
    REDIS_DB=$(echo "$REDIS_URL"   | sed -E 's|.*/([0-9]+)$|\1|; t; s|.*|0|')
    REDIS_PASSWORD=$(echo "$REDIS_URL" | sed -nE 's|redis://[^:]+:([^@]+)@.*|\1|p')

    REDIS_PORT="${REDIS_PORT:-6379}"
    REDIS_DB="${REDIS_DB:-0}"

    CLI_ARGS=(-h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB")
    if [ -n "$REDIS_PASSWORD" ]; then
        CLI_ARGS+=(-a "$REDIS_PASSWORD" --no-auth-warning)
    fi

    echo -e "${BLUE}ℹ️   Connecting to ${REDIS_HOST}:${REDIS_PORT} db=${REDIS_DB}${NC}"

    # Ping
    if ! redis-cli "${CLI_ARGS[@]}" PING | grep -q "PONG"; then
        echo -e "${RED}❌  Cannot reach Redis at ${REDIS_HOST}:${REDIS_PORT}${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓   Redis connection OK${NC}"

    # Count jobs before deletion
    QUEUE_LENGTH=$(redis-cli "${CLI_ARGS[@]}" LLEN job_queue)
    echo -e "${BLUE}ℹ️   Current queue length: ${QUEUE_LENGTH}${NC}"

    if [ "$QUEUE_LENGTH" -eq 0 ]; then
        echo -e "${GREEN}✅  Queue is already empty — nothing to do.${NC}"
        exit 0
    fi

    # Delete the key
    redis-cli "${CLI_ARGS[@]}" DEL job_queue > /dev/null
    echo -e "${GREEN}✓   Deleted job_queue key (${QUEUE_LENGTH} jobs removed)${NC}"

    # Verify
    REMAINING=$(redis-cli "${CLI_ARGS[@]}" LLEN job_queue)
    if [ "$REMAINING" -eq 0 ]; then
        echo -e "${GREEN}✓   Verification passed — queue is empty${NC}"
        echo ""
        echo -e "${GREEN}✅  Queue cleared successfully. The API should now be able to start.${NC}"
    else
        echo -e "${RED}❌  Verification failed — ${REMAINING} jobs still remain${NC}"
        exit 1
    fi

else
    echo -e "${RED}❌  Neither admin_clear_queue.py nor redis-cli is available.${NC}"
    echo "    Install redis-tools (apt install redis-tools) or ensure Python 3 is in PATH."
    exit 1
fi
