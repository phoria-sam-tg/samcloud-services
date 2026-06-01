#!/usr/bin/env bash
# Launch samcloud-services model gateway with ada-wsl profile.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

set -a
. "$HERE/env"
set +a

: "${SC_TOKEN_PATH:=$HOME/.samcloud/services/${SC_SERVICE_NAME:-model-service}.token}"
if [ -f "$SC_TOKEN_PATH" ]; then
    SC_TOKEN="$(cat "$SC_TOKEN_PATH")"
    export SC_TOKEN
fi

cd "$REPO"
exec .venv/bin/python -m uvicorn ollama.server:app \
    --host 0.0.0.0 --port "${SERVICE_PORT:-8800}"
