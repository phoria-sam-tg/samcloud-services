#!/usr/bin/env bash
# Launch the samcloud-services model gateway with the ada-wsl profile.
#
# Deliberately does NOT read the token or export SC_TOKEN. `ada/env` names
# SC_TOKEN_FILE and the process reads that file itself, so the bearer never
# enters the environment and is not inherited by anything the gateway spawns
# (#845). The previous version of this script did `SC_TOKEN="$(cat …)"; export
# SC_TOKEN`, which is the exact shape that made a token readable through
# `ps eww` to any process of the same uid.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

set -a
. "$HERE/env"
set +a

cd "$REPO"
exec .venv/bin/python -m uvicorn ollama.server:app \
    --host 0.0.0.0 --port "${SERVICE_PORT:-8800}"
