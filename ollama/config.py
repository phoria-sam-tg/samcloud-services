"""Environment-driven configuration.

All values that used to be hardcoded to slice-test/stg live here now.
Defaults target the production samcloud registry and the
claude-services-slice device.
"""

import os
import shutil
from pathlib import Path


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# --- samcloud registry ---
SC_BASE = _env("SC_BASE", "https://cloud.samtg.xyz/api/v1")
SC_TOKEN = os.environ.get("SC_TOKEN", "")
SC_DEVICE = _env("SC_DEVICE", "claude-services-slice")
SC_SERVICE_NAME = _env("SC_SERVICE_NAME", "model-service")
SC_SERVICE_ID = f"{SC_DEVICE}/{SC_SERVICE_NAME}"
SC_RESOURCE_ID = _env("SC_RESOURCE_ID", f"{SC_DEVICE}/gpu-0")

# --- auth middleware ---
SC_VERIFY_URL = _env("SC_VERIFY_URL", f"{SC_BASE}/auth/verify")
SC_REQUIRED_SCOPE = _env("SC_REQUIRED_SCOPE", f"device:{SC_DEVICE}")
AUTH_ENABLED = _env_bool("AUTH_ENABLED", True)
AUTH_CACHE_TTL = _env_int("AUTH_CACHE_TTL", 300)

# --- http server ---
SERVICE_PORT = _env_int("SERVICE_PORT", 8800)

# --- lease / lifecycle ---
COOLDOWN_SECONDS = _env_int("COOLDOWN_SECONDS", 300)
LEASE_TTL = _env_int("LEASE_TTL", 3600)
LEASE_RENEW_AT = float(_env("LEASE_RENEW_AT", "0.5"))
OLLAMA_KEEP_ALIVE = _env_int("OLLAMA_KEEP_ALIVE", -1)

# --- backends ---
OLLAMA_BASE = _env("OLLAMA_BASE", "http://localhost:11434")
MODELS_DIR = Path(_env("MODELS_DIR", str(Path.home() / "models")))
LLAMA_SERVER_BIN = _env(
    "LLAMA_SERVER_BIN",
    shutil.which("llama-server") or "/opt/homebrew/bin/llama-server",
)
VLM_PORT = _env_int("VLM_PORT", 8801)
VLM_HOST = _env("VLM_HOST", "127.0.0.1")
VLM_PYTHON = _env(
    "VLM_PYTHON",
    str(Path.home() / "code" / "mlx-vlm-server" / ".venv" / "bin" / "python"),
)
VLM_STARTUP_TIMEOUT = _env_int("VLM_STARTUP_TIMEOUT", 120)

# Force-fit a requested model by unloading whatever is resident.
# OFF by default: the contract is "publish what is available and let the
# handshake pick a model that fits", not "evict to satisfy every ask".
AUTO_EVICT = _env_bool("AUTO_EVICT", False)


# --- exo pool (Backend.EXO) ---
# ONE exo instance spanning slice + wafer, serving ONE request at a time.
# Unlike every other backend the gateway owns, we neither start it nor place
# its model: it must be launched from Terminal.app (macOS grants local-network
# access per responsible process, so a headless launch is denied), and the
# resident model is swapped out of band by whoever operates the pool.
EXO_BASE = _env("EXO_BASE", "http://192.168.1.3:52415")
EXO_ENABLED = _env_bool("EXO_ENABLED", True)
EXO_RESOURCE_ID = _env("EXO_RESOURCE_ID", f"{SC_DEVICE}/exo-pool")

# Tier names that route to the pool. Callers ask for a tier, not a model:
# swapping the resident model costs 30s-10min, so the model is a property of
# the instance and naming one in a request would be a promise we cannot keep.
EXO_TIERS = tuple(
    t.strip() for t in _env("EXO_TIERS", "think").split(",") if t.strip()
)

# Ceiling on how long an exclusive lease can outlive the request that took it.
# This is a leak bound, not an expected generation length — the happy path
# releases in a `finally`. It matters because a lease stranded on an exclusive
# resource does not degrade the pool, it closes it.
EXO_LEASE_TTL = _env_int("EXO_LEASE_TTL", 1800)

# Per-model request defaults — the same shape model-service already uses for
# its Ollama `think:false` workaround. On an exclusive resource an unbounded
# generation is not slow, it is a denial of service: one caller owns the pool
# until it stops talking. Cap every request, and let the caller lower it but
# never raise it past the cap.
EXO_MAX_TOKENS = _env_int("EXO_MAX_TOKENS", 2048)
EXO_MODEL_DEFAULTS = {
    # GLM-4.7-Flash has `thinking_toggle`, and at 6bit over two boxes its
    # thinking phase can run for minutes. We keep thinking ON — the tier is
    # called `think` and that is what it is for — and bound the total instead.
    "glm-4.7-flash": {"max_tokens": EXO_MAX_TOKENS},
}
