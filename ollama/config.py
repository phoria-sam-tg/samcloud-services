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

# --- Stage-1 capacity offering (doc #8) ---
# The service self-reports a coarse offering tier as an `offering:<tier>`
# capability, recomputed from live pressure and updated via PATCH on change.
# Good-citizen flex on a laptop shared with brush splat training: drop the tier
# as unified memory fills (training running), restore when it frees.
#
# Signal: LOCAL unified-memory availability (memory_total - memory_used from
# vm_stat/sysctl, same collector as the stats loop). We compute from local
# memory rather than the registry's lease-based available_memory_mb because the
# gateway runs under a service token, which is scope-filtered out of resource
# reads (GET /resources/{id} -> 403). On a unified-memory Mac real memory
# pressure is in any case the truer "can I serve a model" signal, and it also
# captures training that spikes memory without holding a formal lease.
# Thresholds are available-MB, ascending: none < mini < degraded < full.
OFFERING_ENABLED = _env_bool("OFFERING_ENABLED", True)
OFFERING_POLL_SECONDS = _env_int("OFFERING_POLL_SECONDS", 30)
OFFERING_HYSTERESIS = _env_int("OFFERING_HYSTERESIS", 2)  # stable polls before a tier change
OFFERING_MINI_MB = _env_int("OFFERING_MINI_MB", 3500)      # avail < this -> none (qwen3:1.7b ~3.3GB won't fit)
OFFERING_DEGRADED_MB = _env_int("OFFERING_DEGRADED_MB", 6000)  # [mini,degraded) -> mini (tight, mini model only)
OFFERING_FULL_MB = _env_int("OFFERING_FULL_MB", 10000)     # >= this -> full; [degraded,full) -> degraded

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
