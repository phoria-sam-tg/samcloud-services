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
# Good-citizen flex on a box shared with other work: the tier drops as unified
# memory fills and restores when it frees.
#
# Signal: LOCAL unified-memory availability via capacity.collect(), the same
# collector the stats loop and the fit gate use — so `offering:full` means the
# same thing on every box. We compute from local memory rather than the
# registry's lease-based available_memory_mb because the gateway runs under a
# service token, which is scope-filtered out of resource reads
# (GET /resources/{id} -> 403). On a unified-memory Mac real memory pressure is
# in any case the truer "can I serve a model" signal, and it also captures work
# that spikes memory without holding a formal lease.
#
# There are deliberately no MB thresholds here. OFFERING_MINI_MB /
# _DEGRADED_MB / _FULL_MB used to band `total - used` into tiers; they were
# calibrated when the largest model was ~6GB and went stale the moment a 17.5GB
# model landed, so wafer advertised `offering:full` on ~14GiB available against
# a model that could not load without swapping. capacity.offering_tier() now
# derives the tier from what actually fits out of the live catalogue — `full`
# = everything we hold fits, `mini` = exactly one does, `none` = nothing does —
# which stays true as models come and go and needs no per-box tuning. Neither
# box ever overrode the bands in its plist.
OFFERING_ENABLED = _env_bool("OFFERING_ENABLED", True)
OFFERING_POLL_SECONDS = _env_int("OFFERING_POLL_SECONDS", 30)
OFFERING_HYSTERESIS = _env_int("OFFERING_HYSTERESIS", 2)  # stable polls before a tier change

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
