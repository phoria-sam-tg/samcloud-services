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
# registry's lease-based available_memory_mb because on a unified-memory Mac
# real memory pressure is the truer "can I serve a model" signal: it captures
# work that spikes memory without holding a formal lease, and
# available_memory_mb is spec minus leases and never consults utilisation at
# all (#750 — it reported 36864 MB on wafer at 70.6%).
#
# This comment used to give a second reason — that the gateway runs under a
# service token scope-filtered out of resource reads (GET /resources/{id} ->
# 403). That was true when it was written: wafer's server.log has six such 403s
# on 2026-07-02, the day of this workaround, and the endpoint was never called
# again. It is not true now — the gateway runs as a dedicated agent identity
# (wafer-model-service, user 60, scopes exactly ["device:wafer-services"]) and
# reads resources fine (#760). An earlier edit "corrected" the 403 to 404 and
# was wrong to: the code changed under the comment, the comment did not
# misreport it.
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

# How long we will wait for the pool to finish one generation. Two boxes, ring
# pipeline parallelism and a reasoning model: minutes is normal, and the first
# request after a placement is the slowest (cold KV cache).
EXO_GENERATE_TIMEOUT = _env_int("EXO_GENERATE_TIMEOUT", 1500)

# Ceiling on how long an exclusive lease can outlive the request that took it.
# This is a leak bound, not an expected generation length — the happy path
# releases in a `finally`. It matters because a lease stranded on an exclusive
# resource does not degrade the pool, it closes it.
#
# It MUST be longer than EXO_GENERATE_TIMEOUT, and that ordering is the whole
# point of having both. The registry expires an active lease the moment it
# passes `expires_at` and offers no way to extend one; renewing by
# release-then-reacquire would open a window where a third party can take an
# exclusive resource out from under a running generation. So the only safe
# arrangement is a lease that outlives the longest request it can be covering.
# Set them equal (as an earlier version of this did, both at 1800) and a
# generation that runs to its timeout finishes just as its lease lapses — the
# pool silently becomes grantable to someone else while we are still using it,
# which is the exact collision the exclusive lease exists to prevent.
EXO_LEASE_TTL = _env_int("EXO_LEASE_TTL", 1800)

# Enforce the ordering rather than trusting whoever edits the env next. Widening
# the lease is the safe direction: a lease that is too long delays the pool for
# other consumers, while one that is too short breaks exclusivity outright.
_EXO_TTL_MARGIN = 300
if EXO_LEASE_TTL < EXO_GENERATE_TIMEOUT + _EXO_TTL_MARGIN:
    EXO_LEASE_TTL = EXO_GENERATE_TIMEOUT + _EXO_TTL_MARGIN

# Per-model request defaults — the same shape model-service already uses for
# its Ollama `think:false` workaround. On an exclusive resource an unbounded
# generation is not slow, it is a denial of service: one caller owns the pool
# until it stops talking. Cap every request, and let the caller lower it but
# never raise it past the cap.
EXO_MAX_TOKENS = _env_int("EXO_MAX_TOKENS", 2048)

# How long `GET /models` may reuse a pool status reading. Status display only —
# the serving path always reads fresh, so this cannot route a request at a stale
# resident model.
EXO_STATUS_CACHE_S = _env_int("EXO_STATUS_CACHE_S", 5)

# Gap between the two `busy` samples the wedge guard takes before it declines.
# One sample is an observation of a moving state; two a short interval apart
# distinguish "a generation just finished" from "the slot is occupied and is
# not progressing".
EXO_WEDGE_RECHECK_S = float(_env("EXO_WEDGE_RECHECK_S", "1.0"))
EXO_MODEL_DEFAULTS = {
    # GLM-4.7-Flash has `thinking_toggle`, and at 6bit over two boxes its
    # thinking phase can run for minutes. We keep thinking ON — the tier is
    # called `think` and that is what it is for — and bound the total instead.
    "glm-4.7-flash": {"max_tokens": EXO_MAX_TOKENS},
}
