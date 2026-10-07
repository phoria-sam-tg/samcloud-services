"""Environment-driven configuration.

All values that used to be hardcoded to slice-test/stg live here now.
Defaults target the production samcloud registry and the
claude-services-slice device.
"""

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger("model-config")


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

# The credential file the bearer is read from when SC_TOKEN is NOT in the
# environment (#845). Keeping it out of the environment is the whole point: the
# launcher used `set -a` around its env file, so the value was exported and every
# child process the gateway starts inherited it, and `ps eww <pid>` showed it to
# any reader of that uid.
#
# THERE IS DELIBERATELY NO DEFAULT. This was `~/.samcloud/token` and that was
# wrong in a way worth spelling out, because the wrong version looked correct:
#
#   `~/.samcloud/token` is the SEAT's bearer, not a service's. On wafer it exists
#   and holds `claude-wafer-services` — three group scopes — while that gateway's
#   own identity is `wafer-model-service` with `device:wafer-services` alone. With
#   a default, dropping the export would have started the gateway successfully,
#   logged `SC_TOKEN source: file ~/.samcloud/token`, and run it on a WIDER
#   identity than it is entitled to. Not an outage: a silent privilege widening
#   behind a green-looking startup line, which is strictly worse than a failure.
#
# So when SC_TOKEN is absent and SC_TOKEN_FILE is unset, the source is NOT FOUND
# and startup says so. Each box sets SC_TOKEN_FILE explicitly at the path holding
# that gateway's own token.
#
# The rule this encodes is narrower than "a service must not read
# ~/.samcloud/token" (claude-wafer-services' refinement, and correct): a seat
# token must never be reached for IMPLICITLY, as a default or a fallback, where a
# service identity was intended. A seat's own scripts posting as the seat on
# purpose — wafer's nerf/4dgs heartbeats do exactly that — are a different thing
# and are fine.
SC_TOKEN_FILE = _env("SC_TOKEN_FILE", "")


def _read_token_file(path: str) -> str:
    """The bearer from a 0600 credential file, or "" if there is nothing to read.

    Accepts both shapes that exist in practice: a raw token, and a curl header
    file (`Authorization: Bearer <tok>`), because both live side by side in these
    accounts and picking one would make the other fail silently.

    A permissive mode WARNS and still returns the token. Refusing would turn a
    permission bit into a total authentication outage, and the file being readable
    by others is a different problem from this gateway being able to start. The
    warning names the mode so it is actionable.
    """
    try:
        full = os.path.expanduser(path)
        st = os.stat(full)
        mode = st.st_mode & 0o777
        if mode & 0o077:
            log.warning(
                f"credential file {full} is mode {oct(mode)}, not 0600 — readable "
                f"beyond its owner. Using it anyway; refusing would make a "
                f"permission bit an authentication outage (#845)."
            )
        raw = open(full).read()
    except OSError:
        return ""
    for line in raw.splitlines():
        if line.lower().startswith("authorization:"):
            v = line.split(":", 1)[1].strip()
            return v.split(None, 1)[1].strip() if v.lower().startswith("bearer") else v
    return raw.strip()


_sc_token_env = os.environ.get("SC_TOKEN", "")
SC_TOKEN = _sc_token_env or (_read_token_file(SC_TOKEN_FILE) if SC_TOKEN_FILE else "")
# Which source won, because "the fix is deployed" and "the launcher still exports
# it" are indistinguishable without saying so, and an env-sourced token means the
# hygiene change is not actually in effect. Never the value.
SC_TOKEN_SOURCE = ("environment" if _sc_token_env
                   else f"file {SC_TOKEN_FILE}" if SC_TOKEN else "NOT FOUND")
# NOT FOUND is loud, and it names WHICH of the two ways it happened. "unset" is a
# box whose launcher was changed without setting the path; "unreadable or empty"
# is a path that is set and wrong. Those want different fixes and a bare NOT FOUND
# would send someone to the wrong one.
if SC_TOKEN_SOURCE == "NOT FOUND":
    log.error(
        "no samcloud bearer: SC_TOKEN is not in the environment and "
        + (f"SC_TOKEN_FILE={SC_TOKEN_FILE} is unreadable or empty"
           if SC_TOKEN_FILE else
           "SC_TOKEN_FILE is unset — set it to the path of THIS gateway's own "
           "token file. There is no default on purpose: a seat's "
           "~/.samcloud/token is not a fallback for a service identity (#845)")
        + ". Every registry call will fail 401."
    )
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
# When a model lease is re-taken, as a fraction of its TTL. The loop sleeps
# this before each attempt, so it is also how many attempts fall inside one
# lease — and that count is the whole reason for the value.
#
# 0.5 gave exactly ONE: at TTL 3600 the renewal fires at 1800, and if it is
# REFUSED the next attempt lands at 3600, which is when the unextended lease
# expires. One attempt that races the expiry is not a retry, and D2's
# guarantee — a refused renewal keeps a lease the registry still accounts for
# — was therefore bounded at one interval rather than holding (#861 D2,
# claude-wafer-services).
#
# 0.25 gives attempts at 900 / 1800 / 2700 inside 3600: two renewals may be
# missed and a third still lands, with 900s to spare. That is the same
# arithmetic `EXO_LEASE_RENEW_PCT` sets out for the pool lease — three
# attempts inside 80% of the TTL, 2700/3600 = 75% — and it is 25% there for
# the same reason it is here rather than 33%.
#
# The cost is one extra swap per model every 15 minutes instead of 30, each
# briefly double-counting that model in the registry while the new lease is
# granted and the old one released. Negligible against the failure it removes.
LEASE_RENEW_AT = float(_env("LEASE_RENEW_AT", "0.25"))
OLLAMA_KEEP_ALIVE = _env_int("OLLAMA_KEEP_ALIVE", -1)

# --- The served context window (ticket #903) --------------------------------
# What `num_ctx` we ask Ollama for. Nothing used to set it, and the absence was
# not neutral: Ollama DERIVES a default from free VRAM at the moment of load
# and logs it as `vram-based default context`. On slice that read
# `total_vram="48.0 GiB" default_num_ctx=262144` nineteen times — and
# `total_vram="0 B" default_num_ctx=4096` twice, on 2026-08-25 and 2026-08-26,
# when llama-server GPU discovery timed out and Ollama fell back to CPU. So the
# window a caller got was whatever this box happened to have free that minute,
# with a 64x spread across observed loads and no way to find out which one it
# had been handed.
#
# That is worse than a cap, because a cap is at least a number. A harness
# claiming more context than the endpoint serves gets truncated or refused
# while looking configured, which is exactly what #884 was about to be
# configured into. Pinning it makes the window a decision instead of a
# side-effect of memory pressure, and makes it a number we can publish.
#
# Unset (0) keeps the old behaviour — we send no num_ctx and Ollama derives
# one. That is the right default for a box that has not been measured; it is
# not the right setting for a box with a consumer sizing against it.
#
#   OLLAMA_NUM_CTX=131072                       every ollama model
#   OLLAMA_NUM_CTX_MODELS=qwen3.8:27b-mlx=262144,qwen3:1.7b=40960
#
# Per-model wins over the global. Both are CEILINGS, not reservations: the MLX
# engine grows the KV cache as a conversation fills it, measured on slice
# 2026-10-08 as `ollama ps` size_vram climbing 26,681 -> 30,311 MB on one
# resident model at a fixed num_ctx=262144. So raising this number does not
# cost memory at load; a long conversation does, later.
OLLAMA_NUM_CTX = max(0, _env_int("OLLAMA_NUM_CTX", 0))


def _num_ctx_models() -> dict:
    """Parse OLLAMA_NUM_CTX_MODELS — `name=tokens` pairs, comma separated.

    A malformed pair is dropped with a warning rather than taken as 0: 0 is
    "let Ollama derive one", which is the behaviour this setting exists to
    stop, so a typo must not quietly mean it.
    """
    raw = os.environ.get("OLLAMA_NUM_CTX_MODELS", "").strip()
    out: dict[str, int] = {}
    if not raw:
        return out
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        name = name.strip()
        try:
            n = int(value.strip())
        except ValueError:
            log.warning(f"OLLAMA_NUM_CTX_MODELS: not a number, ignoring: {pair!r}")
            continue
        if not name or n <= 0:
            log.warning(f"OLLAMA_NUM_CTX_MODELS: unusable pair, ignoring: {pair!r}")
            continue
        out[name] = n
    return out


OLLAMA_NUM_CTX_MODELS = _num_ctx_models()


def ollama_num_ctx(model: str) -> int:
    """The num_ctx configured for this model, or 0 for "we do not set it".

    Matches the exact name first, then the name without an Ollama `:tag`, so
    `OLLAMA_NUM_CTX_MODELS=qwen3.8:27b-mlx=262144` covers the model however it
    was written and a bare `qwen3.8=…` still covers every tag of it.
    """
    if model in OLLAMA_NUM_CTX_MODELS:
        return OLLAMA_NUM_CTX_MODELS[model]
    base = model.split(":", 1)[0]
    if base in OLLAMA_NUM_CTX_MODELS:
        return OLLAMA_NUM_CTX_MODELS[base]
    return OLLAMA_NUM_CTX

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

# --- The offer (#861) -------------------------------------------------------
# What /warm, /v1/models and /status advertise, and the two things that stop it
# being a lie.
#
# WHY A WINDOW. A single reading is not a stable basis for an offer. Measured
# on an IDLE wafer 2026-09-29, 24 samples over 2 minutes: memory_available_mb
# min 9988, median 10711, max 12938 — a 2950 MB spread from ordinary desktop
# churn, which is larger than several models in the catalogue. On ada it is
# worse and for a different reason: inside ONE continuous render, free memory
# swung 188 -> 2963 MiB, so the spread is not noise around a level, it is the
# work's own oscillation.
#
# So the offer is computed over a trailing window, taking the MINIMUM
# available and the MAXIMUM device-in-use across it. That single choice gives
# both behaviours Sam asked for, with no separate withdraw/restore thresholds:
#
#   withdraw  happens on the first bad reading, because it lands in the window
#             immediately and the minimum drops at once  ("within one reconcile")
#   restore   cannot happen until every reading in the window agrees, because
#             the old trough has to AGE OUT first          ("restores after")
#
# It also answers the objection that "two consecutive readings agree" does not:
# two samples 5s apart can both sit at the top of a render's swing, but they
# cannot both be the minimum of a window longer than the swing. Size the window
# past the oscillation period you are protecting against, not past the noise.
OFFER_WINDOW_S = _env_int("OFFER_WINDOW_S", 60)

# Floor on how often the hardware is actually read behind the offer, and
# therefore a ceiling on how big the window can get.
#
# `/warm` is auth-exempt, on a port bound to 0.0.0.0, and every call used to
# run the collector and append a sample: three subprocesses per request on
# Metal (15.3ms measured on wafer 2026-09-29) and a list pruned by time but
# never by count. So an unauthenticated caller set the sample rate. At ~333
# req/s — which one client on the LAN reaches without trying — that is ~1,000
# process spawns a second on a box whose whole purpose is to be a good
# neighbour to somebody's Unity session, and ~20,000 entries inside a 60s
# window (4.05ms of list arithmetic per call, and the total work grows with
# the SQUARE of the poll rate, because each of those calls also walks it).
#
# Rate-limiting the READ fixes both at once: at one sample per second the
# window holds at most OFFER_WINDOW_S + 1 entries whatever the request rate.
#
# WHY NOT A COUNT CAP, which is the obvious fix and is the unsafe one. Capping
# `_readings` to the N most recent entries evicts the OLDEST first — and the
# oldest is exactly where the trough lives after a render starts. A flood of
# requests would push the low reading out of the window early and restore the
# offer while the render was still running, which is the one direction this
# whole mechanism exists to prevent. Under-sampling is safe; forgetting is not.
#
# One second is invisible to every decision made from this: `stats_loop`
# samples at 15s and the window is 60s, so nothing here resolves anything
# finer. (Those two numbers are coupled — 15s into 60s is what guarantees ~4
# samples on a box with no traffic at all. Lengthening `stats_loop` thins the
# window silently; claude-wafer-services, reviewing PR #27.)
OFFER_MIN_SAMPLE_INTERVAL_S = float(
    _env("OFFER_MIN_SAMPLE_INTERVAL_S", "1.0"))

# Hard ceiling on the window's length, independent of the rate limit above.
#
# The rate limit already bounds it at OFFER_WINDOW_S / interval + 1 = 61 by
# default, so this never engages in normal operation. It is here for the case
# where that reasoning stops holding — the interval lowered, a caller reaching
# `_record_reading` by another path, a future sampler — because the cost of
# being wrong about it is borne by an unauthenticated endpoint.
#
# IT MERGES RATHER THAN DROPS, and that is the whole design. Dropping the
# oldest entries is the obvious cap and the unsafe one: the oldest entry is
# exactly where the trough sits once a render has started, so a flood of
# requests would evict the low reading early and restore the offer while the
# render was still running — the one direction this mechanism exists to
# prevent. So over the cap, the two oldest entries collapse into one carrying
# the MINIMUM available and the MAXIMUM device-in-use of the pair, stamped
# with the LATER of their two timestamps. No extreme is lost, and the merged
# entry ages out no sooner than the newer of its parts would have (it lives
# slightly longer, which delays restoration — the conservative direction).
#
# AND IT IS MEANT NEVER TO FIRE. At the defaults the rate floor above holds a
# natural window near 60 entries against this 120, so in production the cap is
# dead code — which is the point, not an argument for tightening it to 60. It
# exists for the case where the floor stops holding, and a cap that engages in
# normal operation would be merging readings the window was entitled to keep
# separate (claude-wafer-services, reviewing PR #27).
OFFER_MAX_SAMPLES = max(4, _env_int("OFFER_MAX_SAMPLES", 120))

# THE "SOMEONE IS WORKING" GATE, and it is OFF until a box measures its floor.
#
# `available` answers "do I fit". This answers "is anyone working", and they are
# different questions — they coincide only when the other tenant takes the whole
# device, which is every render we have observed so far and cannot be assumed.
# A lighter render holding 20 GB of a 48 GB board clears any floor we could set
# and only this term would see it.
#
# FOREIGN_IDLE_MB is the box's own idle floor: device memory in use when nobody
# is working. It has NO safe default and none is supplied — a wrong floor is
# worse than no gate, because too low permanently withdraws the offer and too
# high never fires. Unset means the gate does not run, which is exactly today's
# behaviour. What is known so far:
#
#   wafer   ~1000-1200 MB idle (WindowServer and friends), ~3% of 36864
#   ada     ONE reading of 5515 MB, 11% of 49140, taken 2026-09-28 22:5x by
#           samclaude-admin. NOT a baseline: a single sample cannot distinguish
#           a floor from a trough, and every subsequent observation has had a
#           46 GB render in it. claude-ada is sampling for a real idle window.
#
# FOREIGN_MARGIN_MB is how far above the floor counts as work rather than
# drift. It wants to clear the idle spread, not the render — a render is three
# orders of magnitude above either floor above and needs no margin to detect.
#
# Parsed by hand rather than through _env_int, because _env_int's contract is
# "fall back to the default", and here there IS no default — falling back to a
# number would be inventing the measurement the comment above says we do not
# have. An unparseable value leaves the gate OFF and says so, which is the only
# safe reading of "the operator meant something we cannot understand".
def _foreign_idle_mb() -> "int | None":
    raw = os.environ.get("FOREIGN_IDLE_MB", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        log.error(
            f"FOREIGN_IDLE_MB={raw!r} is not an integer — the "
            f"work-in-progress gate stays OFF. It has no default: set it to "
            f"this box's measured idle device memory in MB, or leave it unset."
        )
        return None
    if v < 0:
        log.error(
            f"FOREIGN_IDLE_MB={v} is negative — the gate stays OFF. A floor "
            f"below zero fires on every reading and would withdraw this "
            f"node's offer permanently."
        )
        return None
    return v


FOREIGN_IDLE_MB = _foreign_idle_mb()
FOREIGN_MARGIN_MB = max(0, _env_int("FOREIGN_MARGIN_MB", 1024))

# How long a refused lease counts as evidence that somebody else is working.
#
# WHY A LEASE REFUSAL IS A SEPARATE SIGNAL AT ALL, when foreign_mb measures
# the device directly: because there is one case where foreign_mb is wrong in
# the dangerous direction, and it is ada's case. Under WDDM the driver may
# page OUR allocation out to system memory to satisfy a render. If it does,
# `memory.used` shows the render's ~44 GB while `own_mb` still reports the
# model we think we hold — so `foreign = used - own` comes out small and the
# gateway concludes nobody is working, in the middle of a render. The
# registry sees it from the other side and refuses our renewal, because the
# resource really is oversubscribed. One signal covers the other's blind spot.
#
# ONLY CONTENTION COUNTS, never a transport failure. `queued` and `conflict`
# mean the registry considered the request and said the resource is full;
# `error` means we could not ask. ada's gateway has been 404ing every lease
# request for months on a scope it never had (#861), and if that counted, the
# node would read a permanent authentication fault as a permanent render and
# never offer anything — a failure that looks exactly like caution.
#
# Short, because it is corroboration and not the primary reading. The fit
# check against `available` is what actually protects during a render (free
# memory sits at 188-2963 MiB while one runs), and every load attempt and
# every renewal refreshes this. Stale evidence is not evidence.
LEASE_CONTENTION_TTL_S = max(0, _env_int("LEASE_CONTENTION_TTL_S", OFFER_WINDOW_S))

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

# --- speech to text (Backend.WHISPER) ---
# A fourth gateway-owned backend, the same shape as mlx-vlm: a child process on
# its own port, started on demand, leased, and killed after the idle cooldown.
# It is a separate process rather than an import for two reasons that are
# properties of this box, not preferences:
#
#   1. mlx-whisper pulls mlx, numba, llvmlite and scipy — 485MB of wheels that
#      the gateway itself never calls. Keeping them behind an interpreter the
#      gateway only execs means a broken whisper install cannot stop the
#      gateway importing.
#   2. A transcription's peak is 2.5GB (measured below). Killing a process
#      returns that to the OS; dropping a Python reference to an MLX array
#      returns it to MLX's buffer cache, which this box's own history says is
#      not the same thing.
WHISPER_ENABLED = _env_bool("WHISPER_ENABLED", True)
WHISPER_HOST = _env("WHISPER_HOST", "127.0.0.1")
# 8805, and the first choice of 8803 is worth recording because the check that
# cleared it could not see what it was checking. Several samcloud devices are
# separate ACCOUNTS on one Mac, so they share one port space, and
# `lsof -iTCP:8803 -sTCP:LISTEN` run as this account listed nothing while
# ios-dev-slice/tapes-landing was listening on it. The child then started, loaded
# its weights, and died on `[errno 48] address already in use`.
# Two things that do see a port held by another account: `GET /services` on the
# registry, which records the port of every service on the fleet, and a bind.
WHISPER_PORT = _env_int("WHISPER_PORT", 8805)
# The interpreter that has mlx-whisper. Deliberately NOT the gateway's own:
# see requirements-whisper.txt for what goes in it.
WHISPER_PYTHON = _env(
    "WHISPER_PYTHON",
    str(Path.home() / "code" / "mlx-whisper-server" / ".venv" / "bin" / "python"),
)
# Spawn to first healthy /health. Measured on slice with weights already in the
# HF cache: 3-5s. The budget is wide because a model whose weights are NOT yet
# cached downloads them at startup, and 1.6GB over a domestic link is minutes.
WHISPER_STARTUP_TIMEOUT = _env_int("WHISPER_STARTUP_TIMEOUT", 300)
# One transcription. whisper-large-v3-turbo runs ~8.6x realtime on slice
# (measured: 308s of audio in 35.9s), so this covers about four hours of audio.
WHISPER_REQUEST_TIMEOUT = _env_int("WHISPER_REQUEST_TIMEOUT", 1800)
# Refuse an upload larger than this before reading it. It bounds disk in the
# spool and, loosely, the runtime above: 200MB of AAC at 64kbit/s is ~7 hours.
WHISPER_MAX_UPLOAD_MB = _env_int("WHISPER_MAX_UPLOAD_MB", 200)
# Where an upload lands on its way to the child. The gateway writes here and
# deletes in a finally; the child refuses any path that is not inside it, so a
# request cannot name a file it did not upload.
WHISPER_SPOOL_DIR = Path(
    _env("WHISPER_SPOOL_DIR", str(Path.home() / "var" / "samcloud-services" / "spool" / "whisper"))
)
# The child's stdout and stderr, appended. Not DEVNULL: the failures this
# backend has are startup failures — a venv without mlx-whisper, a model id
# that is not a repo, a full disk mid-download — and all of them are only
# legible in the traceback the child prints on its way out. `load_whisper_model`
# reads the tail of this file into the error it raises, so a caller is told why
# and not just that.
WHISPER_LOG_FILE = Path(
    _env("WHISPER_LOG_FILE", str(Path.home() / "var" / "samcloud-services" / "logs" / "whisper-child.log"))
)
# Every audio format the endpoint accepts is decoded by this binary, so it is
# named rather than found on PATH: the gateway and its children are started by
# launchd, whose PATH does not include /opt/homebrew/bin.
FFMPEG_BIN = _env("FFMPEG_BIN", shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")

# Force-fit a requested model by unloading whatever is resident.
# OFF by default: the contract is "publish what is available and let the
# handshake pick a model that fits", not "evict to satisfy every ask".
AUTO_EVICT = _env_bool("AUTO_EVICT", False)


# --- exo pool (Backend.EXO) ---
# ONE exo instance spanning slice + wafer, serving ONE request at a time.
# Unlike every other backend the gateway owns, we neither start it nor place
# its model: the resident model is swapped out of band by whoever operates the
# pool. That part is unchanged. The *launch* half is not:
#
# This used to say the pool had to be started from Terminal.app, because macOS
# grants local-network access per responsible process and a headless launch was
# believed to be denied. Measured 2026-09-22 (#806), as unix `claude-services`
# spawned by sshd with no GUI session and no Terminal: UDP multicast receive on
# exo's discovery group ff12::e0a1:de89 joined on all 14 interfaces and took 23
# announcement packets from the same peers `sam` sees, in the same 12 seconds;
# a node started that way then formed the ring. Both nodes now run under their
# box's services account. What would disconfirm it: a node that discovers no
# peer when started from a LaunchDaemon specifically — the daemon domain is the
# one launch shape not yet exercised, and the earlier claim was about the GUI
# launchd *agent* domain, which is a different thing again.
#
# The endpoint is localhost because each box now runs its own node, so the pool
# is reachable in-process-tree with no LAN hop and no dependence on which node
# holds master. Read `/state` from both ends before concluding a node is out of
# the ring: a node that has just joined an established master serves a stale,
# incomplete replica for up to a minute (#806).
EXO_BASE = _env("EXO_BASE", "http://localhost:52415")
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

# How long a generation may go silent AFTER it has started producing tokens.
#
# Distinct from EXO_GENERATE_TIMEOUT, which bounds the whole request and has to
# be generous because prefill is legitimately slow (42s measured on a
# 12,252-token prompt). This one bounds the gap BETWEEN tokens, and only once
# the first has arrived — so it can be tight without ever cutting a slow start.
#
# Measured on this pool (#830): a healthy decode emits a chunk every ~29ms,
# p99 52-65ms, and the largest gap ever seen between two tokens was 3.511s.
# A blocked decode, by contrast, produces nothing at all — three times on
# 2026-09-25, for 12, 27 and 58 minutes. 60s is ~17x the worst healthy gap and
# still an order of magnitude faster than EXO_GENERATE_TIMEOUT at noticing.
#
# WHY THIS MATTERS BEYOND ONE REQUEST: while we wait, we hold the pool's
# exclusive lease. The placement guard treats "runners busy with no lease" as
# its wedge signal, so a gateway that waits the full 1500s also hides the wedge
# from the thing that can repair it. Aborting here releases the lease and lets
# the guard act, which turns ~26 minutes of unattended recovery into ~2-3.
# exo skips PrefillProgressChunk in its OpenAI adapter, so every SSE line we
# see is already a real token — "first line" and "first token" are the same.
EXO_STALL_TIMEOUT = _env_int("EXO_STALL_TIMEOUT", 60)

# How long the pool may take to produce its FIRST token, as a function of the
# prompt it was given.
#
# EXO_STALL_TIMEOUT bounds the gap BETWEEN tokens and cannot bound the gap before
# the first one, because prefill is legitimately silent — and that is where the
# 2026-09-27 wedge sat. A generation received at 14:17:56 finished prefilling
# 1,463 tokens at 14:17:59.878 and then never reached `Starting decode`: no tokens
# ever, so the inter-token rule never armed, and the only bound left was
# EXO_GENERATE_TIMEOUT. The pool was shut for 28 minutes (25 of them the gateway
# waiting, 3 the guard recovering) for a stall detectable in about 70 seconds.
#
# The deadline has to scale with the prompt or it cuts real prefills: measured on
# this pool, cold prefill runs at 180-405 tok/s (4.89s for 1,474 tokens; 15.29s
# for 6,197; 37.53s for 11,670). 150 tok/s is below every sample, so the rate is
# a floor rather than an average, and the margin covers the KV-cache transition
# that follows prefill — measured at 0.00s and 17.22s, the second being the step
# this wedge died on.
#
#   1,463 tokens  ->  1463/150 + 60  =  ~70s
#   12,288 tokens -> 12288/150 + 60  = ~142s
#
# Never longer than the whole-request budget: a deadline that exceeds it could not
# fire. #830.
EXO_FIRST_TOKEN_RATE_TPS = _env_int("EXO_FIRST_TOKEN_RATE_TPS", 150)
EXO_FIRST_TOKEN_MARGIN_S = _env_int("EXO_FIRST_TOKEN_MARGIN_S", 60)


def exo_first_token_deadline(prompt_tokens: int | None) -> float:
    """Seconds to allow before the first token, or the whole-request budget.

    Returns EXO_GENERATE_TIMEOUT unchanged when the prompt size is unknown, so a
    caller that cannot count tokens keeps exactly today's behaviour rather than
    getting a deadline computed from a guess.
    """
    if not prompt_tokens or prompt_tokens <= 0:
        return float(EXO_GENERATE_TIMEOUT)
    return min(
        prompt_tokens / EXO_FIRST_TOKEN_RATE_TPS + EXO_FIRST_TOKEN_MARGIN_S,
        float(EXO_GENERATE_TIMEOUT),
    )


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
# TWO NUMBERS WITH DIFFERENT JOBS (#827 b1). Stage (a) had one, and it had to
# cover both, which is why it was 1800s:
#
#   EXO_LEASE_TTL          how long before someone may assume we died   LIVENESS
#   EXO_LEASE_MAX_TOTAL_S  the longest we could legitimately need it     EXPOSURE
#
# Liveness is 120s at b2. A dead holder frees the pool in <=120s, against <=1800s
# before #827 — every incident on that ticket was a holder that went away while its
# exclusive lease lived on.
#
# WHAT b2 BUYS AND COSTS, since the number alone hides both:
#   a dead holder's lease lapses in    600s -> 120s
#   the tick becomes                    60s -> 30s   min(60, TTL*25%): the cap stops
#                                                    binding below TTL 240
#   a full-budget 1500s generation
#   depends on                          ~25 -> ~50 consecutive renewals
#
# That last line is the trade: five times faster recovery from a dead holder, in
# exchange for a long generation depending on twice as many successful renewals to
# keep a lease it already holds. Both clauses still hold, and the recovery path is
# asserted rather than hoped for (test_pool_renewal [13]-[14]), which is what made
# this safe to do rather than merely desirable.
EXO_LEASE_TTL = _env_int("EXO_LEASE_TTL", 120)

# The ceiling, counted by the registry from granted_at, passed EXPLICITLY on the
# grant rather than left to the registry default. The ceiling is the first thing
# a reader checks when a lease lapses, so it belongs in the call.
EXO_LEASE_MAX_TOTAL_S = _env_int("EXO_LEASE_MAX_TOTAL_S", 1800)

# THE INVARIANT SWAP, which is the substance of b1 rather than the number.
#
# Stage (a) forced EXO_LEASE_TTL >= EXO_GENERATE_TIMEOUT + 300: a lease had to
# outlive the longest generation, because a holder that cannot renew has nothing
# else to protect it. Renewal makes that obsolete and moves the requirement:
#
#   1. the CEILING must still outlive the longest generation, because renewals
#      cannot walk a lease past granted_at + max_total_s. This is the clause that
#      inherits stage (a)'s job, and getting it wrong lapses a lease mid-work.
#   2. the TTL must leave room for two MISSED renewals — three attempts, the last
#      with time to spare for a slow registry answer. Not the generation's length
#      any more; the renewal interval's.
#
# Widening is the safe direction for both, as before.
_EXO_TTL_MARGIN = 300
if EXO_LEASE_MAX_TOTAL_S < EXO_GENERATE_TIMEOUT + _EXO_TTL_MARGIN:
    EXO_LEASE_MAX_TOTAL_S = EXO_GENERATE_TIMEOUT + _EXO_TTL_MARGIN

# --- Renewal (#827 P3, stage a) ---------------------------------------------
# The invariant above — a lease must outlive the longest generation — is what a
# holder needs when it CANNOT renew. Renewal is what makes it obsolete: with a
# heartbeat, a dead holder frees the pool in one TTL instead of 1800s, and the
# TTL no longer has to cover the work.
#
# This stage adds the renewing and changes NO timing: EXO_LEASE_TTL stays where
# it is, so nothing about exclusivity depends on the new path working. Stage (b)
# lowers it, and only once a renewal has been seen in this log — at which point
# the invariant above must be replaced by one about the RENEWAL INTERVAL, not
# the generation, or a short TTL breaks exclusivity outright.
#
# An integer percent, not a float: `float("")` raises, and an env var set to
# empty is the commonest way an operator "unsets" one.
#
# 25, not 33, and the reason is arithmetic rather than taste. Stage (b)'s
# invariant is TTL >= 3 * interval, so that two missed renewals still leave a
# third attempt inside the lease. But the interval is itself a fraction of the
# TTL, so that invariant is nearly a TAUTOLOGY and does no work at 33%:
#
#   TTL=120  pct=33  ->  interval 39  attempts at 39/78/117   3s of slack
#   TTL=120  pct=25  ->  interval 30  attempts at 30/60/90   30s of slack
#
# At 33% the third attempt lands at 98% of the lease. This gateway's registry
# calls carry a 10s timeout, so one slow answer on that third attempt loses the
# lease outright — and losing it means the pool becomes grantable while we are
# still generating, which is the failure renewal exists to prevent. 25% puts the
# third attempt at 75% and leaves a quarter of the TTL spare.
EXO_LEASE_RENEW_PCT = _env_int("EXO_LEASE_RENEW_PCT", 25)
EXO_LEASE_RENEW_PCT = min(90, max(10, EXO_LEASE_RENEW_PCT))

# Capped, for two reasons. A third of 1800s is 594s, so today only the five
# longest leases of 514 would ever renew and the path would be all but
# unexercised — an untested mechanism that exclusivity is about to depend on.
# And admin's gate for stage (b) is a renewal observed on a lease over 120s;
# at 594s that could wait days. A 60s cap makes any lease past a minute renew
# at least once, which is 35 of 514 by today's traffic, while costing one
# request per minute only while a generation is actually running.
EXO_LEASE_RENEW_MAX_S = _env_int("EXO_LEASE_RENEW_MAX_S", 60)
EXO_LEASE_RENEW_INTERVAL_S = max(
    5, min(EXO_LEASE_RENEW_MAX_S, int(EXO_LEASE_TTL * EXO_LEASE_RENEW_PCT / 100)))

# Clause 2 of the swap, enforced here because it needs the interval. Three
# attempts (two missed renewals survived) must land inside 80% of the lease, so
# the third still has a fifth of the TTL left for a slow registry answer — this
# gateway's calls carry a 10s timeout. At the b1 defaults: interval 60s, attempts
# at 60/120/180s, 180 <= 480. Widen the TTL rather than narrow the interval,
# because a shorter interval costs requests and a longer TTL costs only how fast
# a dead holder is noticed.
_EXO_MIN_TTL = int(EXO_LEASE_RENEW_INTERVAL_S * 3 / 0.8)
if EXO_LEASE_TTL < _EXO_MIN_TTL:
    EXO_LEASE_TTL = _EXO_MIN_TTL

# Per-model request defaults — the same shape model-service already uses for
# its Ollama `think:false` workaround. On an exclusive resource an unbounded
# generation is not slow, it is a denial of service: one caller owns the pool
# until it stops talking. Cap every request, and let the caller lower it but
# never raise it past the cap.
# 2048 was sized for a non-thinking tier and became the floor the caller could
# not get past: `think` is GLM-4.7-Flash-6bit, whose reasoning tokens come out of
# this same budget. Measured 2026-09-22 (#806): a one-line smoke test
# ("what is 17*23") spent 187 of 200 tokens on reasoning and returned
# finish_reason=length with the answer cut off. hermes-exo asks for 6000 and got
# min(6000, 2048) = 2048 — so it received empty content, no error, and burned its
# turn budget on answers that were truncated before they began (#808 timed out at
# the adapter's 600s wall with chars_out: 0).
#
# 8192 is chosen to clear hermes' 6000 with headroom, so the gateway stops being
# the binding constraint, while still bounding one generation: the pool measured
# ~33 tok/s on this model, so 8192 caps a single request at roughly four minutes
# of an exclusive resource. Raise it only with a matching view on how long one
# caller may own the pool.
EXO_MAX_TOKENS = _env_int("EXO_MAX_TOKENS", 8192)

# Largest prompt this gateway will hand to the pool, in tokens (ticket #837).
#
# THIS PROTECTS THE HOST, NOT THE MODEL'S CONTEXT WINDOW. GLM-4.7-Flash
# advertises 202,752 tokens and exo will honestly try to serve them; slice has
# 64 GB and cannot survive the attempt. On 2026-09-25 a 108,753-token prompt
# arrived here, exo began prefilling it, and at 47,104 tokens macOS's GPU driver
# panicked the box instead of failing the allocation ("completeMemory() prepare
# count underflow" @IOGPUMemory.cpp:492, wired ~53 GB of 64). Three hours down.
#
# MEASURED 2026-09-25 on the current placement (2-node ring: slice holds layers
# 13-47, wafer 0-13; the 34-layer shard is on slice, confirmed from the live
# process rather than `hostsByNode`, which reads inverted). Cold prefill through
# this gateway, max_tokens=1, sampling `vm_stat` every second. Slice baseline
# with the model resident and idle: 21.2 GB wired, 23.0 GB available.
#
#     prompt tokens   peak wired    over baseline   available at peak
#         4,096         24.7 GB        +3.5 GB          21.1 GB
#         8,192         29.3 GB        +8.1 GB          ~17 GB
#        16,384         46.1 GB       +24.9 GB           8.5 GB
#
# The cost is NOT linear in prompt length, which is the whole reason this
# constant is measured rather than picked: 4x the tokens cost 7x the memory.
# Fitting peak = 21.2 + 6.32e-4*N + 5.41e-8*N^2 (GB, N tokens) reproduces the
# 8,192 point to within 0.8 GB. Reading the panic level (53 GB wired) off that
# curve puts it at about 19,000 tokens — i.e. a single ~19k-token prompt is
# enough to reach the state that took the box down, with no other load.
#
# The inference, kept separate from the measurement above: the linear term is
# the KV cache itself (this model caches the MLA latent, 576 values per token
# per layer) and the quadratic term is the attention score matrix materialised
# per prefill chunk over the whole sequence so far, retained by MLX's buffer
# cache. What would disconfirm it: a run where peak memory tracks tokens
# linearly, or one where `mx.clear_cache()` between chunks flattens the curve.
#
# 12,288 puts the predicted peak at 36.8 GB wired — 16 GB below the level that
# panicked the host, and it leaves room for the decode that follows, which adds
# to the same cache at the linear rate (EXO_MAX_TOKENS=8192 more tokens is about
# 5 GB on top). It is a per-host number: it moves if the placement moves, if the
# resident model changes, or if anything else large starts running on slice.
#
# Note the *other* box is the tighter one in a different way: wafer carries 13
# of 47 layers with 36 GB total, and was measured at 14.0 GB available with
# 17.0 of 18.4 GB of swap already in use while this ran. A limit sized only off
# slice is not automatically safe for wafer.
EXO_MAX_PROMPT_TOKENS = _env_int("EXO_MAX_PROMPT_TOKENS", 12288)

# Where to find the resident model's own `tokenizer.json`, so the prompt is
# counted with the same tokenizer that will prefill it rather than estimated.
EXO_MODELS_DIR = _env("EXO_MODELS_DIR", "/Users/Shared/exo/models")

# Fallback when that tokenizer is not on disk: characters per token, used as
# `tokens = chars / ratio`. Measured on GLM-4.7-Flash-6bit's tokenizer, chars
# per token by input kind: prose 4.50, Python 3.86, JSON 3.38, log lines 2.76,
# CJK 2.00, base64-like 1.50. 1.5 is the densest of those, so the estimate
# over-counts ordinary prose roughly 3x. That is deliberate: the fallback exists
# to keep refusing safely when we cannot count exactly, and a refusal says which
# method produced its number so an over-count is legible rather than mysterious.
EXO_CHARS_PER_TOKEN = float(_env("EXO_CHARS_PER_TOKEN", "1.5"))

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
