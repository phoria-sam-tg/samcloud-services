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
EXO_LEASE_TTL = _env_int("EXO_LEASE_TTL", 1800)

# Enforce the ordering rather than trusting whoever edits the env next. Widening
# the lease is the safe direction: a lease that is too long delays the pool for
# other consumers, while one that is too short breaks exclusivity outright.
_EXO_TTL_MARGIN = 300
if EXO_LEASE_TTL < EXO_GENERATE_TIMEOUT + _EXO_TTL_MARGIN:
    EXO_LEASE_TTL = EXO_GENERATE_TIMEOUT + _EXO_TTL_MARGIN

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
