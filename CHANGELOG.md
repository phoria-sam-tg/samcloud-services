# Changelog & Working Notes

Project history and current state. This is a living document.

## 2026-09-20 — Capacity migration committed, and the two boxes reconciled

- **The capacity gate is in git.** It had been running in production on both
  inference boxes since 2026-08-30 as uncommitted working-tree state —
  `ollama/capacity.py` untracked, four files modified-not-committed, and a
  drift of `.bak-*` siblings standing in for history. Neither box agent could
  read the other's tree, so the copies diverged unnoticed (ticket #758).
- **`ollama/capacity.py` is the one collector**, byte-identical on every box, so
  an `offering:` advertised by one service means the same as another's. It sizes
  off `free + inactive + speculative`, not `active + wired + compressor`:
  "used" counts pages the compressor is merely sitting on (22.7GiB of 36GiB read
  used on wafer against ~3.9GiB of real process RSS), which refuses models that
  would have fitted and says nothing about whether a load will swap. Page size
  comes from the kernel — assuming 4KiB on a 16KiB machine under-reported slice
  by 4x for weeks. Tool paths are absolute, because the gateway's launchd PATH
  omits `/usr/sbin` and a bare `sysctl` turned every request into a 404.
- **Refusals are answers, not faults.** `InsufficientCapacity` carries
  `need_mb` / `usable_mb` / `available_mb` / `fits_now`, and
  `server._capacity_503()` is the single funnel every load path returns it
  through — including `POST /models/load`, which had no capacity branch at all
  and returned a 500 with a stack-trace string (#756). The class derives from
  `Exception`, not `RuntimeError`, so the pre-existing
  `except RuntimeError -> 409` cannot swallow it and make the missing branch
  look fixed. `ollama/test_capacity_refusal.py` drives all four shapes.
- **Loads fit rather than evict.** `load_ollama_model` checks against what the
  box can hand over instead of unloading whatever is resident to force-fit the
  ask; eviction is opt-in behind `AUTO_EVICT` (default off).
- **Sizes come from Ollama, not from the name.** `memory_estimate_mb()` reads
  the real weight size; the name fallback now anchors its size tag on a digit
  boundary. Substring matching read "qwen3.8:27b-mlx" as 7b and leased 5000MB
  for a model that resides at 17530MB — a 3.5x under-count that told other
  tenants there was room that did not exist. "13b" hit "3b" the same way.
- **The two lineages are merged.** `feat/wafer-gpu-metal-stats` (stats loop,
  Stage-1 offering) and `main` (GGUF short-name routing, #97 mitigations,
  capacity gate) both branched from 0d60324 and each grew the half the other
  lacked. Merged, then wired onto the shared collector: `_collect_stats()` is
  `capacity.collect()`, `stats_loop` pushes `registry_payload()`, and
  `_compute_offering()` uses `capacity.offering_tier()`.
- **The offering MB bands are gone** (`OFFERING_MINI_MB` / `_DEGRADED_MB` /
  `_FULL_MB`, superseding the 2026-07-02 entry below). They banded
  `total - used` into tiers and were calibrated when the largest model was
  ~6GB, so wafer advertised `offering:full` on ~14GiB available against a
  17.5GB model that could not load without swapping. The tier now derives from
  what actually fits out of the live catalogue — `full` = everything we hold
  fits, `mini` = exactly one does, `none` = nothing does — which stays true as
  models come and go and needs no per-box tuning. `OFFERING_ENABLED`,
  `_POLL_SECONDS` and `_HYSTERESIS` remain the per-box knobs.
- **One stats push per box, from the gateway.** `ollama/gpu_stats.py` retired;
  slice's separate `com.samcloud.cs-gpu-stats` launchd job is unloaded in
  favour of `ModelManager.stats_loop()`, which is lifecycle-managed and shares
  the collector. The module had been hollowed out by the migration anyway —
  `collect_stats()` was a passthrough and the helpers under it were dead.
- **Still open:** `llama-server` and `mlx-vlm` have no capacity gate — only
  Ollama does, so `server.py` catches `InsufficientCapacity` around
  `load_llama_model`, which cannot raise it. Deliberately not fixed here:
  gating them starts refusing loads that succeed today, which is a behaviour
  change callers should be told about first (ticket #754).

## 2026-07-02 — Stage-1 capacity offering (flex tier) on wafer

- **`offering:<tier>` self-report (doc #8 Stage 1, ticket #135).** The gateway now
  advertises one of `offering:full | degraded | mini | none` as a `capabilities`
  entry, recomputed every `OFFERING_POLL_SECONDS` (30s) and PATCHed on change so
  the fleet map reflects live capacity. New `offering_loop` / `_compute_offering`
  / `_apply_offering` in `manager.py`; config knobs in `config.py`
  (`OFFERING_ENABLED`, `_POLL_SECONDS`, `_HYSTERESIS`, `_MINI_MB`, `_DEGRADED_MB`,
  `_FULL_MB`). First reading publishes immediately; later changes require the new
  tier to hold `OFFERING_HYSTERESIS` (2) consecutive polls to avoid flapping.
- **Signal = LOCAL unified-memory availability** (`_collect_stats()`'s
  vm_stat/sysctl `total - used`), *not* the registry's lease-based
  `available_memory_mb`. Reason: the gateway runs under a service token, which is
  scope-filtered out of resource reads (`GET /resources/{id}` → 403, dashboard/
  leases return empty). On a unified-memory Mac real memory pressure is the truer
  "can I serve a model" signal anyway, and it also captures training that spikes
  memory without holding a formal lease. Verified end-to-end: full → degraded
  under ~9 GB pressure, restore to full on release, with hysteresis.
- **Good-citizen flex** for the wafer box shared with brush splat training: the
  tier drops as memory fills (training running) and restores when it frees.

## 2026-06-14 — On-demand mlx-vlm + lease fix

- **mlx-vlm is now gateway-owned and on-demand.** Added `load_vlm_model` (spawns
  `python -m mlx_vlm.server`, leases, tracks the process), `match_vlm_model`, and
  VLM branches in `unload`/`ensure_running`. A cold VLM request now spins the
  server up (~7s) instead of 404ing. Routing wired in `_resolve_model` and
  `/models/load` (+ `auto` detection); `/models/unload` resolves partial names.
- **No more adoption / boot-order for VLM.** `discover()` no longer adopts a
  running mlx-vlm; instead `_kill_stray_vlm()` reaps any stray `mlx_vlm.server` on
  startup so the gateway always owns the process. mlx-vlm dropped from reboot order.
- **Lease `purpose` field removed.** Registry now rejects it (`422 extra_forbidden`);
  `request_lease` and callers updated. Leasing was previously failing for all paths.
- VLM config (`VLM_PYTHON`, `VLM_HOST`, `VLM_PORT`, `VLM_STARTUP_TIMEOUT`) in `config.py`.
- **Known follow-up:** on-demand ollama loads still lease an *estimate* (e.g. 2500 MB
  for a ~20 GB model) — the "wider leasing" reconcile-to-actual work (overlaps PR #2)
  should extend to the VLM path too.

## Current State (2026-04-06)

### Services Running on slice-test (M1 Max, 64GB)

| Service | Port | External | Status |
|---------|------|----------|--------|
| model-service | 8800 | models-stg.samtg.xyz | UP — auto-loads models on request |
| vlm-service | 8801 | vlm-stg.samtg.xyz | UP — Gemma 4 31B (nvfp4 MLX) |

### Ollama 0.20.0 — Models Pulled

| Model | Size | Format | Use Case |
|-------|------|--------|----------|
| `qwen3.5:35b-a3b-coding-nvfp4` | 21GB | MLX safetensors nvfp4 | **Primary** — 44.9 tok/s decode, coding/agent tasks |
| `hermes3:8b` | 4.7GB | GGUF | Tool calling — built for Hermes agent |
| `qwen3-coder:30b` | 18GB | GGUF | Heavy reasoning + agentic tool calling |
| `glm4` | 5.5GB | GGUF | 198K context, tool support |
| `qwen3.5:35b-a3b` | 23GB | GGUF | Legacy — superseded by nvfp4 MLX version |
| `qwen3.5:latest` | 6.6GB | GGUF | 9.7B dense variant |
| `qwen3.5-notk:latest` | 23GB | GGUF | No-think variant |
| `qwen2.5:1.5b` | 986MB | GGUF | Lightweight test model |

### Vision Model (vlm-service :8801)

Gemma 4 31B (nvfp4, MLX-native via mlx-vlm) — 18.6 tok/s, 18.7GB peak.

### Architecture

```
Consumer (Hermes agent, any OpenAI SDK client)
  │  POST /v1/chat/completions
  │  Authorization: Bearer <samcloud-token>
  ▼
model-service :8800 (FastAPI)
  │  SAMcloud auth verification (cached 5min)
  │  _resolve_model() — partial match + auto-load if not loaded
  │  OpenAI ↔ Ollama message translation (tool_calls, arguments, thinking)
  ▼
Ollama :11434 (native /api/chat)
  │  MLX backend (Apple Silicon)
  │  keep_alive=-1 (our lease system manages memory)
  ▼
GPU (Apple M1 Max, 64GB unified memory)
  │  SAMcloud lease for memory accounting
  └── Cooldown: 5 min idle → unload → free GPU → next request auto-reloads
```

### What Works

- **OpenAI-compatible proxy** — any OpenAI SDK client works (`OPENAI_BASE_URL=http://host:8800/v1`)
- **Auto-load on request** — model spins up transparently (~10s), no 404, no client retry
- **Cooldown + auto-reload** — models unload after 5min idle, reload on next request
- **Tool calling** — structured tool_calls with Ollama → OpenAI format translation
- **Multi-turn conversations** — OpenAI ↔ Ollama message format conversion
- **SAMcloud auth** — token verification with scope checking and 5min cache
- **GPU lease management** — request/renew/release via SAMcloud
- **Process adoption** — discovers running models on startup, pins with keep_alive=-1
- **Async streaming** — aiohttp for proper connection cleanup on client disconnect
- **Service discovery** — `GET /service-docs` returns structured docs + full guide

### What's Next

**Immediate (from testing feedback):**
- LaunchAgent/watchdog for auto-restart (model-service has needed manual restarts)
- Integrate vlm-service into ModelManager (currently standalone, no auth)

**Short term:**
- Request queuing — queue requests during model load, serve when ready
- Session-aware cooldown — track active sessions, don't unload mid-conversation
- Multi-model scheduling — queue for model B while A is loading

**Medium term (from satellite agent spec):**
- Build Docker container image for Hermes agents
- SAMcloud enrollment flow for containers
- Agent orchestration (provision, start, stop, destroy)
- Cross-agent messaging via SAMcloud events

**Longer term (from ticket #71 interview):**
- Per-service auth scoping (not just device scope)
- Atomic lease renewal (not release-and-re-request)
- Rate limiting per caller
- Multi-host model services

---

## Timeline

### 2026-03-31 — Initial Setup
- Enrolled as `claude-services` on SAMcloud
- Built model service: SAMcloud client, Ollama client, llama-server client, unified ModelManager, FastAPI server
- Tested full resource lease lifecycle

### 2026-04-02 — Lease Incident + MLX
- Ticket #42: 25GB lease sat indefinitely for 2 days. Added lease renewal loop, expiry validation
- Upgraded Ollama 0.15.4 → 0.19.0 (MLX support)
- Pulled qwen3.5:35b-a3b — 22.8 tok/s on Metal

### 2026-04-03 — Auth, Rename, Docs
- SAMcloud auth middleware (ticket #46): token verification, 5min cache, scope checking
- Service rename: ollama-manager → model-service (ticket #65)
- `GET /service-docs` convention for service discovery
- Satellite agent spec written (SPEC-satellite-agents.md)
- Responded to architecture interview (ticket #71)

### 2026-04-04 — Tool Calling + VLM
- VLM service: Gemma 4 31B (nvfp4 MLX) on mlx-vlm, 18.6 tok/s
- Ollama think bug workaround (ticket #69): route through native /api/chat
- Tool calling pipeline: tools passthrough, arguments format (string not object), message translation
- Model keep_alive=-1 to prevent Ollama eviction (ticket #74)
- Pulled tool-calling models: hermes3:8b, qwen3-coder:30b, glm4 (ticket #76)
- Published to GitHub: github.com/phoria-sam-tg/samcloud-services

### 2026-04-05 — Streaming Fixes + Auto-Load
- think:false was suppressing tool_calls (tickets #77, #79) — scoped to qwen3.5-only, then removed entirely
- Multi-turn tool calling: OpenAI → Ollama message format translation
- Connection leak fix: switched to aiohttp for async streaming (ticket #81)
- Model eviction handling: explicit unload before loading new model

### 2026-04-06 — Transparent Auto-Load
- Pulled qwen3.5:35b-a3b-coding-nvfp4 — the actual MLX model, 44.9 tok/s (2x faster)
- Transparent auto-load on request (ticket #90): never 404 for known models, auto-loads on first request, cooldown + auto-reload cycle

---

## SAMcloud Tickets

| # | Summary | Status |
|---|---------|--------|
| 42 | Unbounded GPU lease — 25GB indefinite | Resolved |
| 43 | Cannot read ticket 42 (scope issue) | Resolved |
| 46 | Token verification endpoint for service-to-service auth | Resolved |
| 64 | Model service naming/endpoint confusion | Resolved |
| 65 | Service rename, description field, docs_endpoint | Resolved |
| 66 | DELETE /services returns 500 | Resolved |
| 68 | Reverse tunnel down — external endpoints unreachable | Resolved |
| 69 | Ollama /v1/ ignores think parameter | Resolved (workaround) |
| 71 | Interview: architecture, auth, leasing | Responded |
| 74 | Model keeps unloading during Hermes sessions | Resolved (keep_alive=-1) |
| 76 | Need tool-call-capable models | Resolved |
| 77 | Tools parameter dropped when proxying to Ollama | Resolved |
| 78 | Ticket scope visibility requires manual admin intervention | Filed |
| 79 | Model eviction + streaming tool_calls + empty responses | Resolved |
| 81 | Stale connections accumulate, blocking Ollama | Resolved (aiohttp) |
| 82 | MLX model available — qwen3.5 nvfp4 2x speed | Filed for testing |
| 90 | Auto-reload on request not working | Resolved (transparent auto-load) |

## Repo

**GitHub:** github.com/phoria-sam-tg/samcloud-services
**Commits:** 15 on main
**Stack:** Python 3.10, FastAPI, httpx, aiohttp, Ollama 0.20, mlx-vlm 0.4.3
