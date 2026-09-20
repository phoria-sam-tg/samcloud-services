# Development Brief

This is **samcloud-services** — infrastructure for managed model inference on a SAMcloud network.

## What This Does

A FastAPI gateway (`ollama/server.py`) that unifies Ollama (MLX), llama-server (llama.cpp)
and mlx-vlm (vision-language) behind an OpenAI-compatible API, with SAMcloud resource
leasing for GPU memory management. Models spin up on demand, unload after 5 min idle.

A fourth backend, the **exo pool** (`Backend.EXO`), is reached the same way but owned
differently: the gateway neither starts it nor places its model, and holds an exclusive
lease around each generation. It cannot restart itself after a reboot — macOS grants
local-network access per responsible process, so a headless launch is denied and the
pool must be started from a Terminal on the host. Known limitation, documented not
solved.

## SAMcloud Identity

Defaults are now env-driven via `ollama/config.py`. Production defaults:

- **Device**: `claude-services-slice` (`SC_DEVICE`)
- **Service**: `claude-services-slice/model-service` (`SC_SERVICE_NAME`)
- **Resource**: `claude-services-slice/gpu-0` (`SC_RESOURCE_ID`)
- **Registry**: `https://cloud.samtg.xyz/api/v1` (`SC_BASE`)
- **Required scope**: `device:claude-services-slice` (`SC_REQUIRED_SCOPE`) — tune per deployment
- **Token**: always via `SC_TOKEN` env — never hardcode. Use the service token (`sc_service_...`) issued at registration, not a user token.

Staging (legacy) used `slice-test/*` identities pointing at `stg.samtg.xyz:9443` — see the `migration` branch for the pre-migration architecture.

## Key Files

| File | What to know |
|------|-------------|
| `ollama/config.py` | Env-driven config. Every SAMcloud identity + backend path reads from here |
| `ollama/server.py` | FastAPI server + auth middleware. Thin routing — delegates to manager |
| `ollama/manager.py` | Core logic. `ModelManager` handles lifecycle, leases, cooldown, adoption. Discovery is resilient to any backend being down |
| `ollama/samcloud.py` | SAMcloud API client. All registry calls go through this |
| `ollama/ollama_client.py` | Ollama API. Note: `chat()` passes `**kwargs` so `think=False` works |
| `ollama/llama_client.py` | llama-server process management. `discover_running()` parses `ps aux` |
| `ollama/exo_client.py` | The exo pool. Already OpenAI-compatible, so chat is a proxy. `resident_model()` reads `/state`; non-streaming goes through `chat_collect()` because exo's own `stream:false` returns no body |

## Run / Test

```bash
SC_TOKEN=<token> python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800
cd ollama && python test_lifecycle.py   # Full lease cycle
cd ollama && python test_cooldown.py    # Idle unload verification
python ollama/test_exo_lease.py        # Pool lease verdict + resident model (no network)
```

## Conventions

- All SAMcloud API calls go through `SamcloudClient` (never raw httpx)
- All model operations go through `ModelManager` (server.py is thin routing)
- `GET /service-docs` is the service discovery convention — auth-exempt, returns structured JSON + full markdown guide
- Auth via SAMcloud token verification (`GET /auth/verify?scope=$SC_REQUIRED_SCOPE`)

## Architecture Decisions

- **Bounded TTL leases** — renewed every 30 min, not indefinite (ticket #42)
- **Health-bound revocation** — stale services should lose leases
- **Actual VRAM** — use `ollama ps` sizes, not file-size estimates
- **Agents lease, not register** — resource registration is device-daemon territory
- **Ollama think workaround** — route through native `/api/chat` with `think:false`, translate to OpenAI format ourselves (ticket #69)
- **On-demand mlx-vlm, gateway-owned** — the gateway starts/stops `mlx_vlm.server` itself (`load_vlm_model` + `match_vlm_model`), rather than adopting a pre-started process. On startup it reaps any stray `mlx_vlm.server`. No boot-order dependency; one VLM per `VLM_PORT`
- **The pool is leased per generation, not per residency** (ticket #770) — `Backend.EXO`
  is the one backend the gateway does not own. ONE exo instance spans slice and wafer,
  serves ONE request at a time, and is placed out of band; a model swap costs 30s-10min,
  so callers ask for a **tier** (`model: "think"`) and get whatever is resident, read
  from `/state` at request time rather than hardcoded. Its samcloud lease is
  **exclusive** and means "the pool is TAKEN", not that bytes are reserved — its pages
  are already wired and already counted by each box's own `capacity.py`, and
  `memory_mb` must be `null` (see below). Acquired around the generation, released in a
  `finally`.
- **A queued lease is not a granted lease** (ticket #770) — `_request_lease` used to
  return an id for any response that did not raise, so a queued lease read as granted.
  Harmless on shared `gpu-0`; on an exclusive resource it is the whole bug. The verdict
  is a `LeaseOutcome`, and it comes from the **body** rather than the status code:
  the registry grants with a plain `200`, not the `201` the API index documents, so
  code-only logic fails in one direction or the other.
- **Never send `memory_mb` for the pool** (ticket #770) — the registry queues when
  `memory_mb > available`, and `available` is `total - leased` where `total` is read
  from `vram_mb`/`gpu_memory_mb`/`unified_memory_mb`/`ram_mb` in the resource specs.
  `exo-pool` carries none, so `total` is 0 and *any* byte count queues forever no
  matter how idle the pool is.
- **`EXO_LEASE_TTL` must exceed `EXO_GENERATE_TIMEOUT`** — the registry expires a lease
  on time and cannot extend one, and renewing by release-then-reacquire would open a
  window for a third party to take an exclusive resource mid-generation. `config.py`
  clamps the ordering rather than trusting the env.
- **Three pillars** — SAMcloud provides routing, resources, and auth

## Current State

See [CHANGELOG.md](CHANGELOG.md) for project history and working notes.
