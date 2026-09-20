# Development Brief

This is **samcloud-services** — infrastructure for managed model inference on a SAMcloud network.

## What This Does

A FastAPI gateway (`ollama/server.py`) that unifies Ollama (MLX), llama-server (llama.cpp)
and mlx-vlm (vision-language) behind an OpenAI-compatible API, with SAMcloud resource
leasing for GPU memory management. Models spin up on demand, unload after 5 min idle.

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
| `ollama/manager.py` | Core logic. `ModelManager` handles lifecycle, leases, cooldown, adoption, the fit gate on load, and the stats + offering background loops. Discovery is resilient to any backend being down |
| `ollama/capacity.py` | **The one capacity signal.** Byte-identical on every box so an `offering:` means the same thing fleet-wide. Every fit decision, offering tier and resource-stats push reads from here — do not re-derive memory anywhere else |
| `ollama/samcloud.py` | SAMcloud API client. All registry calls go through this |
| `ollama/ollama_client.py` | Ollama API. Note: `chat()` passes `**kwargs` so `think=False` works |
| `ollama/llama_client.py` | llama-server process management. `discover_running()` parses `ps aux` |

## Run / Test

```bash
SC_TOKEN=<token> python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800

# From the repo ROOT — relative imports, so `cd ollama && python <file>.py`
# raises ImportError: attempted relative import with no known parent package.
python -m ollama.test_capacity_refusal   # Refusal path: 503 + the numbers, not 500

# Older tests, from inside ollama/ — these lease and load for real.
cd ollama && python test_lifecycle.py   # Full lease cycle
cd ollama && python test_cooldown.py    # Idle unload verification
```

`test_capacity_refusal` loads nothing and leases nothing — the refusal precedes
both — so it is the one safe to run against a live box. It reads real memory, so
it is a live check of the gate and not only of the shape. It self-skips if the
box genuinely has room for a 40GB model.

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
- **Fit, don't evict** — a load checks against what the box can hand over now; it does not unload whatever is resident to force-fit the ask. Eviction is opt-in behind `AUTO_EVICT` (default off). The contract is "here is what fits, pick one"
- **A refusal is an answer, not a fault** — `capacity.InsufficientCapacity` carries `need_mb`/`usable_mb`/`available_mb`/`fits_now`, and every load path returns it through the single `server._capacity_503()` funnel so callers cannot tell the backends apart by how they decline. It derives from `Exception`, **not** `RuntimeError`, so the `except RuntimeError -> 409` on `/models/load` cannot swallow it (ticket #756)
- **Size off `available`, not `used`** — `free + inactive + speculative`. macOS "used" counts pages the compressor is merely holding and is not a fit signal. Page size comes from the kernel; Apple silicon is 16KiB, and assuming 4KiB under-reported a box by 4x
- **Offering tier derives from the catalogue** — `full` = everything we hold fits, `mini` = exactly one does, `none` = nothing does. Fixed MB bands go stale the moment the catalogue changes (ticket #135, doc #8)
- **One stats push per box, from the gateway** — `ModelManager.stats_loop()`, lifecycle-managed. Not a separate daemon; `capacity.registry_payload()` narrows the reading to what the registry's strict schema accepts
- **Three pillars** — SAMcloud provides routing, resources, and auth

## Current State

See [CHANGELOG.md](CHANGELOG.md) for project history and working notes.
