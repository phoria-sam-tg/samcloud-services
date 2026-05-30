# Development Brief

This is **samcloud-services** — infrastructure for managed model inference on a SAMcloud network.

## What This Does

A FastAPI gateway (`ollama/server.py`) that unifies Ollama (MLX) and llama-server (llama.cpp)
behind an OpenAI-compatible API, with SAMcloud resource leasing for GPU memory management.
Models spin up on demand, unload after 5 min idle.

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

## Run / Test

```bash
SC_TOKEN=<token> python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800
cd ollama && python test_lifecycle.py   # Full lease cycle
cd ollama && python test_cooldown.py    # Idle unload verification
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
- **Three pillars** — SAMcloud provides routing, resources, and auth

## Current State

See [CHANGELOG.md](CHANGELOG.md) for project history and working notes.
