# SAMcloud Model Services

Infrastructure for managing model inference services on the SAMcloud home network.
The main component is the **model service** (registered as `ollama-manager`, pending rename to
`model-service` per ticket #65) — a FastAPI inference gateway that unifies Ollama (MLX) and
llama-server (llama.cpp Metal) behind an OpenAI-compatible API, with SAMcloud resource leasing
for GPU memory management and SAMcloud token auth for service-to-service access.

## Identity

- **User**: `claude-services` (user:17, role: agent, scope: `device:slice-test`)
- **Service**: `slice-test/ollama-manager` (port 8800, subdomain `models-stg`) — pending rename to `model-service`
- **SAMcloud**: `https://stg.samtg.xyz:9443/api/v1`
- **Token**: Always set via `SC_TOKEN` env var — do not hardcode tokens

## Service Convention

All services should expose `GET /service-docs` (auth-exempt) returning:
- `description`: short text summary
- `auth`: method, required scope, exempt paths
- `endpoints`: catalogue with descriptions and body schemas
- `guide`: full markdown documentation from repo README
- `loaded_models` / runtime state

## Key Files

| File | Purpose |
|------|---------|
| `ollama/server.py` | FastAPI server on :8800 — OpenAI-compatible endpoints |
| `ollama/manager.py` | Core `ModelManager` — lease lifecycle, cooldown, health, adoption |
| `ollama/samcloud.py` | SAMcloud API client (resources, leases, services, health) |
| `ollama/ollama_client.py` | Ollama API client (pull, load, unload, generate, chat) |
| `ollama/llama_client.py` | llama-server process management and API client |

## Run

```bash
SC_TOKEN=<token> python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800
```

## Test

```bash
cd ollama && python test_lifecycle.py   # Full lease cycle: pull, lease, load, infer, unload, release
cd ollama && python test_cooldown.py    # Verifies idle models get unloaded and leases released
```

## Three Pillars (SAMcloud integration)

1. **Routing** — SAMcloud service discovery, DNS, subdomains
2. **Resources** — SAMcloud lease management for GPU memory (ticket #41)
3. **Auth** — SAMcloud token verification for service-to-service access (ticket #46)

## Architecture Decisions (from ticket #42)

- Leases must have bounded TTL — services renew periodically (every 30min)
- If a service stops reporting health, its leases should be revoked
- Memory estimates should use actual VRAM (from `ollama ps` / process stats), not file sizes
- Resource registration (GPU specs) is device-daemon territory — agents only lease, not register
- Lease renewal is release-and-re-request (brief gap — known limitation under contention)
