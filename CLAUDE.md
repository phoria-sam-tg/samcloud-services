# SAMcloud Model Services

Infrastructure for managing model inference services on the SAMcloud home network.
The main component is `ollama-manager` — a FastAPI service that unifies Ollama and llama-server
backends behind an OpenAI-compatible API, with SAMcloud resource leasing for GPU memory management.

## Identity

- **User**: `claude-services` (user:17, role: agent, scope: `device:slice-test`)
- **Service**: `slice-test/ollama-manager` (port 8800, subdomain `models-stg`)
- **SAMcloud**: `https://stg.samtg.xyz:9443/api/v1`
- **Token**: Always set via `SC_TOKEN` env var — do not hardcode tokens

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

## Architecture Decisions (from ticket #42)

- Leases must have bounded TTL — services renew periodically (every 30min)
- If a service stops reporting health, its leases should be revoked
- Memory estimates should use actual VRAM (from `ollama ps` / process stats), not file sizes
- Resource registration (GPU specs) is device-daemon territory — agents only lease, not register
- Lease renewal is release-and-re-request (brief gap — known limitation under contention)
