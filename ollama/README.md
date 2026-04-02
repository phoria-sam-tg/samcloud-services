# ollama-manager — Unified Model Service

Manages multiple model backends (Ollama + llama-server) behind a single OpenAI-compatible API
with SAMcloud resource leasing for GPU memory management on slice-test (M1 Max, 64GB).

## Modules

| Module | Purpose |
|--------|---------|
| `samcloud.py` | SAMcloud API client — resources, leases, services, health reporting |
| `ollama_client.py` | Ollama API client — pull, load, unload, generate, chat, model info |
| `llama_client.py` | llama-server (llama.cpp) — process discovery, start, stop, health, GGUF model registry |
| `manager.py` | `ModelManager` — unified lifecycle, lease management, cooldown, adoption |
| `server.py` | FastAPI on :8800 — OpenAI-compatible routing to the correct backend |

## Model Lifecycle

```
Request arrives (/v1/chat/completions)
  │
  ├─ Model already loaded? → touch (reset cooldown), serve
  │
  └─ Not loaded:
       1. Estimate memory (or use actual VRAM after load)
       2. Request lease from SAMcloud (gpu-0, memory_mb, TTL=1hr)
       3. Pull model if needed (Ollama) or start process (llama-server)
       4. Load into GPU memory
       5. Serve request
       │
       └─ Idle for 5 minutes (no requests)
            1. Unload model (Ollama keep_alive=0 / kill llama-server)
            2. Release lease on SAMcloud
            3. GPU memory freed
```

## Background Tasks

| Task | Interval | Purpose |
|------|----------|---------|
| Cooldown monitor | 60s check, 5min threshold | Unloads idle models, releases leases |
| Health reporter | 60s | `POST /services/slice-test/ollama-manager/health` to SAMcloud |
| Lease renewal | 30min (50% of 1hr TTL) | Release and re-request leases to prevent expiry |

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SC_TOKEN` | — | SAMcloud agent token (required) |
| `SERVICE_PORT` | `8800` | FastAPI listen port |

### Constants (manager.py)

| Constant | Value | Purpose |
|----------|-------|---------|
| `RESOURCE_ID` | `slice-test/gpu-0` | SAMcloud resource to lease from |
| `SERVICE_ID` | `slice-test/ollama-manager` | Service identity for leases |
| `COOLDOWN_SECONDS` | `300` (5min) | Idle time before auto-unload |
| `LEASE_TTL` | `3600` (1hr) | Lease duration |
| `LEASE_RENEW_AT` | `0.5` | Renew at 50% of TTL elapsed |

## Backends

### Ollama (port 11434)

- Version 0.19.0 with MLX support on Apple Silicon
- Flash attention enabled (`OLLAMA_FLASH_ATTENTION=1`)
- KV cache quantization (`OLLAMA_KV_CACHE_TYPE=q8_0`)
- Models pulled from Ollama hub (e.g. `qwen3.5:35b-a3b`, `qwen2.5:1.5b`)
- Load/unload via Ollama's `keep_alive` mechanism

### llama-server (llama.cpp)

- Version b8500 at `/opt/homebrew/bin/llama-server`
- Metal GPU acceleration (99 layers offloaded)
- Flash attention, quantized KV cache (q4_0)
- GGUF models from `/Users/sam/models/`

| Model File | Params | Quant | ~VRAM |
|-----------|--------|-------|-------|
| `Qwen3-32B-Q6_K.gguf` | 32B | Q6_K | 25GB |
| `Qwen3.5-35B-A3B-Q4_K_M.gguf` | 35B MoE | Q4_K_M | 21GB |

## OpenAI Compatibility

The server exposes standard OpenAI endpoints:

- `POST /v1/chat/completions` — Chat (streaming and non-streaming)
- `POST /v1/completions` — Completion

Model name resolution uses **case-insensitive partial matching** via `_resolve_model()`.
Agents can use friendly names:

- `qwen3-32b` matches `Qwen3-32B-Q6_K` (llama-server)
- `qwen3.5` matches `qwen3.5:35b-a3b` (Ollama)

Requests are proxied to the correct backend transparently.

## Process Adoption

On startup, `ModelManager.discover()` finds already-running model processes:

1. **llama-server**: Parsed from `ps aux` — extracts model path, port, PID, flags
2. **Ollama models**: Queried from `GET /api/ps` — extracts name, VRAM size

Adopted models are tracked as `managed=False` — the manager monitors them and claims
leases but will not kill them on cooldown unless `force=True`.

`claim_leases()` then requests SAMcloud leases for all adopted models retroactively.

## Testing

### test_lifecycle.py

Full 8-step integration test:
1. Check resource baseline
2. Pull a small model
3. Request lease from SAMcloud
4. Load into Ollama
5. Run inference
6. Unload model
7. Release lease
8. Verify resource state restored

### test_cooldown.py

Cooldown verification with a short (30s) threshold:
1. Load model via manager
2. Run inference
3. Wait for cooldown
4. Trigger cooldown check
5. Verify model unloaded and lease released
