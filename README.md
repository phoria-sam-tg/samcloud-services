# samcloud-services

Managed model inference for Apple Silicon Macs on a [SAMcloud](https://github.com/phoria-sam-tg) network.

A FastAPI gateway that unifies [Ollama](https://ollama.com) (MLX) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (Metal) behind a single **OpenAI-compatible API**, with centralized GPU memory management via SAMcloud resource leasing.

Models spin up on demand and unload after idle timeout, freeing GPU memory for other workloads.

## Features

- **OpenAI-compatible** — `POST /v1/chat/completions` works with any OpenAI SDK client
- **Multi-backend** — Routes to Ollama or llama-server transparently based on model type
- **GPU memory leasing** — Requests SAMcloud leases before loading, releases on unload
- **Auto spin-up/cooldown** — Models load on first request, unload after 5 min idle
- **Process adoption** — Discovers and manages already-running model processes
- **SAMcloud auth** — Verifies caller identity via SAMcloud token verification
- **MLX + Metal** — Ollama 0.19 MLX for Apple Silicon, llama.cpp Metal for GGUF models
- **Partial model matching** — Use `qwen3-32b` instead of `Qwen3-32B-Q6_K`

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A SAMcloud registry instance (for leasing and auth)

### Install

```bash
git clone https://github.com/phoria-sam-tg/samcloud-services.git
cd samcloud-services
pip install fastapi uvicorn httpx
```

### Run

```bash
export SC_TOKEN=<your-samcloud-agent-token>
python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800
```

On startup the service will:
1. Discover any running llama-server or Ollama model processes
2. Claim SAMcloud GPU memory leases for each
3. Start background health reporting and lease renewal
4. Begin serving on port 8800

### Use

```bash
# Load a model
curl -X POST http://localhost:8800/models/load \
  -H "Authorization: Bearer $SC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5:35b-a3b", "backend": "ollama"}'

# Chat (OpenAI-compatible)
curl -X POST http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer $SC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "messages": [{"role": "user", "content": "Hello"}]}'

# Check what's loaded
curl -H "Authorization: Bearer $SC_TOKEN" http://localhost:8800/models
```

## API Reference

All endpoints except `/health` and `/service-docs` require `Authorization: Bearer <token>`.

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `POST` | `/v1/completions` | OpenAI-compatible text completion |

Model names use **case-insensitive partial matching**:
- `qwen3-32b` matches `Qwen3-32B-Q6_K` (llama-server)
- `qwen3.5` matches `qwen3.5:35b-a3b` (Ollama)

### Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/models/load` | Load a model — pulls if needed, requests GPU lease |
| `POST` | `/models/unload` | Unload a model — releases GPU lease |
| `GET` | `/models` | List managed and available models |
| `GET` | `/status` | Full status: backends, models, leases, resource utilisation |

### Discovery

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check |
| `GET` | `/service-docs` | No | Structured service documentation (JSON) |

### Load Request Body

```json
{
  "model": "qwen3.5:35b-a3b",
  "backend": "auto",
  "port": 8000,
  "ctx_size": 12288,
  "gpu_layers": 99
}
```

`backend`: `"auto"` (default), `"ollama"`, or `"llama-server"`. Auto selects llama-server for `.gguf` files, Ollama for everything else.

### Chat Request Body

```json
{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

## Architecture

```
                    SAMcloud Registry
                   ┌─────────────────────┐
                   │  auth/verify         │
                   │  resources/leases    │
                   │  services/health     │
                   └────────┬────────────┘
                            │
               health, leases, auth verification
                            │
    ┌───────────────────────┴──────────────────────┐
    │  model-service :8800                         │
    │  ┌────────────────────────────────────────┐  │
    │  │  FastAPI + Auth Middleware              │  │
    │  │  POST /v1/chat/completions             │  │
    │  │  POST /v1/completions                  │  │
    │  │  GET  /models  /status  /health        │  │
    │  └──────────┬─────────────────────┬───────┘  │
    │             │                     │          │
    │  ┌──────────┴──────┐  ┌──────────┴───────┐  │
    │  │ Ollama :11434   │  │ llama-server     │  │
    │  │ MLX backend     │  │ llama.cpp Metal  │  │
    │  └─────────────────┘  └──────────────────┘  │
    └──────────────────────────────────────────────┘
```

## Model Lifecycle

```
Request arrives
  │
  ├─ Model loaded? → serve (reset cooldown timer)
  │
  └─ Not loaded:
       1. Estimate GPU memory needed
       2. Request lease from SAMcloud
       3. Pull model (Ollama) or start process (llama-server)
       4. Load into GPU → serve
       │
       └─ No requests for 5 minutes
            1. Unload model
            2. Release SAMcloud lease
            3. GPU memory freed
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SC_TOKEN` | — | SAMcloud agent token (**required**) |
| `SERVICE_PORT` | `8800` | Server listen port |
| `SC_VERIFY_URL` | `https://stg.samtg.xyz:9443/api/v1/auth/verify` | SAMcloud auth endpoint |
| `SC_REQUIRED_SCOPE` | `device:slice-test` | Required scope for callers |
| `AUTH_ENABLED` | `true` | Set `false` to disable auth (development only) |

### Tuning (manager.py constants)

| Constant | Default | Description |
|----------|---------|-------------|
| `COOLDOWN_SECONDS` | `300` (5 min) | Idle time before auto-unload |
| `LEASE_TTL` | `3600` (1 hr) | Lease duration in seconds |
| `LEASE_RENEW_AT` | `0.5` | Renew at 50% of TTL |
| `RESOURCE_ID` | `slice-test/gpu-0` | SAMcloud resource to lease |

## SAMcloud Integration

The service integrates with SAMcloud across three pillars:

1. **Routing** — Registered as a service with DNS subdomain and health endpoint
2. **Resources** — GPU memory leases requested before loading, released on unload, renewed every 30 min
3. **Auth** — Callers verified via `GET /auth/verify` with scope checking and 5-min cache

## Backends

### Ollama (MLX)

- Ollama 0.19+ with Apple MLX framework
- Flash attention, KV cache quantisation
- Handles the Ollama 0.19 think bug (routes through native `/api/chat` with `think:false` and translates to OpenAI format)

### llama-server (Metal)

- llama.cpp with Metal GPU acceleration
- Full GPU offloading, flash attention, quantised KV cache
- GGUF models from a configurable directory

## Project Structure

```
samcloud-services/
├── README.md                    # This file
├── CLAUDE.md                    # Development brief for AI assistants
├── SPEC-satellite-agents.md     # Spec: isolated agent environments
├── CHANGELOG.md                 # Project history and working notes
├── requirements.txt             # Python dependencies
└── ollama/
    ├── README.md                # Module documentation
    ├── server.py                # FastAPI server + auth middleware
    ├── manager.py               # ModelManager — lifecycle, leases, cooldown
    ├── samcloud.py              # SAMcloud API client
    ├── ollama_client.py         # Ollama backend client
    ├── llama_client.py          # llama-server backend client
    ├── test_lifecycle.py        # Integration test: full lease cycle
    └── test_cooldown.py         # Integration test: idle unload
```

## Testing

```bash
cd ollama
python test_lifecycle.py    # Pull → lease → load → infer → unload → release
python test_cooldown.py     # Load → idle → auto-unload → lease released
```

## Related

- [SPEC-satellite-agents.md](SPEC-satellite-agents.md) — Design spec for running isolated Hermes agent instances in Docker that consume this model service
- [SAMcloud](https://github.com/phoria-sam-tg) — Network services registry with device enrollment, resource leasing, and service discovery

## License

MIT
