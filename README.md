# SAMcloud Services

Services running on the SAMcloud home network, managed from slice-test (M1 Max, 64GB).

## Architecture

```
                    SAMcloud Registry
                   stg.samtg.xyz:9443
                  ┌─────────────────────┐
                  │  /api/v1            │
                  │  - services         │
                  │  - resources/leases │
                  │  - devices          │
                  │  - tickets          │
                  └────────┬────────────┘
                           │
              health, leases, registration
                           │
    slice-test ────────────┘
    ┌──────────────────────────────────────────┐
    │                                          │
    │  ollama-manager :8800                    │
    │  ┌────────────────────────────────────┐  │
    │  │ FastAPI (OpenAI-compatible)        │  │
    │  │ POST /v1/chat/completions         │  │
    │  │ POST /v1/completions              │  │
    │  │ GET  /models  /status  /health    │  │
    │  │ POST /models/load  /models/unload │  │
    │  └──────────┬─────────────────────┬──┘  │
    │             │                     │      │
    │  ┌──────────┴──────┐  ┌──────────┴───┐  │
    │  │ Ollama :11434   │  │ llama-server │  │
    │  │ v0.19 + MLX     │  │ llama.cpp    │  │
    │  │                 │  │ b8500+Metal  │  │
    │  │ qwen3.5:35b-a3b │  │ Qwen3-32B   │  │
    │  │ qwen2.5:1.5b    │  │ (port 8000)  │  │
    │  └─────────────────┘  └──────────────┘  │
    │                                          │
    └──────────────────────────────────────────┘
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/status` | Full status — backends, models, leases, resource utilisation |
| `GET` | `/models` | List managed + available models |
| `POST` | `/models/load` | Load a model (auto-leases memory, pulls if needed) |
| `POST` | `/models/unload` | Unload a model (releases lease) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (routes to correct backend) |
| `POST` | `/v1/completions` | OpenAI-compatible completion |

## Quick Start

```bash
# Set your SAMcloud agent token
export SC_TOKEN=<your-token>

# Start the service
python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800

# Load a model
curl -X POST http://localhost:8800/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5:35b-a3b", "backend": "ollama"}'

# Chat
curl -X POST http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Currently Serving

| Model | Backend | VRAM | Speed |
|-------|---------|------|-------|
| Qwen3.5-35B-A3B | Ollama (MLX) | ~31GB | ~23 tok/s |
| Qwen3-32B-Q6_K | llama-server (Metal) | ~25GB | ~4.5 tok/s |

Models spin up on demand and unload after 5 minutes of inactivity, releasing GPU memory leases.

## Project Structure

```
services/
├── CLAUDE.md                    # Project brief for Claude sessions
├── README.md                    # This file
├── SPEC-satellite-agents.md     # Spec: isolated agent environments
├── .gitignore
└── ollama/
    ├── README.md                # Model service documentation
    ├── server.py                # FastAPI server (OpenAI-compatible)
    ├── manager.py               # ModelManager (lifecycle, leases, cooldown)
    ├── samcloud.py              # SAMcloud API client
    ├── ollama_client.py         # Ollama backend client
    ├── llama_client.py          # llama-server backend client
    ├── test_lifecycle.py        # Integration test: full lease cycle
    └── test_cooldown.py         # Integration test: idle unload
```

## SAMcloud Integration

This service registers as `slice-test/ollama-manager` on SAMcloud and:
- Reports health every 60 seconds
- Requests GPU memory leases before loading models
- Releases leases when models are unloaded (idle cooldown or shutdown)
- Renews leases every 30 minutes to prevent indefinite locks

See [SPEC-satellite-agents.md](SPEC-satellite-agents.md) for the next phase:
isolated Hermes agent instances consuming these model services.
