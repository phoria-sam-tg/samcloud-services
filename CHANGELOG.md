# Changelog & Working Notes

Project history and current state. This is a living document.

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
