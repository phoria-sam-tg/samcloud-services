# Changelog & Working Notes

Project history and current state. This is a living document — updated as things evolve.

## Current State (2026-04-04)

### Services Running on slice-test (M1 Max, 64GB)

| Service | Port | External | What | Backend |
|---------|------|----------|------|---------|
| model-service | 8800 | models-stg.samtg.xyz | Inference gateway (all Ollama + llama-server models) | FastAPI proxy |
| vlm-service | 8801 | vlm-stg.samtg.xyz | Gemma 4 31B vision-language | mlx-vlm (MLX nvfp4) |

### Available Models (via model-service :8800)

| Model | Size | Tool calling | Use case |
|-------|------|-------------|----------|
| `hermes3:8b` | 4.7GB | Native (built for Hermes) | Primary agent model, fast, reliable |
| `qwen3-coder:30b` | 18GB | Agentic | Heavy reasoning + tool calling |
| `glm4` | 5.5GB | Yes | 198K context, good tool support |
| `qwen3.5:35b-a3b` | 23GB | Broken in Ollama | Text/reasoning only |
| `qwen2.5:1.5b` | 986MB | No | Lightweight test model |

### Vision Model (vlm-service :8801)

| Model | Size | Speed | Backend |
|-------|------|-------|---------|
| Gemma 4 31B (nvfp4) | 18.7GB | 18.6 tok/s | mlx-vlm, MLX-native quant |

### SAMcloud Registrations

- `slice-test/model-service` — inference gateway, tool-calling, samcloud-auth, service-docs
- `slice-test/vlm-service` — vision-language, Gemma 4 31B, MLX

### Model Files

| Location | Model | Size |
|----------|-------|------|
| `/Users/sam/models/Qwen3-32B-Q6_K.gguf` | Qwen3-32B | 25GB |
| `~/.ollama/models/` | hermes3:8b, qwen3-coder:30b, glm4, qwen3.5:35b-a3b, qwen2.5:1.5b | ~52GB |
| `~/.cache/huggingface/` | Gemma 4 31B nvfp4, Qwen2.5-VL-7B | ~15GB |
| `~/.ollama/models/` qwen2.5:1.5b | Qwen2.5-1.5B | 940MB |
| `~/.cache/huggingface/` Qwen2.5-VL-7B-4bit | Qwen2.5-VL-7B | ~5GB |

---

## Timeline

### 2026-03-31 — Initial Setup

- Enrolled as `claude-services` on SAMcloud (agent, user:17, scope: device:slice-test)
- Built ollama-manager: SAMcloud client, Ollama client, llama-server client, unified ModelManager, FastAPI server
- Registered `slice-test/ollama-manager` on SAMcloud
- Tested full resource lease lifecycle: pull -> lease -> load -> infer -> unload -> release

### 2026-04-02 — Ticket #42 Response

- Admin flagged: 25GB lease sat indefinitely with no expiry for 2 days
- Root causes: no lease renewal, no health-bound cleanup, memory over-estimated
- Fixed: added lease renewal loop (30 min), expiry validation, clean shutdown
- Upgraded Ollama 0.15.4 -> 0.19.0 (MLX support, +57% prefill, +93% decode)
- Pulled qwen3.5:35b-a3b (MLX, 23GB) — 22.8 tok/s decode on M1 Max

### 2026-04-03 — Auth, Rename, Docs

- **Auth middleware** (ticket #46): SAMcloud token verification on all endpoints, 5-min cache, scope checking
- **Service rename** (ticket #65): ollama-manager -> model-service. Added `description` and `docs_endpoint` fields to SAMcloud
- **`GET /service-docs`** endpoint: public, auth-exempt, returns structured JSON + full markdown guide. Convention for all services.
- **Ticket #64** response: documented model service topology for satellite agent consumers
- **Satellite agent spec** written: isolated Hermes instances in Docker consuming model services
- Wrote CLAUDE.md, README.md, ollama/README.md

### 2026-04-04 — VLM, Think Bug, Tool Calling, Gemma 4

- **VLM service**: installed mlx-vlm 0.4.3, now serving Gemma 4 31B (nvfp4, MLX-native) on port 8801. 18.6 tok/s, 18.7GB peak.
- **Ollama think bug** (ticket #69): Ollama 0.19 `/v1/chat/completions` ignores `think` param — empty content, all output in reasoning field. Workaround: route through native `/api/chat` with `think:false`, translate to OpenAI format in our proxy
- **Model unloading fix** (ticket #74): Models were dropping after 5min (Ollama default keep_alive). Fixed: `keep_alive=-1` on all loads, adopted models pinned on startup, auto-reload if model dropped between requests.
- **Tool-calling models** (ticket #76): Pulled hermes3:8b (native tool calls for Hermes agent), qwen3-coder:30b (agentic reasoning), glm4 (198K context). These support structured tool_calls for autonomous agent workflows.
- **Ollama upgraded** 0.19.0 -> 0.20.0
- **Tunnel fix** (ticket #68): reverse SSH tunnel on port 10005 was down, external endpoints unreachable. Fixed by admin + claude-containers.
- **Cleanup**: removed duplicate Qwen3.5 GGUF (21GB), old Llama-2-7B, SDXL turbo cache, partial downloads. Freed ~30GB disk
- **GitHub**: repo published at github.com/phoria-sam-tg/samcloud-services

---

## Known Issues

- Lease renewal is release-and-re-request (brief gap under contention)
- Ollama `/v1/` think bug — worked around in our proxy, upstream not fixed
- vlm-service (mlx-vlm) is standalone, not managed by ModelManager yet
- Auth disabled on vlm-service (no middleware integrated)
- qwen3.5 tool calling broken in Ollama — use hermes3:8b or qwen3-coder:30b for agent tool calls

## SAMcloud Tickets

| # | Summary | Status |
|---|---------|--------|
| 42 | Unbounded GPU lease — 25GB indefinite | Resolved |
| 43 | Cannot read ticket 42 (scope issue) | Resolved |
| 46 | RFC: Token verification for service-to-service auth | Resolved |
| 64 | Model service naming/endpoint confusion | Resolved |
| 65 | Service rename, description field, docs_endpoint | Resolved |
| 66 | DELETE /services returns 500 | Resolved |
| 68 | Reverse tunnel down — external endpoints unreachable | Resolved |
| 69 | Ollama /v1/ ignores think parameter for qwen3.5 | Resolved (workaround) |
| 71 | Interview: architecture, auth, leasing, service-docs | Responded |
| 74 | Model keeps unloading during Hermes sessions | Resolved (keep_alive=-1) |
| 76 | Need tool-call-capable models for agent containers | Resolved |
