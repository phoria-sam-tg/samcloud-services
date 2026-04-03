# Changelog & Working Notes

Project history and current state. This is a living document — updated as things evolve.

## Current State (2026-04-04)

### Services Running on slice-test (M1 Max, 64GB)

| Service | Port | External | Model | Backend |
|---------|------|----------|-------|---------|
| model-service | 8800 | models-stg.samtg.xyz | Qwen3-32B-Q6_K | llama-server (Metal) |
| model-service | 8800 | models-stg.samtg.xyz | qwen3.5:35b-a3b | Ollama (MLX) |
| vlm-service | 8801 | vlm-stg.samtg.xyz | Qwen2.5-VL-7B | mlx-vlm (MLX) |

### SAMcloud Registrations

- `slice-test/model-service` — inference gateway, OpenAI-compatible, samcloud-auth
- `slice-test/vlm-service` — vision-language model, Qwen2.5-VL-7B

### Model Files

| Location | Model | Size |
|----------|-------|------|
| `/Users/sam/models/Qwen3-32B-Q6_K.gguf` | Qwen3-32B | 25GB |
| `~/.ollama/models/` qwen3.5:35b-a3b | Qwen3.5-35B MoE | 22GB |
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

### 2026-04-04 — VLM, Think Bug, Cleanup

- **VLM service**: installed mlx-vlm 0.4.3, serving Qwen2.5-VL-7B on port 8801, registered as `slice-test/vlm-service`
- **Ollama think bug** (ticket #69): Ollama 0.19 `/v1/chat/completions` ignores `think` param — empty content, all output in reasoning field. Workaround: route through native `/api/chat` with `think:false`, translate to OpenAI format in our proxy
- **Tunnel fix** (ticket #68): reverse SSH tunnel on port 10005 was down, external endpoints unreachable. Fixed by admin + claude-containers.
- **Cleanup**: removed duplicate Qwen3.5 GGUF (21GB), old Llama-2-7B, SDXL turbo cache, partial downloads. Freed ~30GB disk

---

## Known Issues

- Lease renewal is release-and-re-request (brief gap under contention)
- Ollama 0.19 `/v1/` think bug — worked around in our proxy, upstream not fixed
- vlm-service (mlx-vlm) is standalone, not managed by ModelManager yet
- Auth disabled on vlm-service (no middleware integrated)

## SAMcloud Tickets Filed

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
