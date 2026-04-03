# ollama/ — Model Service Module

The core inference gateway. Manages multiple model backends behind a single OpenAI-compatible API
with SAMcloud resource leasing for GPU memory.

## Modules

| Module | Responsibility |
|--------|---------------|
| `server.py` | FastAPI app, auth middleware, OpenAI-compatible endpoints, backend routing |
| `manager.py` | `ModelManager` — model lifecycle, lease management, cooldown, health, process adoption |
| `samcloud.py` | `SamcloudClient` — SAMcloud API (resources, leases, services, health, auth) |
| `ollama_client.py` | `OllamaClient` — Ollama API (pull, load, unload, generate, chat) |
| `llama_client.py` | `LlamaServerClient` — llama-server process management (discover, start, stop, health) |
| `test_lifecycle.py` | Integration test: full pull -> lease -> load -> infer -> unload -> release cycle |
| `test_cooldown.py` | Integration test: load -> idle -> auto-unload -> lease released |

## How Requests Flow

```
Client
  │  POST /v1/chat/completions  {model: "qwen3.5", messages: [...]}
  │  Authorization: Bearer sc_agent_xxx
  ▼
SamcloudAuthMiddleware
  │  → GET /auth/verify?scope=device:slice-test (cached 5 min)
  │  → 401 / 403 / proceed
  ▼
_resolve_model("qwen3.5")
  │  → case-insensitive partial match → finds "qwen3.5:35b-a3b" (Backend.OLLAMA)
  ▼
Backend routing
  ├─ OLLAMA:  native /api/chat with think:false → translate to OpenAI format
  └─ LLAMA:   forward to llama-server /v1/chat/completions directly
```

## Model Lifecycle

```
Not loaded → estimate memory → request SAMcloud lease → pull/start → load → serve
                                                                        │
Idle 5 min ────────────────────── unload ← release lease ← cooldown check
```

The `ModelManager` runs three background tasks:

| Task | Interval | What it does |
|------|----------|-------------|
| Cooldown | 60s check | Unloads models idle > `COOLDOWN_SECONDS` (default 5 min) |
| Health | 60s | Reports health to SAMcloud registry |
| Lease renewal | 30 min | Release and re-request leases (prevents indefinite locks) |

## Process Adoption

On startup, `ModelManager.discover()` finds already-running processes:

- **llama-server**: parsed from `ps aux` — extracts `--model`, `--port`, PID
- **Ollama models**: queried from `GET /api/ps` — name, VRAM size

Adopted models are `managed=False` — monitored and lease-tracked, but NOT killed on cooldown
unless explicitly forced.

## Auth Middleware

`SamcloudAuthMiddleware` in `server.py`:

1. Extracts `Authorization: Bearer <token>` from request
2. Checks in-memory cache (keyed by token hash, 5 min TTL)
3. On miss: calls SAMcloud `GET /auth/verify?scope=<required_scope>`
4. 200 → proceed (caches result), 401 → reject, 403 → out of scope

Exempt paths: `/health`, `/service-docs`

## Ollama Think Bug Workaround

Ollama 0.19's `/v1/chat/completions` ignores the `think` parameter for reasoning models
like Qwen3.5, returning empty `content` with all output in a `reasoning` field.

The fix: for Ollama models, we route through the native `/api/chat` endpoint with `think:false`
(which works correctly) and translate the response to OpenAI format ourselves — both streaming
(SSE `data:` lines) and non-streaming.

This is transparent to consumers. See ticket #69 for details.

## Configuration

See [root README](../README.md#configuration) for environment variables and tuning constants.

## Testing

Both tests require a running SAMcloud registry and Ollama instance.

```bash
cd ollama
python test_lifecycle.py    # ~30s — pulls a small model, full cycle
python test_cooldown.py     # ~45s — loads, waits 30s, verifies unload
```
