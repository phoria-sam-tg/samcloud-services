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
| `whisper_client.py` | `WhisperClient` — the gateway's half of the whisper child (`/health`, `/transcribe`) |
| `whisper_server.py` | The whisper child itself. Run by `WHISPER_PYTHON` as a script, so it imports nothing from this package |
| `test_lifecycle.py` | Integration test: full pull -> lease -> load -> infer -> unload -> release cycle |
| `test_cooldown.py` | Integration test: load -> idle -> auto-unload -> lease released |
| `test_whisper_routing.py` | The transcription endpoint: name resolution, refusals, the five response formats |
| `test_whisper_kill_guard.py` | What `_kill_stray_whisper` may and may not SIGTERM |
| `test_whisper_child.py` | The child: spool path handling, ffmpeg decoding, numpy in JSON. Needs `WHISPER_PYTHON` |

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

Transcription is a separate path, and separate on purpose — a transcription model
cannot answer a chat and `_resolve_model` cannot reach one:

```
Client
  │  POST /v1/audio/transcriptions   multipart: file=@note.m4a  model=whisper-1
  ▼
SamcloudAuthMiddleware              (the same auth as every other route)
  ▼
validate                            response_format, timestamp_granularities[]
  ▼
_resolve_whisper("whisper-1")
  │  → exact match on a catalogue name or an alias → whisper-large-v3-turbo
  │  → fit gate → lease → spawn the child under WHISPER_PYTHON → poll /health
  ▼
spool the upload                    WHISPER_SPOOL_DIR, bounded, unlinked in a finally
  ▼
child: ffmpeg → 16kHz mono float32 → mlx_whisper.transcribe
  ▼
render                              json | text | verbose_json | srt | vtt
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

The transcription tests need neither, and none of them loads a model:

```bash
python -m ollama.test_whisper_routing        # from the repo root
python3 ollama/test_whisper_kill_guard.py
$WHISPER_PYTHON ollama/test_whisper_child.py
```
