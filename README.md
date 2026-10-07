# samcloud-services

Managed model inference for Apple Silicon Macs on a [SAMcloud](https://github.com/phoria-sam-tg) network.

A FastAPI gateway that unifies [Ollama](https://ollama.com) (MLX), [llama.cpp](https://github.com/ggml-org/llama.cpp) (Metal) and [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (speech to text) behind a single **OpenAI-compatible API**, with centralized GPU memory management via SAMcloud resource leasing.

Models spin up on demand and unload after idle timeout, freeing GPU memory for other workloads.

## Features

- **OpenAI-compatible** — `POST /v1/chat/completions` and `POST /v1/audio/transcriptions` work with any OpenAI SDK client
- **Multi-backend** — Routes to Ollama or llama-server transparently based on model type
- **GPU memory leasing** — Requests SAMcloud leases before loading, releases on unload
- **Auto spin-up/cooldown** — Models load on first request, unload after 5 min idle
- **Process adoption** — Discovers and manages already-running model processes
- **SAMcloud auth** — Verifies caller identity via SAMcloud token verification
- **MLX + Metal** — Ollama 0.19 MLX for Apple Silicon, llama.cpp Metal for GGUF models
- **Partial model matching** — Use `qwen3-32b` instead of `Qwen3-32B-Q6_K`
- **Speech to text** — mlx-whisper behind the same auth, `whisper-1` accepted as a model name

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
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible speech to text (multipart) |

Model names use **case-insensitive partial matching**:
- `qwen3-32b` matches `Qwen3-32B-Q6_K` (llama-server)
- `qwen3.5` matches `qwen3.5:35b-a3b` (Ollama)

The transcription route is the exception: it matches **exactly**, on a catalogue
name (`whisper-large-v3-turbo`, `whisper-small`) or an alias (`whisper-1`,
`whisper`). A transcription model has no chat route and a chat model cannot
transcribe, so each endpoint refuses what the other serves.

```bash
curl -s https://models-cs.samtg.xyz/v1/audio/transcriptions \
  -H @"$HOME/.samcloud/token.hdr" \
  -F file=@walkthrough.m4a \
  -F model=whisper-1
# {"text": "Okay, shed shelf two. There are three orange 20-volt batteries..."}
```

| Form field | Default | Description |
|---|---|---|
| `file` | — | the audio, in any format ffmpeg reads (**required**) |
| `model` | `whisper-large-v3-turbo` | catalogue name or alias |
| `language` | auto-detect | ISO-639-1 code, e.g. `en` |
| `prompt` | — | names and spellings to bias towards |
| `response_format` | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | the fallback ladder | a single temperature, if you want one |
| `timestamp_granularities[]` | `segment` | `segment` or `word`; needs `verbose_json` |

### Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/models/load` | Load a model — pulls if needed, requests GPU lease |
| `POST` | `/models/unload` | Unload a model — releases GPU lease |
| `GET` | `/models` | List managed and available models |
| `GET` | `/v1/models/{id}` | One model's entry, including its `context_length` |
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

`backend`: `"auto"` (default), `"ollama"`, `"llama-server"`, or `"mlx-vlm"`. Auto selects llama-server for `.gguf` files, mlx-vlm for recognised vision-language model names (e.g. `qwen2.5-vl`, `gemma-4`), and Ollama for everything else.

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
    │  ┌────────────┐ ┌────────────┐ ┌──────────┐ │
    │  │ Ollama     │ │ llama-srv  │ │ mlx-vlm  │ │
    │  │ :11434 MLX │ │ llama.cpp  │ │ :8801 VL │ │
    │  └────────────┘ └────────────┘ └──────────┘ │
    └──────────────────────────────────────────────┘

All three backends spin up on demand and are torn down on idle cooldown.
The gateway **owns** each backend process it starts (mlx-vlm included) — it
is not pre-started out of band, and there is no boot-order dependency.
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

All defaults are env-driven via `ollama/config.py` and target the production
samcloud registry + the `claude-services-slice` device.

| Variable | Default | Description |
|----------|---------|-------------|
| `SC_TOKEN` | — | SAMcloud agent token (**required**) |
| `SC_BASE` | `https://cloud.samtg.xyz/api/v1` | SAMcloud registry base URL |
| `SC_DEVICE` | `claude-services-slice` | Device identity |
| `SERVICE_PORT` | `8800` | Server listen port |
| `SC_VERIFY_URL` | `${SC_BASE}/auth/verify` | SAMcloud auth endpoint |
| `SC_REQUIRED_SCOPE` | `device:${SC_DEVICE}` | Required scope for callers |
| `AUTH_ENABLED` | `true` | Set `false` to disable auth (development only) |
| `OLLAMA_NUM_CTX` | unset | Served context window for every Ollama model, in tokens. Unset = Ollama derives one from free VRAM |
| `OLLAMA_NUM_CTX_MODELS` | unset | Per-model override: `qwen3.8:27b-mlx=262144,qwen3:1.7b=40960`. Wins over `OLLAMA_NUM_CTX` |
| `OLLAMA_GENERATE_TIMEOUT` | `1800` | Whole streaming request, seconds. Enforced in-process, not by aiohttp |
| `OLLAMA_STALL_TIMEOUT` | `60` | Max silence **between** tokens, once the first has arrived |
| `OLLAMA_FIRST_TOKEN_RATE_TPS` | **unset** | Measured prefill rate, tok/s. Unset = no first-token deadline |
| `OLLAMA_FIRST_TOKEN_MARGIN_S` | `90` | Added to the scaled first-token budget |
| `OLLAMA_FIRST_TOKEN_MIN_S` | `300` | Floor under the first-token budget, so a short prompt is never worse off |
| `OLLAMA_CHARS_PER_TOKEN` | `1.5` | Fallback ratio when an Ollama model has no tokenizer |
| `VLM_PYTHON` | `~/code/mlx-vlm-server/.venv/bin/python` | Python that runs `mlx_vlm.server` |
| `VLM_HOST` | `127.0.0.1` | Host the on-demand mlx-vlm server binds |
| `VLM_PORT` | `8801` | Port for the on-demand mlx-vlm server |
| `VLM_STARTUP_TIMEOUT` | `120` | Seconds to wait for mlx-vlm to become healthy |
| `WHISPER_ENABLED` | `true` | Advertise transcription models on `/v1/models` |
| `WHISPER_PYTHON` | `~/code/mlx-whisper-server/.venv/bin/python` | Python that runs `ollama/whisper_server.py` |
| `WHISPER_HOST` | `127.0.0.1` | Host the on-demand whisper child binds |
| `WHISPER_PORT` | `8805` | Port for the on-demand whisper child |
| `WHISPER_STARTUP_TIMEOUT` | `300` | Seconds to wait for the child to become healthy |
| `WHISPER_REQUEST_TIMEOUT` | `1800` | Seconds one transcription may take |
| `WHISPER_MAX_UPLOAD_MB` | `200` | Upload limit, refused with `413` |
| `WHISPER_SPOOL_DIR` | `~/var/samcloud-services/spool/whisper` | Where an upload lands on its way to the child |
| `WHISPER_LOG_FILE` | `~/var/samcloud-services/logs/whisper-child.log` | The child's stdout and stderr |
| `FFMPEG_BIN` | `$(which ffmpeg)` or `/opt/homebrew/bin/ffmpeg` | Decodes every upload |

### Tuning (manager.py constants)

| Constant | Default | Description |
|----------|---------|-------------|
| `COOLDOWN_SECONDS` | `300` (5 min) | Idle time before auto-unload |
| `LEASE_TTL` | `3600` (1 hr) | Lease duration in seconds |
| `LEASE_RENEW_AT` | `0.5` | Renew at 50% of TTL |
| `RESOURCE_ID` | `slice-test/gpu-0` | SAMcloud resource to lease |

## The served context window

**Size your harness against `context_length` on `/v1/models`. Do not pick a
number.** Every entry that we can answer for carries it:

```bash
curl -s https://models-cs.samtg.xyz/v1/models/qwen3.8:27b-mlx \
  -H @"$HOME/.samcloud/token.hdr"
# {"id": "qwen3.8:27b-mlx", "status": "resident",
#  "memory_mb": 31187, "context_length": 262144}
```

On a **resident** model that is what the live instance actually has, read from
`ollama ps`. On a **loadable** one it is the window we pin, and it is **absent**
where we pin nothing — because Ollama then derives the window from free VRAM at
load time and a prediction of that would be a guess. An absent field means "we
do not know", which is **not** the same as "no limit".

Why this exists (#903): nothing used to set `num_ctx`, and the absence was not
neutral. Ollama logs the window it picks as `vram-based default context`, and
slice's own log has it as 262,144 nineteen times — and as **4,096 twice**, on
2026-08-25 and 2026-08-26, when llama-server GPU discovery timed out and Ollama
fell back to CPU. That is a 64x spread in what a caller got, with no endpoint
reporting which one. `/v1/models` carried `memory_mb` and nothing else;
`/models/info` and `/v1/models/{id}` both 404'd. So a consumer sizing against
this gateway had to guess, and one shipped a hand-picked 65,536 — a window this
box has never served.

### The window is not the usable prompt, and the limit is time

**`context_length` is a memory bound. There is a second, tighter bound made of
time**, and a consumer sizing against the first will be cut by the second.

From `claude-wafer-services`' sweep on the same model (#904), cold prefill
~74.7 tok/s — a floor, since a cold prefill pages weights in as it goes:

| prompt | prefill alone | vs `OLLAMA_GENERATE_TIMEOUT` (1800s) |
|---|---|---|
| 8,000 | ~107s | 6% |
| 32,000 | ~428s | 24% |
| **70,000** | **~937s** | **52%** — #884's trigger |
| 131,072 | ~1755s | 98% |
| **262,144** | **~3509s** | **195% — cannot complete** |

So the published window is 262,144 and the **time-usable** prompt is **~125,000
tokens**, leaving room to decode. 70,000 fits with less headroom than it reads.

Two things follow. **A box that wants full-window prompts must raise
`OLLAMA_GENERATE_TIMEOUT`** — the ceiling is the binding constraint well before
the window is. And retrospectively, under the 300s cap that #904 replaced, any
prompt over **~22,000 tokens** could not complete at all: prefill alone
exceeded the wall. The 65,536 `hermes-assistant` believed it had was never
usable, and neither was 32,000.

Rates are per box. Re-derive from `_log_ollama_timings` on the box you are
sizing for rather than carrying these across.

`OLLAMA_NUM_CTX_MODELS` makes the window a decision instead of a side-effect of
memory pressure. Two things it is not:

- **not a reservation.** The MLX engine grows the KV cache as a conversation
  fills it — measured on slice 2026-10-08, `ollama ps` climbing 26,681 →
  30,311 → 31,187 MB on one resident model at a fixed 262,144. Raising the
  number costs nothing at load; a long conversation costs memory later.
- **not a limit we can exceed.** A configured value above the window the
  weights declare is clamped to the weights and logged once, rather than passed
  to Ollama to allocate for.

One operational note: every path that touches a model sends the same `num_ctx`,
because **Ollama keys a loaded instance by its options** — a request whose
`num_ctx` differs from the resident instance's reloads the model. Adoption is
the deliberate exception: on startup the gateway re-applies the *instance's own*
window, so a gateway restart cannot bounce a model mid-job to change a number.
The configured window takes effect on the next real load.

## Streaming deadlines

**A timeout is only meaningful relative to every other timeout on the path, and
the one that fires first should be the one that reports best.**
(`claude-containers`, #904.) This gateway used to fire first and report worst.

The owned Ollama streaming path had one bound, aiohttp's `total=` at 300s, and
`total` bounds **elapsed time**. So a generation streaming tokens steadily was
killed at 300s for being long — the one thing that is not a fault. Measured on
slice 2026-10-08, counting `Stream error` in `server.log` rather than Ollama's
status column: **18 cut-offs in 81 `/api/chat` requests, 22%**, and the longest
request that actually completed ran **285.0s — 15 seconds** under the wall.

(The status column undercounts: our abort races the response completion, so 11
were logged `500 | 5m0s` and **6 were logged `200 | 5m0s`** — the same event as
a success. Count the gateway's own log, not Ollama's.)

Three bounds now, the structure `exo_client` arrived at on #830:

| | what it bounds | default |
|---|---|---|
| `OLLAMA_FIRST_TOKEN_*` | silence **before** token 1 — prefill, legitimately silent | **unarmed** |
| `OLLAMA_STALL_TIMEOUT` | silence **between** tokens — a stopped generation | `60` |
| `OLLAMA_GENERATE_TIMEOUT` | the whole request | `1800` |

Why not one silence bound for both: prefill produces nothing at all, for longer
the longer the prompt. `sock_read=60` would kill a 70k-token request at 60s
instead of 300s — strictly worse, and it would look like a stall. So the
first-token budget scales with the prompt and the inter-token budget does not.

**The first-token deadline is unarmed by default and setting
`OLLAMA_FIRST_TOKEN_RATE_TPS` is what enables it** — the MLX runner emits no
per-request timings, so this box has no measured prefill rate and any default
would be an invented number. Unarmed, a silent prefill falls back to the
whole-request budget. The inter-token bound is armed regardless, so the single
inference slot (#97) is covered either way; it is also what makes a generous
whole-request ceiling affordable, since raising the ceiling alone would trade a
5-minute truncation for a 30-minute slot occupation on a wedge.

`total=None` is passed to aiohttp, which is **not** "no ceiling":
`OLLAMA_GENERATE_TIMEOUT` is enforced in our own loop. aiohttp's `total` and our
`wait_for` both raise `asyncio.TimeoutError`, and one `except` cannot tell them
apart — owning the deadline is what makes the cause nameable.

The sync (`httpx`) path is **unchanged and was always correct**: httpx `read` is
a per-chunk idle bound, i.e. already silence. The bug was transcribing its `300`
into aiohttp's wall-clock `total=` — the intent was always "300s of silence" and
the shape was lost in the translation.

Every failure now terminates the stream properly. A cut-off used to log and stop
yielding: no `[DONE]`, no `finish_reason`, and `TimeoutError` stringifies to the
empty string so even our log read `Stream error for <model>:` and nothing. Now:

```json
{"error": {"message": "...", "type": "model_stalled", "deadline": "first_token",
           "deadline_s": 1257.0, "tokens_before_stall": 0, "silent_for_s": 1257.0},
 "choices": [{"delta": {}, "finish_reason": "stalled"}]}
```

`model_stalled` / `request_timeout` / `stream_error`, each with its own
`finish_reason`, then `[DONE]`.

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

### mlx-vlm (vision-language)

- Apple MLX vision-language models (Qwen2.5-VL, Gemma 4, etc.)
- The gateway starts `mlx_vlm.server` on demand, leases GPU memory, and stops
  it on idle cooldown — it owns the process end to end (no adoption, no
  boot-order pre-start). On startup any stray `mlx_vlm.server` is reaped so the
  gateway always begins from a clean owned state.
- Configured via `VLM_PYTHON` / `VLM_HOST` / `VLM_PORT` (see Configuration)

### mlx-whisper (speech to text)

- `whisper-large-v3-turbo` (the default) and `whisper-small`, from `mlx-community`
- Owned exactly like mlx-vlm: the gateway starts `ollama/whisper_server.py` under
  `WHISPER_PYTHON` on demand, leases the memory the model was measured to peak
  at, and kills it on idle cooldown. A stray child is reaped on startup.
- A separate interpreter, not the gateway's: see `requirements-whisper.txt` for
  what goes in it and why `torch` comes straight back out.
- One transcription at a time, and the cooldown loop leaves a model alone while
  a request is in flight — a one-hour walkthrough runs about seven minutes,
  which outlives the five-minute idle timer.

## Project Structure

```
samcloud-services/
├── README.md                    # This file
├── CLAUDE.md                    # Development brief for AI assistants
├── SPEC-satellite-agents.md     # Spec: isolated agent environments
├── CHANGELOG.md                 # Project history and working notes
├── requirements.txt             # Python dependencies (the gateway)
├── requirements-whisper.txt     # Python dependencies (the whisper child's venv)
└── ollama/
    ├── README.md                # Module documentation
    ├── server.py                # FastAPI server + auth middleware
    ├── manager.py               # ModelManager — lifecycle, leases, cooldown
    ├── samcloud.py              # SAMcloud API client
    ├── ollama_client.py         # Ollama backend client
    ├── llama_client.py          # llama-server backend client
    ├── whisper_client.py        # The gateway's half of the whisper child
    ├── whisper_server.py        # The whisper child itself — run by WHISPER_PYTHON
    ├── test_lifecycle.py        # Integration test: full lease cycle
    └── test_cooldown.py         # Integration test: idle unload
```

## Testing

```bash
cd ollama
python test_lifecycle.py    # Pull → lease → load → infer → unload → release
python test_cooldown.py     # Load → idle → auto-unload → lease released
```

Transcription, from the repo root — none of these load a model or reach the
registry:

```bash
python -m ollama.test_whisper_routing        # the endpoint: names, refusals, five formats
python3 ollama/test_whisper_kill_guard.py    # what the stray-child reaper may signal
$WHISPER_PYTHON ollama/test_whisper_child.py # the child: paths, ffmpeg, numpy in JSON
```

The last one needs the child's interpreter. Give it
`WHISPER_TEST_AUDIO=<file>` to add a real transcription on top.

## Related

- [SPEC-satellite-agents.md](SPEC-satellite-agents.md) — Design spec for running isolated Hermes agent instances in Docker that consume this model service
- [SAMcloud](https://github.com/phoria-sam-tg) — Network services registry with device enrollment, resource leasing, and service discovery

## License

MIT
