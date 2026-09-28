# Development Brief

This is **samcloud-services** — infrastructure for managed model inference on a SAMcloud network.

## What This Does

A FastAPI gateway (`ollama/server.py`) that unifies Ollama (MLX), llama-server (llama.cpp),
mlx-vlm (vision-language) and mlx-whisper (speech to text) behind an OpenAI-compatible
API, with SAMcloud resource leasing for GPU memory management. Models spin up on demand,
unload after 5 min idle.

A fifth backend, the **exo pool** (`Backend.EXO`), is reached the same way but owned
differently: the gateway neither starts it nor places its model, and holds an exclusive
lease around each generation. Placing the model is still out of band — that limitation
stands. Starting the node is not: the pool ran as `sam` and was believed to need
Terminal.app for local-network access, and as of 2026-09-22 (#806) both nodes run
headless under their box's services account (`claude-services` on slice,
`wafer-services` on wafer) from a shared store at `/Users/Shared/exo`. See
`ollama/config.py` for what was measured.

## SAMcloud Identity

Every identity is env-driven via `ollama/config.py`. **This runs on more than one
box — the values below are the code's fallbacks, not a description of production.**
They happen to spell out `claude-services-slice`, so an agent reading them on any
other box is reading the wrong box's identity. Read the box's own env file
(`~/.config/samcloud-services/env`, 600, sourced by the start script) before
assuming any of them.

| | `config.py` fallback | wafer-services runs | claude-services-slice runs |
|---|---|---|---|
| `SC_DEVICE` | `claude-services-slice` | `wafer-services` | `claude-services-slice` |
| `SC_RESOURCE_ID` | `<device>/gpu-0` | `wafer-services/gpu-metal` | `claude-services-slice/gpu-0` |
| `SC_REQUIRED_SCOPE` | `device:<device>` | `group:services` | `group:services` |
| `SC_BASE` | `https://cloud.samtg.xyz/api/v1` | same | same |
| `SC_SERVICE_NAME` | `model-service` | same | same |

`SC_REQUIRED_SCOPE` is the one row no box runs as written. It is what the auth
middleware checks every caller's token against, so the fallback describes a
narrower access policy than either gateway actually enforces — read it as a safe
default for a new box, not as a description of this one.

**Token**: always via `SC_TOKEN` env — never hardcode; a hardcoded copy is what
silently 401'd a box for fourteen days after a rotation (#760). It need not be a
service token: wafer's gateway runs under a dedicated *agent* identity
(`wafer-model-service`, scopes exactly `["device:wafer-services"]`), which is why
it can read its own resource where a scope-filtered service token could not.

Staging (legacy) used `slice-test/*` identities pointing at `stg.samtg.xyz:9443` — see the `migration` branch for the pre-migration architecture.

## Key Files

| File | What to know |
|------|-------------|
| `ollama/config.py` | Env-driven config. Every SAMcloud identity + backend path reads from here |
| `ollama/server.py` | FastAPI server + auth middleware. Thin routing — delegates to manager |
| `ollama/manager.py` | Core logic. `ModelManager` handles lifecycle, leases, cooldown, adoption, the fit gate on load, and the stats + offering background loops. Discovery is resilient to any backend being down |
| `ollama/capacity.py` | **The one capacity signal.** Byte-identical on every box so an `offering:` means the same thing fleet-wide. Every fit decision, offering tier and resource-stats push reads from here — do not re-derive memory anywhere else |
| `ollama/prompt_size.py` | **The prompt gate for the pool** (#837). Counts a prompt with the resident model's own tokenizer before dispatch and refuses over `EXO_MAX_PROMPT_TOKENS`. Falls back to a conservative chars/token estimate |
| `ollama/samcloud.py` | SAMcloud API client. All registry calls go through this |
| `ollama/ollama_client.py` | Ollama API. Note: `chat()` passes `**kwargs` so `think=False` works |
| `ollama/llama_client.py` | llama-server process management. `discover_running()` parses `ps aux` |
| `ollama/whisper_client.py` | The gateway's half of the whisper child — `/health` sync (the spawn loop polls it), `/transcribe` async (a transcription runs for minutes) |
| `ollama/whisper_server.py` | The whisper child. Executed as a SCRIPT by `WHISPER_PYTHON`, so it imports nothing from this package — a relative import here is an ImportError at startup. Decodes with the ffmpeg the gateway names, never PATH |
| `ollama/exo_client.py` | The exo pool. Already OpenAI-compatible, so chat is a proxy. `resident_model()` reads `/state`; non-streaming goes through `chat_collect()` because exo's own `stream:false` returns no body |

## Run / Test

```bash
SC_TOKEN=<token> python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800

# From the repo ROOT — relative imports, so `cd ollama && python <file>.py`
# raises ImportError: attempted relative import with no known parent package.
python -m ollama.test_capacity_refusal   # Refusal path: 503 + the numbers, not 500

# Older tests, from inside ollama/ — these lease and load for real.
cd ollama && python test_lifecycle.py   # Full lease cycle
cd ollama && python test_cooldown.py    # Idle unload verification
python ollama/test_exo_lease.py        # Pool lease verdict + resident model (no network)
python -m ollama.test_prompt_size      # Prompt gate: 413 shape, and that a refusal takes no lease
python ollama/test_exo_stall.py        # Abort a generation that starts then stops (no network)

# Transcription. None of these loads a model or reaches the registry.
python -m ollama.test_whisper_routing        # Endpoint: names, refusals, five formats, cooldown hold
python3 ollama/test_whisper_kill_guard.py    # What the stray-child reaper may signal
$WHISPER_PYTHON ollama/test_whisper_child.py # Child: spool paths, ffmpeg, numpy in JSON
```

The child test runs under the CHILD's interpreter, not the gateway's — it imports
`whisper_server`, which imports numpy. `WHISPER_TEST_AUDIO=<file>` adds a real
transcription to it. Setting up that interpreter:

```bash
uv venv --python 3.12 ~/code/mlx-whisper-server/.venv
uv pip install --python ~/code/mlx-whisper-server/.venv/bin/python -r requirements-whisper.txt
uv pip uninstall --python ~/code/mlx-whisper-server/.venv/bin/python torch
```

`test_capacity_refusal` loads nothing and leases nothing — the refusal precedes
both — so it is the one safe to run against a live box. It reads real memory, so
it is a live check of the gate and not only of the shape. It self-skips if the
box genuinely has room for a 40GB model — note that the skip returns success
having run step 1 only, so a green result on a very large box has not exercised
the 503 path the test exists for.

## Conventions

- All SAMcloud API calls go through `SamcloudClient` (never raw httpx)
- All model operations go through `ModelManager` (server.py is thin routing)
- `GET /service-docs` is the service discovery convention — auth-exempt, returns structured JSON + full markdown guide
- Auth via SAMcloud token verification (`GET /auth/verify?scope=$SC_REQUIRED_SCOPE`)

## Architecture Decisions

- **Bounded TTL leases** — renewed every 30 min, not indefinite (ticket #42)
- **Health-bound revocation** — stale services should lose leases
- **Actual VRAM** — use `ollama ps` sizes, not file-size estimates
- **Agents lease, not register** — resource registration is device-daemon territory
- **Ollama think workaround** — route through native `/api/chat` with `think:false`, translate to OpenAI format ourselves (ticket #69)
- **On-demand mlx-vlm, gateway-owned** — the gateway starts/stops `mlx_vlm.server` itself (`load_vlm_model` + `match_vlm_model`), rather than adopting a pre-started process. On startup it reaps any stray `mlx_vlm.server`. No boot-order dependency; one VLM per `VLM_PORT`
- **Fit, don't evict** — a load checks against what the box can hand over now; it does not unload whatever is resident to force-fit the ask. Eviction is opt-in behind `AUTO_EVICT` (default off). The contract is "here is what fits, pick one"
- **A refusal is an answer, not a fault** — `capacity.InsufficientCapacity` carries `need_mb`/`usable_mb`/`available_mb`/`fits_now`, and every load path returns it through the single `server._capacity_503()` funnel so callers cannot tell the backends apart by how they decline. It derives from `Exception`, **not** `RuntimeError`, so the `except RuntimeError -> 409` on `/models/load` cannot swallow it (ticket #756)
- **Size off `available`, not `used`** — `free + inactive + speculative`. macOS "used" counts pages the compressor is merely holding and is not a fit signal. Page size comes from the kernel; Apple silicon is 16KiB, and assuming 4KiB under-reported a box by 4x
- **Offering tier derives from the catalogue** — `full` = everything we hold fits, `mini` = exactly one does, `none` = nothing does. Fixed MB bands go stale the moment the catalogue changes (ticket #135, doc #8)
- **One stats push per box, from the gateway** — `ModelManager.stats_loop()`, lifecycle-managed. Not a separate daemon; `capacity.registry_payload()` narrows the reading to what the registry's strict schema accepts
- **The pool is leased per generation, not per residency** (ticket #770) — `Backend.EXO`
  is the one backend the gateway does not own. ONE exo instance spans slice and wafer,
  serves ONE request at a time, and is placed out of band; a model swap costs 30s-10min,
  so callers ask for a **tier** (`model: "think"`) and get whatever is resident, read
  from `/state` at request time rather than hardcoded. Its samcloud lease is
  **exclusive** and means "the pool is TAKEN", not that bytes are reserved — its pages
  are already wired and already counted by each box's own `capacity.py`, and
  `memory_mb` must be `null` (see below). Acquired around the generation, released in a
  `finally`.
- **A queued lease is not a granted lease** (ticket #770) — `_request_lease` used to
  return an id for any response that did not raise, so a queued lease read as granted.
  Harmless on shared `gpu-0`; on an exclusive resource it is the whole bug. The verdict
  is a `LeaseOutcome`, and it comes from the **body** rather than the status code:
  the registry grants with a plain `200`, not the `201` the API index documents, so
  code-only logic fails in one direction or the other.
- **Never send `memory_mb` for the pool** (ticket #770) — the registry queues when
  `memory_mb > available`, and `available` is `total - leased` where `total` is read
  from `vram_mb`/`gpu_memory_mb`/`unified_memory_mb`/`ram_mb` in the resource specs.
  `exo-pool` carries none, so `total` is 0 and *any* byte count queues forever no
  matter how idle the pool is.
- **`EXO_LEASE_TTL` must exceed `EXO_GENERATE_TIMEOUT`** — the registry expires a lease
  on time and cannot extend one, and renewing by release-then-reacquire would open a
  window for a third party to take an exclusive resource mid-generation. `config.py`
  clamps the ordering rather than trusting the env.
- **Comments carry causation badly** (ticket #770) — two false facts were written
  into this codebase in one session and had to be pulled back out: that exo's
  `stream: false` "returns 200 headers and then no body, ever", and that a client
  failed because it "omits `stream`". Both were **real measurements with wrong
  conclusions attached** — the first probed a pool that was already occupied, the
  second trusted a client's own pre-send dump instead of the socket. Neither
  survived contact with a second instrument.
  A measurement ages well; the causal story attached to it does not, and a comment
  is where the two become indistinguishable to a reader who was not there and
  cannot check. So: record what was measured and how, keep the inference visibly
  separate from it, and prefer "measured X under conditions Y" to "X because Y".
  If a comment asserts *why*, it should say what would disconfirm it.
- **A prompt is measured before it is dispatched** (ticket #837) — the pool's
  backend is two Macs, and a prompt long enough to fill their memory does not
  fail an allocation, it panics the host: a 108,753-token prompt through this
  gateway took slice down for three hours on 2026-09-25. `prompt_size.check()`
  runs at the top of the `Backend.EXO` branch, **before the lease and before any
  `/state` read**, and refuses over `EXO_MAX_PROMPT_TOKENS` with a 413 that
  names the limit. Two things it must keep saying: the limit protects the
  **host**, not the model's context window (GLM-4.7-Flash advertises 202,752
  tokens and exo will try to serve them), and the number is **measured** —
  prefill memory grows faster than linearly with prompt length, so 4x the tokens
  cost 7x the memory and an intuited limit lands in the wrong place. It is a
  property of the boxes and the placement: re-measure when either changes.
- **Speech to text is a child process, not an import** (ticket #858) —
  `Backend.WHISPER` is owned exactly as mlx-vlm is: `load_whisper_model` spawns
  `ollama/whisper_server.py` under `WHISPER_PYTHON`, leases, polls `/health`, and
  kills it on cooldown. Two measurements decided against importing mlx-whisper
  into the gateway: the wheels are 485MB the gateway never calls, and a
  transcription peaks at 2.5GB, which a killed process returns to the OS and a
  dropped Python reference returns to MLX's buffer cache. The child is polled for
  its **model**, not just a 200 — a leftover child answers `/health` while holding
  the other model.
- **The audio and chat name spaces do not meet** (ticket #858) — `match_whisper_model`
  matches exactly, like `match_exo_tier` and unlike every other matcher here, and
  `_resolve_model` skips `Backend.WHISPER` entries in both its exact and substring
  passes. A transcription model shares `mgr.models` so leases, cooldown, status and
  shutdown cover it for free, and that is precisely what put it in reach of a chat
  request for a backend with no chat route.
- **A request can outlive the cooldown** (ticket #858) — a transcription is the
  first request here that runs longer than `COOLDOWN_SECONDS`: a one-hour
  walkthrough takes about seven minutes at the measured 8.6x realtime, against a
  300s idle timer. `ManagedModel.in_flight` is held for the call and
  `check_cooldowns` skips a model that has one, so the loop cannot kill a child
  mid-sentence and turn a working request into a 502. `last_used` is stamped at the
  END of the call — stamping it at the start leaves a long transcription looking
  idle while it runs.
- **Whisper memory is measured, and the measurement names its own limits** (ticket
  #858) — `WHISPER_MODELS[*]["memory_mb"]` is peak MLX allocation on slice over
  44.7s of narration, alongside word error rate against a written script:
  turbo 2.8% / 2507MB, turbo-q4 4.2% / 2088MB, small 4.9% / 1455MB. The audio was
  macOS `say`, so it is clean, close-mic'd and unaccented: the ORDER should hold on
  real speech and the absolute rates are a floor. Re-measure on a real walkthrough
  before changing the default on the strength of these.
- **Three pillars** — SAMcloud provides routing, resources, and auth

## Current State

See [CHANGELOG.md](CHANGELOG.md) for project history and working notes.
