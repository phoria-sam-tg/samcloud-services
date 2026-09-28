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
| `ollama/capacity.py` | **The one capacity signal.** One contract on every box so an `offering:` means the same thing fleet-wide, with two arms under it — `metal` (`vm_stat`/`sysctl`/`ioreg`) and `cuda` (`nvidia-smi`), selected by `SC_CAPACITY_BACKEND` or detected. It is no longer byte-identical across boxes and must not become so; what is identical is the question `memory_available_mb` answers. Every fit decision, offering tier and resource-stats push reads from here — do not re-derive memory anywhere else |
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
python -m ollama.test_capacity_backends  # Both capacity arms meet one contract (no GPU needed)
python -m ollama.test_elastic_offering   # The offer: windowed, measured, same at every endpoint
python -m ollama.test_work_gate          # Work wins: the gate, its signals, what it must not break

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

**The two invocation forms are not interchangeable, and the exit code does not
tell you which mistake you made.** Everything under `python -m ollama.<name>`
uses relative imports and exits 1 with `ImportError` if run as
`python ollama/<name>.py`; the older files (`test_cooldown`, `test_lifecycle`,
`test_whisper_kill_guard`) are the opposite and want the direct path. Both
misinvocations look exactly like a regression from your own branch — this cost
samclaude-admin and claude-wafer-services a minute each on 2026-09-29, on the
same file. Run it the way the list above writes it before believing a failure.

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
- **Size off `available`, not `used`** — on Metal, `free + inactive + speculative`. macOS "used" counts pages the compressor is merely holding and is not a fit signal. Page size comes from the kernel; Apple silicon is 16KiB, and assuming 4KiB under-reported a box by 4x
- **One question, two arms** (ticket #861) — `memory_available_mb` means "what this device can hand to a model right now without swapping, counting every tenant". On Metal that is the line above; on CUDA it is `memory.total - memory.used`, because a discrete card has no compressor and an allocation either holds board memory or it does not. Both are whole-device on purpose: the number is taken from the hardware and never from the registry's `available_memory_mb`, which is `total - leased` and therefore blind to anyone who did not ask it for a lease. Measured 2026-09-28: the registry reported `ada-wsl/gpu-0` 49,140 MB free while the card's own `memory.used` was 48,016 MB of 49,140
- **Two questions, not one** (ticket #861) — `memory_available_mb` answers "do I fit"; `memory_device_inuse_mb` and `capacity.foreign_mb(device_inuse_mb, own_mb)` answer "is anyone working". They are not the same question and only coincide when the other tenant takes the whole device. Measured on ada: inside ONE continuous 46 GB Unreal render, free memory swung 188 → 2963 MiB, so a fit check alone offers up to 1939 MB *mid-render* and logs it as a success; `foreign` reads ~46.5 GB throughout. `device_inuse` is IOAccelerator `"In use system memory"` on Metal (~1 GiB idle, against ~22 GiB of `memory_used_mb`) and `nvidia-smi memory.used` on CUDA (the same number as `memory_used_mb` there, because the board is the accelerator). An earlier draft subtracted from `used`, which would have been garbage on the Metal arm with every test green
- **Attribution comes from our own bookkeeping** (ticket #861) — `own_mb` is never a sum over a process table, because neither arm has one. `nvidia-smi --query-compute-apps` returns zero rows against 46 GB of real usage under WDDM; IOAccelerator publishes one aggregate with no per-process split. The error direction is deliberate — under-counting what we hold invents a stranger and costs us an offer, rather than letting us load on top of somebody's render. `foreign_mb` returns **None**, never 0, when the device figure is unreadable: 0 means "nobody is working", which is the wrong way to fail
- **Never gate on utilisation** (ticket #861) — measured on both arms independently. wafer, 24 samples with Unity active: `compute_pct` min 10 / median 32 / max 49, 15 distinct values. ada, mid-render: `utilization.gpu` fell to **0–4% for 12 seconds** while `memory.used` held flat at 43.6 GB. A zero utilisation reading during an active render is real, not a glitch. Only the memory figures say whether work is present — and `load_avg_1m` is worse than useless on the CUDA arm, where it read 0.13 while a render held 46 GB (it moves independently of GPU state, not merely late)
- **The offer is measured, never the registry's** (ticket #861) — `ModelManager.offering()` is the one source for `/warm`, `/v1/models` and `/status`, so those three cannot disagree with each other or with the load gate. The registry's `available_memory_mb` is spec-minus-leases and cannot see a tenant who took no lease: measured on wafer 2026-09-29, on `main`, no leases and nothing resident, `/status` advertised 36,864 MB while the collector one import away measured 11,066. `/status` now reports the measured number under that name and keeps the registry's as `registry_available_memory_mb`
- **`/v1/models` lists only what can load right now** (ticket #861) — a deliberate departure from the convention, which is to list everything askable. On a box sharing a GPU with somebody's work, "askable" and "serveable" come apart: wafer listed a 17.5 GB model with 11 GB free. Blocked models are omitted so a caller cannot ask for one that was never offered; `/models` carries the whole catalogue with `blocked` and the reason, so nothing is hidden from an operator
- **The offer is computed over a window, not a sample** (ticket #861) — minimum `available` and maximum `device_inuse` across a trailing `OFFER_WINDOW_S`. One choice gives both behaviours: withdrawal on the first bad reading (it enters the window at once), restoration only once the trough has aged out. A single sample cannot do it — `available` spanned 9988–12938 over 24 samples on an *idle* wafer, and 188 → 2963 MiB inside one ada render, so two consecutive samples can both sit at the top of a swing but cannot both be its minimum. The window is a **duration**, not a sample count: a fixed count covers two minutes on an idle box and four seconds under load, i.e. it shortens exactly when the protection matters
- **`/warm` is auth-exempt, so its cost is bounded twice** (ticket #861) — the hardware read is rate-limited to `OFFER_MIN_SAMPLE_INTERVAL_S` (1s), because three subprocesses per request on an unauthenticated endpoint bound to `0.0.0.0` turns an anonymous request rate into a process-spawn rate: ~1,000 spawns/s at ~333 req/s, measured at 15.3ms per collect on wafer. And `_readings` is capped at `OFFER_MAX_SAMPLES` — **by merging, never by dropping**. Dropping the oldest is the obvious cap and the unsafe one: after a render starts the oldest entry *is* the trough, so evicting it restores the offer mid-render. Over the cap, the oldest pair collapses into one carrying their min-available and max-in-use at the later timestamp, so no extreme is lost and nothing ages out sooner
- **A failed capacity read still ages the window** (ticket #861) — `_record_reading` prunes before it appends, and is called even when there is nothing to append. The CUDA arm *raises* where Metal degrades to `None` fields, and an earlier version pruned only on the way past an append: a collector that started failing stopped pruning, so the last good reading stayed and the node advertised a number nobody could still measure. Now the offer collapses to nothing within one window. Found by `test_elastic_offering` step 13, not by inspection
- **Work wins** (ticket #861) — on a box that shares a GPU with somebody's work, their work takes precedence over ours. `offering()` stops advertising; `refuse_if_device_in_use()` is the gate that stops a caller who asks anyway, on every owned backend, and raises `capacity.DeviceInUse` → 503 `device_in_use`. A **resident** model keeps serving (its memory is already spent) and an **idle** one is evicted at once rather than waiting out `COOLDOWN_SECONDS`. `DeviceInUse` carries no `Retry-After` on purpose: a lease expires, a render ends when a person is finished, so the honest answer is "another node", not "wait"
- **The gate is off unless a box has measured its idle floor** (ticket #861) — `FOREIGN_IDLE_MB` has no default and setting it is what enables the whole mechanism, lease signal included. Only ada sets it (5,515 — the higher of two disagreeing idle readings, because a render reads ~46,000 so the gap changes no verdict, and only a floor *under* the true idle level fails invisibly). `work_in_progress()` returns **None** on an unset box: "not asked" is not "nobody is working"
- **A refused lease is a work signal; a broken one is not** (ticket #861) — `queued` and `conflict` mean the registry considered the request and said the resource is full. `error` means we could not ask. ada 404'd every lease request for months on a scope it never had, and counting that would read a permanent authentication fault as a permanent render. It exists as a second signal because `foreign_mb` has one blind spot: if WDDM pages our allocation out to satisfy a render, `own_mb` still claims the model and `foreign` comes out small
- **Lease state changes, residency does not** (ticket #861) — a renewal that cannot get the lease back sets `lease_lost` and leaves the model in `self.models`. Dropping it would make `own_device_mb()` fall by its size while the memory is still held, so `foreign_mb` rises by the same amount in the same instant and the gateway reads its own resident model as another tenant — stepping back from itself, permanently, while looking conservative. Asserted in `test_work_gate`
- **The constants are per-backend; the formula is not** (ticket #861) — `usable = min(fraction × available, available − floor)` is shared. Metal keeps 0.9 / 1024 MiB. The CUDA floor is a *time-derived* quantity with no Metal analogue — how much VRAM a render can claim between our reading and the next reconcile — and is **not measured yet**; it ships as the Metal value with `SC_MIN_HEADROOM_MB` to override, and lands with the ada profile. Don't round it to 4096 because 4 GB is a nice number
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
