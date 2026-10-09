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
| `ollama/ollama_client.py` | Ollama API. Note: `chat()` passes `**kwargs` so `think=False` works. Owns the served context window (#903): `num_ctx_for()` clamps the configured value to what the weights declare, `_with_num_ctx()` puts it on every payload, `running_context_length()` reads what the live instance actually has |
| `ollama/llama_client.py` | llama-server process management. `discover_running()` parses `ps aux` |
| `ollama/whisper_client.py` | The gateway's half of the whisper child — `/health` sync (the spawn loop polls it), `/transcribe` async (a transcription runs for minutes) |
| `ollama/whisper_server.py` | The whisper child. Executed as a SCRIPT by `WHISPER_PYTHON`, so it imports nothing from this package — a relative import here is an ImportError at startup. Decodes with the ffmpeg the gateway names, never PATH |
| `ollama/exo_client.py` | The exo pool. Already OpenAI-compatible, so chat is a proxy. `resident_model()` reads `/state`; non-streaming goes through `chat_collect()` because exo's own `stream:false` returns no body |

## Run / Test

The gateway runs from `~/var/samcloud-services-deploy`, a deploy clone held on a
detached HEAD at a named commit — not from your checkout. `deploy/README.md` has
the why, `deploy/rollout.sh` fills it, and nothing restarts on a merge.


```bash
SC_TOKEN=<token> python -m uvicorn ollama.server:app --host 0.0.0.0 --port 8800

# From the repo ROOT — relative imports, so `cd ollama && python <file>.py`
# raises ImportError: attempted relative import with no known parent package.
python -m ollama.test_capacity_refusal   # Refusal path: 503 + the numbers, not 500
python -m ollama.test_capacity_backends  # Both capacity arms meet one contract (no GPU needed)
python -m ollama.test_elastic_offering   # The offer: windowed, measured, same at every endpoint
python -m ollama.test_work_gate          # Work wins: the gate, its signals, what it must not break
python -m ollama.test_renew_ordering     # A lease swap never leaves the registry accounting 0
python -m ollama.test_lease_reconcile    # The lease says what the model actually occupies
python -m ollama.test_num_ctx            # The served window: pinned, clamped, published (no network)
python -m ollama.test_stream_deadlines   # Silence is bounded, elapsed time is not (no network)
python -m ollama.test_route_bindings     # Every path resolves to its own handler (no network)
python -m ollama.test_proxy_bounds       # Every httpx proxy call is bounded, in shape (no network)

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

## Deploying

**The service runs from a deploy clone, not from a working checkout.**
`deploy/rollout.sh` puts a named commit into `~/var/samcloud-services-deploy`
on a detached HEAD, with `.venv` synced from that commit's
`requirements.txt`; restarting is `restart-when-idle`'s separate, idle-gated
step. See `deploy/README.md`.

Three times in two days a live deploy checkout held somebody's in-progress
work (#861) and **the collision is undetectable from either side** — the
person editing has no signal a service runs from the tree, the service has no
signal its tree is being edited, and `git status` is content. One instance ran
a week through several restarts and was found by `lsof` on the gateway's pid.
A deploy that resets to a commit cannot silently serve a working tree, because
there is no working tree to serve.

One consequence worth knowing: **your checkout's venv is not production's.**
A suite that passes in your checkout can fail in the deploy clone — that is
#862, where `jinja2` was undeclared and one box's venv happened to have it.
Verify against `~/var/samcloud-services-deploy/.venv/bin/python` before
restarting.

## Conventions

- All SAMcloud API calls go through `SamcloudClient` (never raw httpx)
- All model operations go through `ModelManager` (server.py is thin routing)
- `GET /service-docs` is the service discovery convention — auth-exempt, returns structured JSON + full markdown guide
- Auth via SAMcloud token verification (`GET /auth/verify?scope=$SC_REQUIRED_SCOPE`)

## Architecture Decisions

- **Bounded TTL leases** — renewed every 30 min, not indefinite (ticket #42)
- **Health-bound revocation** — stale services should lose leases
- **Actual VRAM** — use `ollama ps` sizes, not file-size estimates, **and tell the registry** (ticket #861). The lease is requested from `max(1024, disk_mb)` before the weights are resident and before their context exists; measured on wafer 2026-09-29, `qwen3:1.7b` reserved 1296 MB and occupied 3354. Under-reserving costs us nothing — nothing enforces a lease — it costs whoever reads `available_memory_mb` next and concludes there is room. `_reconcile_lease` re-leases at the real size past `_RECONCILE_RATIO`, and **does not undo the load** if the registry refuses: the local gate already decided it fits from the hardware, and the registry's figure is the weaker instrument. A refused reconcile marks the model `lease_lost` and feeds the contention signal instead
- **Agents lease, not register** — resource registration is device-daemon territory
- **Ollama think workaround** — route through native `/api/chat` with `think:false`, translate to OpenAI format ourselves (ticket #69)
- **On-demand mlx-vlm, gateway-owned** — the gateway starts/stops `mlx_vlm.server` itself (`load_vlm_model` + `match_vlm_model`), rather than adopting a pre-started process. On startup it reaps any stray `mlx_vlm.server`. No boot-order dependency; one VLM per `VLM_PORT`
- **Fit, don't evict** — a load checks against what the box can hand over now; it does not unload whatever is resident to force-fit the ask. Eviction is opt-in behind `AUTO_EVICT` (default off). The contract is "here is what fits, pick one"
- **A refusal is an answer, not a fault** — `capacity.InsufficientCapacity` carries `need_mb`/`usable_mb`/`available_mb`/`fits_now`, and every load path returns it through the single `server._capacity_503()` funnel so callers cannot tell the backends apart by how they decline. It derives from `Exception`, **not** `RuntimeError`, so the `except RuntimeError -> 409` on `/models/load` cannot swallow it (ticket #756)
- **Size off `available`, not `used`** — on Metal, `free + inactive + speculative`. macOS "used" counts pages the compressor is merely holding and is not a fit signal. Page size comes from the kernel; Apple silicon is 16KiB, and assuming 4KiB under-reported a box by 4x
- **One question, two arms** (ticket #861) — `memory_available_mb` means "what this device can hand to a model right now without swapping, counting every tenant". On Metal that is the line above; on CUDA it is `memory.total - memory.used`, because a discrete card has no compressor and an allocation either holds board memory or it does not. Both are whole-device on purpose: the number is taken from the hardware and never from the registry's `available_memory_mb`, which is `total - leased` and therefore blind to anyone who did not ask it for a lease. Measured 2026-09-28: the registry reported `ada-wsl/gpu-0` 49,140 MB free while the card's own `memory.used` was 48,016 MB of 49,140
- **Every lease swap acquires before it releases** (ticket #861) — `_reconcile_lease` and `_renew_leases` both. Release-first leaves a refused re-request with the model resident and the registry accounting **nothing** for memory that is still held; measured as 1296 → 0 on wafer for a model occupying 3354 MB. A refusal keeps the lease it had. The cost is a moment of double-counting, which can refuse a swap that release-first would have been granted — that lands on keeping a valid lease, so it costs an improvement and never a correct row. `test_renew_ordering` models the registry's own figure, samples it after every grant and release, and asserts the minimum is never 0; it also scans `manager.py` for any release followed by a re-request, so a third site cannot appear quietly
- **In-place renewal cannot replace the swap** (ticket #861) — `POST /leases/{id}/renew` is capped at `granted_at + max_total_s`, and a lease granted without an explicit ceiling gets `max(1800, ttl)` = 3600s for these. So it reaches `at_ceiling` an hour after the load, and a model resident longer than that must be re-taken whatever we do. `_renew_pool_leases` renews in place and must stay that way; the difference is the ceiling, not taste
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
- **The served context window is pinned and published, never derived** (ticket #903) — nothing used to set `num_ctx`, and Ollama answers that by DERIVING one from free VRAM at load time, logged as `vram-based default context`. slice's log has 262,144 nineteen times and **4,096 twice** (2026-08-25/26, llama-server GPU discovery timed out and it fell back to CPU) — a 64x spread in the window a caller got, reported nowhere. `/v1/models` carried `memory_mb` only and `/v1/models/{id}` 404'd, so #884 hand-picked a 65,536 this box has never served. `OLLAMA_NUM_CTX` / `OLLAMA_NUM_CTX_MODELS` pin it; `offering()` publishes it; it is **read back from `ollama ps` after the load**, never assumed from the config, because those are different facts. A loadable model's window is reported only where we pin one — predicting a VRAM-derived default would be the same invented number one layer down. **Absent means unknown, not unlimited**, and consumers are told so on `/service-docs`
- **One num_ctx, every call path** (ticket #903) — Ollama keys a loaded instance by its options, so a request whose `num_ctx` differs from the resident instance's **reloads the model**. One call site that forgets is not a cosmetic inconsistency, it is a 30s reload inside somebody's job, and `/api/ps` would report a window the next request does not get. So `_with_num_ctx` is applied inside `OllamaClient` rather than at the call sites, and `test_num_ctx` reads the source to assert each path routes through it. `unload_model` is excluded on purpose. **Adoption is the deliberate exception**: it re-applies the *adopted instance's own* window, so a gateway restart cannot bounce a resident model to change a number — the configured window lands on the next real load, which `keep_alive` guarantees arrives
- **A timeout is only meaningful relative to every other timeout on the path, and the one that fires first should be the one that reports best** (ticket #904, `claude-containers`' phrasing) — the owned Ollama stream had aiohttp's `total=` at 300s, which bounds ELAPSED TIME, so a generation producing tokens steadily was killed for being long. 18 cut-offs in 81 requests on 2026-10-08 (22%), longest completed request 285.0s against a 300s wall. Three bounds now, the structure `exo_client` reached on #830: a prompt-scaled first-token budget (prefill is silent and its length is the only predictor), a tight inter-token budget, and a whole-request ceiling **tracked in-process** — because aiohttp's `total` and our `wait_for` both raise `asyncio.TimeoutError` and one `except` cannot name the cause. `sock_read` alone would be worse than the bug: at 60s it kills a 70k-token prefill at 60s instead of 300s, and calls it a stall. **Count the gateway's own `Stream error`, not Ollama's status column** — the abort races response completion, so 6 of the 18 were logged `200 | 5m0s`
- **A deadline arms on a measurement, not on a default** (ticket #904) — `OLLAMA_FIRST_TOKEN_RATE_TPS` is unset and setting it is what enables the first-token deadline, the same convention as `FOREIGN_IDLE_MB` (#861). The MLX runner emits no per-request timings and the `slot print_timing` lines in `ollama.log` stop at 2026-09-30 and come from a llama.cpp runner whose largest prompt is 2,371 tokens — so there is no prefill rate for `qwen3.8:27b-mlx` and any default would be the invented number #903 was about. Unarmed falls back to the whole-request budget; the inter-token bound is armed regardless, so the inference slot (#97) is covered. The sync `httpx` path is UNCHANGED and was always right: `read` is a per-chunk idle bound, and the bug was transcribing its 300 into aiohttp's wall-clock `total=`
- **A route is bound to a handler, and nothing was asserting which** (ticket #904) — a helper inserted between `@app.post("/v1/chat/completions")` and `async def chat_completions` bound the HELPER. The path still existed and the route COUNT was unchanged, so 51 checks and nine mutations passed while every POST to Hermes' surface 422'd: `model: str` became a required query parameter and `chunk: dict` became the body. 100% failure, strictly worse than the 22% cap it shipped beside, and it reached `main` and the staged deploy clone — the live process only kept serving because it had loaded the old module three days earlier. `test_route_bindings` pins the whole path→handler table, refuses any `_private` name being routed, asserts the table still covers what the app serves, AND sends one real POST through the ASGI app, because the previous tests exercised the deadline logic while never exercising the route that reaches it. "Assert a count, not a presence" one level up, with the count right and the binding wrong
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
- **A short ring is not an outage, and the tier says so through the capacity model**
  (ticket #870) — one of the pool's two nodes is a **laptop**, so the ring spends
  part of every day short. Sam, 2026-09-30: the tier exists to be a multi-device
  model, so while the ring is short `think` is **unavailable the way any model is
  unavailable while the capacity it needs is taken**, and it returns when the
  capacity does. **No one-node placement and no substitution** — a caller that
  asked for `think` and silently received another model has been told something
  untrue, so the decline carries `alternatives` as *information* and swaps in
  nothing. `pool_ring_short` is a distinct error code from `pool_unavailable`,
  with `transient: true` and `operator_action_required: false`, because the one
  cause that is normal must not arrive looking like the one that needs a person.
- **The tier's residency comes from the pool, never from `self.models`**
  (ticket #870) — `resolve_exo_tier` registers `think` with `managed=False` and
  nothing removes it, so the offer read it as resident for the life of the
  process: measured on slice 2026-09-30 with the ring short and nothing placed,
  `/models` advertised `think` resident with `idle_seconds: 80238`. For every
  other backend residency *is* ours to know, because we started it. `/v1/models`
  now omits the tier when it cannot serve, the same convention as a model blocked
  for capacity, and **still performs no pool read** — `pool_watch_loop` polls on a
  timer and `offering()` reads its snapshot, so a wedged pool cannot make
  discovery hang. A snapshot too stale to trust leaves the tier **listed**: not
  having looked is a fact about us, and a dead loop must not delete a configured
  route.
- **`nodeIdentities` is not a ring size** (ticket #870) — it is the one per-node
  map `apply_node_timed_out` does not filter, so it remembers a departed node and
  looks like the way to tell a short ring from a small one. **It is cleared by
  `APIState.reset`, which fires on `is_new_master` — a node that has to PROMOTE
  itself clears its identities; a node that was already master does not.** A
  restart is one way to reach that branch and not the only one: measured on both
  nodes 2026-10-09, slice logged `Node elected Master - maintaining self` six
  times with `Resetting API State` **zero** times and kept both identities, while
  wafer logged `promoting self` followed by the reset and kept only its own. Read
  live on slice at 00:24Z on 2026-09-30, 17 minutes after its own node restarted
  and with wafer genuinely away: `topology.nodes` and `nodeIdentities` both held
  slice alone, so the subtraction said the ring was whole while a node was missing
  — and the decline then told a caller "the ring is whole and no model is placed
  on it".

  **So which node can attribute a departure depends on which node was master, and
  that is a hazard rather than a nicety.** If the departing node held master, the
  survivor must promote, resets, and keeps only itself — so "who is missing" comes
  back empty on the box that stayed, with no restart involved. Do not read "nobody
  restarted" as "identities are reliable": that inference is what the `#870` guard's
  identities-plus-configuration union exists to survive, and it is reachable by
  election history alone. `EXO_RING_MIN_NODES` has **no default**,
  and setting it is what arms `pool_ring_short`; unset, the reason falls back and
  the message says in terms that whether the ring is whole is *not established*.
  Same shape as `FOREIGN_IDLE_MB`: a guessed value produces a confident wrong
  answer rather than a missing one.
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
- **An httpx `read` is a SILENCE bound, and a prefill is silent** (ticket #911) — so `httpx.AsyncClient(timeout=300)` sets connect/read/write/pool all to 300s and a non-streaming request, which produces no bytes until the generation completes, has its whole generation inside one read. That aborted legitimate replies at 300s: this box's own cold prefills measure 718.7s at 62,777 real tokens, 918.0s at 75,776 and 1,691.0s at 112,682, each emitting nothing. The seven proxy call sites carried the two *opposite* defects — three streams at `timeout=None` (no bound at all, so the caller's bound is the only one and the gateway emits no frame) and four non-streams at `timeout=300` — and one of the four hid inside a `stream: true` branch where the VLM tools path collapses to a non-streaming upstream call. All seven now go through `server._proxy_timeout()`: `read` is the whole-request ceiling, because anything tighter needs a prompt-scaled rate and these paths have no exact token count (`prompt_size.count` falls to an estimate without a `tokenizer.json`, #906); `connect` carries the tight bound as the only one not waiting on inference, being a 127.0.0.1 child this gateway started. A bare number for a timeout is the bug — the gate asserts a `Timeout` object
- **Assert a bound by mutating it, not by grepping for it** (ticket #911) — `server.py` now contains the strings `timeout=None` and `timeout=300` in the docstring explaining why they are gone, so a grep-based check would fail against the fix and pass against the defect. `test_proxy_bounds` replaces `_proxy_timeout` with a sentinel and asserts all seven call sites follow it, which no comment can satisfy, and asserts the count is **seven** so a new unbounded site cannot be added quietly. This is the same trap that let three assertions in `test_stream_deadlines` be satisfied by their own explanatory comments

## Current State

See [CHANGELOG.md](CHANGELOG.md) for project history and working notes.
