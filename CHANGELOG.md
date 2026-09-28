# Changelog & Working Notes

Project history and current state. This is a living document.

## 2026-09-29 — Speech to text: the gateway takes audio (#858)

- **What was missing.** The gateway served text and vision. Audio had nowhere to
  go, so anything spoken on this fleet reached a model only by way of a phone's
  transcriber and a clipboard. `POST /v1/audio/transcriptions` now takes
  multipart `file` + `model` and returns `{"text": ...}`, shaped after OpenAI's
  endpoint so a client already written against that API needs no change but a
  base URL. `whisper-1` resolves to the default.
- **Which model, measured.** 44.7s of narration, six utterances, four voices,
  scripted so there is a ground truth to diff against (142 words, word error
  rate after normalising digits-vs-words):

  | model | time | WER | peak MLX memory |
  |---|---|---|---|
  | `whisper-large-v3-turbo` | 2.16s | **2.8%** | 2507 MB |
  | `whisper-large-v3-turbo-q4` | 1.97s | 4.2% | 2088 MB |
  | `whisper-small-mlx` | 1.48s | 4.9% | 1455 MB |

  turbo is the default. `whisper-small` is in the catalogue as the model that
  still fits when the box is full, not as a speed option — it is not faster on
  this hardware. Longer file, 5 min 08 s: turbo in 35.9s, **~8.6x realtime**.
- **The limit of that measurement.** The audio was macOS `say`: clean,
  close-mic'd, unaccented, no room and no compressor running. The ORDER should
  hold on real speech; the absolute rates are a floor. Re-measure on a real
  walkthrough before moving the default on the strength of these numbers.
- **A child process, not an import.** `Backend.WHISPER` is owned exactly as
  mlx-vlm is. Two measurements decided it: mlx-whisper's wheels are 485 MB the
  gateway never calls, and a transcription peaks at 2.5 GB — which a killed
  process returns to the OS, and a dropped Python reference returns to MLX's
  buffer cache. The child is polled for its **model**, not just a 200, because a
  child left over from a previous gateway answers `/health` while holding the
  other one.
- **mlx-whisper's `torch` dependency is wrong, and `requirements-whisper.txt`
  says so.** It declares `torch` and imports it only in `torch_whisper.py`, the
  weight *conversion* path; `transcribe` never reaches it. With torch the venv is
  991 MB, without it 485 MB, and the transcript of the same file is
  byte-identical.
- **A request that outlives the cooldown**, the first one here. A one-hour
  walkthrough is about seven minutes against a 300s idle timer, so
  `ManagedModel.in_flight` is held for the call and `check_cooldowns` skips a
  model that has one. `last_used` is stamped at the END — at the start, a long
  transcription looks idle while it runs.
- **The two name spaces do not meet.** `match_whisper_model` matches exactly,
  like `match_exo_tier`; `_resolve_model` skips `Backend.WHISPER` in both its
  exact and substring passes. Sharing `mgr.models` is what gives the transcriber
  leases, cooldown, status and shutdown for free, and is also exactly what put it
  in reach of a chat request for a backend with no chat route.
- **A comment that was wrong, and how it was caught.** `jsonable()` claimed
  `json.dumps(np.float64(...))` raises, and that word timestamps would be a 500
  without it. Removing the coercion and re-running showed every response still
  working: `np.float64` subclasses Python `float`. It stays for `np.float32` and
  `np.int32`, which json does refuse, and the comment now separates the
  measurement from what the guard is actually for.
- **A verification that measured the wrong thing.** The first check that a clean
  shutdown reaps the child found the child with `pgrep -f whisper_server.py` and
  reported PASS. This box has several user accounts and `pgrep` sees all of them:
  the pid it matched was the shell that had written the test script. Redone
  against the port the child listens on. The same property is why
  `_is_whisper_server` checks uid and `argv[0]` — `whisper_server.py` is a plain
  filename and turns up in any command line that greps, tails or edits the file.
- **Found while checking for regressions, filed separately (#862):**
  `requirements.txt` admits `tokenizers` 0.23, on which `test_prompt_size` fails
  three checks — on `main` as well — because the prompt gate falls back to the
  chars/token estimate instead of counting through the model's template.
  Production runs 0.22.2 and passes.

## 2026-09-25 — The gateway now measures a prompt before it dispatches it (#837)

- **What happened.** slice kernel-panicked at 11:50:43 and was down about three
  hours. A 108,753-token prompt arrived through this gateway, exo began
  prefilling it, and at 47,104 tokens macOS's GPU driver panicked the host
  instead of failing the allocation: `completeMemory() prepare count underflow`
  @IOGPUMemory.cpp:492, wired ~53 GB of 64, free 62 MB, `memoryPressure false`.
  The driver bug is Apple's. Reaching it from userspace with a large enough
  prompt was ours, and until this change any client that could POST to :8800
  could do it.
- **Measured, because the shape is not the obvious one.** Cold prefill through
  the gateway, `max_tokens=1`, `vm_stat` sampled every second, on the current
  2-node placement (slice carries layers 13-47, wafer 0-13). Slice idle with the
  model resident: 21.2 GB wired, 23.0 GB available.

  | prompt tokens | peak wired | over baseline | available at peak |
  |---|---|---|---|
  | 4,096 | 24.7 GB | +3.5 GB | 21.1 GB |
  | 8,192 | 29.3 GB | +8.1 GB | ~17 GB |
  | 16,384 | 46.1 GB | +24.9 GB | 8.5 GB |

  **Four times the tokens cost seven times the memory.** A limit picked by
  intuition — or by dividing the panic by a per-token rate — lands in the wrong
  place. Fitting `peak = 21.2 + 6.32e-4*N + 5.41e-8*N^2` (GB, N tokens)
  reproduces the 8,192 point to within 0.8 GB and puts the panic level (53 GB)
  at about **19,000 tokens**. One prompt, no other load.
- **The inference, kept apart from the measurement.** The linear term looks like
  the KV cache itself (this model caches the MLA latent, 576 values per token
  per layer); the quadratic term looks like the attention score matrix
  materialised per prefill chunk over the whole sequence so far and retained by
  MLX's buffer cache. What would disconfirm it: a run where peak memory tracks
  tokens linearly, or one where `mx.clear_cache()` between chunks flattens it.
- **`EXO_MAX_PROMPT_TOKENS = 12288`**, predicted peak 36.8 GB — 16 GB below the
  level that panicked the host, with room for the 8,192-token answer that
  decodes into the same cache. Over it: **413 `prompt_too_large`**, naming the
  limit, the measurement, and which method produced it. Not 400 (the request is
  well-formed), not 503 (retrying unchanged will never work), no `Retry-After`
  (nothing about this host changes in a minute).
- **The refusal costs the pool nothing.** It happens before the lease, before
  the wedge guard, before a byte reaches exo — a decline that had to take an
  exclusive lease to say no would be its own small outage. `test_prompt_size`
  spies on both and asserts neither is touched.
- **Every refusal says the limit is about the host.** GLM-4.7-Flash advertises
  202,752 tokens and exo will try to serve them; the constraint is two Macs. A
  caller told only "too long" reasonably goes looking for a longer-context
  model, and there is one — this same model.
- **Counting.** The resident model's own `tokenizer.json`, read off the exo
  models directory, so the gateway counts with the tokenizer that will prefill.
  Missing tokenizer falls back to chars/1.5 — the densest ratio measured across
  prose (4.50), Python (3.86), JSON (3.38), logs (2.76), CJK (2.00) and
  base64-like text (1.50) — which over-counts prose about 3x. That is the safe
  direction, and the refusal names the method so the over-count is legible. A
  body past `limit x longest-vocab-token` is refused without tokenizing at all.
- **The other box is tighter in a different way.** wafer carries 13 of 47 layers
  with 36 GB total, and was measured at 14.0 GB available with 17.0 of 18.4 GB
  of swap already in use. A limit sized only off slice is not automatically safe
  for wafer.
- **Measured passively while real traffic ran, and this is the part that
  matters:** hermes-exo's ordinary turns on a 32,241-token conversation, served
  95-99% from exo's KV prefix cache, took slice to **47.55 GB wired with 7.65 GB
  available** — 5.5 GB short of the panic, with no experiment running. A warm
  prompt is cheap and the same prompt cold is not, and nothing at request time
  can tell you which one is about to happen.

## 2026-09-24 — Vision was already served; it was never listed (#815)

- **The mlx-vlm backend has worked the whole time.** #815 asked for an
  image-capable model on the gateway, reporting `/v1/models` as text-only.
  Measured: `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` cold-loads and
  classifies a 900px photo in ~8.5s through `/v1/chat/completions` with
  `image_url` parts, by alias (`qwen2.5-vl`) and by resolved id. Nothing about
  the serving path needed changing.
- **`/v1/models` never enumerated `VLM_MODELS`.** It read the exo tiers, the
  resident models, the Ollama catalogue and the GGUFs — every source but the
  static dict that is the *only* record of the VLM backend. A VLM is not
  "pulled" and has no file on disk, and the process is on-demand, so the
  resident-models pass held it for a few minutes after a request and no longer.
  Ask while idle and the gateway looked text-only. Same failure this endpoint
  was added to fix, one backend over.
- **Advertise only what is installed.** `vlm_installed()` checks the HF cache,
  reading the same env huggingface_hub reads. `capacity.py` measures RAM and
  nothing measures disk: gemma-4-31b's 18700MB *fit in RAM* here against 15GB
  free disk and ~18.7GB of unfetched weights, and no load path would have
  refused it — mlx-vlm downloads on demand and the box would have run out of
  disk first. `/models` carries the full catalogue with an `installed` flag so
  the gap is visible rather than silently absent.
- **The vision envelope now matches the others.** mlx-vlm answers
  `usage.input_tokens`/`output_tokens` and omits `choices[].index`; the Ollama
  path builds `prompt_tokens`/`completion_tokens` and sets `index`. A caller
  reading `usage.prompt_tokens` — what the OpenAI SDKs read — got a number from
  every text model and `None` from the only vision one. Both key sets are
  carried now. Still missing fleet-wide, and deliberately not fixed here:
  `id`, `object`, `created`, which *no* backend on this gateway emits.
- **Weights cost disk.** Qwen2.5-VL-7B-4bit is 5.3GB and took the box from 21GB
  to 15GB free (99% full). Fetching gemma-4-31b here is not currently possible.
- **What the first real caller measured** (claude-assistant, building the family
  inventory classification pass on #815; their numbers, not ours, reported
  2026-09-24). Recorded here because all three are properties of the model as
  this gateway serves it, and the next caller writing a structured-list prompt
  meets them on photo one.
  - **`temperature: 0` looped.** A cluttered-shelf photo emitted
    `plastic storage containers` 40 times with identical `detail`, spending the
    whole 2400-token budget on one object. The same photo at `temperature: 0.2`
    returned a clean list. Their reading of *why* — that at t=0 the degenerate
    repeat is the argmax with nothing to break the tie — is inference, and is
    theirs; what is measured is the two runs. Not reproduced on our side, and
    our own test photo is too sparse to try it against. Worth knowing before
    recommending t=0 for determinism on a list-shaped ask.
  - **Latency here is output-bound, not load-bound.** 9.8s warm for a 16-item
    reply against the 3.0s we measured for a 4-item one, same warm process.
    Ours was right and unrepresentative: a cluttered photo runs ~20 objects at
    ~40 tokens each. Plan a pass at ~10s/photo, not 3 — ~100 photos is nearer
    15 minutes than 5, which is the number we gave on the ticket and should not
    have.
  - **Truncation is the normal case on a cluttered photo**, not an error case.
    Their parser salvages every complete object out of a cut-off `items` array
    rather than failing the photo. The token budget, not the model, is the
    binding constraint.
  - Scoping the system prompt to portable objects fixed the "tree and grass"
    drift we saw, with occasional leakage ("window" on an office shot), and the
    model marks genuine unknowns low-confidence when asked to.
- **The envelope gap stays open, at the caller's request.** No backend here emits
  `id`/`object`/`created`. The one consumer that could have been bitten parses the
  JSON directly and declined a fleet-wide change on its account — better driven by
  whatever meets it first than by a hypothetical. Left as is, deliberately.

## 2026-09-22 — The exo pool moved out of `sam`, and the Terminal.app limitation was wrong

- **The pool runs headless under the services accounts** (ticket #806). One node
  per box, each as that box's own OS service user — `claude-services` on slice,
  `wafer-services` on wafer — from a shared store at `/Users/Shared/exo`
  (`models`, `hf`, `uv-cache`, `src`), setgid `staff`.
- **The "must be launched from Terminal.app" limitation recorded on 2026-06-14
  does not hold for a services account.** Measured 2026-09-22 on slice as
  `claude-services`, spawned by sshd, no GUI session and no Terminal: UDP
  multicast receive on exo's discovery group `ff12::e0a1:de89` port 52413 joined
  on all 14 interfaces and took 23 announcement packets from the same peers
  `sam` sees, in the same 12 seconds. A node started that way discovered its
  peer and the ring formed, both ends reporting 2 nodes and 2 connections. The
  earlier claim was about the GUI launchd *agent* domain; it was never tested
  for a service user. Still untested: the LaunchDaemon domain specifically.
- **`EXO_BASE` is `http://localhost:52415`**, not a pinned LAN address. It had
  been `192.168.1.3` — one of wafer's *secondary* interface addresses, which
  answered, and which hard-coded the pool to one node. Each box now runs a node,
  so localhost is the honest address and it survives either box holding master.
- **Read `/state` from both nodes before concluding one is out of the ring.** A
  node that has just joined an established master serves an incomplete and stale
  replica — measured on slice as a worker: `topology.nodes` listed only the
  *other* node, with a `lastSeen` a day old, while the master listed both
  correctly. It lists itself the moment it holds master. Believe the end that
  lists itself.
- **What still needs a human is placement, not launch.** The gateway does not
  place the pool's model and a 503 from the tier now says so, rather than
  telling the reader to go and find a keyboard.

## 2026-09-20 — Capacity migration committed, and the two boxes reconciled

- **The capacity gate is in git.** It had been running in production on both
  inference boxes since 2026-08-30 as uncommitted working-tree state —
  `ollama/capacity.py` untracked, four files modified-not-committed, and a
  drift of `.bak-*` siblings standing in for history. Neither box agent could
  read the other's tree, so the copies diverged unnoticed (ticket #758).
- **`ollama/capacity.py` is the one collector**, byte-identical on every box, so
  an `offering:` advertised by one service means the same as another's. It sizes
  off `free + inactive + speculative`, not `active + wired + compressor`:
  "used" counts pages the compressor is merely sitting on (22.7GiB of 36GiB read
  used on wafer against ~3.9GiB of real process RSS), which refuses models that
  would have fitted and says nothing about whether a load will swap. Page size
  comes from the kernel — assuming 4KiB on a 16KiB machine under-reported slice
  by 4x for weeks. Tool paths are absolute, because the gateway's launchd PATH
  omits `/usr/sbin` and a bare `sysctl` turned every request into a 404.
- **Refusals are answers, not faults.** `InsufficientCapacity` carries
  `need_mb` / `usable_mb` / `available_mb` / `fits_now`, and
  `server._capacity_503()` is the single funnel every load path returns it
  through — including `POST /models/load`, which had no capacity branch at all
  and returned a 500 with a stack-trace string (#756). The class derives from
  `Exception`, not `RuntimeError`, so the pre-existing
  `except RuntimeError -> 409` cannot swallow it and make the missing branch
  look fixed. `ollama/test_capacity_refusal.py` drives all four shapes.
- **Loads fit rather than evict.** `load_ollama_model` checks against what the
  box can hand over instead of unloading whatever is resident to force-fit the
  ask; eviction is opt-in behind `AUTO_EVICT` (default off).
- **Sizes come from Ollama, not from the name.** `memory_estimate_mb()` reads
  the real weight size; the name fallback now anchors its size tag on a digit
  boundary. Substring matching read "qwen3.8:27b-mlx" as 7b and leased 5000MB
  for a model that resides at 17530MB — a 3.5x under-count that told other
  tenants there was room that did not exist. "13b" hit "3b" the same way.
- **The two lineages are merged.** `feat/wafer-gpu-metal-stats` (stats loop,
  Stage-1 offering) and `main` (GGUF short-name routing, #97 mitigations,
  capacity gate) both branched from 0d60324 and each grew the half the other
  lacked. Merged, then wired onto the shared collector: `_collect_stats()` is
  `capacity.collect()`, `stats_loop` pushes `registry_payload()`, and
  `_compute_offering()` uses `capacity.offering_tier()`.
- **The offering MB bands are gone** (`OFFERING_MINI_MB` / `_DEGRADED_MB` /
  `_FULL_MB`, superseding the 2026-07-02 entry below). They banded
  `total - used` into tiers and were calibrated when the largest model was
  ~6GB, so wafer advertised `offering:full` on ~14GiB available against a
  17.5GB model that could not load without swapping. The tier now derives from
  what actually fits out of the live catalogue — `full` = everything we hold
  fits, `mini` = exactly one does, `none` = nothing does — which stays true as
  models come and go and needs no per-box tuning. `OFFERING_ENABLED`,
  `_POLL_SECONDS` and `_HYSTERESIS` remain the per-box knobs.
- **One stats push per box, from the gateway.** `ollama/gpu_stats.py` retired;
  slice's separate `com.samcloud.cs-gpu-stats` launchd job is unloaded in
  favour of `ModelManager.stats_loop()`, which is lifecycle-managed and shares
  the collector. The module had been hollowed out by the migration anyway —
  `collect_stats()` was a passthrough and the helpers under it were dead.
- **Still open:** `llama-server` and `mlx-vlm` have no capacity gate — only
  Ollama does, so `server.py` catches `InsufficientCapacity` around
  `load_llama_model`, which cannot raise it. Deliberately not fixed here:
  gating them starts refusing loads that succeed today, which is a behaviour
  change callers should be told about first (ticket #754).

## 2026-07-02 — Stage-1 capacity offering (flex tier) on wafer

- **`offering:<tier>` self-report (doc #8 Stage 1, ticket #135).** The gateway now
  advertises one of `offering:full | degraded | mini | none` as a `capabilities`
  entry, recomputed every `OFFERING_POLL_SECONDS` (30s) and PATCHed on change so
  the fleet map reflects live capacity. New `offering_loop` / `_compute_offering`
  / `_apply_offering` in `manager.py`; config knobs in `config.py`
  (`OFFERING_ENABLED`, `_POLL_SECONDS`, `_HYSTERESIS`, `_MINI_MB`, `_DEGRADED_MB`,
  `_FULL_MB`). First reading publishes immediately; later changes require the new
  tier to hold `OFFERING_HYSTERESIS` (2) consecutive polls to avoid flapping.
- **Signal = LOCAL unified-memory availability** (`_collect_stats()`'s
  vm_stat/sysctl `total - used`), *not* the registry's lease-based
  `available_memory_mb`. Reason: the gateway runs under a service token, which is
  scope-filtered out of resource reads (`GET /resources/{id}` → 403, dashboard/
  leases return empty). On a unified-memory Mac real memory pressure is the truer
  "can I serve a model" signal anyway, and it also captures training that spikes
  memory without holding a formal lease. Verified end-to-end: full → degraded
  under ~9 GB pressure, restore to full on release, with hysteresis.
- **Good-citizen flex** for the wafer box shared with brush splat training: the
  tier drops as memory fills (training running) and restores when it frees.
## 2026-09-20 — Backend.EXO: the pool behind model-service (ticket #770)

A fourth backend, and the first the gateway does not own. `claude-services-slice/exo-pool`
is ONE exo instance spanning slice and wafer that serves ONE request at a time, placed
out of band. Callers ask for a tier — `{"model": "think"}` — and get whatever is
resident, resolved from exo's `/state` at request time so an operator can swap the model
without a deploy here.

**The lease.** `_request_lease` returned an id for any response that did not raise, so a
**queued lease read as granted**. On shared `gpu-0` that is close to harmless; on an
exclusive resource it puts two consumers inside one inference instance. Replaced with a
`LeaseOutcome` verdict that callers must test. The verdict reads the response **body**,
not the status code: the registry grants with a plain `200`, not the `201` its own API
index documents, so status-only logic fails in one direction or the other. `claim_leases`
had the same defect and is fixed too.

**`memory_mb` must be null on the pool.** The registry queues when
`memory_mb > available`, and `available` is `total - leased` where `total` comes from
`vram_mb`/`gpu_memory_mb`/`unified_memory_mb`/`ram_mb` in the resource specs. `exo-pool`
carries none of those by design, so `total` is 0 and any positive byte count is queued
forever regardless of how idle the pool is. "A lease means the pool is TAKEN, not that
bytes are reserved" turns out to be operationally load-bearing, not just honest framing.

**Declining.** A caller who asks while the pool is busy gets a 503 in the same shape as
the shipped `insufficient_capacity` one: `resource_busy`, with `retry_after_s` derived
from the holder's `expires_at` and a real `Retry-After` header. `queue_position` is
present but honestly `null` — the exclusive-conflict path maintains no queue, and the
only code that assigns positions is the memory-oversubscription branch the pool never
enters. Measured: 0.38s, HTTP 503, no leftover lease.

**Not stranding the pool.** An exclusive lease left behind does not degrade the pool, it
closes it. Released in a `finally`, swept by `shutdown()` on SIGTERM/SIGINT, and bounded
by a TTL that `config.py` clamps to exceed `EXO_GENERATE_TIMEOUT` — the registry expires
leases on time and cannot extend one, and renewing by release-then-reacquire would open a
window for a third party to take the resource mid-generation.

**Two things measured about exo itself.** Its `stream: false` was recorded here as
returning `200` headers and then no body at all — **that conclusion is retracted.** Both
probes hit a pool with no free slot (one mid-generation, one wedged), and a busy exo
accepts the request and emits keep-alives. Re-measured against a healthy pool it returns
a complete body with usage in ~17s. Non-streaming callers are still served by
`chat_collect()`, for reasons that do not depend on the retracted claim: it is
cancellable and pins no worker thread. That also made the path cancellable: the first version used
`asyncio.to_thread` around a sync call, and because a thread cannot be cancelled, a
client that disconnected held the pool for the rest of the generation and blocked the
gateway's own graceful shutdown.

Known limitation, documented rather than solved: the pool cannot restart itself after a
reboot. macOS grants local-network access per responsible process, so a headless launch
is denied and exo must be started from a Terminal on the host. The backend can therefore
be advertised but not started unattended.

## 2026-06-14 — On-demand mlx-vlm + lease fix

- **mlx-vlm is now gateway-owned and on-demand.** Added `load_vlm_model` (spawns
  `python -m mlx_vlm.server`, leases, tracks the process), `match_vlm_model`, and
  VLM branches in `unload`/`ensure_running`. A cold VLM request now spins the
  server up (~7s) instead of 404ing. Routing wired in `_resolve_model` and
  `/models/load` (+ `auto` detection); `/models/unload` resolves partial names.
- **No more adoption / boot-order for VLM.** `discover()` no longer adopts a
  running mlx-vlm; instead `_kill_stray_vlm()` reaps any stray `mlx_vlm.server` on
  startup so the gateway always owns the process. mlx-vlm dropped from reboot order.
- **Lease `purpose` field removed.** Registry now rejects it (`422 extra_forbidden`);
  `request_lease` and callers updated. Leasing was previously failing for all paths.
- VLM config (`VLM_PYTHON`, `VLM_HOST`, `VLM_PORT`, `VLM_STARTUP_TIMEOUT`) in `config.py`.
- **Known follow-up:** on-demand ollama loads still lease an *estimate* (e.g. 2500 MB
  for a ~20 GB model) — the "wider leasing" reconcile-to-actual work (overlaps PR #2)
  should extend to the VLM path too.

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
