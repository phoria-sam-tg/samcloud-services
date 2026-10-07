# Changelog & Working Notes

Project history and current state. This is a living document.

## 2026-10-08 — A quotient under a token floor is not a rate (#904)

- **Found by `claude-wafer-services` while sweeping with `num_predict:1`**, and
  it is the `total_duration` trap one column over. `_log_ollama_timings`
  computed `eval_count / eval_duration` unconditionally, so a one-token
  generation reported **time-to-first-token wearing a rate's units**. It would
  have read correctly on real traffic and wildly wrong on every short
  generation — the ones that look harmless.
- **My own first sample proves it twice.** `prefill 15 tok in 1.56s = 10 tok/s`
  and `decode 1 tok in 0.23s = 4.3 tok/s`, against ~75 tok/s prefill on a
  2,676-token prompt. Two orders of magnitude of nonsense, in the field whose
  entire purpose is to be read as a rate and used to size a deadline. So the
  floor applies to **both** columns, not just decode.
- `_MIN_TOKENS_FOR_RATE = 32`. Under it, the duration is logged and the
  quotient refused: `prefill 15 tok in 1.56s (under 32 tok: no rate)`. At
  `eval_count == 1` specifically, `first_token 0.23s (1 token, not a rate)`,
  because that number is a latency and is worth having under its own name. Not
  a filter — the row still appears, it just cannot assert a rate it does not
  support.
- **The window is not the usable prompt.** Applying wafer's 74.7 tok/s: a
  70,000-token prefill is ~937s, which is 52% of the 1800s ceiling, and a full
  262,144-token prefill is ~3509s and **cannot complete**. So the published
  window is a memory bound and there is a second, tighter bound made of time —
  the time-usable prompt is ~125,000 tokens. Recorded in the README beside the
  published window, because a consumer sizing against `context_length` alone
  will be cut by the ceiling.
- **And retrospectively: under the 300s cap, any prompt over ~22,000 tokens
  could not complete at all.** Prefill alone exceeded the wall. The 65,536
  `hermes-assistant` believed it had was never usable; neither was 32,000.

## 2026-10-08 — A route is bound to a handler, and nothing asserted which (#904)

- **`_log_ollama_timings` was inserted between `@app.post("/v1/chat/completions")`
  and `async def chat_completions`, so the decorator bound the HELPER.** The path
  still existed; `chat_completions` became unreachable. `model: str` turned into
  a required **query** parameter and `chunk: dict` into the body, so a normal
  request with a JSON body and no query string was rejected before anything ran.
  **Every POST to Hermes' surface 422'd — 100% failure, strictly worse than the
  22% cap it shipped beside, and it never reached the model.**
- **It got to `main` and into the staged deploy clone.** The only reason the
  service kept serving is that the running process had loaded the old module at
  import three days earlier. Caught by `samclaude-admin` reading the diff, not
  by anything in this repo. The deploy clone was rolled back to `6691982`
  immediately — a fix to `main` alone would have left the landmine staged.
- **Why 51 checks and nine mutations missed it.** Both branches register the
  same number of routes: **the count did not change, the binding did.** Nothing
  asserted which handler a path resolves to, and no test went through the ASGI
  app at all — the deadline logic was tested directly while the route that
  reaches it was not. That is this repo's own "assert a count, not a presence"
  rule one level up, with the count right and the binding wrong. Third time in
  two days that a number measured something other than what it looked like.
- **`test_route_bindings`** pins the whole path -> handler table as an
  independent statement of intent (derived from the app, it would have agreed
  with the bug). Four more properties beyond the table, because the failure mode
  is "a helper lands under a decorator" and any insertion can cause it:
  **no `_private` name may be routed** — which catches it without the table
  being up to date; the table must still cover what the app serves, so a new
  endpoint cannot quietly fall outside it; no path may resolve to two handlers;
  and **one real POST through the ASGI app**, asserting specifically that
  `model` is not demanded as a query parameter. Verified by reintroducing the
  defect: 5 checks fail, including the live 422.
- **And the stale figure in `config.py` is corrected** — it still carried "121 x
  200 and 17 x 500, ALL SEVENTEEN at 5m0s", wrong three ways: the denominator
  spanned two days while every cut-off was on one, five of those 17 were the
  CALLER disconnecting at ~180s, and six of ours were logged `200`. That file is
  what a future reader consults first, so the repo was contradicting itself.

## 2026-10-08 — The first-token budget is queue + prefill, not one rate (#904)

- **The trap, raised by `samclaude-admin` and `claude-containers` on #903 after
  #39 was already approved.** Our first-token clock starts when the POST
  returns headers, which Ollama sends on *accepting* a request — before the
  model is scheduled. So the silence it bounds is **queue + prefill**, while
  `prompt_eval_duration` measures **prefill alone**. Arming the first with the
  second makes contention trip a prefill guard: a request killed for being slow
  when it was only waiting, and reported as a stall.
- **It is the `sock_read` trap a second time** — a more legible failure that is
  wrong more often — and the gap is not absorbable. Measured on this box, 73.8s
  of wall against 1.8s of compute: **~40x**.
- **Three terms, because they have three shapes.** Folding them is what makes
  the number wrong, and the test shows the two models diverge rather than
  arguing it:

  | term | shape | measured from |
  |---|---|---|
  | `..._QUEUE_ALLOWANCE_S` | **absolute** | the `queued ~Ns` field, under real contention |
  | `tokens / ..._RATE_TPS` | **scales** | `prompt_eval_duration` — immune to contention |
  | `..._MARGIN_S` | absolute | post-prefill model behaviour |

  A 15-token prompt waited 72s, so waiting does not scale with the prompt and
  cannot be expressed as a rate. Halving the rate to "absorb" queueing covers
  neither end: it still cannot give a 100-token prompt 120s, and it
  over-allows a 70,000-token one — so the deadline stops firing when it should.
- **Rejected: don't count queue time at all**, by tracking our own in-flight
  requests. It only sees OUR queue. The exo `think` tier and every other
  consumer of this box are invisible to it — the same blind spot `foreign_mb`
  has, for the same reason (#861) — so it would under-allow exactly when
  contention is worst.
- **Arming the rate without a queue term now warns**, naming the failure
  (`reported as a stalled prefill`) rather than the setting. Silent when both
  are set and silent when unarmed, so the shipped default logs nothing.
- **And the 120s datapoint is an upper bound, not a prefill measurement.**
  `waiting for stream response (120s, first_chunk)` is wall-clock from the
  caller's side, so it contains the same queueing; the seat cannot decompose it
  (`api_call_count: 0` for this provider). Read as "the seat waits up to 120s
  before first output at today's context", which is what a timeout has to
  cover — not as "prefill takes 120s", which would size the deadline too
  generously to fire.

## 2026-10-08 — Streaming deadlines: silence is bounded, elapsed time is not (#904)

- **The rule**, `claude-containers`' phrasing and better than anything else on
  the ticket: *a timeout is only meaningful relative to every other timeout on
  the path, and the one that fires first should be the one that reports best.*
  This gateway fired first and reported worst.
- **The defect.** The owned Ollama streaming path had one bound — aiohttp's
  `total=` at 300s — and `total` bounds **elapsed time**. A generation streaming
  tokens steadily was killed at 300s for being long, which is the one thing
  that is not a fault.
- **The numbers, counted from the right log.** 2026-10-08: **18 `Stream error`
  events in 81 `/api/chat` requests, 22%**. Ollama's status column undercounts
  because our abort races the response completion — 11 were recorded
  `500 | 5m0s` and **6 were recorded `200 | 5m0s`**, the same event logged as a
  success. The longest request that actually COMPLETED ran **285.0s**, fifteen
  seconds under the wall. An earlier draft of this said 17 of 138 across two
  days; `samclaude-admin` corrected both the count and the denominator.
- **What it did NOT cause**, because the first version of this entry said it
  did. The 18 cut-offs did not produce 18 visible failures. The seat's adapter
  journal shows three consecutive turns of 56, 50 and 31 minutes, all accepted,
  all posting answers, and no matches for error/timeout/retry/stale in its own
  logs. **A turn makes many requests**; Hermes retried across the boundary. So
  a 5-minute per-request ceiling coexists with 56-minute turns and was absorbed
  below the turn level. The cost was latency and a wasted inference slot, not a
  broken answer.
- **And ours was not even the first cap for most of the window.** Five of the
  500s sit at ~3m0s and match `Client disconnected during stream` to the
  second — eight such disconnects between 00:02 and 00:24, ~3 minutes apart, a
  caller giving up at ~180s and retrying. They stop at 00:24; our first 5m0s
  cut-off is at 01:03. **The caller's bound moved at ~00:30, and that is the
  only reason ours became the visible one.** `min(caller, ours)` is what
  actually bounds a generation, so this change hands the cap back rather than
  removing it.
- **Three bounds, not two.** The structure is `exo_client`'s, which was bitten
  by exactly this on #830, and the constants in `config.py` are the design doc:

  | | bounds | default |
  |---|---|---|
  | `OLLAMA_FIRST_TOKEN_*` | silence **before** token 1 | **unarmed** |
  | `OLLAMA_STALL_TIMEOUT` | silence **between** tokens | `60` |
  | `OLLAMA_GENERATE_TIMEOUT` | the whole request | `1800` |

  **`sock_read` alone would have been worse than the bug**: prefill is silent,
  so a 60s silence bound kills a 70,000-token request at 60s instead of 300s —
  and reports it as a stall. The first-token budget scales with the prompt
  because prompt length is the only thing that predicts how long that silence
  should last; the inter-token budget does not, because once tokens are flowing
  a gap is a stopped generation.
- **`total=None` to aiohttp is not "no ceiling".** `OLLAMA_GENERATE_TIMEOUT` is
  enforced in our own loop with `asyncio.wait_for` per line, because aiohttp's
  `total` and ours both raise `asyncio.TimeoutError` and a single `except`
  cannot tell them apart — owning it is what makes the cause nameable instead
  of an empty `{e}`. The slot still matters (#97), and the inter-token bound is
  what makes a generous ceiling affordable: raising `total` alone would trade a
  5-minute truncation for a 30-minute single-slot occupation on a wedge.
- **The deadline arms on a measurement, not on a default.**
  `OLLAMA_FIRST_TOKEN_RATE_TPS` is **unset** and setting it is what enables the
  first-token deadline — the `FOREIGN_IDLE_MB` convention from #861. Nothing
  measures prefill on this backend: the MLX runner emits no per-request
  timings, and the `slot print_timing` lines stop at 2026-09-30 and come from a
  llama.cpp runner whose largest prompt in the entire log is 2,371 tokens. An
  earlier draft shipped 60 tok/s as a "conservative" default, which is the same
  mistake #903 is about. Unarmed falls back to the whole-request budget.
- **A floor, because the scaled budget was a regression without it.** At any
  plausible rate a short prompt derives *less* than the 300s being replaced —
  8,000 tokens at 60 tok/s is 223s — so a prompt whose prefill took 250s would
  have survived the old cap and died under the new one.
  `OLLAMA_FIRST_TOKEN_MIN_S` is the old cap exactly, which makes the change
  non-regressive by construction rather than by argument. Found by
  `test_stream_deadlines` step 1 failing.
- **The sync path is unchanged and was always correct.** httpx `read` is a
  per-chunk idle bound — already silence. The async methods were written by
  transcribing its `300` into aiohttp's wall-clock `total=`: **the intent was
  always "300s of silence" and the shape was lost in the httpx -> aiohttp
  translation.** Left at 300 and deliberately not wired to the new constants,
  because changing a correct line to look like the fix is how the next reader
  loses track of which one was broken.
- **Every failure terminates the stream.** `model_stalled`, `request_timeout`
  and `stream_error`, each with its own `finish_reason`, each followed by
  `[DONE]`. The generic branch logs `{e!r}` and not `{e}`, because a bare
  `TimeoutError` stringifies to the empty string — which is how 18 cut-offs
  were logged as `Stream error for qwen3.8:27b-mlx:` and then nothing.
- **Two bugs this found in its own making.** The budget-vs-window warning
  divided by the now-unset rate and raised `ZeroDivisionError` **at import**,
  firing exactly when a window is pinned — i.e. on the #903 configuration this
  ships beside, taking the gateway down at startup. Caught by `test_num_ctx`,
  which pins a window, and invisible to `test_stream_deadlines` alone. And the
  unknown-prompt message was unreachable: with no token count the first-token
  budget IS the request budget, so `bounded_by_request` always won.
- **The prefill rate is measurable for free, and now is.** The claim that this
  backend emits no per-request timings is true of `ollama.log` and **not of the
  API**: `/api/chat` returns `prompt_eval_count`, `prompt_eval_duration`,
  `eval_count` and `eval_duration` on the final chunk — verified on the live
  27b — and we were already reading two of those four for `usage` while
  throwing the durations away. `_log_ollama_timings` logs prefill and decode
  rates on both ollama paths, so the rate arrives from REAL hermes traffic
  instead of from a synthetic sweep that would hold the single inference slot
  for half an hour. It also logs our token count beside Ollama's own, which
  turns `prompt_size.count`'s estimate into a checkable number.
- **One number not to read as compute.** `total_duration` was **73.8s** on a
  15-token prompt whose prefill and decode together took **1.8s** — the rest is
  queueing for the single slot. A rate derived from it would be wrong by 40x,
  so it is logged as `wall`. It is also a second argument against elapsed-time
  bounds: a request can spend its whole budget waiting rather than working.
- **Still outstanding.** The measured rate itself, which is also what #903
  needs to know what 70,000 tokens costs. And `wafer-services/model-service` is a
  declared mirror carrying the same `total=300`; the repo fix covers both, its
  rollout does not.

## 2026-10-08 — The served context window: pinned, and reported (#903)

- **The number nobody could read.** A consumer sizing itself against this
  gateway had three ways to ask what context we serve, and all three failed:
  `/v1/models` carried `memory_mb` and nothing else, and `/models/info` and
  `/v1/models/{id}` both 404'd. So `hermes-assistant` (#884) hand-picked
  `model.context_length = 65536` — a window this box has never served — and it
  looked configured. A harness claiming more context than the endpoint serves
  gets truncated or refused; claiming less wastes the window it has.
- **And nothing was setting it.** The assumed 65,536 cap did not exist. Nothing
  in this repo passed `num_ctx`, and Ollama answers that absence by DERIVING a
  default from free VRAM at the moment of load, logged as `vram-based default
  context`. slice's own `ollama.log`:

  | `total_vram` | derived `num_ctx` | occurrences |
  |---|---|---|
  | 48.0 GiB | 262,144 | 19 |
  | 58.0 GiB | 262,144 | 1 |
  | 60.8 GiB | 262,144 | 2 |
  | **0 B** | **4,096** | **2** |

  The two `0 B` rows are 2026-08-25 and 2026-08-26, when llama-server GPU
  discovery timed out, Ollama fell back to CPU and read no VRAM at all. So the
  served window was never a cap — it was a **64x spread** decided by whatever
  the box had free that minute, and reported nowhere. That is worse than a
  wrong number, because a wrong number can at least be found.
- **What it was actually serving.** 262,144 — the model's full native window —
  on every load since the `0 B` incidents. `qwen3.8:27b-mlx` therefore already
  cleared #903's 140,000 crossover by 1.9x, and the entire exchange was about a
  cap that was never imposed. Pinning it changes no behaviour today; it stops
  the window being re-decided by memory pressure on the next load.
- **Pinned.** `OLLAMA_NUM_CTX` (all Ollama models) and `OLLAMA_NUM_CTX_MODELS`
  (`name=tokens`, per-model, wins over the global). Unset keeps the old derive
  behaviour, which is right for a box nobody is sizing against and wrong for
  one that has a consumer. A malformed pair is dropped with a warning and never
  read as `0`, because `0` means "let Ollama derive one" — the behaviour the
  setting exists to stop. A value above the window the weights declare is
  clamped to the weights and logged once, so a stale copy of another box's
  setting cannot become an allocation.
- **One num_ctx, every call path.** Ollama keys a loaded instance by its
  options: a request whose `num_ctx` differs from the resident instance's
  **reloads the model**. So the merge lives in `OllamaClient._with_num_ctx` and
  not at five call sites, `unload_model` is excluded, and `test_num_ctx` reads
  the source to assert each path routes through it rather than trusting that it
  does. A caller's own explicit `num_ctx` still wins — `/models/load` takes one.
- **Adoption keeps the instance's own window.** On startup the gateway
  re-applies `keep_alive` to a model somebody else loaded. Sending OUR `num_ctx`
  there would reload a resident model to change a number nobody was waiting on,
  i.e. a gateway restart would bounce a job. It re-applies the adopted
  instance's own `context_length` instead; the configured window lands on the
  next real load, which `keep_alive` guarantees arrives.
- **Reported, from the truth and not from the config.** `offering()` carries
  `context_length`, so `/warm`, `/v1/models` and `/status` cannot disagree
  about it. On a resident model it is read back from `ollama ps` — after the
  load, never assumed, because what we asked for and what we got are different
  facts. `ensure_running` refreshes it off a read it already makes, so a reload
  Ollama performed under us does not strand a stale number. A **loadable**
  model reports it only where we pin one: predicting a VRAM-derived default
  would be this ticket's invented number one layer down. **Absent means
  unknown, not unlimited**, and `/service-docs` says so.
- **`GET /v1/models/{id}` exists now.** `:path`, so an id with a slash
  (`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`) resolves rather than 404ing on
  the segment boundary. Same contract as the list — 404 for anything not on
  offer, blocked models included — and it accepts the partial name the chat
  route accepts, because a consumer that gets a working completion and a 404
  window has the worst of both.
- **The cost of the clamp, bounded.** `num_ctx_for()` is reached from
  `offering()`, which serves the auth-exempt `/warm` for every model in the
  catalogue. It returns before touching Ollama when nothing is pinned, and when
  something is, `/api/show` is cached by digest and the digest map is read at
  most once per 300s — so an anonymous request rate cannot become an Ollama
  request rate. Same mistake `/warm`'s docstring already records about
  subprocess spawns.
- **Still open, and not fixed here.** `ManagedModel.memory_mb` is read once at
  load and never again, while the MLX KV cache grows with the conversation:
  measured 2026-10-08, the lease said 17,530 MB for a model `ollama ps` put at
  26,681 → 30,311 → 31,187 MB over forty minutes. The local fit gate is
  unaffected (it reads hardware), but the registry is told a number that is
  13 GB light and getting lighter, which is exactly the figure another box
  reads to decide whether it can place work here. Separate ticket.

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
