#!/usr/bin/env python3
"""Streaming deadlines on the owned backends: silence, not elapsed time (#904).

`chat_stream` had one bound, aiohttp's `total=` at 300s, and `total` bounds
ELAPSED TIME. So a generation streaming tokens steadily was killed at 300s for
being long — the one thing that is not a fault. Measured on slice, `/api/chat`
for 2026-10-07/08: 121 x 200 and 17 x 500, **all seventeen 500s at exactly
`5m0s`**, every one of them `hermes-assistant` (#884). Successful durations the
same day reached `4m15s`, so the distribution already touched the ceiling.

WHAT THOSE 17 DID NOT DO, because the first version of this file said they did:
they did not produce 17 visible failures. `samclaude-admin` checked the seat's
adapter journal (#903, 2026-10-08) and found three consecutive turns of 56, 50
and 31 minutes, all accepted, all posting answers, and zero matches for error /
timeout / retry / stale in the seat's own logs. A turn makes MANY requests, and
Hermes retried across the boundary, so a 5-minute per-request ceiling coexists
with 56-minute turns and absorbs below the turn level. The cap is still wrong;
its cost was latency and a wasted inference slot, not a broken answer. Keeping
the distinction because a comment that overstates its evidence is how the
previous wrong number survived.

And the caller was never told. The handler logged and stopped yielding: no
`[DONE]`, no `finish_reason`, and `TimeoutError` stringifies to the empty string
so even our own log read `Stream error for qwen3.8:27b-mlx:` and then nothing.

What this asserts:

  1. a stream that keeps producing is NOT killed for running long — the
     regression itself, driven through the real loop;
  2. silence IS bounded, with the two phases told apart: prefill is silent and
     legitimately slow, a gap between tokens is a stopped generation;
  3. "ran long" and "went silent" are different exceptions, because they call
     for opposite responses;
  4. every terminal path tells the caller, and no path can end a stream without
     `[DONE]`.

The loop runs against a fake aiohttp injected into `sys.modules`, with
sub-second constants, so this exercises the real iteration and deadline
arithmetic rather than reading the source for it. Steps 6-7 are structural,
and say so.

    python -m ollama.test_stream_deadlines
"""

import asyncio
import json
import os
import sys
import types

failures = []
checks = 0


def step(n, msg):
    print(f"\n{'='*66}\n  Step {n}: {msg}\n{'='*66}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


# ---------------------------------------------------------------- fake aiohttp
class FakeTimeout:
    def __init__(self, **kw):
        self.kw = kw
        LAST_TIMEOUT.clear()
        LAST_TIMEOUT.update(kw)


LAST_TIMEOUT: dict = {}


class FakeContent:
    """An async line iterator with a scripted schedule of (delay, line)."""

    def __init__(self, script):
        self._script = list(script)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._script:
            raise StopAsyncIteration
        delay, line = self._script.pop(0)
        await asyncio.sleep(delay)
        if line is None:               # a hang: never returns
            await asyncio.sleep(3600)
        return line


class FakeResp:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeSession:
    def __init__(self, script):
        self._script = script
        self.closed = False

    def post(self, url, json=None, timeout=None):
        return FakeResp(FakeContent(self._script))

    async def close(self):
        self.closed = True


SESSIONS: list = []


def install_fake_aiohttp(script):
    mod = types.ModuleType("aiohttp")
    mod.ClientTimeout = FakeTimeout

    def ClientSession(*a, **kw):
        s = FakeSession(script)
        SESSIONS.append(s)
        return s

    mod.ClientSession = ClientSession
    mod.ClientError = type("ClientError", (Exception,), {})
    sys.modules["aiohttp"] = mod


def chunk(content):
    return json.dumps({"message": {"content": content}}).encode()


def drive(client, script, prompt_tokens):
    """Run _stream_with_deadlines to completion; return (chunks, exception)."""
    install_fake_aiohttp(script)

    async def go():
        out = []
        try:
            async for c in client._stream_with_deadlines(
                "/api/chat", {"model": "m"}, prompt_tokens
            ):
                out.append(c)
        except Exception as e:
            return out, e
        return out, None

    return asyncio.run(go())


def main():
    from . import config, ollama_client
    from .ollama_client import OllamaStalled, OllamaRequestTimeout

    c = ollama_client.OllamaClient.__new__(ollama_client.OllamaClient)
    c.base_url = "http://fake"
    c._http = None
    c._clamped = set()
    c._native_ctx = {}
    c._digests = {}
    c._digests_at = 0.0
    c.num_ctx_for = lambda _m: None          # no pinning: keep payload untouched

    step(1, "the first-token deadline scales with the prompt, and clamps")
    config.OLLAMA_GENERATE_TIMEOUT = 1800
    config.OLLAMA_FIRST_TOKEN_RATE_TPS = 60
    config.OLLAMA_FIRST_TOKEN_MARGIN_S = 90
    config.OLLAMA_FIRST_TOKEN_MIN_S = 300
    config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S = 0
    d8 = config.ollama_first_token_deadline(8000)
    d70 = config.ollama_first_token_deadline(70000)
    check(8000 / 60 + 90 < 300,
          "the scale ALONE would give an 8k prompt 223s — less than the flat "
          "300s being replaced, which is why a floor exists")
    check(d8 == 300,
          f"...and the FLOOR binds at 300s ({d8:.0f}s), not the 223s the scale "
          f"alone would give — otherwise a short prompt gets a TIGHTER budget "
          f"than the flat cap this replaces, which is a regression")
    check(config.ollama_first_token_deadline(100) == 300,
          "a tiny prompt is never worse off than before the change")
    check(d70 > 300, f"the scale only ever BUYS time ({d70:.0f}s for 70k)")
    check(abs(d70 - (70000 / 60 + 90)) < 0.01,
          f"70,000 tokens (#884's trigger) -> {d70:.0f}s")
    check(config.ollama_first_token_deadline(10_000_000) == 1800,
          "clamped to the whole-request budget — a deadline longer than the "
          "request's own could never fire")
    check(config.ollama_first_token_deadline(None) == 1800,
          "unknown prompt -> the whole-request budget, i.e. exactly the "
          "behaviour before this existed")
    check(config.ollama_first_token_deadline(0) == 1800, "0 is also unknown")

    # Sub-second constants for the behavioural steps.
    config.OLLAMA_GENERATE_TIMEOUT = 4
    config.OLLAMA_STALL_TIMEOUT = 1
    config.OLLAMA_FIRST_TOKEN_RATE_TPS = 1000
    config.OLLAMA_FIRST_TOKEN_MARGIN_S = 1
    # Scaled down with the rest. Left at the 300s default it would exceed the
    # 4s request budget, the clamp would collapse the two, and steps 3-4 would
    # test the request timeout instead of the stall — which is exactly the
    # misconfiguration step 10 warns a real box about.
    config.OLLAMA_FIRST_TOKEN_MIN_S = 1

    step(2, "THE REGRESSION: a stream that keeps producing is not killed")
    # 30 chunks, 0.05s apart. Elapsed ~1.5s, which is past several multiples of
    # the 1s inter-token budget — the old `total=` would have cut this, the
    # silence rule must not, because nothing is ever silent for 1s.
    got, err = drive(c, [(0.05, chunk("x")) for _ in range(30)], 100)
    check(err is None, f"no exception on a steady stream (got {err!r})")
    check(len(got) == 30, f"all 30 chunks delivered (got {len(got)})")

    step(3, "silence BEFORE the first token is a first_token stall")
    got, err = drive(c, [(0, None)], 100)   # prompt known -> budget 100/1000+1
    check(isinstance(err, OllamaStalled), f"OllamaStalled (got {type(err).__name__})")
    check(getattr(err, "phase", None) == "first_token",
          f"phase=first_token (got {getattr(err, 'phase', None)!r})")
    check(getattr(err, "tokens", -1) == 0, "zero tokens before it")
    check("Prefill never produced" in str(err),
          "the message says prefill never produced, not 'timed out'")

    step(4, "silence BETWEEN tokens is an inter_token stall, with a count")
    got, err = drive(c, [(0.05, chunk("a")), (0.05, chunk("b")), (0, None)], 100)
    check(isinstance(err, OllamaStalled), f"OllamaStalled (got {type(err).__name__})")
    check(getattr(err, "phase", None) == "inter_token", "phase=inter_token")
    check(getattr(err, "tokens", 0) == 2,
          f"names how many chunks arrived first ({getattr(err, 'tokens', 0)})")
    check(len(got) == 2, "and the chunks before the stall were delivered")

    step(5, "running long is NOT a stall, and is a different exception")
    # Chunks every 0.5s forever: never silent for 1s, so only the 4s
    # whole-request budget can stop it.
    got, err = drive(c, [(0.5, chunk("x")) for _ in range(40)], 100)
    check(isinstance(err, OllamaRequestTimeout),
          f"OllamaRequestTimeout, not OllamaStalled (got {type(err).__name__})")
    check(not isinstance(err, OllamaStalled),
          "...and NOT a stall — it was producing the whole time")
    check(getattr(err, "tokens", 0) >= 5,
          f"reports the chunks it did receive ({getattr(err, 'tokens', 0)})")
    check("NOT a stall" in str(err),
          "the message says so in words, because the fix differs: raise the "
          "budget vs. the runner is stuck")
    check("OLLAMA_GENERATE_TIMEOUT" in str(err),
          "...and names the knob to change")

    step(6, "an unknown prompt size claims no measurement it did not make")
    got, err = drive(c, [(0, None)], None)
    check(isinstance(err, OllamaRequestTimeout),
          f"a silent stream with no token count is a request timeout, not a "
          f"stall (got {type(err).__name__})")
    check("prompt size was unknown" in str(err), "and says why")
    check("No first-token deadline was armed" in str(err),
          "...naming the thing that was not armed, rather than leaving the "
          "reader to infer it from a budget that looks arbitrary")

    step(7, "the session is closed on every path")
    check(all(s.closed for s in SESSIONS),
          f"all {len(SESSIONS)} fake sessions closed in the finally")

    step(13, "queue and prefill are separate terms, because they have "
             "different shapes")
    config.OLLAMA_GENERATE_TIMEOUT = 1800
    config.OLLAMA_FIRST_TOKEN_MIN_S = 0       # isolate the arithmetic
    config.OLLAMA_FIRST_TOKEN_RATE_TPS = 100
    config.OLLAMA_FIRST_TOKEN_MARGIN_S = 0
    config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S = 0
    bare_small = config.ollama_first_token_deadline(100)
    bare_big = config.ollama_first_token_deadline(70000)
    config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S = 120
    q_small = config.ollama_first_token_deadline(100)
    q_big = config.ollama_first_token_deadline(70000)
    check(q_small - bare_small == 120 and q_big - bare_big == 120,
          "the queue term is ABSOLUTE — it adds the same 120s to a 100-token "
          "prompt as to a 70,000-token one, because a 15-token prompt waited "
          "72s for the slot and waiting does not scale with the prompt")
    check(bare_big - bare_small == (70000 - 100) / 100,
          "the prefill term SCALES — that is the part "
          "`prompt_eval_duration` measures, and the only part it measures")
    # If the queue term were folded into the rate instead, it would have to be
    # a much slower rate, which would over-allow long prompts by the same
    # factor it under-allows short ones. Show that the two models diverge.
    config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S = 0
    config.OLLAMA_FIRST_TOKEN_RATE_TPS = 50   # "absorb" queueing by halving it
    folded_small = config.ollama_first_token_deadline(100)
    folded_big = config.ollama_first_token_deadline(70000)
    check(folded_small < 120,
          f"folding queueing into the rate CANNOT cover a short prompt's "
          f"72s wait ({folded_small:.0f}s for 100 tokens) — the absolute "
          f"term is not expressible as a rate")
    check(folded_big > q_big,
          f"...while over-allowing a long one ({folded_big:.0f}s vs "
          f"{q_big:.0f}s), so the deadline stops firing when it should")

    step(14, "arming the rate without a queue allowance is called out")
    import logging as _l
    seen = []
    h = type("H", (_l.Handler,), {"emit": lambda _s, r: seen.append(r.getMessage())})()
    config.OLLAMA_FIRST_TOKEN_RATE_TPS = 100
    config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S = 0
    config.log.addHandler(h)
    config._warn_if_rate_armed_without_queue_allowance()
    config.log.removeHandler(h)
    check(len(seen) == 1 and "reported as a stalled prefill" in seen[0],
          "a rate with no queue term warns: the rate measures compute, the "
          "deadline bounds queue+prefill, and the difference was ~40x here")
    seen.clear()
    config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S = 120
    config.log.addHandler(h)
    config._warn_if_rate_armed_without_queue_allowance()
    config.log.removeHandler(h)
    check(not seen, "both set: silent")
    seen.clear()
    config.OLLAMA_FIRST_TOKEN_RATE_TPS = 0
    config.OLLAMA_FIRST_TOKEN_QUEUE_ALLOWANCE_S = 0
    config.log.addHandler(h)
    config._warn_if_rate_armed_without_queue_allowance()
    config.log.removeHandler(h)
    check(not seen,
          "and UNARMED is silent too — the shipped default must not warn, or "
          "every box logs it forever and nobody reads it")

    step(11, "shipped default: the deadline is NOT armed, because nothing "
             "measures prefill on this backend yet")
    import importlib
    for k in ("OLLAMA_FIRST_TOKEN_RATE_TPS",):
        os.environ.pop(k, None)
    shipped = importlib.reload(config)
    check(shipped.OLLAMA_FIRST_TOKEN_RATE_TPS == 0,
          "OLLAMA_FIRST_TOKEN_RATE_TPS defaults to unset — the MLX runner "
          "emits no per-request timings, so any default would be the same "
          "kind of invented number #903 was about")
    check(shipped.ollama_first_token_deadline(70000)
          == float(shipped.OLLAMA_GENERATE_TIMEOUT),
          "unarmed, a 70k prompt falls back to the whole-request budget "
          "rather than a deadline derived from a guessed rate")
    check(shipped.OLLAMA_STALL_TIMEOUT == 300,
          f"the inter-token default is 300s, not 60 "
          f"({shipped.OLLAMA_STALL_TIMEOUT}). 60 was EXO_STALL_TIMEOUT, a "
          f"measured figure for a different backend, and it aborted 2 of the "
          f"first 6 real requests after the #904 rollout — a 33% abort rate, "
          f"worse than the 22% it replaced. 300 is the per-chunk idle bound "
          f"the sync path always had, i.e. the original intent")
    # The two settings SHIP TOGETHER, so the unarmed rate meets a pinned
    # window on the first box that takes both. That combination divided by
    # zero at import and would have taken the gateway down at startup.
    os.environ["OLLAMA_NUM_CTX_MODELS"] = "qwen3.8:27b-mlx=262144"
    try:
        both = importlib.reload(config)
        check(both.OLLAMA_NUM_CTX_MODELS.get("qwen3.8:27b-mlx") == 262144,
              "a pinned window alongside an UNARMED rate imports cleanly — "
              "the budget-vs-window warning divides by the rate and must not "
              "run when there is none (#903 and #904 ship together)")
    except ZeroDivisionError:
        check(False, "a pinned window alongside an unarmed rate must not "
                     "raise ZeroDivisionError at import")
    finally:
        os.environ.pop("OLLAMA_NUM_CTX_MODELS", None)
        importlib.reload(config)
    # Restore the test's own scaled constants for the steps below.
    config.OLLAMA_GENERATE_TIMEOUT = 1800
    config.OLLAMA_FIRST_TOKEN_MIN_S = 300
    config.OLLAMA_FIRST_TOKEN_RATE_TPS = 60
    config.OLLAMA_FIRST_TOKEN_MARGIN_S = 90

    step(10, "a floor at or above the request budget is called out, not silent")
    config.OLLAMA_GENERATE_TIMEOUT = 300
    config.OLLAMA_FIRST_TOKEN_MIN_S = 300
    check(config.ollama_first_token_deadline(1) == 300,
          "with the floor equal to the budget the two collapse — the "
          "first-token phase becomes unreportable")
    import logging as _logging
    seen_warn = []
    h = type("H", (_logging.Handler,), {"emit": lambda _s, r: seen_warn.append(r.getMessage())})()
    config.log.addHandler(h)
    config._warn_if_first_token_floor_collapses()
    config.log.removeHandler(h)
    check(len(seen_warn) == 1 and "collapses onto" in seen_warn[0],
          "...and config says so at startup rather than leaving it to be "
          "discovered from a mislabelled error")
    config.OLLAMA_GENERATE_TIMEOUT = 1800
    config.OLLAMA_FIRST_TOKEN_MIN_S = 300
    seen_warn.clear()
    config.log.addHandler(h)
    config._warn_if_first_token_floor_collapses()
    config.log.removeHandler(h)
    check(not seen_warn,
          "the healthy default (1800 vs 300) warns about NOTHING — a line "
          "that fires on the normal path teaches its reader to skip it")

    step(8, "structural: the elapsed-time cap is gone, and cannot come back")
    src = open(os.path.join(os.path.dirname(__file__),
                            "ollama_client.py")).read()
    check("total=300" not in src,
          "no `total=300` anywhere in the client")
    check("total=None" in src,
          "aiohttp is given total=None — our budget is enforced in the loop so "
          "it can be told apart from a stall")
    check(src.count("ClientTimeout(") == 1,
          f"exactly one ClientTimeout construction "
          f"({src.count('ClientTimeout(')}) — a second would be a second "
          f"policy nobody is reading")
    check("sock_connect=10" in src,
          "connecting to a dead Ollama is still bounded; that is neither a "
          "stall nor a long generation")
    # The httpx read timeout is a DIFFERENT shape (per-read, i.e. already
    # silence) and must not be collapsed onto the inter-token budget, which
    # would cut a prefill on the non-streaming collect paths.
    check("httpx.Timeout(connect=10, read=300, write=10, pool=10)" in src,
          "the SYNC path is untouched: httpx `read` is a per-chunk idle bound, "
          "so it was always the right shape. The bug was transcribing this "
          "300 into aiohttp's wall-clock `total=` — the intent was always "
          "'300s of silence' and the shape was lost in the translation")
    check("OLLAMA_GENERATE_TIMEOUT" in src and "wait_for" in src,
          "the whole-request ceiling still exists and is enforced in-process; "
          "total=None is not 'no ceiling' (#97: the single inference slot)")

    step(9, "structural: no terminal path leaves the caller without [DONE]")
    srv = open(os.path.join(os.path.dirname(__file__), "server.py")).read()
    body = srv.split("if mm.backend == Backend.OLLAMA:", 1)[1].split(
        "return StreamingResponse(stream(), media_type=\"text/event-stream\")", 1)[0]
    for exc in ("OllamaStalled", "OllamaRequestTimeout"):
        check(f"except {exc} as e:" in body,
              f"the ollama stream handler catches {exc}")
    # Four terminal yields: done, stalled, timeout, generic error. Counted,
    # not merely present — a presence check passes on a duplicate.
    check(body.count('yield "data: [DONE]\\n\\n"') == 4,
          f"exactly 4 [DONE] emissions: finish, stall, timeout, error "
          f"(got {body.count('yield (data: [DONE]'.replace('(', chr(34)))})")
    # The three causes are now passed to the one emitter rather than spelled
    # out per branch, so assert the CALLS, not the literals.
    check(body.count('"model_stalled", e.detail') == 1
          and body.count('"request_timeout", e.detail') == 1
          and body.count('"stream_error", f"{type(e).__name__}: {e}"') == 1,
          "each failure names its own cause to the emitter, so a caller can "
          "tell them apart — noting claude-containers' finding that for "
          "Hermes specifically the three collapse to a generic "
          "provider_stream_error and only the TEXT distinguishes them")
    # The EXACT line, not the substring: the comment above it in server.py
    # also contains `{e!r}`, so a presence check passed on the reverted code
    # when this was mutation-tested. "Assert a count, not a presence" —
    # defeated here by my own comment, which is the whole reason for the rule.
    check(body.count('log.warning(f"Stream error for {mm.name}: {e!r}")') == 1,
          "the generic branch logs {e!r}, not {e} — a bare TimeoutError "
          "stringifies to the empty string, which is how 18 cut-offs were "
          "logged as `Stream error for qwen3.8:27b-mlx:` and then nothing")
    check("prompt_tokens=n_prompt" in body,
          "the counted prompt is passed through, so the deadline is armed")

    step(16, "a failure emits BOTH detector shapes, in order")
    # Hermes has two detectors on two branches, each blind to what the other
    # catches (claude-containers, driven against the installed client):
    #   A choices: [] + flat pair -> _choiceless_chunk, needs NO content
    #   B populated + delta.content + finish_reason "error" -> the text detector
    # B alone works today but makes one field load-bearing, and that field is
    # byte-identical to error_message. A alone is what admin asked for. Both.
    # Imported defensively so this step REPORTS on a tree without the emitter
    # rather than aborting the run — an ImportError halts main() and the named
    # checks never appear, which is the weaker evidence of the two.
    try:
        from .server import _failure_frames, _FAILURE_FINISH_REASON
    except ImportError:
        _failure_frames = None
        _FAILURE_FINISH_REASON = None
    check(_failure_frames is not None,
          "server exposes _failure_frames — the one emitter for the two "
          "detector shapes")
    frames = (_failure_frames("request_timeout", "cap fired after 1800s",
                              {"deadline_s": 1800.0, "elapsed_s": 1801.2})
              if _failure_frames else [])
    check(len(frames) == 2, f"two frames per failure (got {len(frames)})")
    A, B = (frames + [{}, {}])[:2]
    check(A.get("choices") == [],
          "A is choiceless — the path that needs no accumulated content, and "
          "the one a prefill-phase cap can actually reach")
    check(A.get("error_type") == "request_timeout" and A.get("error_message"),
          "...carrying the flat pair _choiceless_chunk reads off the chunk")
    check(A.get("error", {}).get("deadline_s") == 1800.0,
          "...and the nested detail, so deadline_s/elapsed_s survive")
    check((B.get("choices") or [{}])[0].get("delta", {}).get("content")
          == "cap fired after 1800s",
          "B carries NON-EMPTY delta.content — the only field the text "
          "detector reads, which is why it is not duplication of error_message")
    check((B.get("choices") or [{}])[0].get("finish_reason") == "error"
          and _FAILURE_FINISH_REASON == "error",
          'B uses "error", the one value no client reads as normal completion')
    check(bool(frames) and frames.index(A) < frames.index(B),
          "A is emitted FIRST, so it raises in the chunk loop before B is read")

    srv_s = open(os.path.join(os.path.dirname(__file__), "server.py")).read()
    ob = srv_s.split("if mm.backend == Backend.OLLAMA:", 1)[1].split(
        'return StreamingResponse(stream(), media_type="text/event-stream")', 1)[0]
    for bad in ('"finish_reason": "stalled"', '"finish_reason": "timeout"'):
        check(bad not in ob,
              f'no {bad} — an unrecognised reason degrades to "assume normal", '
              f'which is why those frames fell through to success')
    check(ob.count("for frame in _failure_frames(") == 3,
          f'all three branches go through the one emitter, so the two-frame '
          f'sequence cannot be right in two places and wrong in the third '
          f'({ob.count("for frame in _failure_frames(")})')
    check(ob.count('yield "data: [DONE]') == 4,
          f'and every terminal path still ends with [DONE] '
          f'({ob.count(chr(34) + "data: [DONE]")})')
    check('"finish_reason": finish' in ob,
          "the SUCCESS frame is untouched — the rule is about which value, "
          "not about removing the field")
    check("ASSEMBLES" in srv_s and "manufactured success" in srv_s,
          "the source records what the silence actually cost: not a dropped "
          "error but a well-formed response accepted as an answer")

    step(15, "the armed/not-armed line reports the state it is actually in")
    # Caught in the live log a minute after the #904 restart: the branch keyed
    # on `if prompt_tokens:`, so a box with a counted prompt and NO measured
    # rate logged "armed ... / 0 tok/s" — claiming a state it was not in and
    # displaying the zero it had divided by.
    src_c = open(os.path.join(os.path.dirname(__file__),
                              "ollama_client.py")).read()
    fn = src_c.split("async def _stream_with_deadlines", 1)[1].split(
        "\n    async def ", 1)[0]
    check("armed = (prompt_tokens" in fn
          and "config.OLLAMA_FIRST_TOKEN_RATE_TPS > 0" in fn,
          "the log branch keys on whether the deadline is ARMED, not on "
          "whether the prompt was counted")
    check("if armed:" in fn,
          "...and the armed message is behind that condition")
    check('"OLLAMA_FIRST_TOKEN_RATE_TPS is unset"' in fn,
          "the not-armed message says WHICH of the two reasons it was — an "
          "unset rate and an uncountable prompt are different problems")
    check("counted {prompt_tokens} prompt tokens" in fn,
          "...and still reports the count it does have, so the line is not "
          "less informative for being honest")

    step(12, "the rate gets measured from real traffic, not a synthetic sweep")
    check("def _log_ollama_timings(" in srv,
          "the done chunk's timings are logged — `/api/chat` returns "
          "prompt_eval_duration and eval_duration, which we were already "
          "half-reading for `usage` and throwing away")
    check(srv.count("_log_ollama_timings(mm.name, chunk, n_prompt)") == 2,
          f"on BOTH ollama paths, streaming and collect "
          f"({srv.count('_log_ollama_timings(mm.name, chunk, n_prompt)')}) — "
          f"one of two would silently halve the sample")
    fn = srv.split("def _log_ollama_timings(", 1)[1].split("\ndef ", 1)[0]
    check("prompt_eval_duration" in fn and "eval_duration" in fn,
          "both durations, so prefill and decode are separable — which is "
          "what nobody could do when today's 5-minute requests were triaged")
    check("wall" in fn and "queue" in fn,
          "total_duration is labelled `wall` and documented as including "
          "queue wait: it was 73.8s on a prompt whose compute took 1.8s, so a "
          "rate derived from it would be wrong by 40x")
    check("our count" in fn,
          "our token count is logged beside ollama's own, turning the "
          "chars/token estimate into a checkable number")

    # A quotient under a token floor is not a rate — it is fixed overhead
    # wearing a rate's units. Driven, not read: these are real samples.
    from .server import _log_ollama_timings, _MIN_TOKENS_FOR_RATE
    import logging as _lg
    lines = []
    _h = type("H", (_lg.Handler,), {"emit": lambda _s, r: lines.append(r.getMessage())})()
    _log = _lg.getLogger("model-service")
    _log.addHandler(_h)
    # slice, measured: a 15-token prompt and a 1-token generation. Used to
    # claim 10 tok/s prefill and 4.3 tok/s decode.
    _log_ollama_timings("m", {"prompt_eval_count": 15,
                              "prompt_eval_duration": 1561665666,
                              "eval_count": 1, "eval_duration": 230414084,
                              "total_duration": 73825340875}, 18)
    # wafer, measured cold: 2,676 tokens at 74.7 tok/s, num_predict:1.
    _log_ollama_timings("m", {"prompt_eval_count": 2676,
                              "prompt_eval_duration": 35830000000,
                              "eval_count": 1, "eval_duration": 310000000,
                              "total_duration": 60900000000}, 2700)
    # a real generation: both columns earn a rate.
    _log_ollama_timings("m", {"prompt_eval_count": 2000,
                              "prompt_eval_duration": 20000000000,
                              "eval_count": 300, "eval_duration": 12000000000,
                              "total_duration": 32100000000}, 1950)
    _log.removeHandler(_h)
    check(len(lines) == 3, f"three samples logged (got {len(lines)})")
    tiny, cold, real = lines
    check("10 tok/s" not in tiny and "no rate" in tiny,
          "a 15-token prompt does NOT claim 10 tok/s — under the floor it "
          "logs the duration and refuses the quotient")
    check("4.3 tok/s" not in tiny and "first_token 0.23s" in tiny,
          "a 1-token generation is reported as first_token latency, not as a "
          "decode rate (claude-wafer-services, #904: `eval_count/eval_duration`"
          " at ec=1 is the total_duration trap one column over)")
    check("1 token, not a rate" in cold,
          "...including on a long prompt swept with num_predict:1, which is "
          "the exact shape that produced the finding")
    check("74.7 tok/s" in cold,
          f"but the PREFILL on that row does earn its rate — reproduces "
          f"wafer's measured 74.7 tok/s from their own numbers")
    check("queued ~24.8s" in cold,
          "and `queued` independently agrees with their 24.7s weight-load "
          "figure, from the same row")
    check("= 100.0 tok/s" in real and "decode 300 tok" in real
          and "= 25.0 tok/s" in real,
          "a real generation rates both columns")
    check("queued" not in real,
          "...and carries no queued term, because there was none")
    check(_MIN_TOKENS_FOR_RATE >= 32,
          f"the floor is at least 32 tokens ({_MIN_TOKENS_FOR_RATE})")

    # The trusted-rate ceiling, which is PER MODEL and unset by default. The
    # two rows below are both real measurements and they sit inside 1.3x of
    # each other, which is why no flat constant works.
    HIT = {"prompt_eval_count": 76321, "prompt_eval_duration": 22730000000,
           "eval_count": 397, "eval_duration": 33800000000,
           "total_duration": 56600000000}                  # slice, 27b, cached
    FAST = {"prompt_eval_count": 2677, "prompt_eval_duration": 1030000000,
            "eval_count": 1, "eval_duration": 100000000,
            "total_duration": 1200000000}                  # wafer, 1.7b, GENUINE

    lines.clear(); _log.addHandler(_h)
    _log_ollama_timings("qwen3.8:27b-mlx", HIT, 201746)
    _log_ollama_timings("qwen3:1.7b", FAST, 2700)
    _log.removeHandler(_h)
    check(len(lines) == 2, f"two rows (got {len(lines)})")
    check("= 3358" not in lines[0] and "= 2605" not in lines[1],
          "UNSET by default: neither row asserts a rate, because the gateway "
          "cannot tell reuse from a fast model and a flat constant cannot "
          "either — 2,605 genuine vs 3,358 cached is 1.3x apart")
    check("76321 tok in 22.73s" in lines[0]
          and "2677 tok in 1.03s" in lines[1],
          "...but both still log the count and the duration, which is "
          "strictly more than before and all they can support")

    os.environ["OLLAMA_MAX_TOK_S_MODELS"] = "qwen3.8:27b-mlx=400,qwen3:1.7b=6000"
    cfg2 = importlib.reload(config)
    import ollama.server as _srv
    importlib.reload(_srv)
    lines.clear()
    _l2 = _lg.getLogger("model-service"); _l2.addHandler(_h)
    _srv._log_ollama_timings("qwen3.8:27b-mlx", HIT, 201746)
    _srv._log_ollama_timings("qwen3:1.7b", FAST, 2700)
    _l2.removeHandler(_h)
    check("rate not trusted" in lines[0] and "3,358" in lines[0],
          "with a 400 ceiling the 27b cache row is `rate not trusted` and "
          "states the implied figure without asserting it")
    check("cache hit" not in lines[0],
          "NOT labelled `cache hit` — at 961 tok/s a genuine 1.7b prefill and "
          "a partially-cached 27b one are indistinguishable, so the label "
          "must not name a cause the data cannot support")
    check("tok/s" in lines[1] and "rate not trusted" not in lines[1],
          f"...while the 1.7b row DOES earn its rate under a 6000 ceiling — "
          f"a flat 400 would have deleted every prefill that model will ever "
          f"log and blamed a cache that was not there. Row: {lines[1][:120]}")
    os.environ.pop("OLLAMA_MAX_TOK_S_MODELS", None)
    importlib.reload(config); importlib.reload(_srv)
    check("queued ~" in fn,
          "the queue residue is stated, not left as arithmetic — it is the "
          "field the #904 correlation gets sorted on, and a correlation that "
          "needs read-and-subtract per line gets done on three samples")
    check("gap > 0.5" in fn,
          "...and only when it is real, so the normal path does not carry a "
          "`queued ~0.0s` that teaches its reader to skip the line")

    print(f"\n{'='*66}")
    print(f"  {checks - len(failures)}/{checks} checks passed")
    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")
    print(f"{'='*66}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
