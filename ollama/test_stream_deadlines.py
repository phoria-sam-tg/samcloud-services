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
    check(shipped.OLLAMA_STALL_TIMEOUT == 60,
          "the inter-token bound is armed regardless, so the inference slot "
          "is covered while the rate is unmeasured")
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
    check(body.count('"finish_reason": "stalled"') == 1
          and body.count('"finish_reason": "timeout"') == 1
          and body.count('"finish_reason": "error"') == 1,
          "each failure carries its own finish_reason, so a caller can tell "
          "them apart without parsing prose")
    # The EXACT line, not the substring: the comment above it in server.py
    # also contains `{e!r}`, so a presence check passed on the reverted code
    # when this was mutation-tested. "Assert a count, not a presence" —
    # defeated here by my own comment, which is the whole reason for the rule.
    check(body.count('log.warning(f"Stream error for {mm.name}: {e!r}")') == 1,
          "the generic branch logs {e!r}, not {e} — a bare TimeoutError "
          "stringifies to the empty string, which is how 17 cut-offs were "
          "logged as `Stream error for qwen3.8:27b-mlx:` and then nothing")
    check("prompt_tokens=n_prompt" in body,
          "the counted prompt is passed through, so the deadline is armed")

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
