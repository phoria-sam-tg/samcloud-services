"""The gateway abandons a generation that starts and then stops.

WHY THIS EXISTS (#830). A blocked decode produces tokens and then nothing, for
as long as anyone will wait — 12, 27 and 58 minutes on 2026-09-25. The whole
request timeout is necessarily generous (prefill alone measured 42s on a
12,252-token prompt), so waiting it out meant the gateway held the pool's
exclusive lease for up to EXO_GENERATE_TIMEOUT. That matters beyond the one
request: the placement guard's wedge signal is "runners busy with NO lease", so
a patient gateway hides the wedge from the thing that repairs it.

The rule: once tokens have started, a gap longer than EXO_STALL_TIMEOUT ends the
request. Before the first token there is no rule at all — prefill is silent and
legitimately slow, and cutting it would break long prompts.

Run: python ollama/test_exo_stall.py
"""

import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama import config                                    # noqa: E402
from ollama.exo_client import ExoClient, ExoStalled, ExoRequestFailed  # noqa: E402

PASS = FAIL = 0


def check(desc, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"      \033[0;32m✓\033[0m {desc}")
    else:
        FAIL += 1
        print(f"      \033[0;31m✗\033[0m {desc}")
        if detail:
            print(f"        {detail}")


def fake_aiohttp(script):
    """An aiohttp whose response yields `script` — (delay, bytes) pairs."""

    class Content:
        def __aiter__(self):
            return self

        def __init__(self):
            self.i = 0

        async def __anext__(self):
            if self.i >= len(script):
                raise StopAsyncIteration
            delay, line = script[self.i]
            self.i += 1
            await asyncio.sleep(delay)
            return line

    class Resp:
        status = 200
        content = Content()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return ""

    class Session:
        def post(self, *a, **k):
            return Resp()

        async def close(self):
            pass

    mod = types.ModuleType("aiohttp")
    mod.ClientSession = lambda *a, **k: Session()
    mod.ClientTimeout = lambda **k: None
    mod.ClientError = type("ClientError", (Exception,), {})
    return mod


def drain(script, stall=0.25, overall=5, prompt_tokens=None, rate=None, margin=None):
    """Run chat_stream against `script`; return (lines, exception or None)."""
    sys.modules["aiohttp"] = fake_aiohttp(script)
    old_stall, old_total = config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT
    old_rate, old_margin = config.EXO_FIRST_TOKEN_RATE_TPS, config.EXO_FIRST_TOKEN_MARGIN_S
    config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT = stall, overall
    if rate is not None:
        config.EXO_FIRST_TOKEN_RATE_TPS = rate
    if margin is not None:
        config.EXO_FIRST_TOKEN_MARGIN_S = margin
    lines, err = [], None

    async def go():
        nonlocal err
        try:
            async for line in ExoClient().chat_stream(
                "m", [{"role": "user", "content": "x"}], prompt_tokens=prompt_tokens
            ):
                lines.append(line)
        except Exception as e:  # noqa: BLE001 - the test is about which one
            err = e

    try:
        asyncio.run(go())
    finally:
        config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT = old_stall, old_total
        config.EXO_FIRST_TOKEN_RATE_TPS = old_rate
        config.EXO_FIRST_TOKEN_MARGIN_S = old_margin
        sys.modules.pop("aiohttp", None)
    return lines, err


def main():
    print("\n  exo stall abort\n")

    lines, err = drain([(0.01, b"data: a\n"), (0.01, b"data: b\n"), (0.01, b"data: [DONE]\n")])
    check("a steady stream finishes untouched",
          err is None and len([x for x in lines if x]) == 3, f"{err!r} {lines}")

    lines, err = drain([(0.01, b"data: a\n"), (0.01, b"data: b\n"), (1.0, b"data: c\n")])
    check("tokens then silence past the limit -> ExoStalled",
          isinstance(err, ExoStalled), f"got {err!r}")
    check("...and it counts the tokens seen before the stall",
          isinstance(err, ExoStalled) and err.tokens == 2, f"tokens={getattr(err,'tokens',None)}")
    check("...and the caller is told how long it was silent",
          isinstance(err, ExoStalled) and err.silent_for >= 0.2,
          f"silent_for={getattr(err,'silent_for',None)}")
    check("...and the message says stuck, not slow",
          isinstance(err, ExoStalled) and "stuck, not slow" in str(err), str(err)[:160])

    # Prefill: exo sends nothing at all until the first token. A long silence
    # here must NOT trip the stall rule, or every large prompt dies.
    lines, err = drain([(0.6, b"data: first\n"), (0.01, b"data: [DONE]\n")], stall=0.25, overall=5)
    check("silence BEFORE the first token is prefill, not a stall",
          err is None and len([x for x in lines if x]) == 2, f"{err!r} {lines}")

    # Blank lines are SSE terminators, not progress.
    lines, err = drain([(0.01, b"\n"), (0.01, b"\n"), (1.0, b"data: a\n")], stall=0.25, overall=5)
    check("blank lines alone do not arm the stall clock",
          err is None or not isinstance(err, ExoStalled), f"got {err!r}")

    # No token ever, past the whole-request budget: that is the old timeout,
    # and it must stay an ExoRequestFailed rather than becoming a stall.
    lines, err = drain([(1.0, b"data: a\n")], stall=5, overall=0.25)
    check("no answer at all -> the request timeout, not a stall",
          isinstance(err, ExoRequestFailed) and not isinstance(err, ExoStalled), f"got {err!r}")

    # --- the first-token deadline -------------------------------------------
    # EXO_STALL_TIMEOUT cannot bound the silence BEFORE the first token, and that
    # is where the 2026-09-27 wedge sat: prefill finished, decode never started,
    # no token ever, so the inter-token rule never armed and the pool was shut for
    # 28 minutes. The deadline has to scale with the prompt or it cuts real
    # prefills. #830.

    check("the formula is tokens/rate + margin, admin's numbers",
          round(config.exo_first_token_deadline(1463), 1) == 69.8
          and round(config.exo_first_token_deadline(12288), 1) == 141.9,
          f"1463 -> {config.exo_first_token_deadline(1463)}, "
          f"12288 -> {config.exo_first_token_deadline(12288)}")

    check("an unknown prompt size falls back to the whole-request budget",
          config.exo_first_token_deadline(None) == float(config.EXO_GENERATE_TIMEOUT)
          and config.exo_first_token_deadline(0) == float(config.EXO_GENERATE_TIMEOUT),
          "a guessed deadline is worse than the behaviour it replaces")

    check("the deadline can never exceed the whole-request budget",
          config.exo_first_token_deadline(10 ** 7) == float(config.EXO_GENERATE_TIMEOUT),
          "a deadline longer than the request budget could not fire")

    # 100 tokens at rate 1000/s + 0.2s margin = 0.3s; the pool sends nothing.
    lines, err = drain([(2.0, b"data: a\n")], stall=5, overall=5,
                       prompt_tokens=100, rate=1000, margin=0.2)
    check("prefill that never produces a first token -> ExoStalled",
          isinstance(err, ExoStalled), f"got {err!r}")
    check("...named as the first_token deadline, not the inter-token one",
          isinstance(err, ExoStalled) and err.phase == "first_token",
          f"phase={getattr(err, 'phase', None)}")
    check("...and it reports zero tokens seen",
          isinstance(err, ExoStalled) and err.tokens == 0,
          f"tokens={getattr(err, 'tokens', None)}")
    check("...and the message names the formula, so a log says why",
          isinstance(err, ExoStalled) and "first_token deadline" in str(err)
          and "100-token prompt" in str(err), str(err)[:200])

    # The same prompt, answering inside the deadline: must not be cut.
    lines, err = drain([(0.05, b"data: a\n"), (0.05, b"data: [DONE]\n")],
                       stall=5, overall=5, prompt_tokens=100, rate=1000, margin=0.2)
    check("a prefill that answers inside its deadline is untouched",
          err is None and len([x for x in lines if x]) == 2, f"{err!r} {lines}")

    # A long prefill with a generous deadline: the deadline must scale, not bite.
    lines, err = drain([(0.4, b"data: a\n"), (0.01, b"data: [DONE]\n")],
                       stall=5, overall=5, prompt_tokens=6000, rate=1000, margin=0.2)
    check("a larger prompt earns a longer deadline (6000/1000+0.2 = 6.2s)",
          err is None, f"cut a prefill that was inside its scaled deadline: {err!r}")

    # Without a token count the old behaviour must be preserved exactly.
    lines, err = drain([(1.0, b"data: a\n")], stall=5, overall=0.25)
    check("no token count -> still the request timeout, not a stall",
          isinstance(err, ExoRequestFailed) and not isinstance(err, ExoStalled),
          f"got {err!r}")

    check("the inter-token stall still names its own phase",
          isinstance(drain([(0.01, b"data: a\n"), (1.0, b"data: b\n")])[1], ExoStalled)
          and drain([(0.01, b"data: a\n"), (1.0, b"data: b\n")])[1].phase == "inter_token",
          "phase must distinguish the two deadlines")

    # --- the whole-request timeout must not wear the stall's name -----------
    # 2026-09-27, in production: "Pool STALLED mid-stream ... 153 chunks then 6s
    # of silence", logged against a 60s stall budget that 6s cannot have tripped.
    # It fired at exactly 1500s -- aiohttp's session `total`, not the per-line
    # wait. One `except asyncio.TimeoutError` cannot tell the two apart, so the
    # 504 claimed a stall nobody measured, and no log line could be trusted to
    # mean the inter-token rule had ever fired.

    # Chunks arriving steadily, generous per-line budget, short overall budget:
    # the request runs out of time while the stream is healthy.
    lines, err = drain([(0.1, b"data: a\n")] * 10, stall=5, overall=0.35)
    check("request budget expiring mid-stream -> ExoRequestFailed, not a stall",
          isinstance(err, ExoRequestFailed) and not isinstance(err, ExoStalled),
          f"got {err!r} -- a 1500s timeout must not be reported as a stall")
    check("...and it says how many chunks arrived, without claiming silence",
          isinstance(err, ExoRequestFailed) and "chunks received" in str(err)
          and "silence" not in str(err), str(err)[:160])

    # And the per-line rule must still fire when IT is the binding budget.
    lines, err = drain([(0.01, b"data: a\n"), (1.0, b"data: b\n")],
                       stall=0.25, overall=30)
    check("the inter-token rule still fires when it is the tighter budget",
          isinstance(err, ExoStalled) and err.phase == "inter_token",
          f"got {err!r}")

    # --- the wiring itself --------------------------------------------------
    # The deadline was implemented before the prompt counter reached main, so
    # `prompt_tokens` was None and the whole mechanism was inert for a day with
    # nothing in a log or a test saying so. chat_collect is what server.py calls
    # on the non-streaming path; if it drops the argument the feature is off.
    def _collect(script, prompt_tokens, stall=5, overall=5, rate=1000, margin=0.2):
        sys.modules["aiohttp"] = fake_aiohttp(script)
        o = (config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT,
             config.EXO_FIRST_TOKEN_RATE_TPS, config.EXO_FIRST_TOKEN_MARGIN_S)
        (config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT,
         config.EXO_FIRST_TOKEN_RATE_TPS, config.EXO_FIRST_TOKEN_MARGIN_S) = (
            stall, overall, rate, margin)
        err = None
        try:
            asyncio.run(ExoClient().chat_collect(
                "m", [{"role": "user", "content": "x"}], prompt_tokens=prompt_tokens))
        except Exception as e:  # noqa: BLE001
            err = e
        finally:
            (config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT,
             config.EXO_FIRST_TOKEN_RATE_TPS, config.EXO_FIRST_TOKEN_MARGIN_S) = o
            sys.modules.pop("aiohttp", None)
        return err

    err = _collect([(2.0, b"data: a\n")], prompt_tokens=100)
    check("chat_collect forwards prompt_tokens, so the deadline is armed",
          isinstance(err, ExoStalled) and err.phase == "first_token",
          f"got {err!r} — if this is not a first_token stall the wiring is dropped")

    err = _collect([(2.0, b"data: a\n")], prompt_tokens=None, overall=0.3)
    check("...and without it the old whole-request budget still governs",
          isinstance(err, ExoRequestFailed) and not isinstance(err, ExoStalled),
          f"got {err!r}")

    print(f"\n  {'ALL PASSED' if not FAIL else 'FAILED'}: "
          f"exo stall abort {PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
