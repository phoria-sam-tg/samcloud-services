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


def drain(script, stall=0.25, overall=5):
    """Run chat_stream against `script`; return (lines, exception or None)."""
    sys.modules["aiohttp"] = fake_aiohttp(script)
    old_stall, old_total = config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT
    config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT = stall, overall
    lines, err = [], None

    async def go():
        nonlocal err
        try:
            async for line in ExoClient().chat_stream("m", [{"role": "user", "content": "x"}]):
                lines.append(line)
        except Exception as e:  # noqa: BLE001 - the test is about which one
            err = e

    try:
        asyncio.run(go())
    finally:
        config.EXO_STALL_TIMEOUT, config.EXO_GENERATE_TIMEOUT = old_stall, old_total
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

    print(f"\n  {'ALL PASSED' if not FAIL else 'FAILED'}: "
          f"exo stall abort {PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
