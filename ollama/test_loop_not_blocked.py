#!/usr/bin/env python3
"""A non-streaming generation must not run on the event loop (#959).

THE FAILURE THIS PINS. `OllamaClient.chat` is a blocking generator and
`chat_completions` is `async def`. Iterating it in the handler ran the whole
generation on the event loop, so ONE non-streaming request took the entire
gateway dark for its length — every route, `/health` included. Measured on
slice 2026-10-11 22:37Z with one `max_tokens:2` request in flight: four
`/warm` and four `/health` probes, all `http=000` on a 6s timeout, from a
process that had answered `/health` in 0.07s thirty seconds earlier.

It presents as a dead service that cannot be restarted. uvicorn's SIGTERM
handler is a loop callback too, so `kill -TERM` is a no-op for the length of
the generation (measured: survived 120s, cleared by SIGKILL) and launchd's
KeepAlive never fires. The registry meanwhile reads `online`, because the
health reporter is also a loop task: it does not report a failure, it stops
reporting at all.

WHY A TICK COUNT. The assertion cannot be "the handler returns quickly" — it
must not, the generation is genuinely slow. What distinguishes a fix from the
bug is whether OTHER coroutines run while it is slow. So this counts how many
times the loop got control during one deliberately slow generation: many with
the fix, zero or one without it. A `/health` probe through TestClient would
not do it — TestClient drives its own loop and serialises the two calls.

    python -m ollama.test_loop_not_blocked
"""

import asyncio
import os
import sys
import time
from contextlib import contextmanager

if __package__ is None:
    print("run as:  python -m ollama.test_loop_not_blocked")
    raise SystemExit(1)

os.environ["AUTH_ENABLED"] = "0"
os.environ.setdefault("SC_TOKEN", "test")

from . import prompt_size, server                       # noqa: E402
from .manager import Backend                            # noqa: E402

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


# How long the stubbed backend blocks for. Long enough that a loop with any
# freedom at all ticks many times, short enough to run in a suite.
BLOCK_S = 1.0
TICK_S = 0.02


class FakeModel:
    name = "qwen3.8:27b-mlx"
    backend = Backend.OLLAMA
    context_length = 262144
    in_flight = 0


class FakeOllama:
    def __init__(self):
        self.calls = 0

    def chat(self, model, messages, **kwargs):
        """A blocking generator, which is what the real client is."""
        self.calls += 1
        time.sleep(BLOCK_S)          # the prefill, compressed
        yield {"message": {"content": "OK"},
               "done": True, "prompt_eval_count": 19, "eval_count": 2}


class FakeMgr:
    def __init__(self):
        self.ollama = FakeOllama()
        self.models = {FakeModel.name: FakeModel()}
        self.claims = []             # (event, in_flight, thread name)

    @contextmanager
    def serving(self, name):
        """Records WHERE it is entered, which is half the point of the fix."""
        import threading
        self.claims.append(("enter", threading.current_thread().name))
        self.models[name].in_flight += 1
        try:
            yield
        finally:
            self.models[name].in_flight -= 1
            self.claims.append(("exit", threading.current_thread().name))


async def drive(req):
    """Run the handler and count loop ticks while it is in flight."""
    task = asyncio.create_task(server.chat_completions(req))
    ticks, t0 = 0, time.monotonic()
    while not task.done():
        await asyncio.sleep(TICK_S)
        ticks += 1
        if time.monotonic() - t0 > BLOCK_S * 20:
            task.cancel()
            raise AssertionError("handler never finished")
    return await task, ticks, time.monotonic() - t0


def main():
    fake = FakeMgr()
    real_mgr, real_resolve, real_count = server.mgr, server._resolve_model, prompt_size.count
    server.mgr = fake

    async def _resolve(name):
        return fake.models[FakeModel.name]
    server._resolve_model = _resolve
    prompt_size.count = lambda *a, **kw: (19, "stub")

    try:
        req = server.ChatRequest(
            model="qwen3.8:27b-mlx",
            messages=[{"role": "user", "content": "say OK"}],
            max_tokens=2, stream=False,
        )
        body, ticks, elapsed = asyncio.run(drive(req))
        print(f"  one {BLOCK_S}s non-streaming generation: "
              f"{ticks} loop ticks in {elapsed:.2f}s")

        # The assertion. Without the fix the handler runs to completion inside
        # the first scheduling slot and this is 0 or 1.
        check(ticks >= 10,
              f"the loop kept control during the generation ({ticks} ticks, "
              f"floor 10)")
        check(elapsed >= BLOCK_S,
              f"and the generation really did take its time ({elapsed:.2f}s) — "
              f"a fast stub would pass the tick test for the wrong reason")
        check(fake.ollama.calls == 1, "the backend was called exactly once")
        check(body["choices"][0]["message"]["content"] == "OK",
              "and the reply still comes back assembled")
        check(body["usage"]["total_tokens"] == 21,
              f"with usage carried through ({body['usage']})")

        # The claim has to be taken and released ON THE LOOP: `in_flight` is
        # mutated from the event loop alone, and `check_cooldowns` reads it
        # there. A claim entered inside the worker is #907 reopened.
        events = [e for e, _ in fake.claims]
        threads = {t for _, t in fake.claims}
        check(events == ["enter", "exit"],
              f"the claim was taken once and released once ({events})")
        check(threads == {"MainThread"},
              f"both on the event loop's thread, not the worker ({threads})")
        check(fake.models[FakeModel.name].in_flight == 0,
              "and nothing is left claimed")
    finally:
        server.mgr, server._resolve_model = real_mgr, real_resolve
        prompt_size.count = real_count

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
