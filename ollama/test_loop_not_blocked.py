#!/usr/bin/env python3
"""A non-streaming generation must not run on the event loop (#959).

THE FAILURE THIS PINS. `OllamaClient.chat` is a blocking generator and
`chat_completions` is `async def`. Iterating it in the handler ran the whole
generation on the event loop, so ONE non-streaming request took the entire
gateway dark for its length — every route, `/health` included. Measured on
slice 2026-10-11 22:37Z with one `max_tokens:2` request in flight: four
`/warm` and four `/health` probes, all `http=000` on a 6s timeout, from a
process that had answered `/health` in 0.07s thirty seconds earlier.

It presents as a service that is alive and cannot be reached, and it DELAYS
its own restart: uvicorn's signal handler is a loop callback too, so a
`kill -TERM` sent during a blocked generation is not processed while the
generation runs. Measured: 24 `/health` probes over 120s all `http=000` with
the process still up. It is a delay and not a veto — corrected from an earlier
claim of mine that SIGTERM was a no-op cleared only by SIGKILL. Admin's read
of the box shows the graceful shutdown ran at 22:41:34Z once the loop freed,
and launchd's KeepAlive did bring the service back. The registry meanwhile
reads `online` throughout, because the health reporter is a loop task as well:
it does not report a failure, it stops reporting at all.

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

    def claim(self, name):
        """Records WHERE it is taken, which is half the point of the fix."""
        import threading
        self.claims.append(("enter", threading.current_thread().name))
        mm = self.models.get(name)
        if mm is None:
            return None
        mm.in_flight += 1
        return mm

    def release(self, name):
        import threading
        mm = self.models.get(name)
        if mm is None:
            return
        mm.in_flight -= 1
        self.claims.append(("exit", threading.current_thread().name))

    @contextmanager
    def serving(self, name):
        if self.claim(name) is None:
            yield
            return
        try:
            yield
        finally:
            self.release(name)


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
        pass

    # ---------------------------------------------------------------
    # A cancelled caller must not make the model look idle (codex, #959).
    #
    # Cancelling an `asyncio.to_thread` await does not stop the worker. The
    # first version of this fix scoped the claim to a `with` block around the
    # await, so a client that hung up mid-generation dropped `in_flight` to 0
    # while a thread was still driving the model — `check_cooldowns` would then
    # be free to unload and release the lease under live work. That is #907
    # reopened through the door #959's fix opened.
    fake2 = FakeMgr()
    server.mgr = fake2

    async def _resolve2(name):
        return fake2.models[FakeModel.name]
    server._resolve_model = _resolve2
    try:
        req2 = server.ChatRequest(
            model="qwen3.8:27b-mlx",
            messages=[{"role": "user", "content": "say OK"}],
            max_tokens=2, stream=False,
        )

        async def cancel_midway():
            task = asyncio.create_task(server.chat_completions(req2))
            await asyncio.sleep(BLOCK_S / 5)          # worker is running
            task.cancel()
            t0 = time.monotonic()
            try:
                await task
                raised = "nothing"
            except asyncio.CancelledError:
                raised = "CancelledError"
            unwind = time.monotonic() - t0
            during = fake2.models[FakeModel.name].in_flight
            # Now let the worker actually finish.
            await asyncio.sleep(BLOCK_S * 1.5)
            return raised, unwind, during, fake2.models[FakeModel.name].in_flight

        raised, unwind, during, after = asyncio.run(cancel_midway())
        print(f"  cancelled caller: raised {raised} in {unwind:.3f}s, "
              f"in_flight during={during} after={after}")
        check(raised == "CancelledError",
              "the handler still cancels promptly for the caller")
        check(unwind < BLOCK_S / 2,
              f"and does not wait out the worker to do it ({unwind:.3f}s)")
        check(during == 1,
              f"the claim SURVIVES the cancellation while the worker runs "
              f"(in_flight={during})")
        check(after == 0,
              f"and is released when the worker actually ends "
              f"(in_flight={after})")
        threads = {t for _, t in fake2.claims}
        check(threads == {"MainThread"},
              f"both ends still on the loop's thread ({threads})")
    finally:
        server.mgr, server._resolve_model = real_mgr, real_resolve
        prompt_size.count = real_count

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
