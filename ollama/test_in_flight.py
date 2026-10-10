"""#907: a generation holds `in_flight`, so the cooldown cannot unload under it.

WHY THIS FILE EXISTS

`check_cooldowns` has always respected `in_flight` — the comment above it says
so. Until now the flag was claimed on the TRANSCRIPTION path only, and `touch()`
stamps `last_used` once BEFORE the work, so a long generation looked
progressively more idle while it ran. Measured on slice 2026-10-08: **18
unloads, every one 300-355s into an in-flight request**, each releasing the
capacity lease while Ollama kept the weights because its runner was busy. That
is #907's phantom `foreign_mb`.

It could not have fired before #904, and the coincidence is exact:

    COOLDOWN_SECONDS                  300
    the old aiohttp ClientTimeout     300

The timeout killed every Ollama request at the moment the cooldown first became
eligible. Removing it is what exposed this, which is why all 18 are dated the
day #904 shipped. (`claude-wafer-services`, #907.)

WHAT IS ASSERTED, AND HOW

The thing that failed is a LIFETIME, so every check here advances a clock and
runs the cooldown loop against a generation that has not finished. Nothing reads
source text: a file documenting its own history contains every string it ever
got wrong.

    python -m ollama.test_in_flight
"""
import asyncio
import os
import time

if __package__ in (None, ""):
    print("run as:  python -m ollama.test_in_flight")
    raise SystemExit(1)

os.environ["AUTH_ENABLED"] = "0"

from . import manager as mgr_mod  # noqa: E402

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


def section(name):
    print(f"\n{name}")


COOLDOWN = mgr_mod.COOLDOWN_SECONDS


class FakeManager:
    """The real `serving` and `check_cooldowns`, over a hand-built model table.

    Subclassing rather than mocking: `serving` and `check_cooldowns` are the code
    under test and must be the real ones. Everything they reach outward for —
    the offer reading, the foreign-work gate, the unload — is stubbed, because
    this is about a lifetime and not about memory.
    """

    def __init__(self, in_flight=0, last_used_ago=0.0, working=False):
        mm = mgr_mod.ManagedModel.__new__(mgr_mod.ManagedModel)
        mm.name = "qwen3.8:27b-mlx"
        mm.backend = mgr_mod.Backend.OLLAMA
        mm.managed = True
        mm.in_flight = in_flight
        mm.last_used = time.time() - last_used_ago
        mm.request_count = 0
        mm.memory_mb = 17333
        mm.lease_id = "lease_test"
        mm.loaded_at = time.time() - last_used_ago
        mm.port = None
        mm.tier = None
        self.models = {mm.name: mm}
        self.unloaded = []
        self._working = working

    # --- the real methods under test
    serving = mgr_mod.ModelManager.serving
    check_cooldowns = mgr_mod.ModelManager.check_cooldowns
    touch = mgr_mod.ModelManager.touch

    # --- stubs for everything those two reach outward for
    def offer_reading(self):
        return {"memory_device_inuse_mb": 20000}

    def own_device_mb(self):
        return 17530

    def work_in_progress(self, foreign_mb):
        return self._working

    def unload(self, name):
        self.unloaded.append(name)
        self.models.pop(name, None)
        return {"model": name, "unloaded": True}


section("1. the bug, reproduced: an idle-looking model IS unloaded")

m = FakeManager(in_flight=0, last_used_ago=COOLDOWN + 55)
m.check_cooldowns()
check(m.unloaded == ["qwen3.8:27b-mlx"],
      f"in_flight=0 and last_used {COOLDOWN + 55:.0f}s ago -> unloaded "
      f"(this is what happened 18 times)")

section("2. the fix: a generation in flight is not unloaded")

m = FakeManager(in_flight=0, last_used_ago=COOLDOWN + 55)
with m.serving("qwen3.8:27b-mlx"):
    check(m.models["qwen3.8:27b-mlx"].in_flight == 1, "serving() claims in_flight")
    m.check_cooldowns()
    check(m.unloaded == [], "cooldown runs and leaves it alone")
check(m.models["qwen3.8:27b-mlx"].in_flight == 0, "released on exit")

section("3. released even when the generation raises")

m = FakeManager()
try:
    with m.serving("qwen3.8:27b-mlx"):
        raise RuntimeError("the upstream died mid-prefill")
except RuntimeError:
    pass
check(m.models["qwen3.8:27b-mlx"].in_flight == 0,
      "a raising generation does not leak the claim")

section("4. last_used is RE-STAMPED at release, so it is not eligible immediately")

# The half that is not about in_flight: a 549s prefill leaves last_used 549s
# stale the instant it finishes, so the model is eligible on the very next tick.
m = FakeManager(last_used_ago=COOLDOWN + 249)
with m.serving("qwen3.8:27b-mlx"):
    pass
m.check_cooldowns()
check(m.unloaded == [],
      f"a request that just finished after {COOLDOWN + 249:.0f}s survives the next tick")
check(m.models["qwen3.8:27b-mlx"].request_count == 0,
      "and request_count is NOT touched — that is touch()'s job, not this one")

section("5. nesting, because NUM_PARALLEL is not forever 1")

m = FakeManager(last_used_ago=COOLDOWN + 55)
with m.serving("qwen3.8:27b-mlx"):
    with m.serving("qwen3.8:27b-mlx"):
        check(m.models["qwen3.8:27b-mlx"].in_flight == 2, "two claims count to 2")
    check(m.models["qwen3.8:27b-mlx"].in_flight == 1, "inner release leaves one")
    m.check_cooldowns()
    check(m.unloaded == [], "still protected by the outer claim")
check(m.models["qwen3.8:27b-mlx"].in_flight == 0, "both released")

section("6. an unknown model does not raise and does not invent an entry")

m = FakeManager()
with m.serving("no-such-model"):
    pass
check("no-such-model" not in m.models, "no entry created for a name we do not hold")

section("7. foreign work still wins — the claim is not a veto on policy")

# Under foreign work the cooldown collapses to zero, but `in_flight` still
# protects a request mid-flight: that is admin's recorded policy (work wins,
# but cutting a live request needs a measured grace period, which is C2).
m = FakeManager(last_used_ago=1.0, working=True)
m.check_cooldowns()
check(m.unloaded == ["qwen3.8:27b-mlx"],
      "idle 1s but foreign work present -> unloaded (cooldown collapses to 0)")

m = FakeManager(last_used_ago=1.0, working=True)
with m.serving("qwen3.8:27b-mlx"):
    m.check_cooldowns()
    check(m.unloaded == [], "same, but in flight -> survives this tick")

section("8. the wrappers hold the claim across ITERATION, not the call")

from . import server  # noqa: E402

real_mgr = server.mgr
server.mgr = FakeManager(last_used_ago=COOLDOWN + 55)
try:
    async def drive():
        seen = []

        async def agen():
            # The claim must be held HERE, mid-generation — the handler has
            # already returned by this point in the real path.
            seen.append(server.mgr.models["qwen3.8:27b-mlx"].in_flight)
            server.mgr.check_cooldowns()
            seen.append(list(server.mgr.unloaded))
            yield "chunk"

        out = [x async for x in server._held_async("qwen3.8:27b-mlx", agen())]
        return seen, out

    seen, out = asyncio.run(drive())
    check(seen and seen[0] == 1, "_held_async: claimed during iteration")
    check(len(seen) > 1 and seen[1] == [],
          "_held_async: cooldown ran mid-stream and unloaded nothing")
    check(out == ["chunk"], "_held_async: the chunks still come through")
    check(server.mgr.models["qwen3.8:27b-mlx"].in_flight == 0,
          "_held_async: released when the stream ends")

    server.mgr = FakeManager(last_used_ago=COOLDOWN + 55)

    def sgen():
        check(server.mgr.models["qwen3.8:27b-mlx"].in_flight == 1,
              "_held_sync: claimed during iteration")
        server.mgr.check_cooldowns()
        check(server.mgr.unloaded == [],
              "_held_sync: cooldown ran mid-collection and unloaded nothing")
        yield {"done": True}

    got = list(server._held_sync("qwen3.8:27b-mlx", sgen()))
    check(got == [{"done": True}], "_held_sync: the chunks still come through")
    check(server.mgr.models["qwen3.8:27b-mlx"].in_flight == 0,
          "_held_sync: released when the loop ends")
finally:
    server.mgr = real_mgr

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
