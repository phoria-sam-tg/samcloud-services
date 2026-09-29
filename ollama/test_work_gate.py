#!/usr/bin/env python3
"""Work wins: the gate, the signals that feed it, and what it must not break.

`offering()` (#861, PR #27) stops ADVERTISING when another tenant is using the
device. It cannot stop a caller who asks anyway — `POST /models/load` names a
model directly — so a node would withdraw its offer and then load onto the
render for the next caller who ignored it. This is the gate.

Loads nothing, leases nothing, talks to no network. Every reading is chosen
here, because the states being pinned are ones a live box reaches only when
somebody else happens to be rendering.

    python -m ollama.test_work_gate
"""

import os
import sys
import time

os.environ["AUTH_ENABLED"] = "0"
os.environ.setdefault("SC_TOKEN", "test")

from fastapi.testclient import TestClient

from . import capacity, config, manager as manager_mod, server
from .manager import Backend, ManagedModel, ModelManager
from .samcloud import SamcloudClient

# ada, measured 2026-09-29. Idle device memory, and one continuous 46 GB
# Unreal render. The board is 49,140 MiB.
BOARD = 49140
ADA_IDLE_INUSE = 3118          # median of 31 samples, spread 116 MB
ADA_FLOOR = 5515               # the higher of the two idle readings, as set on ada
RENDER_INUSE = 46331           # UnrealEditor pid 15240, from the Windows side

failures, checks = [], 0


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def reading(device_inuse, available=None):
    return {
        "memory_total_mb": BOARD,
        "memory_used_mb": device_inuse,
        "memory_available_mb": BOARD - device_inuse if available is None else available,
        "memory_device_inuse_mb": device_inuse,
        "compute_pct": 0.0,
        "load_avg_1m": 0.0,
    }


class Ada:
    """A manager configured as ada is: CUDA, work gate on, nothing else."""

    def __init__(self, device_inuse=ADA_IDLE_INUSE, floor=ADA_FLOOR):
        self.next = reading(device_inuse)
        self.floor = floor

    def __enter__(self):
        self._collect = capacity.collect
        self._floor = config.FOREIGN_IDLE_MB
        self._interval = config.OFFER_MIN_SAMPLE_INTERVAL_S
        self._installed = manager_mod.hf_model_installed
        capacity.collect = lambda: dict(self.next)
        config.FOREIGN_IDLE_MB = self.floor
        config.OFFER_MIN_SAMPLE_INTERVAL_S = 0.0
        manager_mod.hf_model_installed = lambda repo_id: False
        mgr = ModelManager(sc=SamcloudClient(token="test"))
        mgr.ollama.list_models = lambda: [
            {"name": "qwen3:1.7b", "size": 2000 * 1024 * 1024}]
        mgr.ollama.memory_estimate_mb = lambda n: 2000
        mgr.ollama.load_model = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a model was LOADED while another tenant was working"))
        mgr.llama.available_models = lambda: []
        mgr.sc.resource_dashboard = lambda: []
        mgr.sc.list_leases = lambda **kw: []
        self.mgr = mgr
        return self

    def set(self, device_inuse):
        self.next = reading(device_inuse)
        self.mgr._readings.clear()

    def __exit__(self, *a):
        capacity.collect = self._collect
        config.FOREIGN_IDLE_MB = self._floor
        config.OFFER_MIN_SAMPLE_INTERVAL_S = self._interval
        manager_mod.hf_model_installed = self._installed


def resident(mgr, name="qwen3:1.7b", mb=2000, backend=Backend.OLLAMA, lease="lease_x"):
    now = time.time()
    mm = ManagedModel(name=name, backend=backend, memory_mb=mb, lease_id=lease,
                      port=11434, loaded_at=now, last_used=now)
    mgr.models[name] = mm
    return mm


def main():
    step(1, "a new load is refused while another tenant is working")
    with Ada(device_inuse=RENDER_INUSE) as a:
        try:
            a.mgr.load_ollama_model("qwen3:1.7b")
            check(False, "the load was refused")
        except capacity.DeviceInUse as e:
            check(True, "raised DeviceInUse, not InsufficientCapacity")
            check(e.foreign_mb == RENDER_INUSE,
                  f"carrying what is held by someone else ({e.foreign_mb})")
            check(e.idle_floor_mb == ADA_FLOOR and e.margin_mb is not None,
                  f"and the floor it was judged against ({e.idle_floor_mb})")
            check(e.as_dict()["retry_after_s"] is None,
                  "with NO retry hint — a render ends when a person is done")
            check(e.as_dict()["error"] == "device_in_use",
                  "and a distinct error name, not conflated with the other two")
            print(f"  {e}")

    step(2, "and it is refused even when it would comfortably fit")
    # The whole reason the gate is separate from the fit check. At a render's
    # own peak free memory (2963 MiB of ada's 53 samples) `usable()` offers
    # 1939 MB — enough for a small model, loaded into a gap inside somebody's
    # render, logged as a success.
    with Ada() as a:
        a.next = reading(RENDER_INUSE, available=2963)
        fits = capacity.fits(2000, 2963)
        print(f"  free 2963 -> usable {capacity.usable_mb(2963)}, fits(2000)={fits}")
        try:
            a.mgr.load_ollama_model("qwen3:1.7b")
            check(False, "refused despite fitting")
        except capacity.DeviceInUse:
            check(True, "refused despite the fit check being satisfiable")
        except capacity.InsufficientCapacity:
            check(False, "refused on FIT — the gate did not run first")

    step(3, "a resident model keeps serving; the gate is on NEW loads only")
    with Ada(device_inuse=RENDER_INUSE) as a:
        mm = resident(a.mgr)
        before = mm.request_count
        got = a.mgr.load_ollama_model("qwen3:1.7b")
        check(got is mm, "an already-resident model is handed back")
        check(mm.request_count == before + 1, "and its request is counted")
        # Cutting an in-flight request is admin's policy but needs a measured
        # grace period — that is C2. Refusing a request to a model that is
        # ALREADY loaded would strand a caller for no gain: the memory is
        # already spent.

    step(4, "an idle model is evicted at once when work appears")
    with Ada(device_inuse=RENDER_INUSE) as a:
        mm = resident(a.mgr)
        mm.last_used = time.time()          # idle for zero seconds
        freed = []
        a.mgr.unload = lambda name, force=False: freed.append(name) or {"status": "ok"}
        out = a.mgr.check_cooldowns()
        print(f"  cooldown={config.COOLDOWN_SECONDS}s, idle=0s -> unloaded {freed}")
        check(freed == ["qwen3:1.7b"],
              f"unloaded without waiting out the cooldown ({freed})")

    step(5, "...but not one that is mid-request")
    with Ada(device_inuse=RENDER_INUSE) as a:
        mm = resident(a.mgr)
        mm.in_flight = 1
        freed = []
        a.mgr.unload = lambda name, force=False: freed.append(name) or {"status": "ok"}
        a.mgr.check_cooldowns()
        check(freed == [], f"a request in flight is not cut ({freed})")
        # C2 cuts it, after measuring the grace period admin asked for.

    step(6, "with no work, the cooldown is the ordinary one")
    with Ada() as a:                        # idle device
        mm = resident(a.mgr)
        mm.last_used = time.time()
        freed = []
        a.mgr.unload = lambda name, force=False: freed.append(name) or {"status": "ok"}
        a.mgr.check_cooldowns()
        check(freed == [], f"a freshly-used model stays ({freed})")
        mm.last_used = time.time() - config.COOLDOWN_SECONDS - 1
        a.mgr.check_cooldowns()
        check(freed == ["qwen3:1.7b"], "and goes once genuinely idle")

    step(7, "a refused lease is a work signal; a broken one is not")
    # ada 404'd every lease request for months on a scope it never had. If
    # that counted, the node would read a permanent authentication fault as a
    # permanent render and never offer anything again — a failure that looks
    # exactly like caution.
    with Ada() as a:
        check(a.mgr.work_in_progress(0) is False, "idle box: nobody is working")
        a.mgr.note_lease_contention("error", "qwen3:1.7b")
        check(a.mgr.lease_contention_fresh() is False,
              "a 404 / timeout / dead registry is NOT evidence")
        check(a.mgr.work_in_progress(0) is False, "so the gate stays open")
        a.mgr.note_lease_contention("queued", "qwen3:1.7b")
        check(a.mgr.lease_contention_fresh() is True, "a QUEUED lease is")
        check(a.mgr.work_in_progress(0) is True,
              "and it alone closes the gate, with foreign reading 0")
        a.mgr._lease_contended_at = time.monotonic() - config.LEASE_CONTENTION_TTL_S - 1
        check(a.mgr.work_in_progress(0) is False,
              "stale evidence is not evidence")
        a.mgr.note_lease_contention("conflict", "qwen3:1.7b")
        check(a.mgr.work_in_progress(0) is True, "a 409 conflict counts too")

    step(8, "the gate is OFF on a box that has not measured its floor")
    with Ada(floor=None) as a:
        a.set(RENDER_INUSE)
        check(a.mgr.work_in_progress(RENDER_INUSE) is None,
              "None — not asked, and not False")
        a.mgr.note_lease_contention("queued")
        check(a.mgr.work_in_progress(RENDER_INUSE) is None,
              "the lease signal is off with it: one switch, not two")
        # The load must go through untouched: slice and wafer leave
        # FOREIGN_IDLE_MB unset and must behave exactly as they did.
        a.mgr.refuse_if_device_in_use("qwen3:1.7b")
        check(True, "and refuse_if_device_in_use is a no-op")

    step(9, "a lost lease changes lease state and NOT residency")
    # If marking a model unleased ever dropped it from `self.models`,
    # `own_device_mb()` would fall by its size while the memory is still
    # held, `foreign_mb` would rise by the same amount in the same instant,
    # and the gateway would read its own resident model as another tenant —
    # stepping back from itself, permanently, looking conservative
    # (claude-wafer-services).
    with Ada() as a:
        mm = resident(a.mgr, mb=20000)
        a.mgr.sc.release_lease = lambda lid: None
        a.mgr._request_lease = lambda name, mb: None      # renewal cannot get it back
        own_before = a.mgr.own_device_mb()
        a.mgr._renew_leases()
        own_after = a.mgr.own_device_mb()
        print(f"  own_mb {own_before} -> {own_after}, lease_id={mm.lease_id!r}, "
              f"lease_lost={mm.lease_lost}")
        check(mm.lease_id is None, "the lease is gone")
        check(mm.lease_lost is True, "and that is recorded as a LOSS, not absence")
        check("qwen3:1.7b" in a.mgr.models, "the model is still resident")
        check(own_after == own_before == 20000,
              f"and own_mb is unchanged ({own_before} -> {own_after})")
        # The consequence, stated as the number it protects: with the device
        # at 20000 (our model alone), foreign must read 0 and not 20000.
        check(capacity.foreign_mb(20000, own_after) == 0,
              "so the gateway does not read its own model as a foreign tenant")

    step(10, "over HTTP it is a 503 that names itself")
    with Ada(device_inuse=RENDER_INUSE) as a:
        server.mgr = a.mgr
        client = TestClient(server.app, raise_server_exceptions=False)
        r = client.post("/models/load", json={"model": "qwen3:1.7b", "backend": "ollama"})
        body = r.json().get("detail", {})
        print(f"  POST /models/load -> {r.status_code} {body}")
        check(r.status_code == 503, f"503, not 500 ({r.status_code})")
        check(body.get("error") == "device_in_use",
              f"named distinctly from insufficient_capacity ({body.get('error')})")
        check(body.get("foreign_mb") == RENDER_INUSE, "with the number behind it")
        check("Retry-After" not in r.headers,
              "and no Retry-After — the answer is 'another node', not 'wait'")
        # And the offer agrees with the gate: one source, one verdict.
        o = client.get("/warm").json()
        check(o["loadable"] == [], "/warm offers nothing at the same moment")
        check(o["capacity"]["work_in_progress"] is True, "and says why")

    step(11, "the floor is a backstop, and it is already outside its own data")
    # SC_MIN_HEADROOM_MB=3000 was derived as "above the maximum free memory
    # observed during a render" — 2,963 MiB, from claude-ada's 53 samples of
    # one continuous 46 GB Unreal session. C1's first hour on ada produced a
    # SECOND render shape sitting at 5,064 MB free, above that ceiling. So the
    # derivation is out of date within hours of being written, and this step
    # exists so nobody discovers that by reading a comment that is wrong.
    #
    # The conclusion is NOT to raise the floor (claude-wafer-services): free
    # memory during a render has now been seen from 188 to 5,064 MB, so no
    # ceiling on it separates render from idle. `foreign_mb` does.
    real_floor = capacity.MIN_HEADROOM_MB
    capacity.MIN_HEADROOM_MB = 3000                    # ada's value
    try:
        RENDER1_PEAK, RENDER2_LIVE = 2963, 5064
        u1, u2 = capacity.usable_mb(RENDER1_PEAK), capacity.usable_mb(RENDER2_LIVE)
        print(f"  usable(free={RENDER1_PEAK}) = {u1}   "
              f"usable(free={RENDER2_LIVE}) = {u2}")
        check(u1 == 0, f"zero at the render it was derived from ({u1})")
        # The honest assertion, not the hoped-for one.
        check(u2 > 0,
              f"NONZERO at the second render shape ({u2}MB) — the fit check "
              f"alone would offer that much into a live 44 GB render")
        check(u2 == 2064, f"reproducing ada's live figure exactly ({u2})")
    finally:
        capacity.MIN_HEADROOM_MB = real_floor

    # And what actually refuses, at both: the foreign term.
    with Ada(device_inuse=44076) as f:      # ada's live reading, same moment
        o = f.mgr.offering()
        cap = o["capacity"]
        print(f"  foreign={cap['foreign_mb']} work_in_progress={cap['work_in_progress']}")
        check(cap["work_in_progress"] is True,
              "the work gate refuses where the floor did not")
        check(o["loadable"] == [], f"nothing offered ({o['loadable']})")
        check(all(m["reason"] == "work_in_progress" for m in o["blocked"]),
              "and every model is blocked for THAT reason, not for fit")

    print(f"\n{'='*60}")
    print(f"  {checks} checks run")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
