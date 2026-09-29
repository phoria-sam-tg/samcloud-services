#!/usr/bin/env python3
"""The lease says what the model actually occupies, not what we guessed.

A lease is requested from `memory_estimate_mb` — `max(1024, disk_mb)` —
before the weights are resident and before their context exists. The registry
publishes that number as what this box has claimed and computes
`available_memory_mb` from it. Measured on wafer 2026-09-29
(claude-wafer-services), one resident model:

    qwen3:1.7b   lease reserves 1296 MB   actually occupies 3354 MB

2.6x, on the number every other tenant reads. Under-reserving costs us
nothing — nothing enforces a lease — it costs whoever reads the registry next
and concludes there is room.

Loads nothing, leases nothing, talks to no network.

    python -m ollama.test_lease_reconcile
"""

import os
import sys
import time

os.environ["AUTH_ENABLED"] = "0"
os.environ.setdefault("SC_TOKEN", "test")

from . import capacity, config, manager as manager_mod
from .manager import Backend, ModelManager
from .samcloud import SamcloudClient

ESTIMATE, ACTUAL = 1296, 3354       # wafer's measured pair

failures, checks = [], 0


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


class Box:
    """A manager that loads `qwen3:1.7b` without touching anything real."""

    def __init__(self, grant=True, refusal="queued", floor=None):
        self.grant, self.refusal, self.floor = grant, refusal, floor
        self.leased = []          # (memory_mb) of every lease REQUESTED
        self.released = []        # every lease id released
        self.unloaded = []
        # Ordered log of both, because the ORDER is the property under test:
        # acquire-before-release is what stops the registry passing through 0.
        self.events = []

    def __enter__(self):
        self._collect = capacity.collect
        self._floor = config.FOREIGN_IDLE_MB
        self._interval = config.OFFER_MIN_SAMPLE_INTERVAL_S
        self._installed = manager_mod.hf_model_installed
        config.FOREIGN_IDLE_MB = self.floor
        config.OFFER_MIN_SAMPLE_INTERVAL_S = 0.0
        manager_mod.hf_model_installed = lambda repo_id: False
        capacity.collect = lambda: {
            "memory_total_mb": 65536, "memory_used_mb": 5536,
            "memory_available_mb": 60000, "memory_device_inuse_mb": 500,
            "compute_pct": 0.0, "load_avg_1m": 0.0,
        }
        mgr = ModelManager(sc=SamcloudClient(token="test"))
        mgr.ollama.list_models = lambda: [
            {"name": "qwen3:1.7b", "size": ESTIMATE * 1024 * 1024}]
        mgr.ollama.memory_estimate_mb = lambda n: ESTIMATE
        mgr.ollama.load_model = lambda *a, **k: None
        # What `ollama ps` reports once it is really resident.
        mgr.ollama.list_running = lambda: [
            {"name": "qwen3:1.7b", "size": ACTUAL * 1024 * 1024}]
        mgr.llama.available_models = lambda: []
        mgr.unload = lambda name, force=False: self.unloaded.append(name)

        def _request_lease(model_name, memory_mb):
            self.leased.append(memory_mb)
            self.events.append(("request", memory_mb))
            # The first lease is always granted; the RECONCILE is what this
            # test varies, so `grant` governs any request after the first.
            if len(self.leased) == 1 or self.grant:
                return f"lease_{len(self.leased)}"
            mgr.note_lease_contention(self.refusal, model_name)
            return None

        mgr._request_lease = _request_lease
        def _release(lid):
            self.released.append(lid)
            self.events.append(("release", lid))
        mgr.sc.release_lease = _release
        self.mgr = mgr
        return self

    def __exit__(self, *a):
        capacity.collect = self._collect
        config.FOREIGN_IDLE_MB = self._floor
        config.OFFER_MIN_SAMPLE_INTERVAL_S = self._interval
        manager_mod.hf_model_installed = self._installed


def main():
    step(1, "the registry is told the size the model turned out to be")
    with Box() as b:
        mm = b.mgr.load_ollama_model("qwen3:1.7b")
        print(f"  leases requested: {b.leased}  released: {b.released}")
        check(b.leased == [ESTIMATE, ACTUAL],
              f"leased at the estimate, then re-leased at the actual ({b.leased})")
        check(b.released == ["lease_1"],
              f"the estimate's lease was released ({b.released})")
        # THE ORDER IS THE POINT. Release-first leaves the registry accounting
        # 0 for this model between the two calls, and permanently so if the
        # new request is refused.
        print(f"  events: {b.events}")
        check(b.events == [("request", ESTIMATE), ("request", ACTUAL),
                           ("release", "lease_1")],
              "the new lease was acquired BEFORE the old one was released")
        check(mm.lease_id == "lease_2", f"the model holds the new one ({mm.lease_id})")
        check(mm.lease_lost is False, "and is not marked unleased")
        check(mm.memory_mb == ACTUAL, f"manager records the real size ({mm.memory_mb})")

    step(2, "a small discrepancy is not worth the window")
    # Release-then-reacquire leaves the resource momentarily unheld. Not
    # worth paying for a rounding error.
    with Box() as b:
        b.mgr.ollama.list_running = lambda: [
            {"name": "qwen3:1.7b", "size": int(ESTIMATE * 1.2) * 1024 * 1024}]
        mm = b.mgr.load_ollama_model("qwen3:1.7b")
        print(f"  actual {int(ESTIMATE*1.2)} vs estimate {ESTIMATE}: leases {b.leased}")
        check(b.leased == [ESTIMATE], f"one lease, no reconcile ({b.leased})")
        check(b.released == [], f"and nothing released ({b.released})")
        check(mm.lease_id == "lease_1", "the original lease stands")

    step(3, "a refused reconcile leaves the estimate lease standing")
    # The failure this ordering exists to prevent, measured live on
    # wafer-services/gpu-metal by claude-wafer-services: with release-first,
    # a refused re-request took the registry from accounting 1296 MB to
    # accounting 0 for a model still occupying 3354 — under-accounting 1.63x
    # WORSE than before the reconcile ran, at the moment we learn the truth.
    with Box(grant=False) as b:
        mm = b.mgr.load_ollama_model("qwen3:1.7b")
        print(f"  events: {b.events}")
        check(b.released == [],
              f"the estimate's lease was NOT released ({b.released})")
        check(mm.lease_id == "lease_1",
              f"the model still holds it ({mm.lease_id})")
        check(mm.lease_lost is False,
              "and is not marked lost — it is leased, at the wrong size")
        check(("release", "lease_1") not in b.events,
              "the registry never passes through 0 for this model")

    step(4, "a refused reconcile does NOT undo the load")
    # Deliberately different from the version on `ada-linux-cuda-profile`,
    # which unloaded and raised. The local capacity gate already decided this
    # model fits, from the hardware; the registry's figure is
    # spec-minus-leases and cannot see a tenant who took no lease, which is
    # the whole case #861 exists for. Refusing on a disagreement with the
    # weaker instrument would be a regression.
    with Box(grant=False) as b:
        mm = b.mgr.load_ollama_model("qwen3:1.7b")
        print(f"  leases {b.leased}  released {b.released}  unloaded {b.unloaded}")
        check(b.unloaded == [], f"the model was not unloaded ({b.unloaded})")
        check("qwen3:1.7b" in b.mgr.models, "it is still resident")
        check(mm.memory_mb == ACTUAL,
              f"and still accounted to us at its real size ({mm.memory_mb})")
        # The invariant from C1, restated on this path: residency is what
        # own_mb counts, and it must not move when lease state does.
        check(b.mgr.own_device_mb() == ACTUAL,
              f"own_mb still counts it ({b.mgr.own_device_mb()})")
        check(capacity.foreign_mb(ACTUAL, b.mgr.own_device_mb()) == 0,
              "so the gateway does not read its own model as a foreign tenant")

    step(5, "a queued reconcile is a work signal; a broken one is not")
    with Box(grant=False, refusal="queued", floor=5515) as b:
        b.mgr.load_ollama_model("qwen3:1.7b")
        check(b.mgr.lease_contention_fresh() is True,
              "queued: the registry considered it and said the resource is full")
        check(b.mgr.work_in_progress(0) is True, "so the node steps back")
    with Box(grant=False, refusal="error", floor=5515) as b:
        b.mgr.load_ollama_model("qwen3:1.7b")
        check(b.mgr.lease_contention_fresh() is False,
              "error: we could not ask, which says nothing about the device")
        check(b.mgr.work_in_progress(0) is False,
              "so a 404 or a dead registry does not close the gate")
        # This is the case ada was in for months. Counting it would have read
        # a permanent authentication fault as a permanent render.

    step(6, "on a box with no declared floor it is a log line and nothing more")
    with Box(grant=False, floor=None) as b:
        b.mgr.load_ollama_model("qwen3:1.7b")
        check(b.mgr.lease_contention_fresh() is True, "the refusal is recorded")
        check(b.mgr.work_in_progress(0) is None,
              "but nothing is gated — slice and wafer are unaffected")

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
