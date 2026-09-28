#!/usr/bin/env python3
"""The offer is measured, windowed, and the same at every endpoint (#861).

Three endpoints used to advertise the registry's `available_memory_mb`, which
is spec-minus-leases and never consults utilisation — so it cannot see a tenant
who took no lease, which is every tenant that matters on a work machine.
Measured on wafer 2026-09-29, on `main`, with no leases held and no models
resident: `/status` said 36,864 MB while the collector one import away measured
11,066. Callers scrape that.

Loads nothing, leases nothing, talks to no network: `capacity.collect` and the
backend catalogues are stubbed, so every reading in here is one this test
chose. That is the point — the failures being pinned are ones a live box
reaches only when somebody else happens to be rendering.

    python -m ollama.test_elastic_offering
"""

import os
import sys
import time

os.environ["AUTH_ENABLED"] = "0"          # test process only, before import
os.environ.setdefault("SC_TOKEN", "test")

from fastapi.testclient import TestClient

from . import capacity, config, manager as manager_mod, server
from .manager import Backend, ManagedModel, ModelManager
from .samcloud import SamcloudClient

failures = []
checks = 0

# A stand-in catalogue with a spread wide enough to straddle any plausible
# reading: one model that fits almost anywhere, one that fits nowhere.
CATALOGUE = [
    {"name": "small:1b", "size": 2000 * 1024 * 1024},
    {"name": "mid:8b", "size": 6000 * 1024 * 1024},
    {"name": "big:70b", "size": 40000 * 1024 * 1024},
]


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def reading(available, device_inuse=500, total=65536):
    return {
        "memory_total_mb": total,
        "memory_used_mb": total - available,
        "memory_available_mb": available,
        "memory_device_inuse_mb": device_inuse,
        "compute_pct": 0.0,
        "load_avg_1m": 0.0,
    }


class Fixture:
    """A manager whose every input is chosen here, and restored on exit."""

    def __init__(self, available=40000, device_inuse=500, catalogue=CATALOGUE):
        self.next = reading(available, device_inuse)
        self.catalogue = catalogue

    def __enter__(self):
        self._collect = capacity.collect
        self._idle = config.FOREIGN_IDLE_MB
        self._vlm = manager_mod.hf_model_installed
        capacity.collect = lambda: dict(self.next)
        # slice genuinely has a VLM on disk, so without this the catalogue
        # under test is whatever this box happens to hold — the test would
        # pass or fail depending on the machine running it.
        manager_mod.hf_model_installed = lambda repo_id: False
        mgr = ModelManager(sc=SamcloudClient(token="test"))
        mgr.ollama.list_models = lambda: list(self.catalogue)
        mgr.llama.available_models = lambda: []
        mgr.sc.resource_dashboard = lambda: []
        mgr.sc.list_leases = lambda **kw: []
        self.mgr = mgr
        return self

    def set(self, available, device_inuse=500):
        self.next = reading(available, device_inuse)

    def raises(self, exc=RuntimeError("nvidia-smi: no devices were found")):
        """Make the next collect() blow up, as the CUDA arm does on failure."""
        def _boom():
            raise exc
        capacity.collect = _boom

    def __exit__(self, *a):
        capacity.collect = self._collect
        config.FOREIGN_IDLE_MB = self._idle
        manager_mod.hf_model_installed = self._vlm


def main():
    step(1, "the offer is the window MINIMUM, not the latest reading")
    # The failure: `available` was measured spanning 9988-12938 over 24 samples
    # on an IDLE wafer — 2950 MB of desktop churn, wider than several models in
    # the catalogue. A per-request comparison against one sample flaps across
    # the whole offer rather than at its margin.
    with Fixture(available=40000) as f:
        f.mgr.offer_reading()
        f.set(available=5000)
        r = f.mgr.offer_reading()
        print(f"  saw 40000 then 5000 -> offer sees {r['memory_available_mb']}")
        check(r["memory_available_mb"] == 5000,
              "a DROP is taken immediately — withdrawal within one reconcile")
        f.set(available=40000)
        r = f.mgr.offer_reading()
        print(f"  back to 40000 -> offer still sees {r['memory_available_mb']}")
        check(r["memory_available_mb"] == 5000,
              "a RISE is not, while the trough is still in the window")
        check(r["samples"] == 3, f"all three readings are in it ({r['samples']})")

    step(2, "restoration waits for the trough to age out")
    with Fixture(available=40000) as f:
        f.set(available=5000)
        f.mgr.offer_reading()
        # Age every recorded reading past the window rather than sleeping for
        # it: the rule under test is "older than OFFER_WINDOW_S is dropped",
        # and a test that waits 60s to assert it is a test nobody runs.
        with f.mgr._readings_lock:
            f.mgr._readings = [
                (t - config.OFFER_WINDOW_S - 1, r) for t, r in f.mgr._readings
            ]
        f.set(available=40000)
        r = f.mgr.offer_reading()
        print(f"  trough aged out -> offer sees {r['memory_available_mb']}")
        check(r["memory_available_mb"] == 40000, "the offer restores")
        check(r["samples"] == 1, f"on the fresh reading alone ({r['samples']})")

    step(3, "device-in-use takes the MAXIMUM over the same window")
    # The mirror of step 1, and it has to be the other extreme: one sample
    # showing the accelerator busy is enough to say somebody is working, where
    # one sample showing it idle proves nothing. ada measured utilisation
    # dropping to 0-4% for twelve seconds mid-render.
    with Fixture(available=40000, device_inuse=500) as f:
        f.mgr.offer_reading()
        f.set(available=40000, device_inuse=46000)
        f.mgr.offer_reading()
        f.set(available=40000, device_inuse=500)
        r = f.mgr.offer_reading()
        print(f"  500, 46000, 500 -> offer sees {r['memory_device_inuse_mb']}")
        check(r["memory_device_inuse_mb"] == 46000,
              "the busy sample wins — an idle instant does not clear it")

    step(4, "buckets: loadable is what actually fits, blocked says how short")
    with Fixture(available=10000) as f:
        o = f.mgr.offering()
        names = lambda b: [m["name"] for m in o[b]]
        print(f"  available=10000 usable={o['capacity']['usable_mb']}")
        print(f"  loadable={names('loadable')} blocked={names('blocked')}")
        check(names("loadable") == ["mid:8b", "small:1b"],
              f"the two that fit, largest first ({names('loadable')})")
        check(names("blocked") == ["big:70b"], f"the one that does not")
        big = o["blocked"][0]
        check(big["reason"] == "insufficient_capacity", "with the reason")
        check(big["short_by_mb"] == 40000 - o["capacity"]["usable_mb"],
              f"and how much would have to free ({big['short_by_mb']})")

    step(5, "the offer uses the same estimate as the load gate")
    # An offer computed by a different rule from the gate promises models the
    # gate then refuses. cd366c8 did exactly that — its buckets used
    # disk x1.1 + a 4 GB KV floor while the gate used the disk size.
    with Fixture(available=60000) as f:
        o = f.mgr.offering()
        offered = {m["name"]: m["need_mb"] for m in o["loadable"] + o["blocked"]}
        gate = {m["name"]: f.mgr.ollama.memory_estimate_mb(m["name"])
                for m in CATALOGUE}
        print(f"  offer={offered}")
        print(f"  gate ={gate}")
        check(offered == gate,
              "every catalogue entry is sized identically by both")
        check(len(offered) == len(CATALOGUE) == 3,
              f"and all {len(CATALOGUE)} were compared ({len(offered)})")

    step(6, "work_in_progress is tri-state, and unset means NOT ASKED")
    with Fixture(available=40000, device_inuse=46000) as f:
        config.FOREIGN_IDLE_MB = None
        o = f.mgr.offering()
        check(o["capacity"]["work_in_progress"] is None,
              "no declared floor -> None, never False")
        check(len(o["loadable"]) > 0,
              "and the offer is unchanged — B does not gate by default")
        # False would say "we checked and nobody is working" on a box holding
        # 46 GB for somebody else. The distinction is the whole reason
        # FOREIGN_IDLE_MB has no default.
        check(f.mgr.work_in_progress(None) is None, "the predicate agrees")

        config.FOREIGN_IDLE_MB = 5515
        check(f.mgr.work_in_progress(46000) is True,
              "46000 over a 5515 floor is work")
        check(f.mgr.work_in_progress(5515) is False,
              "the floor itself is not")
        check(f.mgr.work_in_progress(5515 + config.FOREIGN_MARGIN_MB) is False,
              "nor is the floor plus exactly the margin")
        check(f.mgr.work_in_progress(None) is True,
              "and 'we cannot tell' is treated as 'someone is'")

    step(7, "when someone is working, nothing is on offer — fit or no fit")
    with Fixture(available=40000, device_inuse=46000) as f:
        config.FOREIGN_IDLE_MB = 5515
        o = f.mgr.offering()
        print(f"  available=40000 (fits everything), foreign="
              f"{o['capacity']['foreign_mb']}")
        check(o["loadable"] == [], "loadable is empty despite 40 GB free")
        check(len(o["blocked"]) == 3, f"all three blocked ({len(o['blocked'])})")
        reasons = {m["reason"] for m in o["blocked"]}
        check(reasons == {"work_in_progress"},
              f"for the right reason ({reasons})")
        # short_by_mb would invite a caller to wait for a number to move, when
        # what has to change is that the other tenant stops.
        check(all("short_by_mb" not in m for m in o["blocked"]),
              "and without a short_by_mb that would mean nothing")

    step(8, "own_mb counts what we hold, and excludes the pool")
    with Fixture() as f:
        now = time.time()
        f.mgr.models["a"] = ManagedModel(
            name="a", backend=Backend.OLLAMA, memory_mb=3000,
            lease_id=None, port=11434, loaded_at=now, last_used=now)
        f.mgr.models["think"] = ManagedModel(
            name="think", backend=Backend.EXO, memory_mb=30000,
            lease_id=None, port=52415, loaded_at=now, last_used=now)
        print(f"  own_device_mb={f.mgr.own_device_mb()}")
        check(f.mgr.own_device_mb() == 3000,
              "3000 of ours, and the pool's 30000 excluded")
        # Counting the pool would subtract memory we did not allocate from the
        # foreign figure — the over-counting direction, which HIDES a tenant.
        check(capacity.foreign_mb(46000, f.mgr.own_device_mb()) == 43000,
              "so foreign stays honest (43000, not 13000)")

    step(9, "over HTTP: /warm is auth-exempt and carries its own reading")
    with Fixture(available=10000) as f:
        server.mgr = f.mgr
        client = TestClient(server.app, raise_server_exceptions=False)
        r = client.get("/warm")
        print(f"  GET /warm -> {r.status_code}")
        check(r.status_code == 200, f"200 with no credential ({r.status_code})")
        body = r.json()
        check(set(body) >= {"resident", "loadable", "blocked", "capacity"},
              f"the four keys ({sorted(body)})")
        cap = body["capacity"]
        check(cap["available_mb"] == 10000,
              f"the reading it decided on is in the body ({cap['available_mb']})")
        check("window_s" in cap and "samples" in cap,
              "and says how it was measured, not just what it concluded")
        check("/warm" in server.AUTH_EXEMPT_PATHS, "listed as exempt")

    step(10, "/v1/models omits what cannot load; /models keeps everything")
    with Fixture(available=10000) as f:
        server.mgr = f.mgr
        client = TestClient(server.app, raise_server_exceptions=False)
        data = client.get("/v1/models").json()["data"]
        ids = [d["id"] for d in data]
        print(f"  /v1/models -> {ids}")
        check("big:70b" not in ids,
              "the 40 GB model is absent with 10 GB free — the whole point")
        check("mid:8b" in ids and "small:1b" in ids, "the two that fit are there")
        statuses = {d["id"]: d.get("status") for d in data}
        check(statuses.get("mid:8b") == "loadable",
              f"each entry says which it is ({statuses})")
        check(all("object" in d and d["object"] == "model" for d in data),
              "and the OpenAI shape is intact")
        # Nothing is hidden from an operator: /models carries the blocked list
        # with the reason, which is what makes omission here safe.
        offering = client.get("/models").json()["offering"]
        check([m["name"] for m in offering["blocked"]] == ["big:70b"],
              "/models still shows it, blocked, with the reason")

    step(11, "/status advertises the measured number, not the registry's")
    # The bug in one assertion. wafer's registry said 36864 with no leases
    # held while the box had 11066; whichever number lands under the name
    # callers read is the one that matters.
    with Fixture(available=11066) as f:
        f.mgr.sc.resource_dashboard = lambda: [{
            "id": f.mgr.sc and __import__("ollama.config", fromlist=["x"]).SC_RESOURCE_ID,
            "memory_pct": 0, "compute_pct": 0, "health": "green",
            "available_memory_mb": 36864,
        }]
        res = f.mgr.status()["resource"]
        print(f"  {res}")
        check(res["available_memory_mb"] == 11066,
              f"available_memory_mb is measured ({res['available_memory_mb']})")
        check(res.get("registry_available_memory_mb") == 36864,
              "the registry's view is kept, under a name that says whose it is")
        check(res["available_memory_mb"] != res["registry_available_memory_mb"],
              "and the 3.3x gap is visible rather than resolved silently")

    step(12, "every owned backend is bucketed, not just ollama")
    # A whisper child and a VLM occupy the same device as a chat model, so
    # "can this box start it right now" is one question with one answer. If a
    # backend were exempt, the offer would depend on which backend a model
    # happens to live in rather than on what the box can do.
    with Fixture(available=3000) as f:
        manager_mod.hf_model_installed = lambda repo_id: True
        o = f.mgr.offering()
        by_backend = {}
        for m in o["loadable"] + o["blocked"]:
            by_backend.setdefault(m["backend"], []).append(m["name"])
        print(f"  {by_backend}")
        check("mlx-whisper" in by_backend,
              f"transcription is in the buckets ({sorted(by_backend)})")
        check("mlx-vlm" in by_backend, "so is vision")
        check("ollama" in by_backend, "alongside ollama")
        # 3000 MB free -> usable 1976. whisper-small (1600) fits; the
        # large-v3-turbo (2600) and everything bigger does not.
        loadable = {m["name"] for m in o["loadable"]}
        print(f"  usable={o['capacity']['usable_mb']} loadable={loadable}")
        check("whisper-small" in loadable,
              f"the 1600 MB transcriber fits in 1976 ({loadable})")
        check("whisper-large-v3-turbo" not in loadable,
              "the 2600 MB one does not, and is blocked like any other model")

    step(13, "a collector that raises ends in offering nothing")
    # The Metal arm degrades to None fields; the CUDA arm RAISES, because
    # nvidia-smi runs with check=True and a 10s timeout (samclaude-admin, on
    # merging A). Both have to reach the same place: offer nothing. Readings
    # already in the window keep their value for at most OFFER_WINDOW_S, which
    # is the bound on how long a dead collector can keep an offer alive.
    with Fixture(available=40000) as f:
        f.mgr.offer_reading()
        f.raises()
        r = f.mgr.offer_reading()
        print(f"  collector raising, window still warm -> {r}")
        check(r["memory_available_mb"] == 40000,
              "the last good reading stands while it is inside the window")
        with f.mgr._readings_lock:
            f.mgr._readings = [
                (t - config.OFFER_WINDOW_S - 1, x) for t, x in f.mgr._readings
            ]
        r = f.mgr.offer_reading()
        print(f"  ...and once it ages out -> {r}")
        check(r["memory_available_mb"] == 0,
              f"available collapses to 0 ({r['memory_available_mb']})")
        check(r["memory_device_inuse_mb"] is None,
              "device_inuse is None — 'we cannot tell', not 'nothing is in use'")
        o = f.mgr.offering()
        check(o["loadable"] == [],
              f"so nothing is on offer ({[m['name'] for m in o['loadable']]})")
        check(len(o["blocked"]) == 3, "everything is blocked instead")

    step(14, "reading the hardware does not stall the event loop")
    # capacity.collect() is subprocesses — vm_stat/sysctl/ioreg, or one
    # nvidia-smi — measured at 22ms per call on slice with no cache. Awaited
    # inline that is 22ms of dead loop per request on an endpoint built to be
    # polled, and it delays stats_loop and offering_loop with it. The test:
    # make offering() take 100ms of BLOCKING time and check a concurrent
    # coroutine still gets scheduled while the handler runs.
    import asyncio

    with Fixture() as f:
        server.mgr = f.mgr
        real_offering = f.mgr.offering

        def slow_offering():
            time.sleep(0.1)                      # blocking, not awaitable
            return real_offering()

        f.mgr.offering = slow_offering
        ticks = {"n": 0}

        async def ticker():
            while True:
                ticks["n"] += 1
                await asyncio.sleep(0.005)

        async def drive():
            t = asyncio.create_task(ticker())
            await asyncio.sleep(0.01)
            before = ticks["n"]
            await server.warm()
            during = ticks["n"] - before
            t.cancel()
            return during

        during = asyncio.run(drive())
        print(f"  ticker ran {during} times during a 100ms offering()")
        # Inline, the loop is frozen and this is 0 or 1. Off the loop, a 5ms
        # ticker gets ~20 turns; anything past a handful proves it yielded.
        check(during >= 5,
              f"the loop kept running while the hardware was read ({during})")

    print(f"\n{'='*60}")
    print(f"  {checks} checks run")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for x in failures:
            print(f"    - {x}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
