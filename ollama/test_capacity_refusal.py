#!/usr/bin/env python3
"""Regression test for the capacity refusal path.

`manager` raised `capacity.InsufficientCapacity` and `server` caught it by the
same name from the 2026-08-30 capacity migration until 2026-09-20, and nothing
defined it. So every refusal raised AttributeError — and because evaluating an
`except` clause is what raised it, the error escaped the whole `try` without
reaching the fallback underneath, and a capacity answer reached the caller as a
bare 500 with no detail. The gate itself was never wrong. It just could not say
why, which left "this model does not fit" and "this gateway is broken" looking
identical from outside.

Loads nothing and leases nothing: the refusal happens before either. Reads real
memory off this box, so it is a live check of the gate as well as of the shape.

    python -m ollama.test_capacity_refusal
"""

import os
import sys

os.environ["AUTH_ENABLED"] = "0"          # test process only, before import
os.environ.setdefault("SC_TOKEN", "test")

from fastapi.testclient import TestClient

from . import capacity, server
from .manager import ModelManager
from .samcloud import SamcloudClient

FORTY_GB_MB = 40960

failures = []


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def _manager() -> ModelManager:
    """A manager that never talks to the registry — the refusal precedes it."""
    mgr = ModelManager(sc=SamcloudClient(token="test"))
    mgr.ollama.memory_estimate_mb = lambda name: FORTY_GB_MB
    return mgr


def main():
    step(1, "the exception exists and carries the reading")
    check(hasattr(capacity, "InsufficientCapacity"),
          "capacity.InsufficientCapacity is defined")
    check(issubclass(capacity.InsufficientCapacity, Exception),
          "it is catchable as Exception")

    step(2, "a 40 GB ask is refused against this box's real memory")
    avail = capacity.collect()["memory_available_mb"]
    print(f"  available={avail}MB usable={capacity.usable_mb(avail)}MB")
    if capacity.fits(FORTY_GB_MB, avail):
        print("  SKIP — 40GB actually fits right now; nothing to refuse")
        return 0

    try:
        _manager().load_ollama_model("qwen3:1.7b")
        check(False, "load was refused")
    except capacity.InsufficientCapacity as e:
        check(True, "load raised InsufficientCapacity, not AttributeError")
        check(e.need_mb == FORTY_GB_MB, f"need_mb carried ({e.need_mb})")
        check(e.available_mb == e.available_mb and e.usable_mb is not None,
              f"usable_mb carried ({e.usable_mb})")
        check(isinstance(e.fits_now, list), f"fits_now carried ({e.fits_now})")
        check(str(e).strip() != "", "detail is human-readable")
        print(f"  detail: {e}")

    step(3, "the except clause no longer escapes its own try")
    # The original defect, isolated: the fallback below must be reachable.
    reached = {"fallback": False}
    try:
        try:
            raise capacity.InsufficientCapacity("x")
        except capacity.InsufficientCapacity:
            pass
        try:
            raise RuntimeError("some other fault")
        except capacity.InsufficientCapacity:
            check(False, "wrong branch taken")
        except Exception:
            reached["fallback"] = True
    except AttributeError as e:
        check(False, f"clause evaluation still raises: {e}")
    check(reached["fallback"], "a non-capacity error still reaches the fallback")

    step(4, "over HTTP, a refusal is a 503 with the numbers — not a 500")
    server.mgr = _manager()               # no lifespan: nothing adopted, nothing leased
    client = TestClient(server.app, raise_server_exceptions=False)
    r = client.post("/models/load", json={"model": "qwen3:1.7b", "backend": "ollama"})
    print(f"  POST /models/load -> {r.status_code}")
    check(r.status_code == 503, f"status is 503, got {r.status_code}")
    body = r.json().get("detail", {})
    print(f"  body: {body}")
    check(isinstance(body, dict) and body.get("error") == "insufficient_capacity",
          "body names the reason")
    check(body.get("need_mb") == FORTY_GB_MB, "body carries need_mb")
    check("fits_now" in body, "body carries fits_now")

    print(f"\n{'='*60}")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
