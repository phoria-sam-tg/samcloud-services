#!/usr/bin/env python3
"""The pool lease is renewed IN PLACE, and a failed renewal never drops it.

#827: every incident so far is a holder that went away while its exclusive lease
lived on. Renewal is the fix, and it has one way to be written wrong that this
file exists to prevent — `manager._renew_leases()`, one function above the new
code, renews the MODEL leases by releasing and re-requesting. On a shared memory
lease that is survivable. On the exclusive single-slot pool it is a window in
which another caller can take the pool mid-generation, which is the fault #827
removes. So the central assertion here is negative: renewal calls renew and
nothing else.

Run: python3 ollama/test_pool_renewal.py
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.manager import ModelManager                      # noqa: E402
from ollama import config                                    # noqa: E402
import ollama.manager as _mgr_mod                            # noqa: E402
import httpx                                                 # noqa: E402

passed = failed = 0


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        failed += 1
        print(f"  \033[31m✗\033[0m {label}  {detail}")


class Recorder:
    """A samcloud client that records which lease calls were made, in order."""

    def __init__(self, renew=None):
        self.calls = []
        self._renew = renew

    def renew_lease(self, lease_id):
        self.calls.append(("renew", lease_id))
        if self._renew:
            raise self._renew
        return {"status": "renewed"}

    def release_lease(self, lease_id):
        self.calls.append(("release", lease_id))
        return {"status": "released"}

    def request_lease(self, **kw):
        self.calls.append(("request", kw.get("resource_id")))
        return {"lease_id": "lease_new", "status_code": 200}


def manager_holding(lease_id, held_for_s, sc):
    m = ModelManager(sc=sc)
    with m._pool_lock:
        m._pool_leases.add(lease_id)
        m._pool_lease_acquired[lease_id] = time.monotonic() - held_for_s
    return m


def logs_from(fn):
    """Capture (level, message) pairs emitted while fn() runs."""
    seen = []
    real = {lvl: getattr(_mgr_mod.log, lvl) for lvl in ("info", "warning", "error")}
    for lvl in real:
        setattr(_mgr_mod.log, lvl,
                (lambda l: lambda msg, *a, **k: seen.append((l, str(msg))))(lvl))
    try:
        fn()
    finally:
        for lvl, f in real.items():
            setattr(_mgr_mod.log, lvl, f)
    return seen


def http_error(code):
    req = httpx.Request("POST", "http://x/leases/l/renew")
    return httpx.HTTPStatusError("boom", request=req,
                                 response=httpx.Response(code, request=req))


def main():
    print("pool lease renewal (#827 P3 stage a)")

    print("  [1] renewal extends IN PLACE — it never releases and re-requests")
    sc = Recorder()
    m = manager_holding("lease_A", 300, sc)
    logs_from(m._renew_pool_leases)
    check("renew_lease was called for the held lease",
          ("renew", "lease_A") in sc.calls, str(sc.calls))
    # The whole point. _renew_leases() above does exactly this and must not be
    # the model for an exclusive lease.
    check("NOTHING was released", all(c[0] != "release" for c in sc.calls), str(sc.calls))
    check("and no new lease was requested",
          all(c[0] != "request" for c in sc.calls), str(sc.calls))
    check("the lease is still held afterwards",
          "lease_A" in m._pool_leases, str(m._pool_leases))

    print("  [2] the log line carries the age — the gate for stage (b)")
    # admin gates the TTL drop on 'a lease over 120s seen renewing in its log',
    # so the age has to be IN the line, not inferable from timestamps.
    sc = Recorder()
    m = manager_holding("lease_B", 187, sc)
    seen = logs_from(m._renew_pool_leases)
    line = next((msg for lvl, msg in seen if "renewed in place" in msg), "")
    check("it says renewed in place", bool(line), str(seen))
    check("it states a held-time over 120s", "187s held" in line, line)
    check("logged at info, not warning", any(
        lvl == "info" and "renewed in place" in msg for lvl, msg in seen), str(seen))

    print("  [3] a failed renewal KEEPS the lease — dropping it is worse")
    sc = Recorder(renew=http_error(503))
    m = manager_holding("lease_C", 90, sc)
    seen = logs_from(m._renew_pool_leases)
    check("the lease is still held", "lease_C" in m._pool_leases, str(m._pool_leases))
    check("nothing was released", all(c[0] != "release" for c in sc.calls), str(sc.calls))
    check("and it warns rather than failing silently",
          any(lvl == "warning" for lvl, _ in seen), str(seen))

    print("  [4] a 404 means the registry reaped us mid-generation: ERROR")
    sc = Recorder(renew=http_error(404))
    m = manager_holding("lease_D", 400, sc)
    seen = logs_from(m._renew_pool_leases)
    errs = [msg for lvl, msg in seen if lvl == "error"]
    check("it is an error, not a warning", len(errs) == 1, str(seen))
    check("and it says exclusivity is gone",
          errs and "no longer exclusively" in errs[0], str(errs))
    check("the lease is still tracked, so shutdown still tries to release it",
          "lease_D" in m._pool_leases, str(m._pool_leases))

    print("  [5] a released lease is not renewed, and leaves no bookkeeping behind")
    sc = Recorder()
    m = manager_holding("lease_E", 10, sc)
    m.release_pool("lease_E")
    sc.calls.clear()
    logs_from(m._renew_pool_leases)
    check("no renewal for a lease we gave back",
          sc.calls == [], str(sc.calls))
    check("and its acquire time was forgotten",
          "lease_E" not in m._pool_lease_acquired, str(m._pool_lease_acquired))

    print("  [6] an unknown acquire time still renews, and says so")
    sc = Recorder()
    m = ModelManager(sc=sc)
    with m._pool_lock:
        m._pool_leases.add("lease_F")          # no acquire time recorded
    seen = logs_from(m._renew_pool_leases)
    check("it renews anyway", ("renew", "lease_F") in sc.calls, str(sc.calls))
    check("and reports the age as unknown rather than 0s",
          any("unknown held" in msg for _, msg in seen), str(seen))

    print("  [7] the interval is capped, so the path is exercised before it is trusted")
    check("interval is at most the cap",
          config.EXO_LEASE_RENEW_INTERVAL_S <= config.EXO_LEASE_RENEW_MAX_S,
          f"{config.EXO_LEASE_RENEW_INTERVAL_S} > {config.EXO_LEASE_RENEW_MAX_S}")
    check("and never degenerate",
          config.EXO_LEASE_RENEW_INTERVAL_S >= 5, str(config.EXO_LEASE_RENEW_INTERVAL_S))
    # A third of 1800s is 594s: without the cap only the 5 longest leases of 514
    # would ever renew, and stage (b)'s gate could wait days.
    check("uncapped it would be far longer than a minute",
          int(config.EXO_LEASE_TTL * config.EXO_LEASE_RENEW_PCT / 100) > 60,
          f"ttl={config.EXO_LEASE_TTL} pct={config.EXO_LEASE_RENEW_PCT}")
    check("the percentage is clamped into a sane band",
          10 <= config.EXO_LEASE_RENEW_PCT <= 90, str(config.EXO_LEASE_RENEW_PCT))

    print("  [8] stage (a) changes no timing — exclusivity cannot depend on the new path")
    check("the TTL still outlives the longest generation",
          config.EXO_LEASE_TTL >= config.EXO_GENERATE_TIMEOUT + 300,
          f"ttl={config.EXO_LEASE_TTL} gen={config.EXO_GENERATE_TIMEOUT}")

    print()
    if failed:
        print(f"  FAILED: {failed} failed, {passed} passed")
        sys.exit(1)
    print(f"  ALL PASSED: pool lease renewal {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
