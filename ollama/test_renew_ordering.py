#!/usr/bin/env python3
"""A lease swap never leaves the registry accounting zero for a resident model.

PR #29 established this on `_reconcile_lease`: acquire the new lease, release
the old only on success, and a refusal keeps what you had. `_renew_leases` is
the same kind of swap and kept the old ordering — release, then re-request,
and on a refusal `lease_id = None` with the model still on the GPU. Measured
on wafer as 1296 -> 0 for a model occupying 3354 MB
(claude-wafer-services), and far wider than the reconcile's exposure: the
reconcile fires only past `_RECONCILE_RATIO`, this fires every
`LEASE_TTL * LEASE_RENEW_AT` on every leased model, forever.

The test models the registry's own accounting rather than the call order, and
samples it after every event, because "never passes through zero" is a claim
about the registry's figure and not about the sequence we happen to emit.

    python -m ollama.test_renew_ordering
"""

import os
import re
import sys
import time

os.environ["AUTH_ENABLED"] = "0"
os.environ.setdefault("SC_TOKEN", "test")

from . import capacity, config, manager as manager_mod
from .manager import Backend, ManagedModel, ModelManager
from .samcloud import SamcloudClient

MODEL, SIZE = "qwen3:1.7b", 3354

failures, checks = [], 0


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


class Registry:
    """The registry's lease rows for one resource, and its running total.

    `accounted` is sampled after EVERY grant and release, so a figure that
    dips to zero between two calls cannot hide inside one tick.
    """

    def __init__(self, grant=True):
        self.rows: dict = {}
        self.grant = grant
        self.n = 0
        self.accounted: list = []
        self.events: list = []

    def request(self, name, mb):
        self.n += 1
        # The FIRST grant is always given — it is the load, not the renewal
        # under test. `grant` governs the swap.
        if self.n == 1 or self.grant:
            lid = f"lease_{self.n}"
            self.rows[lid] = mb
            self.events.append(("request", mb, "granted"))
            self._sample()
            return lid
        self.events.append(("request", mb, "refused"))
        self._sample()
        return None

    def release(self, lid):
        self.rows.pop(lid, None)
        self.events.append(("release", lid, None))
        self._sample()

    def _sample(self):
        self.accounted.append(sum(self.rows.values()))


def box(grant=True):
    reg = Registry(grant=grant)
    mgr = ModelManager(sc=SamcloudClient(token="test"))
    mgr._request_lease = lambda name, mb: reg.request(name, mb)
    mgr.sc.release_lease = lambda lid: reg.release(lid)
    now = time.time()
    lid = reg.request(MODEL, SIZE)          # the load's own lease
    mgr.models[MODEL] = ManagedModel(
        name=MODEL, backend=Backend.OLLAMA, memory_mb=SIZE, lease_id=lid,
        port=11434, loaded_at=now, last_used=now)
    reg.accounted.clear()                   # measure the RENEWAL, not the load
    return mgr, reg


def main():
    step(1, "a granted renewal: acquire, then release — never zero")
    mgr, reg = box(grant=True)
    mgr._renew_leases()
    mm = mgr.models[MODEL]
    print(f"  events   {reg.events[1:]}")
    print(f"  accounted {reg.accounted}")
    check([e[0] for e in reg.events[1:]] == ["request", "release"],
          "the new lease is acquired BEFORE the old is released")
    check(min(reg.accounted) > 0,
          f"the registry never accounts 0 (min {min(reg.accounted)})")
    check(min(reg.accounted) == SIZE,
          f"and never dips below the model's size (min {min(reg.accounted)})")
    check(max(reg.accounted) == SIZE * 2,
          f"it double-counts briefly instead, which is the safe direction "
          f"(max {max(reg.accounted)})")
    check(sum(reg.rows.values()) == SIZE, "and settles back to one lease")
    check(mm.lease_id == "lease_2", f"the model holds the new one ({mm.lease_id})")
    check(mm.lease_id is not None, "and is leased throughout")

    step(2, "a refused renewal keeps the lease it had")
    mgr, reg = box(grant=False)
    before = mgr.models[MODEL].lease_id
    mgr._renew_leases()
    mm = mgr.models[MODEL]
    print(f"  events   {reg.events[1:]}")
    print(f"  accounted {reg.accounted}")
    check([e[0] for e in reg.events[1:]] == ["request"],
          "nothing was released")
    check(mm.lease_id == before,
          f"the model still holds {before} ({mm.lease_id})")
    check(min(reg.accounted) == SIZE,
          f"the registry still accounts its full size (min {min(reg.accounted)})")
    check(mm.lease_id is not None,
          "it is still leased, at the right size")
    check(MODEL in mgr.models, "residency is untouched, as on every path")

    step(3, "the same refusal on the OLD ordering reads zero")
    # Not a hypothetical: this is the shape that was on main, reproduced here
    # so the test states what it is protecting rather than only that the new
    # code passes.
    mgr, reg = box(grant=False)
    mm = mgr.models[MODEL]
    old_id = mm.lease_id
    mgr.sc.release_lease(old_id)                    # release first ...
    new_id = mgr._request_lease(MODEL, SIZE)        # ... then re-request
    print(f"  accounted {reg.accounted}  new_id={new_id}")
    check(0 in reg.accounted,
          f"release-first passes through 0 ({reg.accounted})")
    check(new_id is None and sum(reg.rows.values()) == 0,
          "and stays there, for a model still resident")

    step(4, "no swap anywhere in manager.py releases before it requests")
    # The half that bounds the sweep (claude-wafer-services): the comment says
    # where to look, a grep over every call site of the primitive says where to
    # stop. This is that grep, kept as a check so a third site cannot appear
    # quietly. If a future site legitimately must release first, widen this
    # with the reason rather than deleting it.
    src = open(os.path.join(os.path.dirname(__file__), "manager.py")).read()
    lines = src.splitlines()
    rel = re.compile(r"\b(?:sc\.release_lease|_release_lease_quietly)\s*\(")
    req = re.compile(r"\b_request_lease\s*\(")
    offenders = []
    for i, line in enumerate(lines):
        if rel.search(line) and "def " not in line:
            window = "\n".join(lines[i + 1:i + 7])
            if req.search(window):
                offenders.append(i + 1)
    releases = [i + 1 for i, l in enumerate(lines)
                if rel.search(l) and "def " not in l]
    print(f"  {len(releases)} release sites, {len(offenders)} release-then-request")
    check(len(releases) >= 10,
          f"the scan actually found the call sites ({len(releases)}) — a "
          f"pattern that matched nothing would pass this step silently")
    check(offenders == [],
          f"no site releases and then re-requests (offending lines: {offenders})")

    step(5, "a refused renewal is actually retried, more than once")
    # D2 keeps the old lease on a refusal, which is worth nothing unless a
    # retry lands before that lease expires. At LEASE_RENEW_AT 0.5 it did not
    # (claude-wafer-services): the loop sleeps the interval FIRST and renews a
    # lease it did not extend, so the single retry landed exactly at expiry.
    # 0.25 is the value that makes the guarantee hold, and it is the same
    # arithmetic EXO_LEASE_RENEW_PCT already uses for the pool lease.
    ttl = config.LEASE_TTL
    interval = int(ttl * config.LEASE_RENEW_AT)
    attempts = [n * interval for n in range(1, 10) if n * interval < ttl]
    print(f"  TTL {ttl}s, interval {interval}s -> attempts inside the lease: "
          f"{attempts}, expiry at {ttl}")
    check(interval < ttl,
          f"a renewal is attempted before expiry ({interval} < {ttl})")
    n = len(attempts)
    check(n >= 3,
          f"two renewals can be missed and a third still lands "
          f"({n} attempt{'' if n == 1 else 's'} at "
          f"LEASE_RENEW_AT={config.LEASE_RENEW_AT})")
    # Guarded, because this step exists to FAIL on a bad constant and an
    # IndexError is not a failure report: at LEASE_RENEW_AT=0.5 `attempts` has
    # one entry, `attempts[2]` raised, and the run ended with a traceback —
    # no summary line, and the docstring check below never ran. A test whose
    # regression path crashes tells you less than one that prints FAIL.
    third = attempts[2] if n >= 3 else None
    check(third is not None and third <= ttl * 0.8,
          f"the third lands inside 80% of the TTL ({third} <= {ttl * 0.8}) "
          f"— EXO_LEASE_RENEW_PCT's rule, applied here"
          if third is not None else
          f"there is no third attempt to place inside 80% of the TTL "
          f"({n} inside {ttl}s)")
    # The docstring works the timeline through in seconds. Pin it against the
    # live constants so the example cannot drift away from them: a changed TTL
    # or fraction fails here rather than leaving a plausible, wrong worked
    # example in the file.
    doc = ModelManager._renew_leases.__doc__ or ""
    missing = [t for t in attempts[:3] + [ttl] if f"t={t}" not in doc]
    # attempts[:3] is short when the constant is wrong; the check below then
    # verifies less than it does normally, so say how much it verified.
    check(not missing,
          f"and the docstring's worked timeline names all {len(attempts[:3]) + 1} "
          f"of {attempts[:3] + [ttl]} (missing: {missing})")

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
