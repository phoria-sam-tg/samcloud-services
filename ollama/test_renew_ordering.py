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

    step(5, "how long a refused renewal's guarantee actually lasts")
    # D2 keeps the old lease on a refusal, which is only worth anything if a
    # retry lands before that lease expires. It does not, quite
    # (claude-wafer-services): the loop sleeps the interval FIRST, and an
    # unextended lease expires exactly when the next attempt fires.
    ttl = config.LEASE_TTL
    interval = int(ttl * config.LEASE_RENEW_AT)
    attempts = [n * interval for n in range(1, 6) if n * interval < ttl]
    print(f"  TTL {ttl}s, interval {interval}s -> attempts inside the lease: "
          f"{attempts}, expiry at {ttl}")
    check(interval < ttl,
          f"a renewal is at least attempted before expiry ({interval} < {ttl})")
    # The honest statement, asserted rather than described. If someone lowers
    # LEASE_RENEW_AT to 0.25 this flips to three and the docstring's example
    # needs updating with it — which is the point of pinning it here.
    check(len(attempts) == 1,
          f"at LEASE_RENEW_AT={config.LEASE_RENEW_AT} a refusal gets exactly "
          f"{len(attempts)} attempt, and the retry races the expiry rather "
          f"than preceding it")
    doc = ModelManager._renew_leases.__doc__ or ""
    check("ONE retry" in doc or "one retry" in doc.lower(),
          "and the docstring says so rather than promising indefinitely")

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
