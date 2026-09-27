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
import asyncio
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

    def __init__(self, renew=None, reply=None):
        self.calls = []
        self._renew = renew
        # The stage (a) reality: accepted, capped, extending nothing.
        self.reply = reply if reply is not None else {
            "lease_id": "x", "status": "active", "extended_by_s": 0,
            "capped": True, "max_total_s": 1800, "reason": "at_ceiling"}

    def renew_lease(self, lease_id, ttl_seconds):
        # ttl_seconds is REQUIRED by the endpoint; recording it means a caller
        # that stops sending it fails here rather than with a 422 in production.
        self.calls.append(("renew", lease_id, ttl_seconds))
        if self._renew:
            raise self._renew
        return dict(self.reply)

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


def http_error(code, body=None):
    """An HTTPStatusError shaped like the real endpoint's refusal.

    `body` matters. samclaude-admin, reviewing PR #35: #33's 404 for a lease that
    is no longer active carries `expires_at` in its body, and a renewer that
    sniffed the body for that key read a reaped lease as alive. This client
    branches on the STATUS CODE, so it should be immune — but "should be" is the
    phrase that has been wrong all day, so the real body goes in the fake and the
    immunity is asserted rather than reasoned about.
    """
    req = httpx.Request("POST", "http://x/leases/l/renew")
    return httpx.HTTPStatusError(
        "boom", request=req,
        response=httpx.Response(code, request=req, json=body) if body is not None
        else httpx.Response(code, request=req))


# Verbatim from #33's handler, the branch for status != "active".
#
# PROVENANCE: this and the `sc.reply` dicts below are hand-written COPIES of
# response shapes owned by ANOTHER REPO — samcloud's registry/main.py, the
# /renew and /leases handlers, as of samcloud v0.12.54. They are stale by
# default, not authoritative. samcloud pins its own key set on the producer
# side; the client-side defence against a rename is the totality check in
# _renew_pool_leases, not these dicts. Do not treat a green suite here as
# evidence that the plane still answers this way.
REAPED_404_BODY = {
    "detail": "Lease lease_x is expired, not active",
    "status": "expired",
    "expires_at": "2026-09-27T08:30:00+00:00",
}


def main():
    print("pool lease renewal (#827 P3 stage a)")

    print("  [1] renewal extends IN PLACE — it never releases and re-requests")
    sc = Recorder()
    m = manager_holding("lease_A", 300, sc)
    logs_from(m._renew_pool_leases)
    check("renew_lease was called for the held lease",
          any(c[0] == "renew" and c[1] == "lease_A" for c in sc.calls), str(sc.calls))
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
    line = next((msg for lvl, msg in seen if "187s held" in msg), "")
    check("it states a held-time over 120s", bool(line), str(seen))
    check("logged at info, not warning",
          any(lvl == "info" and "187s held" in msg for lvl, msg in seen), str(seen))
    check("it sends the REQUIRED ttl_seconds with the renewal",
          any(c[0] == "renew" and c[2] == config.EXO_LEASE_TTL for c in sc.calls),
          str(sc.calls))

    print("  [2b] a capped renewal must NOT read as an extension")
    # Stage (a) grants 1800s and the ceiling is granted_at+1800, so every renewal
    # is accepted and extends nothing. A line saying 'renewed' for that is a
    # renewal that silently did nothing looking exactly like one that silently
    # failed — the shape of every defect this ticket turned up.
    check("it says there was no extension", "NO extension" in line, line)
    check("and names the ceiling that capped it", "1800s ceiling" in line, line)
    check("and says why it is expected, not a fault",
          "stage (b)" in line, line)

    print("  [2c] a real extension says so, with the amount")
    sc = Recorder(reply={"extended_by_s": 60, "capped": False, "reason": "extended",
                         "expires_at": "2026-09-27T08:00:00Z", "max_total_s": 1800})
    m = manager_holding("lease_B2", 200, sc)
    seen = logs_from(m._renew_pool_leases)
    line2 = next((msg for lvl, msg in seen if "200s held" in msg), "")
    check("it reports the extension", "+60s" in line2, line2)
    check("and does not claim there was none", "NO extension" not in line2, line2)

    print("  [2d] no extension and NOT capped is a different thing, and says so")
    # The registry returns extended_by_s 0 with capped False for a lease that has
    # no expiry at all. Treating "no extension" as "ceiling" would mislabel it —
    # the same assumption-in-a-branch that the else-clause above used to make.
    sc = Recorder(reply={"extended_by_s": 0, "capped": False, "max_total_s": 1800,
                         "reason": "indefinite",
                         "note": "indefinite lease — nothing to renew"})
    m = manager_holding("lease_B3", 140, sc)
    seen = logs_from(m._renew_pool_leases)
    line3 = next((msg for lvl, msg in seen if "140s held" in msg), "")
    check("it does not claim the ceiling capped it",
          "ceiling" not in line3, line3)
    check("it warns rather than reporting business as usual",
          any(lvl == "warning" and "140s held" in msg for lvl, msg in seen), str(seen))
    check("and it repeats what the registry actually said",
          "nothing to renew" in line3, line3)

    print("  [2e] capped AND extended together: the last extension before the ceiling")
    # samclaude-admin, reviewing #33: on the final renewal target lands on the
    # ceiling, which is still later than the current expiry — so extended_by_s > 0
    # and capped is true at the same time. Branching on capped first would print
    # "NO extension" for a renewal that extended.
    sc = Recorder(reply={"extended_by_s": 25, "capped": True, "max_total_s": 1800,
                         "reason": "extended", "expires_at": "2026-09-27T08:30:00Z"})
    m = manager_holding("lease_B4", 1775, sc)
    seen = logs_from(m._renew_pool_leases)
    line4 = next((msg for lvl, msg in seen if "1775s held" in msg), "")
    check("the extension is reported, not swallowed by capped",
          "+25s" in line4, line4)
    check("it does NOT say there was no extension",
          "NO extension" not in line4, line4)
    check("and it warns that this is the last one",
          "LAST extension" in line4 and "1800s ceiling" in line4, line4)

    print("  [2f] already_later is a MISCONFIGURATION, not an anomaly")
    # #33 gives this its own reason. It means the renewal TTL is smaller than the
    # granted one, so every renewal extends nothing and the lease lapses on its
    # original expiry while the loop reports success — liveness off while looking
    # on. Unreachable at a constant TTL, which is exactly why it must be loud if
    # it ever appears.
    sc = Recorder(reply={"extended_by_s": 0, "capped": False,
                         "reason": "already_later", "max_total_s": 1800})
    m = manager_holding("lease_B5", 300, sc)
    seen = logs_from(m._renew_pool_leases)
    errs = [msg for lvl, msg in seen if lvl == "error"]
    check("it is an error, not a warning", len(errs) == 1, str(seen))
    check("and it says renewals are doing nothing",
          errs and "NOT EXTENDING" in errs[0], str(errs))
    check("it lists the candidate causes rather than asserting one",
          errs and "Either the renewal TTL is below the granted one" in errs[0]
          and "grant exceeded the registry ceiling" in errs[0], str(errs))
    # @claude-wafer-services, 08:00: the adapter grants TURN_TIMEOUT=5400s, three
    # times the ceiling, and gets already_later forever. Blaming the renewal TTL
    # would be "a correct message for the wrong reason" there. The gateway can
    # only hit the first cause, but the message is what a reader has.
    check("and does not assert the cause that only applies to us",
          errs and "so the renewal TTL is smaller" not in errs[0], str(errs))

    print("  [2f2] ttl_below_granted (#34) names its one cause")
    # #34 splits already_later. This half has exactly one cause, so unlike the
    # pre-split value it can assert it rather than list candidates.
    sc = Recorder(reply={"extended_by_s": 0, "capped": False,
                         "reason": "ttl_below_granted", "max_total_s": 1800,
                         "expires_at": "2026-09-27T09:00:00Z"})
    m = manager_holding("lease_B7", 320, sc)
    seen = logs_from(m._renew_pool_leases)
    errs = [msg for lvl, msg in seen if lvl == "error"]
    check("it is an error", len(errs) == 1, str(seen))
    check("it names the single cause", errs and "granted for" in errs[0], str(errs))
    check("and reports the expiry the lease will actually lapse on",
          errs and "2026-09-27T09:00:00Z" in errs[0], str(errs))
    # The pre-#34 value must keep the two-cause wording: the gateway may meet
    # either release, and mapping one onto the other claims precision the
    # response does not carry.
    sc = Recorder(reply={"extended_by_s": 0, "capped": False,
                         "reason": "already_later", "max_total_s": 1800})
    m = manager_holding("lease_B8", 320, sc)
    errs = [m2 for l, m2 in logs_from(m._renew_pool_leases) if l == "error"]
    check("the pre-split value still lists both candidates",
          errs and "Either the renewal TTL is below the granted one" in errs[0], str(errs))

    print("  [2g] a response with no `reason` is a CONTRACT VIOLATION, not an inference")
    # A quiet fallback to the old two-boolean inference is the failure this whole
    # ticket keeps turning up, so the inference announces itself.
    # It used to infer the reason from extended_by_s/capped. That guess cannot
    # tell an OLD plane from a RENAMED field, so it is gone: absent is a third
    # outcome, not a falsy one.
    sc = Recorder(reply={"extended_by_s": 0, "capped": True, "max_total_s": 1800})
    m = manager_holding("lease_B6", 250, sc)
    seen = logs_from(m._renew_pool_leases)
    errs = [msg for lvl, msg in seen if lvl == "error"]
    check("it is an error naming the missing key", len(errs) == 1 and "reason" in errs[0], str(seen))
    check("it says contract violation", errs and "contract violation" in errs[0], str(errs))
    check("it does NOT guess a branch",
          not any("NO extension" in msg or "+" in msg for _, msg in seen), str(seen))

    print("  [11] a RENAMED field is caught — the case no fixture can catch")
    # The whole point. A plane that renamed extended_by_s produces a response the
    # old code read as a real zero: "NO extension" forever, 62 checks green.
    for renamed, gone in (({"extension_s": 60, "capped": False, "reason": "extended",
                            "max_total_s": 1800}, "extended_by_s"),
                          ({"extended_by_s": 60, "was_capped": False, "reason": "extended",
                            "max_total_s": 1800}, "capped")):
        sc = Recorder(reply=renamed)
        m = manager_holding(f"lease_R_{gone}", 300, sc)
        seen = logs_from(m._renew_pool_leases)
        errs = [msg for lvl, msg in seen if lvl == "error"]
        check(f"a renamed {gone} is an error, not a silent branch",
              len(errs) == 1 and gone in errs[0], str(seen))
        check(f"and it lists the keys that DID arrive, so the rename is visible",
              errs and "Keys present" in errs[0], str(errs))

    print("  [2h] at_ceiling logs once, and a CHANGE always logs")
    # admin, reviewing #35: past the ceiling the line repeats forever. At stage (a)
    # every renewal is capped, so a 1500s generation would print ~25 identical
    # lines. Suppress the repeat — but only the repeat.
    sc = Recorder()                                   # default reply: at_ceiling
    m = manager_holding("lease_B9", 200, sc)
    first = logs_from(m._renew_pool_leases)
    second = logs_from(m._renew_pool_leases)
    third = logs_from(m._renew_pool_leases)
    check("the first capped renewal logs",
          any("NO extension" in msg for _, msg in first), str(first))
    check("the second does not repeat it", second == [], str(second))
    check("nor the third", third == [], str(third))
    # The important half: silence must mean "unchanged", never "new but hidden".
    sc.reply = {"extended_by_s": 40, "capped": False, "reason": "extended",
                "max_total_s": 1800, "expires_at": "2026-09-27T09:00:00Z"}
    changed = logs_from(m._renew_pool_leases)
    check("a change of reason logs immediately",
          any("+40s" in msg for _, msg in changed), str(changed))
    sc.reply = {"extended_by_s": 0, "capped": False, "reason": "ttl_below_granted",
                "max_total_s": 1800}
    broke = logs_from(m._renew_pool_leases)
    check("and a change to an error state is never suppressed",
          any(lvl == "error" for lvl, _ in broke), str(broke))

    print("  [12] a lease younger than the threshold is skipped, not renewed")
    # 21:15:19Z: the heartbeat fired 486ms after a grant, `now+600` was microseconds
    # ahead of `granted+600`, the registry's int() of that was 0, and it reported
    # `ttl_below_granted` — a caller misconfiguration that had not happened. The
    # request could not achieve anything, so it is not sent.
    sc = Recorder()
    m = manager_holding("lease_young", 0.4, sc)      # 400ms old
    seen = logs_from(m._renew_pool_leases)
    check("no renewal request was sent", sc.calls == [], str(sc.calls))
    check("nothing logged at info, warning or error",
          not any(lvl in ("info", "warning", "error") for lvl, _ in seen), str(seen))
    check("the lease is still held", "lease_young" in m._pool_leases, str(m._pool_leases))

    print("  [12b] and a lease past the threshold is renewed as normal")
    sc = Recorder()
    m = manager_holding("lease_old_enough", 3.0, sc)
    seen = logs_from(m._renew_pool_leases)
    check("the renewal was sent",
          any(c[0] == "renew" for c in sc.calls), str(sc.calls))
    check("and it logged", any(lvl == "info" for lvl, _ in seen), str(seen))

    print("  [12c] the threshold is small enough not to disturb clause 2")
    # Skipping a whole interval would push the worst-case gap from 60s to ~120s
    # and leave a third of clause 2's margin. 2s leaves it untouched.
    iv = config.EXO_LEASE_RENEW_INTERVAL_S
    check("the skip is far below one interval",
          ModelManager._RENEW_MIN_AGE_S < iv / 10,
          f"{ModelManager._RENEW_MIN_AGE_S}s vs interval {iv}s")
    check("so three attempts still fit inside 80% of the TTL",
          iv * 3 + ModelManager._RENEW_MIN_AGE_S <= config.EXO_LEASE_TTL * 0.8,
          f"{iv*3}+{ModelManager._RENEW_MIN_AGE_S} vs {config.EXO_LEASE_TTL*0.8:.0f}")

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

    print("  [4b] 403 and 422 are NOT transient: renewal is simply not working")
    # These repeat identically every interval. Liveness would be off while the
    # code looks like it is running — #843 exactly. So they are errors, not
    # warnings, and they say that nothing is being kept alive.
    for code, word in ((403, "not the holder"), (422, "bad request body")):
        sc = Recorder(renew=http_error(code))
        m = manager_holding(f"lease_{code}", 150, sc)
        seen = logs_from(m._renew_pool_leases)
        errs = [msg for lvl, msg in seen if lvl == "error"]
        check(f"{code} is logged at error", len(errs) == 1, str(seen))
        check(f"{code} says renewal is not working and names why",
              errs and "NOT WORKING" in errs[0] and word in errs[0], str(errs))
        check(f"{code} still keeps the lease",
              f"lease_{code}" in m._pool_leases, str(m._pool_leases))

    print("  [4c] the real 404 body, which carries expires_at, is still a reap")
    # PR #35 had this defect: its refusal check tested for expires_at in the body
    # and #33's 404 carries it, so a reaped lease read as alive and the action ran
    # on against a grantable pool. This client keys on the status code instead, so
    # the same body must still reach the ERROR branch.
    sc = Recorder(renew=http_error(404, REAPED_404_BODY))
    m = manager_holding("lease_reaped", 410, sc)
    seen = logs_from(m._renew_pool_leases)
    errs = [msg for lvl, msg in seen if lvl == "error"]
    check("a 404 carrying expires_at is an error, not a success",
          len(errs) == 1, str(seen))
    check("and it still says exclusivity is gone",
          errs and "no longer exclusively" in errs[0], str(errs))
    check("nothing was logged at info — a reap must not read as a renewal",
          not any(lvl == "info" for lvl, _ in seen), str(seen))
    check("and the lease stays tracked so shutdown still releases it",
          "lease_reaped" in m._pool_leases, str(m._pool_leases))

    print("  [4d] a 404 on a lease we NO LONGER HOLD is our own race, not a reap")
    # claude-wafer-services, #827: the tick reads the held map under _pool_lock and
    # POSTs outside it — deliberately, since holding a lock across a 10s HTTP call
    # is the wrong trade. A generation finishing in that window releases its lease
    # and the renewal 404s on a lease we no longer own. Median lease 9.4s against a
    # 60s tick, so this is the NORMAL path, and reporting it as a reap would be a
    # false alarm that trains its reader to skip the line.
    class RaceThenGone:
        def __init__(self, mgr): self.mgr = mgr
        def renew_lease(self, lease_id, ttl_seconds):
            # exactly the race: release lands while the POST is in flight
            with self.mgr._pool_lock:
                self.mgr._pool_leases.discard(lease_id)
            raise http_error(404, REAPED_404_BODY)
    m = ModelManager(sc=None)
    m.sc = RaceThenGone(m)
    with m._pool_lock:
        m._pool_leases.add("lease_raced")
        m._pool_lease_acquired["lease_raced"] = time.monotonic() - 70
    seen = logs_from(m._renew_pool_leases)
    check("no error — this is benign",
          not any(lvl == "error" for lvl, _ in seen), str(seen))
    check("and no warning either", not any(lvl == "warning" for lvl, _ in seen), str(seen))
    check("nothing claims the pool was reaped",
          not any("reaped" in msg.lower() or "GONE" in msg for _, msg in seen), str(seen))

    print("  [4e] a 404 on a lease we STILL HOLD is the reap the error exists for")
    sc = Recorder(renew=http_error(404, REAPED_404_BODY))
    m = manager_holding("lease_reaped_held", 410, sc)
    seen = logs_from(m._renew_pool_leases)
    errs = [msg for lvl, msg in seen if lvl == "error"]
    check("it is an error", len(errs) == 1, str(seen))
    check("and it says we still hold it, so the distinction is visible",
          errs and "STILL HOLD" in errs[0], str(errs))
    check("and still names exclusivity", errs and "exclusively" in errs[0], str(errs))

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
    check("it renews anyway", any(c[0] == "renew" and c[1] == "lease_F" for c in sc.calls), str(sc.calls))
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

    print("  [8] b1's invariant swap: the CEILING inherits the old TTL clause")
    # Stage (a) required TTL >= GENERATE_TIMEOUT + 300, because a holder that
    # cannot renew has nothing else protecting it. b1 moves that requirement onto
    # the ceiling, which renewals cannot walk past — getting THIS wrong lapses a
    # lease mid-generation, which is what the old clause existed to prevent.
    check("the ceiling outlives the longest generation",
          config.EXO_LEASE_MAX_TOTAL_S >= config.EXO_GENERATE_TIMEOUT + 300,
          f"ceiling={config.EXO_LEASE_MAX_TOTAL_S} gen={config.EXO_GENERATE_TIMEOUT}")
    check("the TTL is now SHORTER than the longest generation, which is the point",
          config.EXO_LEASE_TTL < config.EXO_GENERATE_TIMEOUT,
          f"ttl={config.EXO_LEASE_TTL} gen={config.EXO_GENERATE_TIMEOUT}")
    check("and three renewal attempts fit inside 80% of the TTL",
          config.EXO_LEASE_RENEW_INTERVAL_S * 3 <= config.EXO_LEASE_TTL * 0.8,
          f"3x{config.EXO_LEASE_RENEW_INTERVAL_S} vs {config.EXO_LEASE_TTL*0.8:.0f}")

    print("  [8b] the grant sends both numbers, explicitly")
    sc = Recorder()
    m = ModelManager(sc=sc)
    # acquire_pool builds the payload; assert what it would send rather than
    # restating it, so a change to either constant follows automatically.
    sent = {}
    def fake_request_lease(**kw):
        sent.update(kw)
        return {"lease_id": "lease_z", "status": "active", "status_code": 200}
    sc.request_lease = fake_request_lease
    m.exo = type("E", (), {"pool_status": staticmethod(lambda: {"ok": True})})()
    try:
        m.acquire_pool("test")
    except Exception:
        pass
    check("ttl_seconds is the liveness number",
          sent.get("ttl_seconds") == config.EXO_LEASE_TTL, str(sent.get("ttl_seconds")))
    check("max_total_s is the exposure number, sent explicitly",
          sent.get("max_total_s") == config.EXO_LEASE_MAX_TOTAL_S, str(sent.get("max_total_s")))
    check("and they are NOT the same number any more",
          config.EXO_LEASE_TTL != config.EXO_LEASE_MAX_TOTAL_S,
          f"{config.EXO_LEASE_TTL} vs {config.EXO_LEASE_MAX_TOTAL_S}")

    print("  [9] two missed renewals leave real slack, not 3 seconds of it")
    # Stage (b)'s invariant is TTL >= 3 * interval. Because the interval is a
    # fraction of the TTL, that invariant is nearly a tautology and does no work
    # unless the fraction is small enough: at 33% the third attempt lands at 98%
    # of the lease, and one slow registry answer (10s timeout) loses it. This is
    # the assertion that would fail if someone raised the percentage back.
    for ttl in (120, 300, 600, 1800):
        iv = max(5, min(config.EXO_LEASE_RENEW_MAX_S,
                        int(ttl * config.EXO_LEASE_RENEW_PCT / 100)))
        third = iv * 3
        check(f"at TTL={ttl}s the third attempt is at {third}s, <=80% of the lease",
              third <= ttl * 0.8, f"interval={iv} third={third} ttl={ttl}")

    print("  [10] a slow renewal must NOT stall the event loop")
    # samclaude-admin, reviewing #10: _renew_pool_leases makes a synchronous
    # httpx POST with the plane's 10s timeout, every interval, for as long as
    # a generation is streaming — on the same loop as that stream. Run on the
    # loop it stalls every token for the duration. And the streaming path is
    # where the 60s inter-token rule lives, so a slow renewal could trip this
    # gateway's own stall detector against a healthy generation.
    #
    # Drives the REAL pool_renewal_loop with a short interval and a fake that
    # takes 2s, and measures a concurrent 0.1s ticker. Against a loop that
    # calls _renew_pool_leases directly, the ticker stalls for ~2s.
    class Slow:
            def renew_lease(self, lease_id, ttl_seconds):
                time.sleep(2.0)
                return {"extended_by_s": 0, "capped": True, "reason": "at_ceiling",
                        "max_total_s": 1800}

    async def scenario():
            real = config.EXO_LEASE_RENEW_INTERVAL_S
            config.EXO_LEASE_RENEW_INTERVAL_S = 0.05
            try:
                m = ModelManager(sc=Slow())
                with m._pool_lock:
                    m._pool_leases.add("lease_slow")
                    m._pool_lease_acquired["lease_slow"] = time.monotonic() - 200
                task = asyncio.create_task(m.pool_renewal_loop())
                await asyncio.sleep(0.12)          # let the renewal start
                gaps, prev = [], time.monotonic()
                for _ in range(6):
                    await asyncio.sleep(0.1)
                    now = time.monotonic()
                    gaps.append(now - prev)
                    prev = now
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return max(gaps)
            finally:
                config.EXO_LEASE_RENEW_INTERVAL_S = real



    worst = asyncio.run(scenario())
    check(f"the worst tick gap stayed under 0.5s (was {worst:.2f}s)",
              worst < 0.5, f"worst gap {worst:.2f}s — the loop was blocked")

    print()
    if failed:
        print(f"  FAILED: {failed} failed, {passed} passed")
        sys.exit(1)
    print(f"  ALL PASSED: pool lease renewal {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
