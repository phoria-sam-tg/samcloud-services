"""A release that fails is remembered and retried (#938).

Measured on slice 2026-10-09: `DELETE /leases/lease_40ceb8ea4112d095` answered
**502** at 22:36:21Z, one second inside the 0.12.81 deploy's registry restart.
The old code logged it and dropped the id, so the lease read `active` with
17,505 MB for the rest of its 3600s TTL -- and `gpu-0` showed two active leases
totalling 34,838 MB for one 18 GB model.

The window is about a second, so this is not a thing care prevents: every deploy
restarts the registry. The asymmetry the same second demonstrated is the point --
the only other call caught was a periodic stats push, which the next tick
overwrote. **In a cutover, periodic calls self-heal and one-shot state
transitions orphan.**

Run from the repo ROOT: python -m ollama.test_lease_release_retry
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.manager import ModelManager, ManagedModel, Backend   # noqa: E402
from ollama import config                                        # noqa: E402

PASS = FAIL = 0


def check(desc, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[0;32mok\033[0m   {desc}")
    else:
        FAIL += 1
        print(f"  \033[0;31mFAIL\033[0m {desc}")
        if detail:
            print(f"       {detail}")


def http_error(status):
    """An exception shaped like httpx's, carrying a response with a status."""
    e = RuntimeError(f"Server error '{status}'")
    e.response = MagicMock()
    e.response.status_code = status
    return e


def mgr(release_effects):
    """`release_effects` is a list consumed one per release_lease call."""
    sc = MagicMock()
    calls = []

    def _rel(lid):
        calls.append(lid)
        eff = release_effects.pop(0) if release_effects else None
        if isinstance(eff, Exception):
            raise eff
        return {"status": "released"}
    sc.release_lease.side_effect = _rel
    m = ModelManager(sc=sc)
    m.exo = MagicMock()
    return m, calls


def main():
    print("\n  a failed lease release is remembered and retried\n")

    # --- the 502 that started this --------------------------------------
    m, calls = mgr([http_error(502)])
    m._release_lease_quietly("lease_abc")
    check("a 502 on release leaves the id queued, not dropped",
          list(m._unreleased) == ["lease_abc"], str(m._unreleased))
    check("...and it was actually attempted once", calls == ["lease_abc"], str(calls))

    m.sc.release_lease.side_effect = lambda lid: {"status": "released"}
    m._retry_unreleased()
    check("the retry releases it and clears the queue",
          m._unreleased == {}, str(m._unreleased))

    # --- a success is not queued ----------------------------------------
    m, calls = mgr([None])
    m._release_lease_quietly("lease_ok")
    check("a successful release queues nothing", m._unreleased == {}, str(m._unreleased))

    # --- 404 is GONE, which is the answer we wanted ---------------------
    m, calls = mgr([http_error(404)])
    m._release_lease_quietly("lease_404")
    check("a 404 counts as released — the registry no longer accounts for it",
          m._unreleased == {},
          "retrying a lease that cannot be released would never drain: "
          + str(m._unreleased))

    # --- the queue does not grow without bound --------------------------
    m, calls = mgr([])
    m.sc.release_lease.side_effect = lambda lid: (_ for _ in ()).throw(http_error(502))
    m._release_lease_quietly("lease_stuck")
    check("a persistently failing release stays queued meanwhile",
          list(m._unreleased) == ["lease_stuck"])
    # age it past the TTL
    import time as _t
    m._unreleased["lease_stuck"] = (_t.monotonic() - (config.LEASE_TTL + 10),
                                    config.LEASE_TTL)
    m._retry_unreleased()
    check("past its TTL the entry is dropped — the registry has expired it anyway",
          m._unreleased == {},
          "a queue that never drains would grow for the life of the process: "
          + str(m._unreleased))

    # THE GIVE-UP CLOCK IS THAT LEASE'S OWN. These ids are MODEL leases under
    # LEASE_TTL (3600s); the exo pool's bound is EXO_LEASE_MAX_TOTAL_S (1800s).
    # Using the pool's number gave up at the half-way point of the very window
    # #938 measured — one hour of a wrong ledger — leaving the second half
    # uncovered. One row each side of 1800s.
    assert config.EXO_LEASE_MAX_TOTAL_S < config.LEASE_TTL, "fixture assumes 1800 < 3600"
    mid = (config.EXO_LEASE_MAX_TOTAL_S + config.LEASE_TTL) / 2   # ~2700s

    m, calls = mgr([])
    m.sc.release_lease.side_effect = lambda lid: (_ for _ in ()).throw(http_error(502))
    m._release_lease_quietly("lease_mid")
    m._unreleased["lease_mid"] = (_t.monotonic() - mid, config.LEASE_TTL)
    m._retry_unreleased()
    check(f"an entry aged {int(mid)}s — past the POOL bound, inside its own TTL — "
          f"is still retried",
          list(m._unreleased) == ["lease_mid"],
          "the old clock dropped this, abandoning half the window: "
          + str(m._unreleased))

    m, calls = mgr([])
    m.sc.release_lease.side_effect = lambda lid: (_ for _ in ()).throw(http_error(502))
    m._release_lease_quietly("lease_old")
    m._unreleased["lease_old"] = (_t.monotonic() - (config.LEASE_TTL + 60),
                                  config.LEASE_TTL)
    m._retry_unreleased()
    check("...and one past its own TTL is dropped",
          m._unreleased == {}, str(m._unreleased))

    # An explicit TTL is honoured, so a pool lease reaching this path gets its
    # own bound rather than the model default.
    m, calls = mgr([])
    m.sc.release_lease.side_effect = lambda lid: (_ for _ in ()).throw(http_error(502))
    m._release_lease_quietly("lease_pool", ttl_seconds=config.EXO_LEASE_MAX_TOTAL_S)
    check("an explicit ttl_seconds is carried on the entry",
          m._unreleased["lease_pool"][1] == config.EXO_LEASE_MAX_TOTAL_S,
          str(m._unreleased))

    # --- the same id is not queued twice --------------------------------
    m, calls = mgr([])
    m.sc.release_lease.side_effect = lambda lid: (_ for _ in ()).throw(http_error(502))
    m._release_lease_quietly("lease_dup")
    first = m._unreleased["lease_dup"][0]
    m._release_lease_quietly("lease_dup")
    check("a repeat failure does not reset the entry's age",
          list(m._unreleased) == ["lease_dup"] and m._unreleased["lease_dup"][0] == first,
          "resetting the clock would make the TTL give-up unreachable: "
          + str(m._unreleased))

    # --- None is not a lease --------------------------------------------
    m, calls = mgr([])
    m._release_lease_quietly(None)
    check("None releases nothing and queues nothing",
          calls == [] and m._unreleased == {})

    # --- THE UNLOAD PATH, which is where the orphan was born ------------
    # `del self.models[name]` is unconditional, so the release there is the last
    # moment the id exists. It must go through the remembering path.
    m, calls = mgr([http_error(502)])
    m.models["q"] = ManagedModel(name="q", backend=Backend.OLLAMA, memory_mb=4006,
                                 lease_id="lease_unload", port=11434,
                                 loaded_at=0.0, last_used=0.0, request_count=1)
    m.ollama = MagicMock()
    try:
        m.unload("q")
    except Exception as e:
        check("unload() ran", False, f"raised {type(e).__name__}: {e}")
    else:
        check("unload() with a failing release still forgets the model",
              "q" not in m.models)
        check("...but REMEMBERS the lease it could not release",
              list(m._unreleased) == ["lease_unload"],
              "this is the exact shape that orphaned 17,505 MB: " + str(m._unreleased))

    # --- the retry is driven by a loop that already exists --------------
    import inspect
    src = inspect.getsource(ModelManager.lease_renewal_loop)
    check("lease_renewal_loop drives the retry, rather than a new task",
          "_retry_unreleased" in src, src[-200:])

    print(f"\n  {'all checks passed' if not FAIL else 'FAILED'}: "
          f"{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
