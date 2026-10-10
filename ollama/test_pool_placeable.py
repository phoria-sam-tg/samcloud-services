"""The pool tier's offer distinguishes "not placed" from "cannot be placed" (#801).

Criterion 3 of task 801 asks the listing to show `think` as available when it is
resident **or can be placed**. Before this, `pool_offer()` had one answer for
every not-resident pool -- `blocked` -- which is a true statement about now and a
silent one about whether a request could make the tier serve. That distinction is
what on-demand placement turns on, so not-resident is three answers:

  placeable                 exo says a ring cycle has room for it
  pool_unplaceable          exo says it does not
  pool_placement_unknown    WE DID NOT FIND OUT

The last one is the case these tests exist for. A probe that fails must not read
as a refusal, and an absent key must not read as a failed probe.

Run from the repo ROOT: python -m ollama.test_pool_placeable
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.manager import ModelManager            # noqa: E402
from ollama.exo_client import ExoClient            # noqa: E402
from ollama import config                          # noqa: E402

PASS = FAIL = 0
TIER = config.EXO_TIERS[0] if config.EXO_TIERS else "think"
WANT = "mlx-community/gpt-oss-120b-MXFP4-Q8"


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


def mgr(record=None, record_raises=False, placeable=None, probe_calls=None):
    """A manager whose registry returns `record` and whose exo answers `placeable`."""
    sc = MagicMock()
    if record_raises:
        sc.get_resource.side_effect = RuntimeError("registry down")
    else:
        sc.get_resource.return_value = record
    m = ModelManager(sc=sc)
    m.exo = MagicMock(spec=ExoClient)

    def _probe(model_id):
        if probe_calls is not None:
            probe_calls.append(model_id)
        return placeable
    m.exo.placement_available.side_effect = _probe
    return m


def view(m, status):
    """Install `status` as a fresh view, the way pool_watch_loop would."""
    import time
    m._pool_view = (time.monotonic(), status)


NOT_READY = {"ready": False, "resident_model": None, "busy": False,
             "instances": [], "unavailable_reason": "no_instance"}


def main():
    print("\n  pool tier: placeable vs unplaceable vs unknown\n")

    # --- the three not-resident answers ---------------------------------
    m = mgr(record={"model": WANT}, placeable=True)
    view(m, {**NOT_READY, "placeable": True, "desired_model": WANT,
             "placeable_reason": None})
    o = m.pool_offer()
    check("exo says there is room -> state `placeable`, not `blocked`",
          o["state"] == "placeable", str(o))
    check("...and it names the model a request would place",
          o.get("model") == WANT, str(o))
    check("...and reports not busy, since nothing is running on it",
          o.get("busy") is False, str(o))

    m = mgr(record={"model": WANT}, placeable=False)
    view(m, {**NOT_READY, "placeable": False, "desired_model": WANT,
             "placeable_reason": None})
    o = m.pool_offer()
    check("exo says no cycle has room -> blocked, reason pool_unplaceable",
          o["state"] == "blocked" and o["reason"] == "pool_unplaceable", str(o))

    m = mgr(record={"model": WANT}, placeable=None)
    view(m, {**NOT_READY, "placeable": None, "desired_model": WANT,
             "placeable_reason": None})
    o = m.pool_offer()
    check("the probe failed -> pool_placement_unknown, NOT pool_unplaceable",
          o["reason"] == "pool_placement_unknown",
          "a probe we could not complete must never read as exo's refusal: " + str(o))
    check("...and the detail says so in words",
          "not established" in o["detail"], str(o))

    # --- an ABSENT key is not a failed probe ----------------------------
    m = mgr(record={"model": WANT})
    view(m, {"ready": True, "resident_model": WANT, "busy": False, "instances": []})
    o = m.pool_offer()
    check("pool ready -> `resident`, and no placeable key is consulted",
          o["state"] == "resident", str(o))

    m = mgr(record={"model": WANT})
    view(m, dict(NOT_READY))          # no `placeable` key at all
    o = m.pool_offer()
    check("not ready and the key is ABSENT -> the original blocked reason",
          o["state"] == "blocked" and o["reason"] == "pool_no_instance",
          "absent must not collapse into the None that means `probed and could "
          "not tell`: " + str(o))

    # --- staleness still wins -------------------------------------------
    import time as _t
    m = mgr(record={"model": WANT})
    m._pool_view = (_t.monotonic() - (config.EXO_POOL_VIEW_MAX_AGE_S + 5),
                    {**NOT_READY, "placeable": True, "desired_model": WANT})
    o = m.pool_offer()
    check("a STALE view reads as unknown even when it said placeable",
          o["state"] == "unknown" and o["reason"] == "pool_view_stale", str(o))

    # --- the desired model comes from the record ------------------------
    m = mgr(record={"model": WANT})
    check("_pool_desired_model reads the record's model field",
          m._pool_desired_model() == WANT)

    m = mgr(record={"id": "x"})
    check("a record with NO model key -> None, not a guess",
          m._pool_desired_model() is None)

    m = mgr(record={"model": "   "})
    check("a record naming only whitespace -> None",
          m._pool_desired_model() is None)

    m = mgr(record=None)
    check("a null record -> None",
          m._pool_desired_model() is None)

    m = mgr(record_raises=True)
    check("an unreadable registry -> None, and it does not raise",
          m._pool_desired_model() is None)

    # the cache: one registry call, not one per probe
    m = mgr(record={"model": WANT})
    m._pool_desired_model(); m._pool_desired_model(); m._pool_desired_model()
    check("the desired model is cached, so the watch loop does not poll the registry",
          m.sc.get_resource.call_count == 1,
          f"calls={m.sc.get_resource.call_count}")

    # --- no model named -> exo is never asked ---------------------------
    calls = []
    m = mgr(record={"id": "x"}, placeable=True, probe_calls=calls)
    placeable, model, why = m._probe_placeable()
    check("no model in the record -> exo is NOT asked",
          calls == [] and placeable is None and why == "desired_model_unknown",
          f"calls={calls} placeable={placeable} why={why}")

    calls = []
    m = mgr(record={"model": WANT}, placeable=True, probe_calls=calls)
    placeable, model, why = m._probe_placeable()
    check("a named model IS asked about, by its record id",
          calls == [WANT] and placeable is True, f"calls={calls}")

    # --- the watch loop must not probe a READY pool ---------------------
    # While the pool is resident the planner is being asked whether a SECOND
    # copy fits, which on a 61 GB placement always answers no -- recording
    # `unplaceable` for a tier that is serving perfectly well.
    calls = []
    m = mgr(record={"model": WANT}, placeable=False, probe_calls=calls)
    m.exo.pool_status.return_value = {"ready": True, "resident_model": WANT,
                                      "busy": False, "instances": []}

    asyncio.run(one_poll_for(m))
    check("a READY pool is never probed for placement",
          calls == [], f"the planner was asked {len(calls)} time(s) about a "
                       f"pool that is already serving: {calls}")
    check("...and the view carries no placeable key for it",
          m._pool_view is not None and "placeable" not in m._pool_view[1],
          str(m._pool_view))

    calls = []
    m = mgr(record={"model": WANT}, placeable=True, probe_calls=calls)
    m.exo.pool_status.return_value = dict(NOT_READY)
    asyncio.run(one_poll_for(m))
    check("a NOT-ready pool IS probed, once per poll",
          calls == [WANT], f"calls={calls}")
    check("...and the result is recorded on the view",
          m._pool_view[1].get("placeable") is True, str(m._pool_view[1]))

    # --- the probe's own three-valued contract --------------------------
    for code, body, want, desc in [
        (200, {"MlxRingInstance": {"instanceId": "x"}}, True,
         "200 with a spec -> True"),
        (400, {"error": {"message": "No cycles found with sufficient memory"}}, False,
         "400 `sufficient memory` -> False, a real capacity verdict"),
        (400, {"error": {"message": "unknown model_id"}}, None,
         "a DIFFERENT 400 -> None; we asked the question wrong"),
        # A 200 IS NOT A YES. exo answers some refusals 200-with-an-error-body —
        # the shape that had the placement guard POST 47 error bodies back as
        # placements. `bool({"error": ...})` is True, so before this the probe
        # advertised `placeable` and the offer promised a placement that was
        # never coming, with no row in this matrix to catch it.
        (200, {"error": {"message": "No cycles found with sufficient memory"}}, False,
         "200 carrying a capacity refusal -> False, NOT True"),
        (200, {"detail": "x"}, None,
         "200 carrying a detail refusal -> None, NOT True"),
        (200, [1, 2], None, "200 carrying a list -> None, not a truthy yes"),
        (503, {"detail": "nope"}, None, "503 -> None"),
        (200, None, None, "200 with an empty body -> None"),
    ]:
        c = ExoClient.__new__(ExoClient)
        resp = MagicMock()
        resp.status_code = code
        if body is None:
            resp.json.side_effect = ValueError("not json")
        else:
            resp.json.return_value = body
        c._http = MagicMock()
        c._http.get.return_value = resp
        check(f"placement_available: {desc}",
              c.placement_available(WANT) is want,
              f"got {c.placement_available(WANT)!r}")

    c = ExoClient.__new__(ExoClient)
    c._http = MagicMock()
    c._http.get.side_effect = RuntimeError("connection refused")
    check("placement_available: a transport failure -> None, never False",
          c.placement_available(WANT) is None)

    # --- criterion 3: the reason names the tenant, not the planner -------
    # "No cycles found with sufficient memory" is the planner's answer and a
    # caller can do nothing with it. On slice the gateway is the only thing that
    # loads a model, so it knows what has to clear and can say so.
    from ollama.manager import ManagedModel, Backend
    import time as _time

    def with_resident(*specs):
        m = mgr(record={"model": WANT}, placeable=False)
        for name, mb, idle in specs:
            m.models[name] = ManagedModel(
                name=name, backend=Backend.OLLAMA, memory_mb=mb, lease_id="l",
                port=11434, loaded_at=_time.time() - 600,
                last_used=_time.time() - idle, request_count=3)
        view(m, {**NOT_READY, "placeable": False, "desired_model": WANT,
                 "placeable_reason": None})
        return m, m.pool_offer()

    m, o = with_resident(("qwen3.8:27b-mlx", 17530, 42))
    check("64. the reason NAMES the resident model, not just 'no room'",
          "qwen3.8:27b-mlx" in o["detail"], o["detail"][:160])
    check("64. ...with its size, so a caller can see what has to clear",
          "17,530 MB" in o["detail"], o["detail"][:160])
    check("64. ...and says the tier returns when it clears, not that it is a fault",
          "becomes available when that clears" in o["detail"]
          and "nothing to report" in o["detail"], o["detail"][:200])
    check("64. ...and the blockers are machine-readable too",
          [b["name"] for b in o.get("blockers") or []] == ["qwen3.8:27b-mlx"],
          str(o.get("blockers")))
    check("64. ...while the reason code stays pool_unplaceable",
          o["reason"] == "pool_unplaceable" and o["state"] == "blocked", str(o))

    m, o = with_resident(("small", 1200, 5), ("qwen3.8:27b-mlx", 17530, 42))
    check("65. several residents -> largest first, which is what has to clear",
          [b["name"] for b in o["blockers"]] == ["qwen3.8:27b-mlx", "small"],
          str(o["blockers"]))
    check("65. ...and both are named in the sentence",
          "qwen3.8:27b-mlx" in o["detail"] and "small" in o["detail"]
          and "are resident" in o["detail"], o["detail"][:200])

    # NOTHING of ours resident, and the planner still refuses. A different fact:
    # there is no clearance coming, so a caller must not wait for one. This is
    # the case measured on slice 2026-10-09T22:17Z — pool cleared by hand, 0
    # instances, and the 120b still unplaceable because the ring was 14 GiB short.
    m = mgr(record={"model": WANT}, placeable=False)
    view(m, {**NOT_READY, "placeable": False, "desired_model": WANT,
             "placeable_reason": None})
    o = m.pool_offer()
    check("66. nothing of ours resident -> says the shortfall is the RING's",
          "nothing of ours is resident" in o["detail"]
          and "rather than a tenant to wait for" in o["detail"], o["detail"][:200])
    check("66. ...and does not invent a blocker",
          o.get("blockers") == [], str(o.get("blockers")))

    # The tier must never list itself as its own blocker — `resolve_exo_tier`
    # registers it on first use and nothing removes it.
    m = mgr(record={"model": WANT}, placeable=False)
    m.models[TIER] = ManagedModel(name=TIER, backend=Backend.EXO, memory_mb=0,
                                  lease_id=None, port=52415,
                                  loaded_at=_time.time(), last_used=_time.time(),
                                  request_count=1, managed=False)
    view(m, {**NOT_READY, "placeable": False, "desired_model": WANT,
             "placeable_reason": None})
    o = m.pool_offer()
    check("67. the exo tier is never its own blocker",
          o.get("blockers") == [] and "nothing of ours is resident" in o["detail"],
          str(o.get("blockers")) + " / " + o["detail"][:120])

    print(f"\n  {'all checks passed' if not FAIL else 'FAILED'}: "
          f"{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


async def _one_poll(m):
    task = asyncio.create_task(m.pool_watch_loop())
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def one_poll_for(m):
    return _one_poll(m)


if __name__ == "__main__":
    sys.exit(main())
