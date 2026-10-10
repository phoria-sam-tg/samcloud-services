"""Placing and unplacing an exo instance — the two calls that change the pool (#801 step 2).

Before this the gateway could only READ the pool: `ExoClient` had state,
pool_status, resident_model, list_models and the chat paths, and nothing that
could create or remove an instance. `exo-placement-guard.sh` did that. Criteria 1
and 2 of task 801 need both halves in the gateway, and they land together on
purpose: a release path without a place path is a ratchet that converts "holds 61
GiB indefinitely" into "gone until a human notices", on a timer.

The cases are mostly about refusals that look like answers:

  a planner refusal arrives as a 200-shaped body, `{"error": {...}}`
  a POST's 200 means QUEUED, not placed
  a DELETE's False means "we do not know", not "already gone"

Run from the repo ROOT: python -m ollama.test_pool_place
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.exo_client import ExoClient            # noqa: E402
from ollama import config                          # noqa: E402

PASS = FAIL = 0
MODEL = "mlx-community/gpt-oss-120b-MXFP4-Q8"
SPEC = {"MlxRingInstance": {"instanceId": "abc", "shardAssignments": {"modelId": MODEL}}}


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


def client(status=200, body=None, raises=False):
    c = ExoClient.__new__(ExoClient)
    c._http = MagicMock()
    if raises:
        for m in ("get", "post", "delete"):
            getattr(c._http, m).side_effect = RuntimeError("connection refused")
        return c
    r = MagicMock()
    r.status_code = status
    if body is _NOJSON:
        r.json.side_effect = ValueError("not json")
    else:
        r.json.return_value = body
    for m in ("get", "post", "delete"):
        getattr(c._http, m).return_value = r
    return c


class _NoJson:
    pass


_NOJSON = _NoJson()


def main():
    print("\n  exo place / unplace\n")

    # --- placement_spec: a refusal is 200-shaped -------------------------
    check("a real spec comes back as a dict",
          client(200, SPEC).placement_spec(MODEL) == SPEC)

    check("{\"error\": ...} is NOT a spec, however well-formed",
          client(200, {"error": {"message": "No cycles found with sufficient memory"}})
          .placement_spec(MODEL) is None,
          "this is the body that became 47 junk POSTs on 2026-10-07/08")

    check("{\"detail\": ...} is not a spec either",
          client(200, {"detail": "nope"}).placement_spec(MODEL) is None)

    check("an empty dict is not a spec",
          client(200, {}).placement_spec(MODEL) is None)

    check("a list is not a spec",
          client(200, [1, 2]).placement_spec(MODEL) is None)

    check("a 400 is not a spec",
          client(400, {"error": {"message": "x"}}).placement_spec(MODEL) is None)

    check("an unparseable body is not a spec",
          client(200, _NOJSON).placement_spec(MODEL) is None)

    check("a transport failure is not a spec",
          client(raises=True).placement_spec(MODEL) is None)

    c = client(200, SPEC)
    c.placement_spec(MODEL, min_nodes=2)
    _, kw = c._http.get.call_args
    check("min_nodes is passed to the planner when given",
          kw["params"].get("min_nodes") == 2, str(kw))
    c = client(200, SPEC)
    c.placement_spec(MODEL)
    _, kw = c._http.get.call_args
    check("...and omitted when not, rather than sent as None",
          "min_nodes" not in kw["params"], str(kw))

    # --- place: the body shape and what success means -------------------
    c = client(200, {"commandId": "cmd-1"})
    cid = c.place(SPEC)
    _, kw = c._http.post.call_args
    check("place() returns exo's command id", cid == "cmd-1", repr(cid))
    check("...and wraps the spec as {\"instance\": spec}, not the bare spec",
          list(kw["json"].keys()) == ["instance"] and kw["json"]["instance"] is SPEC,
          str(kw.get("json"))[:120])

    check("command_id (snake) is accepted too",
          client(200, {"command_id": "cmd-2"}).place(SPEC) == "cmd-2")

    check("a 200 with NO command id is not a placement",
          client(200, {"ok": True}).place(SPEC) is None,
          "a 200 means queued; without a command id there is nothing to watch")

    check("a non-200 is not a placement",
          client(500, {"commandId": "x"}).place(SPEC) is None)

    check("an unparseable 200 is not a placement",
          client(200, _NOJSON).place(SPEC) is None)

    check("a transport failure is not a placement",
          client(raises=True).place(SPEC) is None)

    # --- unplace: only a 200 is true ------------------------------------
    check("unplace() is True on 200", client(200, {}).unplace("abc") is True)
    check("...False on 404 — NOT read as `already gone`",
          client(404, {"detail": "no such instance"}).unplace("abc") is False)
    check("...False on 500", client(500, {}).unplace("abc") is False)
    check("...False on a transport failure", client(raises=True).unplace("abc") is False)

    c = client(200, {})
    c.unplace("inst-7")
    args, _ = c._http.delete.call_args
    check("...and it deletes the instance it was given",
          args[0] == "/instance/inst-7", str(args))

    # --- the two timeouts are distinct ----------------------------------
    check("the mutating calls get their own, longer timeout",
          config.EXO_PLACE_TIMEOUT_S > config.EXO_PLACEMENT_TIMEOUT_S,
          f"place={config.EXO_PLACE_TIMEOUT_S} probe={config.EXO_PLACEMENT_TIMEOUT_S}")

    c = client(200, {"commandId": "x"})
    c.place(SPEC)
    _, kw = c._http.post.call_args
    check("...and place() uses it",
          kw.get("timeout") == config.EXO_PLACE_TIMEOUT_S, str(kw.get("timeout")))

    # --- the ratchet guard ----------------------------------------------
    # Both halves exist on the client, so nothing can ship a release path
    # without a rebuild path. This is an assertion about the module, not a
    # behaviour test, and it is here because the ordering is the risk.
    for name in ("placement_spec", "place", "unplace"):
        check(f"ExoClient.{name} exists", hasattr(ExoClient, name))

    print(f"\n  {'all checks passed' if not FAIL else 'FAILED'}: "
          f"{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
