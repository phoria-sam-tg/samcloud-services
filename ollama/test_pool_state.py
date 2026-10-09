"""#801 step 1: the background pool-state view, and that it never invents a verdict.

WHAT THIS IS FOR

Discovery needs to answer *can this tier be served* without reading exo inline —
`list_models_openai` promises "No pool read at all, so a wedged pool cannot make
discovery hang", and `pool_status_cached` cannot be used there because a cache
MISS still fetches a few hundred KB of `/state` on the request path. So the
answer is maintained out of band and read from memory.

THE PROPERTY THAT MATTERS MOST IS THE THREE-VALUED ONE

`unknown` is a state, not a default. A probe that fails must NOT resolve to
`unplaceable`, because a consumer reads that as "exo says it cannot be served"
when the truth is "we did not find out". On #903, four separate missing lookups
became values inside one evening — `context_length: null` for an absent key,
`host`/`base_url` nulls read as a scope policy, five unresolved callers printed
as `None`, and a `detail` field read as 0 chars because the schema calls it
something else. Each was caught by somebody other than its author, and the last
would have reported another seat as having destroyed a task.

So most of this file drives the failure paths rather than the happy one, and the
check it exists for is that a broken probe and a real "no" are distinguishable.

READ-ONLY BY CONSTRUCTION. The view cannot place or release anything: #801's
criterion 2 is gated on criterion 1, nobody owns the create path, and a view
that could act would be release-without-rebuild — the ratchet both other seats
argued against.

    python -m ollama.test_pool_state
"""
import os
import time

if __package__ in (None, ""):
    print("run as:  python -m ollama.test_pool_state")
    raise SystemExit(1)

os.environ["AUTH_ENABLED"] = "0"

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}")


def section(name):
    print(f"\n{name}")


from . import config            # noqa: E402
from . import manager as mgr_mod  # noqa: E402


class _Resp:
    def __init__(self, code, payload=None, raise_on_json=False):
        self.status_code = code
        self._p = payload
        self._raise = raise_on_json

    def json(self):
        if self._raise:
            raise ValueError("not json")
        return self._p


class _Http:
    """Stands in for the exo client's HTTP session. Records what it was asked."""

    def __init__(self, resp=None, boom=None):
        self.resp, self.boom, self.calls = resp, boom, []

    def get(self, path, params=None, timeout=None):
        self.calls.append((path, params, timeout))
        if self.boom:
            raise self.boom
        return self.resp


def _client(resp=None, boom=None):
    from . import exo_client
    c = exo_client.ExoClient.__new__(exo_client.ExoClient)
    c._http = _Http(resp, boom)
    return c


# ------------------------------------------------- 1. the probe is 3-valued
section("1. placement_available: True / False / None, and None is not False")

MODEL = "mlx-community/gpt-oss-120b-MXFP4-Q8"

c = _client(_Resp(200, {"MlxRingInstance": {"instanceId": "x"}}))
check(c.placement_available(MODEL) is True, "200 with a body -> True")

c = _client(_Resp(400, {"error": {"message":
                                  "No cycles found with sufficient memory"}}))
check(c.placement_available(MODEL) is False,
      "400 'No cycles found with sufficient memory' -> False, a real capacity no")

# Everything below must be None. These are the cases that would otherwise
# become a verdict.
for label, cl in (
        ("connection refused", _client(boom=OSError("refused"))),
        ("timeout", _client(boom=TimeoutError("timed out"))),
        ("500 from exo", _client(_Resp(500, {"error": "boom"}))),
        ("503 from exo", _client(_Resp(503, None))),
        ("404 — route gone", _client(_Resp(404, None))),
        ("200 but unparseable body", _client(_Resp(200, None, raise_on_json=True))),
        ("400 for a DIFFERENT reason", _client(_Resp(400, {"error": {
            "message": "Field required: model_id"}}))),
        ("400 with an unreadable body", _client(_Resp(400, None,
                                                     raise_on_json=True))),
):
    got = cl.placement_available(MODEL)
    check(got is None, f"{label} -> None (got {got!r}) — NOT False")

# The 400-for-another-reason case is the sharpest: it is a 400, like the real
# capacity answer, and it means something completely different.
c = _client(_Resp(400, {"error": {"message": "Field required: model_id"}}))
check(c.placement_available(MODEL) is not False,
      "a malformed-request 400 is not read as 'cannot be placed' — we asked "
      "the question wrong, which is not the pool saying no")

c = _client(_Resp(200, {"MlxRingInstance": {}}))
c.placement_available(MODEL)
path, params, timeout = c._http.calls[0]
check(path == "/instance/placement", "probes exo's own placement planner")
check(params == {"model_id": MODEL}, "passes the model as a query param")
check(timeout == config.EXO_PLACEMENT_TIMEOUT_S,
      f"bounded at EXO_PLACEMENT_TIMEOUT_S ({timeout}s) — a planner, not a lookup")

# ------------------------------------------------- 2. read-only
section("2. the probe is READ-ONLY, asserted on the verb")

c = _client(_Resp(200, {"MlxRingInstance": {}}))
c.placement_available(MODEL)
check(all(hasattr(c._http, "get") for _ in [0]), "only `get` is used")
check(not hasattr(c._http, "post_called"),
      "nothing posts — criterion 2 is gated on criterion 1 and the create path "
      "has no owner, so a view that could act would be a ratchet")


# ------------------------------------------------- 3. the manager view
section("3. pool_state(): no network, and `unknown` survives every failure")


class _Mgr:
    pool_state = mgr_mod.ModelManager.pool_state
    refresh_pool_state = mgr_mod.ModelManager.refresh_pool_state

    def __init__(self, exo=None):
        self.exo = exo
        self._pool_state = None
        self._pool_last_model = None


class _Exo:
    def __init__(self, status=None, boom=None, avail=None):
        self._status, self._boom, self._avail = status, boom, avail
        self.placement_calls = 0

    def pool_status(self):
        if self._boom:
            raise self._boom
        return self._status

    def placement_available(self, model):
        self.placement_calls += 1
        return self._avail


m = _Mgr()
st = m.pool_state()
check(st["state"] == "unknown", "before any probe -> unknown")
check(st["checked_at"] is None and st["age_s"] is None,
      "and it says so rather than reporting an age it does not have")

m = _Mgr(_Exo(status={"resident_model": MODEL, "ready": True, "busy": False}))
m.refresh_pool_state()
st = m.pool_state()
check(st["state"] == "resident", "resident+ready -> resident")
check(st["model"] == MODEL, "and names the model")
check(isinstance(st["age_s"], float), "carries age_s, so staleness is visible")
check(m._pool_last_model == MODEL, "remembers the model for a later probe")

m = _Mgr(_Exo(boom=OSError("pool unreachable")))
m.refresh_pool_state()
st = m.pool_state()
check(st["state"] == "unknown",
      "pool unreachable -> unknown, NOT unplaceable: we learned nothing about "
      "placement, only that we could not ask")
check("unreachable" in (st.get("reason") or ""), "and the reason says which")

# not resident, nothing ever seen -> unknown, not a guess
m = _Mgr(_Exo(status={"resident_model": None, "ready": False}))
m.refresh_pool_state()
check(m.pool_state()["state"] == "unknown",
      "not resident and no model ever seen -> unknown rather than probing for "
      "a model we would have had to invent")
check(m.exo.placement_calls == 0, "and no probe is made with a guessed model")

# not resident, model known -> the probe decides, three ways
for avail, want in ((True, "placeable"), (False, "unplaceable"), (None, "unknown")):
    e = _Exo(status={"resident_model": None, "ready": False}, avail=avail)
    m = _Mgr(e)
    m._pool_last_model = MODEL
    m.refresh_pool_state()
    got = m.pool_state()["state"]
    check(got == want, f"placement_available={avail!r} -> {want} (got {got})")
    check(e.placement_calls == 1, f"  probed once for {want}")

e = _Exo(status={"resident_model": None, "ready": False}, avail=None)
m = _Mgr(e)
m._pool_last_model = MODEL
m.refresh_pool_state()
check("probe failed" in (m.pool_state().get("reason") or ""),
      "a failed probe says so, so `unknown` is never silent about why")

section("4. the three states are distinguishable, which is the whole point")

states = set()
for avail in (True, False, None):
    e = _Exo(status={"resident_model": None, "ready": False}, avail=avail)
    m = _Mgr(e)
    m._pool_last_model = MODEL
    m.refresh_pool_state()
    states.add(m.pool_state()["state"])
check(states == {"placeable", "unplaceable", "unknown"},
      f"three inputs give three distinct states ({sorted(states)}) — a "
      f"two-valued view would collapse 'did not find out' into 'no'")

section("5. a box with no pool acquires no loop and no verdict")

m = _Mgr(None)
m.refresh_pool_state()
st = m.pool_state()
check(st["state"] == "unknown" and "not enabled" in (st.get("reason") or ""),
      "no exo client -> unknown, reason names it (wafer runs this gateway "
      "with EXO_ENABLED=0 and fronts no pool)")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
