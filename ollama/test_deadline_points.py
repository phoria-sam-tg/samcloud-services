"""#908: the deadline primitives are published, and published as points.

WHY THIS FILE EXISTS

Four shapes were proposed for "what can this box prefill inside its deadline"
and all four were rejected on measurement: a derived `serveable` (the quantity
lies outside the data, and the four defensible derivations spread 1.7x), a
fitted curve (worst residual +40.2% over 22 points), a scalar rate (the average
rate declines with N, 104.3 tok/s at 8,903 against 66.6 at 112,682), and a
decode reserve (the consumer's reply distribution, which moved 5,087 -> 7,223
inside one hour).

So the thing that can regress here is not a number being wrong — it is one of
those four shapes coming BACK, as a convenience, because points are awkward to
consume. Most of what follows asserts absences, and that is deliberate.

NOTHING HERE READS SOURCE TEXT. Three assertions in `test_stream_deadlines.py`
were once satisfied by their own explanatory comments, and this package's files
document their own history, so they contain every string they ever got wrong.
Every check below either calls a function or drives the ASGI app.

    python -m ollama.test_deadline_points
"""
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    print("run as:  python -m ollama.test_deadline_points")
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


from . import config          # noqa: E402
from . import deadline_points as dp   # noqa: E402

MEASURED = "qwen3.8:27b-mlx"

# --------------------------------------------------------------- the block
section("1. the block carries the three primitives and nothing derived")

b = dp.for_model(MEASURED, 1800)
check(b["generate_timeout_s"] == 1800,
      "generate_timeout_s is the value PASSED IN, not a second copy of config")
check(isinstance(b["prefill_points"], list) and b["prefill_points"],
      "prefill_points is a non-empty list for a measured model")
check(isinstance(b["decode_points"], list) and b["decode_points"],
      "decode_points is a non-empty list for a measured model")

# The four rejected shapes, asserted as ABSENT. A future edit that adds any of
# them has to delete a check that says why it was rejected.
for banned in ("serveable", "serveable_tokens", "max_prompt_tokens",
               "prefill_rate", "prefill_tok_per_s", "decode_reserve",
               "reserve_tokens", "curve", "fit", "coefficients"):
    check(banned not in b,
          f"no `{banned}` — a derived single figure hides the spread inside itself")

check(not any(k.endswith("_rate") for k in b),
      "no field named *_rate at all: a scalar in a schema reads as a constant")

# ------------------------------------------------------------ the empty set
section("2. an unmeasured model publishes EMPTY, not absent")

u = dp.for_model("no-such-model-anywhere", 1800)
check(u["prefill_points"] == [], "unmeasured -> prefill_points == []")
check(u["decode_points"] == [], "unmeasured -> decode_points == []")
check("prefill_points" in u and "decode_points" in u,
      "the KEYS are present — absence resolves to a default in every consumer "
      "chain read so far, and the default is the largest number available")
check(u["generate_timeout_s"] == 1800,
      "and the timeout is still published: a box that pins no window can still "
      "state a deadline")

# ------------------------------------------------------------- provenance
section("3. every point carries n, a date, and where it came from")

for i, p in enumerate(b["prefill_points"]):
    tag = f"prefill[{i}] N={p.get('prompt_tokens')}"
    check(isinstance(p.get("n"), int) and p["n"] >= 1, f"{tag}: n >= 1")
    check(bool(p.get("measured")), f"{tag}: carries the date measured")
    check(p.get("source") in ("seat", "probe"), f"{tag}: source is seat|probe")
    lo, hi = p["tok_per_s_range"]
    check(lo <= p["tok_per_s"] <= hi, f"{tag}: the rate lies inside its spread")
    check(p["prefill_seconds"] > 0 and p["prompt_tokens"] > 0,
          f"{tag}: positive")

# A probe prompt is 2.222 chars/token of pasted log against the seat's measured
# 3.74-3.95, so it is a TIME observation and not a density sample. The
# distinction has to survive in the data, not only in the prose.
check(any(p["source"] == "probe" for p in b["prefill_points"]),
      "probe points are labelled as such rather than mixed into seat traffic")
check(any(p["source"] == "seat" for p in b["prefill_points"]),
      "and seat traffic is present")

section("4. decode points carry REPLY LENGTH, which is the whole point")

for i, p in enumerate(b["decode_points"]):
    tag = f"decode[{i}] ctx={p.get('context_tokens')}"
    check(isinstance(p.get("reply_tokens"), int) and p["reply_tokens"] > 0,
          f"{tag}: reply_tokens present — a rate without it is not comparable")
    rlo, rhi = p["reply_tokens_range"]
    check(rlo <= p["reply_tokens"] <= rhi, f"{tag}: reply length inside its range")
    check(bool(p.get("cls")), f"{tag}: carries its reply-length class")
    check(isinstance(p.get("n"), int) and p["n"] >= 1, f"{tag}: n >= 1")

classes = {p["cls"] for p in b["decode_points"]}
check(len(classes) >= 2,
      "more than one reply-length class is published, so the confound is "
      "visible rather than averaged away")

# The two classes overlap in rate (short 10.0-21.7, long 13.6-24.2). A consumer
# handed rates alone would read the confound as the signal, so assert the
# overlap exists in the published data and was not tidied out.
longs = [p for p in b["decode_points"] if p["cls"] == "reply>=2000"]
shorts = [p for p in b["decode_points"] if p["cls"] != "reply>=2000"]
if longs and shorts:
    lo_long = min(p["decode_tok_per_s_range"][0] for p in longs)
    hi_short = max(p["decode_tok_per_s_range"][1] for p in shorts)
    check(hi_short > lo_long,
          f"the classes OVERLAP in rate ({hi_short} > {lo_long}) — which is why "
          f"reply length is published beside it")

# ---------------------------------------------------------------- the notes
section("5. the units note says real tokens, and names the consumer's job")

units = b["units"].lower()
check("prompt_eval_count" in b["units"],
      "names the field these came from, so a consumer can reproduce it")
check("estimate" in units,
      "says the consumer's own budget may be an ESTIMATE")
check("above or below" in units,
      "and that the correction factor may go EITHER way — a consumer told only "
      "to 'divide' assumes over-counting, and two estimators on one prompt "
      "differed 1.79x in both directions")
check("cannot compute" in units,
      "and that this gateway cannot compute that factor for them")
check("interpretation" in b and "extrapolat" in b["interpretation"].lower(),
      "the interpretation note warns where the data stops")

# ------------------------------------------------------- monotone and sane
section("6. the point set is ordered and does not imply a constant rate")

ns = [p["prompt_tokens"] for p in b["prefill_points"]]
check(ns == sorted(ns), "prefill points are ordered by size, so a scan is a scan")
check(len(set(ns)) == len(ns), "no duplicate sizes")

secs = [p["prefill_seconds"] for p in b["prefill_points"]]
check(secs == sorted(secs), "and time increases with size")

# THE CENTRAL ASSERTION. If the published points implied one rate, a consumer
# would be right to collapse them into a scalar and the whole ticket would be
# moot. They must not.
rates = [p["prompt_tokens"] / p["prefill_seconds"] for p in b["prefill_points"]]
spread = max(rates) / min(rates)
check(spread > 1.2,
      f"the implied rate varies {spread:.2f}x across the set, so no scalar "
      f"describes this box")
check(rates[0] > rates[-1],
      "and it DECLINES with size: a rate measured at small N is an upper bound "
      "at large N, not a candidate")

# Convexity is what lets a consumer bracket a deadline without a fit, and it
# holds over COMPARABLE observations only. The published set is not a curve:
# each of the low points is the slowest observation in a band, the bands carry
# different n, and an extremum over 7 draws is more extreme than one over 3. So
# 31,955 (n=3, 11.025 ms/tok) sits ABOVE 39,782 (n=7, 10.635) and the full set
# is not monotone — which is this ticket's own failure mode one level down:
# two numbers compared across a boundary neither was measured over.
#
# Asserted where the bracket actually applies: the tail from 44,905 up, which
# is the region any deadline binds in and where the #906 convexity argument was
# verified by chord test.
tail = [p for p in b["prefill_points"] if p["prompt_tokens"] >= 44905]
per_tok = [p["prefill_seconds"] / p["prompt_tokens"] for p in tail]
check(len(tail) >= 4 and per_tok == sorted(per_tok),
      f"seconds-per-token is non-decreasing over the tail (n={len(tail)} points "
      f"from 44,905) — CONVEX there, which is what makes a chord one-sided")

# And the head must NOT be read as a curve. The only thing that lets a consumer
# tell the difference is `n` on every point, so that is asserted rather than
# assumed — if n were dropped, the non-monotone head would look like data.
check(all(isinstance(p.get("n"), int) for p in b["prefill_points"]),
      "every point carries n, so a consumer can see which are comparable — "
      "band extrema over unequal n are not a curve and must not be chorded")

# --------------------------------------------------------------- the route
section("7. published on the OpenAI surface, same shape on list and retrieve")

from fastapi.testclient import TestClient  # noqa: E402

# Bare, NOT `with`. The app's lifespan adopts whatever Ollama holds and unloads
# it on shutdown, so a context-managed TestClient evicts the production model —
# measured on slice 2026-10-09 15:22, mid-service. A gate must be runnable on
# the box it describes.
client = TestClient(config and __import__(
    "ollama.server", fromlist=["app"]).app, raise_server_exceptions=False)
from . import server  # noqa: E402


class _StubMgr:
    """Enough manager for discovery, with one measured and one unmeasured id."""

    def offering(self):
        return {"resident": [{"name": MEASURED, "backend": "ollama",
                              "memory_mb": 17505, "context_length": 262144}],
                "loadable": [{"name": "qwen3:1.7b", "backend": "ollama",
                              "need_mb": 1400, "context_length": 40960}]}


real = server.mgr
try:
    server.mgr = _StubMgr()
    r = client.get("/v1/models")
    check(r.status_code == 200, f"GET /v1/models -> 200 (got {r.status_code})")
    entries = {e["id"]: e for e in r.json()["data"]} if r.status_code == 200 else {}

    check(MEASURED in entries, f"{MEASURED} listed")
    if MEASURED in entries:
        d = entries[MEASURED].get("deadlines")
        check(isinstance(d, dict), "the measured entry carries a deadlines block")
        check(d and d["generate_timeout_s"] == config.OLLAMA_GENERATE_TIMEOUT,
              "and its timeout is the one the SERVING path enforces")
        check(d and len(d["prefill_points"]) == len(b["prefill_points"]),
              "with the full point set, not a summary")
        # The window and the points answer different questions and the ratio is
        # ~3x. Both must be present so a consumer cannot mistake one for the other.
        check("context_length" in entries[MEASURED],
              "context_length is ALSO published — the two answer different "
              "questions and a consumer needs both to see the gap")

    if "qwen3:1.7b" in entries:
        d2 = entries["qwen3:1.7b"].get("deadlines")
        check(isinstance(d2, dict) and d2["prefill_points"] == [],
              "an unmeasured LISTED model publishes an empty set, not a "
              "missing block")

    # Same entry shape on retrieve, which is the contract /v1/models/{id} states.
    r2 = client.get(f"/v1/models/{MEASURED}")
    check(r2.status_code == 200, f"GET /v1/models/{{id}} -> 200 (got {r2.status_code})")
    if r2.status_code == 200 and MEASURED in entries:
        check(r2.json().get("deadlines") == entries[MEASURED].get("deadlines"),
              "retrieve and list agree exactly — they promise the same entry shape")
finally:
    server.mgr = real

# ----------------------------------------------------------- reproducible
section("8. the curation is reproducible, not asserted")

root = Path(server.__file__).resolve().parent.parent
tool = root / "deploy" / "deadline-points.py"
check(tool.exists(), "deploy/deadline-points.py ships beside the data")
if tool.exists():
    p = subprocess.run([sys.executable, str(tool), "--help"],
                       capture_output=True, text=True, cwd=root)
    check(p.returncode == 0, f"it runs (--help exit {p.returncode})")
    check("--cold-max" in p.stdout,
          "and the cache filter is an ARGUMENT, not a buried constant")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
