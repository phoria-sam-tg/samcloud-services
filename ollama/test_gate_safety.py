"""An instrument must not disturb the thing it measures (#905).

WHY THIS FILE EXISTS

`python -m ollama.test_loaded_commit` was run on a live box on 2026-10-09 and it
EVICTED THE PRODUCTION MODEL. Not through a bug in what it asserted — through how
it built its client:

    with TestClient(server.app) as c:      # <- runs the app's LIFESPAN
        startup   adopts whatever Ollama currently holds
        shutdown  UNLOADS it

The resident 27b went from `keep_alive=-1` to an `expires_at` three minutes in
the past, mid-service, while the real gateway was holding a 118,272-token
request. Every assertion in that file passed. Nothing it asserted was wrong.

`claude-containers` audited their own instruments against the same shape on #905
and found them clean — but noted the reason was accidental: *"it is the absence
of a lifespan that saves it, and it was not a decision I made deliberately."*
That is the case for this file. Nine of the ten gates in this package already
construct `TestClient` bare, and none of them does so because a rule said to.

SO THE RULE IS CHECKED RATHER THAN REMEMBERED

Starlette's `TestClient` runs the app's lifespan **only** when used as a context
manager, so the whole property is syntactic: no gate may use `TestClient` as a
`with` target. That is enforced here over the AST.

NOT A GREP, AND ON THIS PROPERTY THAT IS NOT A PREFERENCE. `test_loaded_commit`
now contains the exact string `with TestClient(server.app)` inside the comment
explaining why it must not do that. A text search fails against the fix and
passes against the defect — the same inversion #911's gate had to avoid, and the
same one that let three assertions in `test_stream_deadlines` be satisfied by
their own explanatory comments.

AND THE DETECTOR IS TESTED ON A KNOWN VIOLATION

A scanner that reports "0 problems" is indistinguishable from a scanner that
cannot see. So section 1 runs the detector against synthetic sources that DO
violate the rule and asserts it fires, before section 2 trusts it to say the
package is clean. Two of today's checks passed vacuously before being fixed —
an undrained `StreamingResponse` that never constructed a client, and an empty
`seen` list that satisfied every loop — so this is the third instance of the
same lesson and it is cheap to apply.

    python -m ollama.test_gate_safety
"""
import ast
from pathlib import Path

if __package__ in (None, ""):
    print("run as:  python -m ollama.test_gate_safety")
    raise SystemExit(1)

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


# The names that start an app's lifespan when used as a context manager.
# `LifespanManager` is asgi-lifespan's explicit form; it is not used here today
# and is listed so that reaching for it does not quietly bypass this gate.
_LIFESPAN_CTX = {"TestClient", "LifespanManager"}


def _called_name(node: ast.AST):
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def lifespan_context_uses(source: str) -> list:
    """(name, lineno) for every lifespan-starting call used as a `with` target.

    Syntactic on purpose: the property being checked IS syntactic, because
    Starlette keys the lifespan off `__enter__`.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                name = _called_name(item.context_expr)
                if name in _LIFESPAN_CTX:
                    found.append((name, node.lineno))
    return found


# ------------------------------------------------- 1. the detector works
section("1. the detector fires on known violations (else section 2 is vacuous)")

VIOLATIONS = {
    "the exact shape that evicted the model":
        "from fastapi.testclient import TestClient\n"
        "with TestClient(server.app) as c:\n"
        "    c.get('/status')\n",
    "async form":
        "async def f():\n"
        "    async with TestClient(app) as c:\n"
        "        pass\n",
    "attribute form, e.g. module.TestClient(...)":
        "with testclient.TestClient(app) as c:\n    pass\n",
    "asgi-lifespan's explicit manager":
        "with LifespanManager(app):\n    pass\n",
    "nested inside a function and a try":
        "def g():\n    try:\n        with TestClient(app) as c:\n"
        "            pass\n    finally:\n        pass\n",
    "one of several with-items":
        "with open('x') as f, TestClient(app) as c:\n    pass\n",
}
for label, src in VIOLATIONS.items():
    check(bool(lifespan_context_uses(src)), f"fires: {label}")

CLEAN = {
    "bare construction, the correct form":
        "c = TestClient(server.app)\nc.get('/status')\n",
    "bare with kwargs":
        "client = TestClient(server.app, raise_server_exceptions=False)\n",
    "the STRING in a comment — the case a grep gets backwards":
        "# `with TestClient(server.app)` on a live box EVICTS THE MODEL.\n"
        "c = TestClient(server.app)\n",
    "the string in a docstring":
        '"""Do not write `with TestClient(app)` here."""\n'
        "c = TestClient(app)\n",
    "an unrelated with-block":
        "with open('x') as f:\n    pass\n",
    "a non-call context expression":
        "with mgr.serving('m'):\n    pass\n",
}
for label, src in CLEAN.items():
    check(not lifespan_context_uses(src), f"silent: {label}")

# The comment case, asserted as the contrast rather than left implied.
_comment_src = CLEAN["the STRING in a comment — the case a grep gets backwards"]
check("with TestClient(server.app)" in _comment_src
      and not lifespan_context_uses(_comment_src),
      "and that source CONTAINS the offending text while being clean — which "
      "is exactly why this is an AST walk")

# ------------------------------------------------- 2. the package is clean
section("2. no gate in this package starts the app's lifespan")

pkg = Path(__file__).resolve().parent
gates = sorted(pkg.glob("test_*.py"))

# A glob that matches nothing would make every check below pass. Asserted,
# because that is the vacuous-pass shape twice already today.
check(len(gates) >= 8,
      f"found {len(gates)} gate files to scan (a bad glob would pass vacuously)")
check(any(g.name == "test_gate_safety.py" for g in gates),
      "including this file, so the scan is demonstrably reaching the directory")

users = 0
for g in gates:
    src = g.read_text()
    hits = lifespan_context_uses(src)
    if "TestClient" in src:
        users += 1
    check(not hits,
          f"{g.name}: no lifespan context manager"
          + (f" (found {hits})" if hits else ""))

check(users >= 5,
      f"{users} gates do use TestClient — so the rule is load-bearing rather "
      f"than trivially satisfied by nobody using it")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
