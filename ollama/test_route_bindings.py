#!/usr/bin/env python3
"""Every path resolves to the handler it is named for (#904).

WHY THIS EXISTS, and it is the most expensive lesson on this ticket. A helper
function was inserted BETWEEN `@app.post("/v1/chat/completions")` and
`async def chat_completions`, so the decorator bound the HELPER. The path still
existed, the route count was unchanged, and `chat_completions` became
unreachable. Every POST to Hermes' surface then 422'd — `model: str` became a
required *query* parameter and `chunk: dict` became the body — so a normal
request was rejected before anything ran.

**100% failure, and strictly worse than the 22% cap it shipped beside.** It
reached `main` and was staged into the live deploy clone; the only reason the
service kept serving is that the running process had loaded the old module at
import, three days earlier. `samclaude-admin` caught it by reading the diff.

Why 51 checks and nine mutations did not. Both branches register the same number
of routes — the COUNT did not change, the BINDING did. Nothing asserted which
handler a path resolves to, and no test went through the ASGI app at all: the
deadline logic was tested directly while the route that reaches it was not.
That is "assert a count, not a presence" one level up, with the count right.

So this file asserts the whole table rather than the one path that broke, since
the failure mode is "a helper lands under a decorator" and ANY insertion can
cause it.

    python -m ollama.test_route_bindings
"""

import os
import sys

os.environ.setdefault("SC_TOKEN", "test-token-not-used")
os.environ.setdefault("AUTH_ENABLED", "0")

failures = []
checks = 0


def step(n, msg):
    print(f"\n{'='*66}\n  Step {n}: {msg}\n{'='*66}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


# path -> the handler that path MUST resolve to. Written out rather than derived,
# because deriving it from the app is what makes the bug invisible: the table has
# to be the independent statement of intent.
# **AND the method**, which an earlier version of this file left unpinned
# (samclaude-admin, #904): a handler rebound POST -> GET on the same path still
# passed. Same class as the defect this file exists for — the thing that
# changed was not the thing being asserted.
EXPECTED = {
    "/health": ("health", "GET"),
    "/warm": ("warm", "GET"),
    "/service-docs": ("service_docs", "GET"),
    "/status": ("status", "GET"),
    "/models": ("list_models", "GET"),
    "/v1/models": ("list_models_openai", "GET"),
    "/v1/models/{model_id:path}": ("retrieve_model_openai", "GET"),
    "/models/load": ("load_model", "POST"),
    "/models/unload": ("unload_model", "POST"),
    "/v1/chat/completions": ("chat_completions", "POST"),
    "/v1/completions": ("completions", "POST"),
    "/v1/audio/transcriptions": ("create_transcription", "POST"),
}


def main():
    from .server import app

    table = {}
    methods = {}
    for r in app.routes:
        path = getattr(r, "path", None)
        ep = getattr(r, "endpoint", None)
        if path and ep is not None:
            table.setdefault(path, set()).add(ep.__name__)
            # HEAD and OPTIONS are added by Starlette for a GET; they are not
            # a second binding and must not read as one.
            verbs = {m for m in (getattr(r, "methods", None) or set())
                     if m not in ("HEAD", "OPTIONS")}
            methods.setdefault(path, set()).update(verbs)

    step(1, "every named path resolves to its own handler, on its own method")
    for path, (want, verb) in sorted(EXPECTED.items()):
        got = table.get(path)
        check(got == {want},
              f"{path} -> {want}" + (f"  (got {got})" if got != {want} else ""))
        got_m = methods.get(path)
        check(got_m == {verb},
              f"{path} is {verb}"
              + (f"  (got {sorted(got_m) if got_m else got_m})"
                 if got_m != {verb} else ""))

    step(2, "the one that broke, stated on its own")
    check(table.get("/v1/chat/completions") == {"chat_completions"},
          "/v1/chat/completions is bound to chat_completions and NOT to a "
          "helper that happened to be defined under its decorator")

    step(3, "no handler is bound to more than one path by accident")
    # A decorator landing on the wrong function usually shows up here too: the
    # helper gets a path, and whatever the helper was near loses one.
    for path, names in sorted(table.items()):
        check(len(names) == 1,
              f"{path} resolves to exactly one handler ({sorted(names)})")

    step(4, "no private helper is routed")
    # The actual shape of the defect: a name starting with `_` should never be
    # reachable over HTTP. This is the check that is independent of the table
    # above, so it still fires if someone adds a route AND forgets the table.
    for path, names in sorted(table.items()):
        private = sorted(n for n in names if n.startswith("_"))
        check(not private,
              f"{path} is not served by a private helper"
              + (f"  ({private})" if private else ""))

    step(5, "the table covers what the app actually serves")
    # So a new endpoint cannot be added without being named here — otherwise
    # this file silently stops covering the surface it is meant to pin.
    served = {p for p in table if not p.startswith("/openapi")
              and p not in ("/docs", "/redoc", "/docs/oauth2-redirect")}
    missing = sorted(served - set(EXPECTED))
    check(not missing,
          f"every served path is named in EXPECTED (unnamed: {missing})")
    stale = sorted(set(EXPECTED) - served)
    check(not stale, f"EXPECTED names no path the app lost (stale: {stale})")

    step(6, "a real request goes THROUGH the ASGI app, not past it")
    # The gap samclaude-admin named: the deadline logic was tested directly
    # while the route that reaches it was not. Inspecting `app.routes` is
    # necessary and not sufficient — this sends the request.
    #
    # No lifespan, so `mgr` is None and the handler raises. That is fine and is
    # the POINT: anything that is not a routing rejection means the request
    # reached `chat_completions`. The signal we are looking for is the specific
    # 422 the misbinding produces, where `model: str` became a required QUERY
    # parameter and `chunk: dict` became the body.
    try:
        from fastapi.testclient import TestClient
    except Exception as e:                                  # pragma: no cover
        check(False, f"fastapi.testclient unavailable, step skipped: {e}")
    else:
        body = {"model": "qwen3.8:27b-mlx",
                "messages": [{"role": "user", "content": "hi"}]}
        status, detail = None, ""
        try:
            r = TestClient(app, raise_server_exceptions=False).post(
                "/v1/chat/completions", json=body)
            status = r.status_code
            try:
                detail = str(r.json())
            except Exception:
                detail = r.text[:200]
        except Exception as e:
            # The handler blowing up on mgr=None unwinds to here on some
            # versions. Reaching the handler at all is what we are asserting.
            status, detail = 500, f"{type(e).__name__}: {e}"
        check(status != 422,
              f"a normal Hermes-shaped POST is not rejected by routing "
              f"(status {status})")
        check("'loc': ['query', 'model']" not in detail,
              "...and specifically not with `model` demanded as a QUERY "
              "parameter, which is the exact signature of the misbinding")

    print(f"\n{'='*66}")
    print(f"  {checks - len(failures)}/{checks} checks passed")
    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")
    print(f"{'='*66}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
