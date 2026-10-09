"""An empty SC_REQUIRED_SCOPE refuses every caller, instead of admitting them all.

The defect (#914): `SamcloudAuthMiddleware.dispatch` appended `?scope=` to the
verify URL only `if self.required_scope`, so an empty one asked /auth/verify no
scope question at all and EVERY valid token on the plane was admitted. The
widening was total — not "wider than intended" — and the config that produced it
reads as a narrowing, so someone tightening a scope got the opposite.

These drive the REAL middleware. The fake /auth/verify derives its grant rule
from `verify_token_endpoint` in samcloud's `registry/main.py`, including the
prefix fallback (#915) — a fake that cannot refuse cannot test a contract, and
one that refuses for different reasons than the plane tests nothing useful.

Run from the repo root: python -m ollama.test_no_auth_policy
"""
import asyncio
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Before importing server. `server.AUTH_ENABLED` is read at module level from
# config, so a stray AUTH_ENABLED=0 in the shell would short-circuit dispatch
# before it ever reached the guard and every check below would pass vacuously.
os.environ.pop("AUTH_ENABLED", None)

passed = failed = 0


def step(n, label):
    print(f"\n  [{n}] {label}")


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        failed += 1
        print(f"  \033[31m✗\033[0m {label}  {detail}")


class _StubResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload


class _StubClient:
    """The fake plane. Its grant rule is the real endpoint's, prefix included."""

    def __init__(self, cap, token_scopes):
        self._cap, self._scopes = cap, token_scopes

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        self._cap.append(url)
        scope = url.split("?scope=", 1)[1] if "?scope=" in url else None
        if scope is not None:
            granted = ("*" in self._scopes
                       or scope in self._scopes
                       or any(scope.startswith(s) for s in self._scopes))
            if not granted:
                return _StubResp(403)
        return _StubResp(200, {"username": "some-caller", "role": "agent",
                               "scopes": self._scopes})


class _Httpx:
    def __init__(self, cap, token_scopes):
        self._cap, self._scopes = cap, token_scopes

    def AsyncClient(self, **kw):
        return _StubClient(self._cap, self._scopes)


class _URL:
    def __init__(self, p):
        self.path = p


class _State:
    pass


class _Req:
    def __init__(self, path="/v1/models", auth="Bearer sc_agent_caller"):
        self.url = _URL(path)
        self.headers = {"Authorization": auth}
        self.state = _State()


def drive(required_scope, token_scopes, path="/v1/models",
          auth_enabled=True, prime_cache_with=None):
    """Run the real middleware once.

    Returns (verify_urls_fetched, allowed, status, body).
    `prime_cache_with` injects an identity into the instance cache under the
    request's own token, which is how [5] reaches the cache branch.
    """
    from ollama import server
    cap = []
    real_httpx, real_enabled = server.httpx, server.AUTH_ENABLED
    server.httpx = _Httpx(cap, token_scopes)
    server.AUTH_ENABLED = auth_enabled
    try:
        mw = server.SamcloudAuthMiddleware(
            app=None, verify_url="https://plane.example/auth/verify",
            required_scope=required_scope,
        )
        req = _Req(path=path)
        if prime_cache_with is not None:
            mw._set_cached(req.headers["Authorization"], prime_cache_with)

        async def call_next(_req):
            return "ALLOWED"

        out = asyncio.run(mw.dispatch(req, call_next))
    finally:
        server.httpx, server.AUTH_ENABLED = real_httpx, real_enabled
    allowed = out == "ALLOWED"
    status = getattr(out, "status_code", None)
    body = b""
    if hasattr(out, "body"):
        body = out.body
    return cap, allowed, status, body


def main():
    print("An empty SC_REQUIRED_SCOPE refuses rather than admits (#914)")

    step(1, "an empty scope refuses, and never asks the plane anything")
    cap, allowed, status, body = drive("", ["group:services"])
    check("not admitted", not allowed, f"allowed={allowed}")
    check(f"503, not 200 or 500 ({status})", status == 503, str(status))
    check("the plane was not contacted at all",
          len(cap) == 0, f"{len(cap)} url(s): {cap}")
    check("the body names the variable the operator must set",
          b"SC_REQUIRED_SCOPE" in body, body[:120].decode(errors="replace"))
    check("and carries a machine-readable error key",
          b"no_auth_policy" in body, body[:120].decode(errors="replace"))

    step(2, "the refusal does not depend on who is asking")
    # This is the defect's whole shape: with an empty scope the old code admitted
    # EVERY valid token, so a test that only tried an under-privileged caller
    # would pass on the bug. A `*` holder is the most-privileged caller there is.
    for scopes, who in (([], "a token with no scopes"),
                        (["group:services"], "a scoped token"),
                        (["*"], "a token holding `*`")):
        _, allowed, status, _ = drive("", scopes)
        check(f"{who} is refused ({status})", not allowed and status == 503,
              f"allowed={allowed} status={status}")

    step(3, "a configured scope still admits a caller who holds it")
    # The control. Without this the suite would pass on a middleware that
    # refuses everything, which is a different bug with the same test result.
    cap, allowed, status, _ = drive("group:services", ["group:services"])
    check("admitted", allowed, f"status={status}")
    check("exactly one verify call", len(cap) == 1, f"{len(cap)}: {cap}")
    check("and it carried the scope question",
          cap and cap[0].endswith("?scope=group:services"), str(cap))

    step(4, "a configured scope still refuses a caller who lacks it")
    _, allowed, status, _ = drive("group:services", ["group:other"])
    check(f"403 from the plane, unchanged ({status})",
          not allowed and status == 403, f"allowed={allowed} status={status}")

    step(5, "the refusal precedes the cache, not just the verify call")
    # The placement check. A guard at the URL-building site would sit AFTER
    # `_get_cached`, so a box that had already cached a caller would refuse new
    # tokens and keep admitting cached ones — a half-closed door that reads as
    # closed. This check fails on that implementation and passes on this one.
    cap, allowed, status, _ = drive(
        "", ["group:services"],
        prime_cache_with={"username": "already-verified", "role": "agent"},
    )
    check("a cached caller is refused too", not allowed, f"allowed={allowed}")
    check(f"with the same 503 ({status})", status == 503, str(status))
    check("still without contacting the plane", len(cap) == 0, str(cap))

    step(6, "auth-exempt paths still answer, so the box stays diagnosable")
    # /warm is exempt on purpose, and a misconfigured box is exactly when
    # somebody needs to read it. Refusing it would make the failure harder to
    # see without making anything safer: it discloses less than auth does.
    for p in ("/warm", "/health", "/service-docs"):
        _, allowed, status, _ = drive("", ["group:services"], path=p)
        check(f"{p} answers ({status})", allowed, f"status={status}")

    step(7, "with auth off, an empty scope is not a defect")
    # AUTH_ENABLED=false is the documented off-switch. The scope is then unused
    # rather than unusable, and refusing would break a supported configuration.
    _, allowed, status, _ = drive("", [], auth_enabled=False)
    check("admitted with AUTH_ENABLED=false", allowed, f"status={status}")

    step(8, "config says so once at startup, and only when it matters")
    import logging
    recs = []

    class _Grab(logging.Handler):
        def emit(self, r):
            recs.append(r)

    # `_env` coalesces falsy to the default, so no env value can make
    # SC_REQUIRED_SCOPE empty today (measured; that is why this guard is
    # pre-emptive). So the healthy path is what there is to assert here: the
    # error must NOT fire on a configured box, or its reader learns to skip it.
    from ollama import config as _c
    h = _Grab()
    logging.getLogger(_c.__name__).addHandler(h)
    try:
        importlib.reload(_c)
    finally:
        logging.getLogger(_c.__name__).removeHandler(h)
    errs = [r for r in recs if r.levelno >= logging.ERROR
            and "SC_REQUIRED_SCOPE" in r.getMessage()]
    check("a configured box logs no scope error on the healthy path",
          len(errs) == 0, f"{len(errs)}: {[r.getMessage()[:60] for r in errs]}")
    check("and the guard exists in config, conditional on AUTH_ENABLED",
          "AUTH_ENABLED and not SC_REQUIRED_SCOPE" in
          (Path(__file__).resolve().parent / "config.py").read_text(),
          "the startup check is missing or unconditional")

    print("\n" + "=" * 60)
    print(f"  {passed} checks run")
    print("  all checks passed" if not failed else f"  {failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
