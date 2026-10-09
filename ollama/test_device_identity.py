#!/usr/bin/env python3
"""SC_DEVICE comes from the environment or from nowhere — never from a peer.

#865: `SC_DEVICE` defaulted to `claude-services-slice`, a real peer. Three more
identities derive from it (`SC_SERVICE_ID`, `SC_RESOURCE_ID`,
`SC_REQUIRED_SCOPE`, and `EXO_RESOURCE_ID` 800 lines later), so any process that
imported this module without an env file composed a plausible identity belonging
to another machine and wrote registry rows under it. It never showed a symptom
because the one box the default is correct on is the box it was written on.

The population that hits it is not gateways, which have env files, but probes
and harnesses run without one — it fired on wafer on 2026-09-29 in a script
written by the agent who had just audited the defect (claude-wafer-services).

The sharp edge, and why [5] and [6] exist: the derivations go EMPTY when the
device is unknown, but `SC_REQUIRED_SCOPE` must not, because the middleware that
reads it appends `?scope=` only `if self.required_scope`. Empty there asks
/auth/verify no scope question at all and admits every valid token on the plane.
Refusing callers is correct for an unconfigured box; admitting all of them is
the opposite, and it would have been a silent auth widening introduced BY the
fix for a silent misidentification.

Run from the repo root: python -m ollama.test_device_identity
"""
import asyncio
import importlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Before importing server: these tests are about the auth path, and a stray
# AUTH_ENABLED=0 inherited from the shell would make [6] pass vacuously by
# short-circuiting dispatch before it ever composes a verify URL.
os.environ.pop("AUTH_ENABLED", None)

passed = failed = 0

OLD_DEFAULT = "claude-services-slice"   # the peer this ticket is about


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        failed += 1
        print(f"  \033[31m✗\033[0m {label}  {detail}")


def attr(mod, name, default="<absent>"):
    """A MISSING attribute is a failure to report, not a crash.

    On the pre-#865 config `SC_DEVICE_SOURCE` does not exist, and letting that
    raise in [1] would abort the run before [2]-[6] — the checks that actually
    describe the defect — ever execute. Reverting the change must show WHICH
    guarantees break, not just a non-zero exit.
    """
    return getattr(mod, name, default)


def load(**env):
    """Re-import config with a given environment. Returns the module."""
    for k in ("SC_DEVICE", "SC_SERVICE_NAME", "SC_RESOURCE_ID",
              "SC_REQUIRED_SCOPE", "EXO_RESOURCE_ID"):
        os.environ.pop(k, None)
    for k, v in env.items():
        if v is not None:
            os.environ[k] = v
    from ollama import config
    return importlib.reload(config)


# ---- a fake /auth/verify that refuses what the real one refuses ----------
# Derived from registry/main.py, not from memory: `AuthContext.has_scope` is
# `"*" in scopes or scope in scopes`, the endpoint resolves a `device:<id>`
# scope through group membership on a device that must exist, and there is a
# prefix fallback over the token's own scopes. A fake that granted everything
# could not show that the sentinel refuses, which is the whole contract here.
class _StubResp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._b = body or {}

    def json(self):
        return self._b


class _StubClient:
    def __init__(self, cap, token_scopes):
        self._cap = cap
        self._scopes = token_scopes

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        self._cap.append(url)
        scope = None
        if "?scope=" in url:
            scope = url.split("?scope=", 1)[1]
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


def drive_auth(required_scope, token_scopes):
    """Run the REAL middleware once. Returns (verify_urls, allowed, status)."""
    from ollama import server
    cap = []
    real_httpx = server.httpx
    server.httpx = _Httpx(cap, token_scopes)
    try:
        mw = server.SamcloudAuthMiddleware(
            app=None, verify_url="https://plane.example/auth/verify",
            required_scope=required_scope,
        )

        async def call_next(_req):
            return "ALLOWED"

        out = asyncio.run(mw.dispatch(_Req(), call_next))
    finally:
        server.httpx = real_httpx
    allowed = out == "ALLOWED"
    status = getattr(out, "status_code", None)
    return cap, allowed, status


def main():
    print("SC_DEVICE identity (#865)")

    print("  [1] the environment is the only source, and the source says so")
    c = load(SC_DEVICE="wafer-services")
    check("device is what the env said", c.SC_DEVICE == "wafer-services", c.SC_DEVICE)
    check("source names the environment", attr(c, "SC_DEVICE_SOURCE") == "environment",
          attr(c, "SC_DEVICE_SOURCE"))
    check("service_id composes from it",
          c.SC_SERVICE_ID == "wafer-services/model-service", c.SC_SERVICE_ID)
    check("resource composes from it",
          c.SC_RESOURCE_ID == "wafer-services/gpu-0", c.SC_RESOURCE_ID)
    check("required scope composes from it",
          c.SC_REQUIRED_SCOPE == "device:wafer-services", c.SC_REQUIRED_SCOPE)
    check("exo pool composes from it",
          c.EXO_RESOURCE_ID == "wafer-services/exo-pool", c.EXO_RESOURCE_ID)

    # This is the regression that matters, and it FAILS against the version with
    # a `claude-services-slice` default: that version resolves a real peer here
    # and every check below it still passes, because the composed ids are all
    # well-formed. A test that only asked "did an identity resolve?" would have
    # passed on the broken version.
    print(f"  [2] unset is NOT SET, and specifically not {OLD_DEFAULT}")
    c = load(SC_DEVICE=None)
    check("device is empty", c.SC_DEVICE == "", repr(c.SC_DEVICE))
    check("source is NOT SET, so the log can say why",
          attr(c, "SC_DEVICE_SOURCE") == "NOT SET", attr(c, "SC_DEVICE_SOURCE"))
    check(f"and specifically not the peer {OLD_DEFAULT}",
          c.SC_DEVICE != OLD_DEFAULT, c.SC_DEVICE)

    print("  [3] the derivations refuse to compose rather than half-composing")
    # `/model-service` and `/gpu-0` are worse than empty: the registry would
    # accept them as names, and they read like values in a log.
    for name in ("SC_SERVICE_ID", "SC_RESOURCE_ID", "EXO_RESOURCE_ID"):
        v = getattr(c, name)
        check(f"{name} is empty", v == "", repr(v))
        check(f"{name} did not half-compose a leading /", not v.startswith("/"), repr(v))
        check(f"{name} does not name the peer", OLD_DEFAULT not in v, repr(v))

    print("  [4] an independently set resource is still honoured")
    # A box may name its resource directly; only the COMPOSED fallback is gone.
    c = load(SC_DEVICE=None, SC_RESOURCE_ID="wafer-services/gpu-metal",
             EXO_RESOURCE_ID="wafer-services/exo-pool")
    check("explicit resource wins over an unknown device",
          c.SC_RESOURCE_ID == "wafer-services/gpu-metal", c.SC_RESOURCE_ID)
    check("explicit exo resource wins too",
          c.EXO_RESOURCE_ID == "wafer-services/exo-pool", c.EXO_RESOURCE_ID)

    print("  [5] SC_REQUIRED_SCOPE never goes empty, in either state")
    for label, kw in (("device set", {"SC_DEVICE": "wafer-services"}),
                      ("device unset", {"SC_DEVICE": None})):
        cc = load(**kw)
        check(f"non-empty with {label}", bool(cc.SC_REQUIRED_SCOPE),
              repr(cc.SC_REQUIRED_SCOPE))
    # THE #914/#916 INTERACTION, pinned here because this PR introduces the
    # strip that could break it. `_env_id` strips and THEN coalesces, so a
    # whitespace-only scope falls back to the default. The unsafe shape --
    # `(v if v is not None else default).strip()` -- would yield '' and reach
    # the middleware's fail-open, turning a diagnosability fix into an auth
    # widening (samclaude-services, #914). Driven through the real `_env_id`,
    # not a constructed middleware, because the env path is the claim.
    for raw in ("", " ", "\t\n  ", "   "):
        for dev in ("testbox", None):
            cc = load(SC_DEVICE=dev, SC_REQUIRED_SCOPE=raw)
            check(f"scope non-empty for {raw!r} with device={dev!r}",
                  bool(cc.SC_REQUIRED_SCOPE), repr(cc.SC_REQUIRED_SCOPE))

    c = load(SC_DEVICE=None)
    check("unset device demands a scope naming no real device",
          c.SC_REQUIRED_SCOPE == "device:SC_DEVICE-is-unset", c.SC_REQUIRED_SCOPE)
    check("which is not a device scope anyone could hold",
          OLD_DEFAULT not in c.SC_REQUIRED_SCOPE, c.SC_REQUIRED_SCOPE)

    print("  [6] against the real middleware: empty would admit, the sentinel refuses")
    # The claim in config.py's comment, made falsifiable. Same caller token in
    # both rows; only the configured scope differs.
    caller = ["group:services"]
    urls, allowed, status = drive_auth("", caller)
    check("an EMPTY scope asks /auth/verify no scope question",
          urls and "?scope=" not in urls[0], str(urls))
    check("...and therefore ADMITS the caller — the fail-open this avoids",
          allowed, f"allowed={allowed} status={status}")
    check("exactly one verify call was made", len(urls) == 1, str(urls))

    urls, allowed, status = drive_auth("device:SC_DEVICE-is-unset", caller)
    check("the sentinel does ask the scope question",
          urls and urls[0].endswith("?scope=device:SC_DEVICE-is-unset"), str(urls))
    check("...and the caller is refused 403", (not allowed) and status == 403,
          f"allowed={allowed} status={status}")

    # Control: the same machinery must still ADMIT a caller who holds the scope,
    # or [6] would pass on a middleware that refuses everything.
    urls, allowed, status = drive_auth("group:services", caller)
    check("control: a caller holding the required scope is admitted",
          allowed, f"allowed={allowed} status={status}")

    print("  [7] no default anywhere in config.py names a peer")
    src = Path(__file__).resolve().parent.joinpath("config.py").read_text()
    # Count, not presence: a second _env default reintroducing the peer would
    # hide behind a presence check that the first one satisfied.
    peer_defaults = re.findall(
        r'_env(?:_id)?\(\s*"[A-Z0-9_]+"\s*,\s*[^)]*' + re.escape(OLD_DEFAULT), src)
    check(f"zero _env defaults mention {OLD_DEFAULT}",
          len(peer_defaults) == 0, str(peer_defaults))
    sc_device_lines = re.findall(r'^SC_DEVICE = .*$', src, re.M)
    check("exactly one SC_DEVICE assignment", len(sc_device_lines) == 1,
          str(sc_device_lines))
    check("and it defaults to empty",
          sc_device_lines and sc_device_lines[0] == 'SC_DEVICE = _env_id("SC_DEVICE", "")',
          str(sc_device_lines))

    print("  [8] a padded identity is stripped, so the source cannot lie")
    # `_env` does not strip and " " is TRUTHY, so without _env_id a whitespace
    # device passes the unset guard and SC_DEVICE_SOURCE reports `environment`
    # -- the field added to separate configured from inherited would assert a
    # human chose it (samclaude-services, reviewing #865).
    c = load(SC_DEVICE="wafer-services ")
    check("a trailing space is stripped off the device",
          c.SC_DEVICE == "wafer-services", repr(c.SC_DEVICE))
    check("so the derivations do not carry it",
          c.SC_SERVICE_ID == "wafer-services/model-service", repr(c.SC_SERVICE_ID))
    check("nor does the scope",
          c.SC_REQUIRED_SCOPE == "device:wafer-services", repr(c.SC_REQUIRED_SCOPE))

    c = load(SC_DEVICE="  \t\n  ")
    check("a WHITESPACE-ONLY device is unset, not configured",
          c.SC_DEVICE == "", repr(c.SC_DEVICE))
    check("and the source says NOT SET rather than environment",
          attr(c, "SC_DEVICE_SOURCE") == "NOT SET", attr(c, "SC_DEVICE_SOURCE"))
    check("so the derivations refuse to compose",
          (c.SC_SERVICE_ID, c.SC_RESOURCE_ID, c.EXO_RESOURCE_ID) == ("", "", ""),
          f"{c.SC_SERVICE_ID!r} {c.SC_RESOURCE_ID!r} {c.EXO_RESOURCE_ID!r}")
    check("and the scope still fails closed",
          c.SC_REQUIRED_SCOPE == "device:SC_DEVICE-is-unset", c.SC_REQUIRED_SCOPE)

    c = load(SC_DEVICE=None, SC_RESOURCE_ID=" wafer-services/gpu-metal\n")
    check("an independently set resource is stripped too",
          c.SC_RESOURCE_ID == "wafer-services/gpu-metal", repr(c.SC_RESOURCE_ID))

    # Control: stripping must not touch a value that was already clean, or the
    # checks above would pass on a helper that mangles every id.
    c = load(SC_DEVICE="wafer-services", SC_SERVICE_NAME="model-service")
    check("control: a clean identity is returned unchanged",
          (c.SC_DEVICE, c.SC_SERVICE_ID) == ("wafer-services",
                                             "wafer-services/model-service"),
          f"{c.SC_DEVICE!r} {c.SC_SERVICE_ID!r}")

    print()
    if failed:
        print(f"  FAILED: {failed} failed, {passed} passed")
        sys.exit(1)
    print(f"  ALL PASSED: SC_DEVICE identity {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
