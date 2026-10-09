"""#911: every httpx proxy call is bounded, and bounded in the right SHAPE.

WHY THIS FILE EXISTS

#904 replaced `aiohttp ClientTimeout(total=300)` with three measured bounds on
the OWNED Ollama path. The seven httpx proxy call sites were not touched and
carried the two opposite defects:

    3 streams      timeout=None    no bound at all — a wedged upstream hangs
                                   forever and the gateway emits no frame
    4 non-streams  timeout=300     #904's defect on a different transport: a
                                   non-streaming request produces no bytes until
                                   the generation completes, so the whole
                                   generation sits inside ONE read and a
                                   legitimate reply past 300s is aborted

One of the four was hiding inside a `stream: true` branch (the VLM tools path
collapses to a non-streaming upstream call), so a caller asking for a stream was
capped at 300s with nothing in the request to say so.

NOTHING HERE GREPS SOURCE TEXT, and on this file that is not a style
preference — `server.py` now contains the strings `timeout=None` and
`timeout=300` in the docstring that explains why they are gone. A grep-based
assertion would fail against the fix and pass against the defect. So the seam is
a FAKE httpx injected into the module, and every check reads the timeout the
code actually handed to the client.

    python -m ollama.test_proxy_bounds
"""
import asyncio
import os

if __package__ in (None, ""):
    print("run as:  python -m ollama.test_proxy_bounds")
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
from . import server          # noqa: E402
from .manager import Backend  # noqa: E402

# ------------------------------------------------------------- the bound
section("1. the shape of the bound, read off the object the code builds")

t = server._proxy_timeout()
check(t.connect == config.OLLAMA_PROXY_CONNECT_S,
      f"connect is tight ({t.connect}s) — the one bound not waiting on "
      f"inference, to a 127.0.0.1 child this gateway started")
check(t.read == config.OLLAMA_GENERATE_TIMEOUT,
      f"read is the whole-request ceiling ({t.read}s), not something tighter")
check(t.write == config.OLLAMA_PROXY_WRITE_S, "write is bounded")
check(t.pool == config.OLLAMA_PROXY_WRITE_S, "pool is bounded")

# The two defects, as properties of the value rather than of the source text.
check(t.read is not None and t.connect is not None,
      "nothing is None — an unbounded proxy makes the caller's own bound the "
      "only one and the gateway emits no frame at all")
check(t.read > 300,
      f"read ({t.read}s) EXCEEDS 300 — a 62,777-token cold prefill measured "
      f"718.7s on this box, and a prefill emits no bytes, so the whole prefill "
      f"sits inside one read")

# Measured cold prefills with no bytes emitted. A silence bound below these
# cuts work this box demonstrably completes.
for n, secs in ((62777, 718.7), (75776, 918.0), (112682, 1691.0)):
    check(t.read >= secs,
          f"read covers the measured {n:,}-token prefill ({secs}s)")

check(t.connect < 60,
      "and connect is NOT given the same slack — a local connect that takes a "
      "minute is a dead process, not a busy one")

# ------------------------------------------------------- the seam
section("2. what the proxy paths actually hand to httpx")


class _FakeResponse:
    status_code = 200

    def __init__(self, lines=()):
        self._lines = list(lines)

    def json(self):
        return {"choices": [{"message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}]}

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    """Records every timeout it is handed. One instance list, module-wide."""
    seen: list = []

    def __init__(self, *a, **kw):
        self.ctor_kwargs = kw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method, url, **kw):
        _FakeClient.seen.append(("stream", url, kw.get("timeout", "ABSENT")))
        return _FakeResponse(['data: {"choices":[{"delta":{"content":"x"}}]}',
                              "", "data: [DONE]"])

    async def post(self, url, **kw):
        _FakeClient.seen.append(("post", url, kw.get("timeout", "ABSENT")))
        return _FakeResponse()


class _FakeHttpx:
    AsyncClient = _FakeClient
    Timeout = server.httpx.Timeout          # the real one: we assert on it

    class HTTPStatusError(Exception):
        pass

    class RequestError(Exception):
        pass

    class ReadTimeout(Exception):
        pass


class _MM:
    def __init__(self, backend):
        self.name = "proxy-model"
        self.backend = backend
        self.port = 9999
        self.managed = True
        self.in_flight = 0
        self.request_count = 0
        self.lease_id = None
        self.tier = None


class _StubMgr:
    def touch(self, name):
        pass

    def status(self):
        return {"models": []}


async def _drive(backend, stream, tools=None, route="chat"):
    """Run one proxy branch and return the timeouts it handed to httpx.

    `route` covers both endpoints: the seven call sites are split 5 in
    `chat_completions` and 2 in `completions`, and a gate that drove only the
    first would have left the `/v1/completions` pair unasserted — which is how
    all seven came to disagree with each other in the first place.
    """
    _FakeClient.seen = []
    real_httpx, real_mgr, real_resolve = server.httpx, server.mgr, server._resolve_model
    try:
        server.httpx = _FakeHttpx
        server.mgr = _StubMgr()

        async def fake_resolve(name):
            return _MM(backend)
        server._resolve_model = fake_resolve

        if route == "chat":
            req = server.ChatRequest(
                model="proxy-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=stream, tools=tools)
            out = await server.chat_completions(req)
        else:
            req = server.CompletionRequest(model="proxy-model", prompt="hi",
                                           stream=stream)
            out = await server.completions(req)
        # A StreamingResponse does nothing until iterated — the whole point of
        # #907's wrappers — so the generator is drained here or no client is
        # ever constructed and the check would pass vacuously.
        body = getattr(out, "body_iterator", None)
        if body is not None:
            async for _ in body:
                pass
        return list(_FakeClient.seen)
    finally:
        server.httpx, server.mgr, server._resolve_model = \
            real_httpx, real_mgr, real_resolve


def _assert_bounded(seen, label):
    check(bool(seen), f"{label}: a client call was actually made "
                      f"(an empty list would pass every check below)")
    for kind, url, to in seen:
        check(to != "ABSENT", f"{label}: {kind} passes a timeout at all")
        check(to is not None, f"{label}: {kind} timeout is not None")
        check(not isinstance(to, (int, float)),
              f"{label}: {kind} timeout is a Timeout object, not a bare number "
              f"— a bare number sets connect/read/write/pool all the same, "
              f"which is how 300 became a generation bound")
        if hasattr(to, "read"):
            check(to.read == config.OLLAMA_GENERATE_TIMEOUT,
                  f"{label}: {kind} read is the ceiling ({to.read}s)")
            check(to.connect == config.OLLAMA_PROXY_CONNECT_S,
                  f"{label}: {kind} connect is tight ({to.connect}s)")


for backend, stream, tools, label in (
        (Backend.LLAMA, True, None, "llama-server stream"),
        (Backend.LLAMA, False, None, "llama-server non-stream"),
        (Backend.VLM, True, None, "VLM stream"),
        (Backend.VLM, False, None, "VLM non-stream"),
        # The one that hid: a stream request whose upstream call is NOT a
        # stream, so it carried the 300s generation bound while the caller
        # believed it had asked for a stream.
        (Backend.VLM, True, [{"type": "function",
                              "function": {"name": "f", "parameters": {}}}],
         "VLM stream WITH TOOLS (collapses to a non-stream upstream call)"),
):
    section(f"   {label}")
    try:
        _assert_bounded(asyncio.run(_drive(backend, stream, tools)), label)
    except Exception as e:
        check(False, f"{label}: driving raised {e!r}")

# /v1/completions — the other two sites, same contract, different endpoint.
for stream, label in ((True, "/v1/completions stream"),
                      (False, "/v1/completions non-stream")):
    section(f"   {label}")
    try:
        _assert_bounded(
            asyncio.run(_drive(Backend.LLAMA, stream, route="completions")),
            label)
    except Exception as e:
        check(False, f"{label}: driving raised {e!r}")

# ------------------------------------------------- one bound, one place
section("3. every site goes through the one helper")

# Asserted by MUTATING the helper and observing every call site follow it. A
# site with its own literal would not move, and this cannot be satisfied by a
# comment the way a grep can.
real_helper = server._proxy_timeout
SENTINEL = server.httpx.Timeout(connect=1, read=2, write=3, pool=4)
try:
    server._proxy_timeout = lambda: SENTINEL
    moved = total = 0
    cases = [("chat", Backend.LLAMA, True, None),
             ("chat", Backend.LLAMA, False, None),
             ("chat", Backend.VLM, True, None),
             ("chat", Backend.VLM, False, None),
             ("chat", Backend.VLM, True, [{"type": "function",
                                           "function": {"name": "f",
                                                        "parameters": {}}}]),
             ("completions", Backend.LLAMA, True, None),
             ("completions", Backend.LLAMA, False, None)]
    for route, backend, stream, tools in cases:
        for kind, url, to in asyncio.run(
                _drive(backend, stream, tools, route=route)):
            total += 1
            if to is SENTINEL:
                moved += 1
    # 7 call sites exist in server.py; this drives all 7.
    check(total == 7,
          f"all SEVEN call sites were reached ({total}) — the count is asserted "
          f"so a new unbounded site cannot be added without failing here")
    check(total > 0 and moved == total,
          f"and all {total} followed the helper when it changed ({moved}/{total}) "
          f"— none carries its own literal")
finally:
    server._proxy_timeout = real_helper

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
