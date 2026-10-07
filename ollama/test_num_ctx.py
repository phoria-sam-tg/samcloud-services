#!/usr/bin/env python3
"""The served context window: pinned, clamped, and published (#903).

Nothing used to set `num_ctx`, and the absence was not neutral. Ollama DERIVES
a default from free VRAM at load time; slice's own log has
`total_vram="48.0 GiB" default_num_ctx=262144` nineteen times and
`total_vram="0 B" default_num_ctx=4096` twice, from the two days llama-server
GPU discovery timed out and Ollama fell back to CPU. A 64x spread in the window
a caller got, with no endpoint reporting which one it was handed — so #884
hand-picked 65,536, a number this box has never served.

Three things are asserted here, and the third is the one with the longer life:

  1. the configured window reaches EVERY Ollama call path, because Ollama keys
     a loaded instance by its options and one forgetful call site is a reload
     in the middle of somebody's job;
  2. a configured value above the model's native window is clamped, not passed
     on, and adoption re-applies the instance's OWN window so a gateway restart
     cannot bounce a resident model to change a number;
  3. the window is PUBLISHED, so a consumer can size against it instead of
     inventing one.

No network, no model, no registry. The client is driven against a fake Ollama.

    python -m ollama.test_num_ctx
"""

import importlib
import os
import sys

failures = []
checks = 0


def step(n, msg):
    print(f"\n{'='*64}\n  Step {n}: {msg}\n{'='*64}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


NATIVE = 262144
MODEL = "qwen3.8:27b-mlx"


def fresh(env: dict):
    """Re-import config and ollama_client under a given environment."""
    for k in ("OLLAMA_NUM_CTX", "OLLAMA_NUM_CTX_MODELS"):
        os.environ.pop(k, None)
    os.environ.update(env)
    from . import config, ollama_client
    importlib.reload(config)
    importlib.reload(ollama_client)
    return config, ollama_client


class FakeHTTP:
    """Records every payload the client sends, answers like Ollama."""

    def __init__(self):
        self.posts = []
        self.streams = []
        self.tags_calls = 0
        self.show_calls = 0

    class _Resp:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

        def iter_lines(self):
            return iter(())

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def get(self, path, **kw):
        if path == "/api/tags":
            self.tags_calls += 1
            return self._Resp({"models": [
                {"name": MODEL, "digest": "abc", "size": 18174721847},
                {"name": "qwen3:1.7b", "digest": "def", "size": 1359318745},
            ]})
        if path == "/api/ps":
            return self._Resp({"models": [
                {"name": MODEL, "size": 31784263680, "context_length": NATIVE},
            ]})
        return self._Resp({})

    def post(self, path, json=None, **kw):
        if path == "/api/show":
            self.show_calls += 1
            return self._Resp({"model_info": {"qwen3_5.context_length": NATIVE}})
        self.posts.append((path, json))
        return self._Resp({})

    def stream(self, method, path, json=None, **kw):
        self.streams.append((path, json))
        return self._Resp({})


def client(mod, http):
    c = mod.OllamaClient.__new__(mod.OllamaClient)
    c.base_url = "http://fake"
    c._http = http
    c._clamped = set()
    c._native_ctx = {}
    c._digests = {}
    c._digests_at = 0.0
    return c


def main():
    step(1, "unset is the old behaviour: no num_ctx, and no extra Ollama reads")
    cfg, oc = fresh({})
    check(cfg.ollama_num_ctx(MODEL) == 0, "ollama_num_ctx is 0 when unset")
    http = FakeHTTP()
    c = client(oc, http)
    check(c.num_ctx_for(MODEL) is None, "num_ctx_for -> None when unset")
    check(http.tags_calls == 0 and http.show_calls == 0,
          "an unpinned box pays NOTHING to answer it (offering() calls this "
          "per model on the auth-exempt /warm)")
    list(c.chat(MODEL, [{"role": "user", "content": "hi"}]))
    _, payload = http.streams[-1]
    check("options" not in payload, "no options injected when unset")

    step(2, "a pinned window reaches every call path")
    cfg, oc = fresh({"OLLAMA_NUM_CTX_MODELS": f"{MODEL}=140000"})
    check(cfg.ollama_num_ctx(MODEL) == 140000, "per-model value is read")
    http = FakeHTTP()
    c = client(oc, http)
    list(c.chat(MODEL, [{"role": "user", "content": "hi"}]))
    list(c.generate(MODEL, "hi"))
    c.load_model(MODEL, keep_alive=300)
    seen = [p for _, p in http.streams] + [p for pa, p in http.posts
                                           if pa == "/api/generate"]
    check(len(seen) == 3, f"three call paths exercised (got {len(seen)})")
    for p in seen:
        check((p.get("options") or {}).get("num_ctx") == 140000,
              f"num_ctx=140000 present on {sorted(p.keys())}")

    step(3, "the async paths too — they build their own payloads")
    src = open(os.path.join(os.path.dirname(__file__),
                            "ollama_client.py")).read()
    for fn in ("chat_stream", "generate_stream", "generate", "chat"):
        body = src.split(f"def {fn}(", 1)[1].split("\n    def ", 1)[0]
        check("_with_num_ctx" in body,
              f"{fn}() routes its payload through _with_num_ctx")
    unload = src.split("def unload_model(", 1)[1].split("\n    def ", 1)[0]
    check("_with_num_ctx" not in unload,
          "unload_model() does NOT — keep_alive:0 needs no window, and "
          "injecting one would make an unload look like a load")

    step(4, "a caller's own num_ctx wins over the configured one")
    http = FakeHTTP()
    c = client(oc, http)
    list(c.chat(MODEL, [], options={"num_ctx": 8192}))
    _, p = http.streams[-1]
    check(p["options"]["num_ctx"] == 8192,
          "an explicit num_ctx is left alone (operators use /models/load)")

    step(5, "above the native window is clamped, not passed on")
    cfg, oc = fresh({"OLLAMA_NUM_CTX": "999999"})
    http = FakeHTTP()
    c = client(oc, http)
    check(c.num_ctx_for(MODEL) == NATIVE,
          f"999999 clamped to the declared {NATIVE}")
    check(c.native_context_length(MODEL) == NATIVE,
          "native window read from /api/show, architecture-prefixed key")
    before = http.show_calls
    for _ in range(20):
        c.num_ctx_for(MODEL)
    check(http.show_calls == before,
          f"/api/show cached by digest ({http.show_calls} calls for 21 asks)")
    check(http.tags_calls <= 2,
          f"/api/tags TTL-cached, not per call ({http.tags_calls} calls)")

    step(6, "a malformed override is dropped, never read as 0")
    cfg, _ = fresh({"OLLAMA_NUM_CTX": "131072",
                    "OLLAMA_NUM_CTX_MODELS": "good=98304,broken=,=5,junk=abc"})
    check(cfg.ollama_num_ctx("good") == 98304, "the good pair is kept")
    for bad in ("broken", "junk"):
        check(cfg.ollama_num_ctx(bad) == 131072,
              f"'{bad}' falls back to the global, NOT to 0 — 0 means "
              f"'let Ollama derive one', which is what pinning exists to stop")
    check(cfg.ollama_num_ctx("qwen3.8:27b-mlx") == 131072,
          "an unlisted model takes the global")
    cfg, _ = fresh({"OLLAMA_NUM_CTX_MODELS": "qwen3.8=262144"})
    check(cfg.ollama_num_ctx("qwen3.8:27b-mlx") == 262144,
          "a bare name covers every tag of it")

    step(7, "adoption re-applies the instance's OWN window")
    mgr_src = open(os.path.join(os.path.dirname(__file__),
                                "manager.py")).read()
    adopt = mgr_src.split("# Discover running Ollama models", 1)[1].split(
        "# mlx-vlm is NOT adopted", 1)[0]
    check("num_ctx=adopted_ctx" in adopt,
          "adoption passes the running instance's context_length, so a "
          "gateway restart cannot reload a model to change a number")
    check("context_length=adopted_ctx" in adopt,
          "and records it, so /warm reports it immediately after a restart")

    step(8, "the window is published where a consumer can read it")
    srv = open(os.path.join(os.path.dirname(__file__), "server.py")).read()
    v1 = srv.split('@app.get("/v1/models")', 1)[1].split(
        '@app.get("/v1/models/{model_id:path}")', 1)[0]
    check('entry["context_length"] = context_length' in v1,
          "/v1/models carries context_length per entry")
    check('m.get("context_length")' in v1,
          "...sourced from offering(), the one place the three endpoints agree")
    check('@app.get("/v1/models/{model_id:path}")' in srv,
          "/v1/models/{id} exists — it used to 404, which is how a consumer "
          "ended up guessing")
    check('{model_id:path}' in srv,
          ":path, so an id with a slash (mlx-community/...) resolves")
    off = mgr_src.split("def offering(", 1)[1].split("\n    def ", 1)[0]
    check('"context_length": mm.context_length' in off,
          "offering() reports the resident instance's real window")
    check("prospective_ctx" in off,
          "...and a loadable model's window only where we pin it")

    step(9, "a resident window is never a guess")
    mm_src = mgr_src.split("class ManagedModel", 1)[1].split("\n@", 1)[0]
    check("context_length: Optional[int] = None" in mm_src,
          "ManagedModel carries it, defaulting to None")
    load = mgr_src.split("def load_ollama_model(", 1)[1].split(
        "\n    def ", 1)[0]
    check("served_ctx" in load and 'm.get("context_length")' in load,
          "load_ollama_model reads the window back from `ollama ps` after "
          "the load instead of assuming it got what it asked for")
    ensure = mgr_src.split("# Check if Ollama still has it loaded", 1)[1].split(
        "\n    def ", 1)[0]
    check("mm.context_length = int(ctx)" in ensure,
          "ensure_running refreshes it off a read it already makes, so a "
          "reload we did not perform does not strand a stale number")

    print(f"\n{'='*64}")
    print(f"  {checks - len(failures)}/{checks} checks passed")
    if failures:
        print("  FAILURES:")
        for f in failures:
            print(f"    - {f}")
    print(f"{'='*64}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
