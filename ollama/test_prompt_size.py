#!/usr/bin/env python3
"""Regression test for the prompt-size gate on the pool (ticket #837).

On 2026-09-25 a 108,753-token prompt reached the exo pool through this gateway,
exo began prefilling it, and macOS's GPU driver panicked slice rather than
failing the allocation — three hours down. Nothing in the gateway measured a
prompt before dispatching it, so any client could do that, and one did.

What this pins down:

  1. The prompt is counted through the resident model's own chat template and
     tokenizer — the same two files exo uses — so our number and exo's mean the
     same thing.
  6. Tool definitions count. This template renders every tool inline, and a
     counter that reads `messages` alone read a live request as 11,742 tokens
     that exo then prefilled as 18,118.
  2. A prompt over the limit is refused — 413, with the limit, the measurement
     and the method in the body.
  3. The refusal says the limit is about the HOST and not the model's context
     window, and names the model's real window, because the obvious wrong move
     after being told "too long" is to go looking for a longer-context model.
  4. The refusal costs the pool nothing: no lease is requested, no `/state` is
     read, no byte reaches exo. A decline that had to take an exclusive lease
     to say no would be its own small outage.
  5. With the tokenizer missing, the estimate still refuses (over-counting, in
     the safe direction) rather than failing open.

Leases nothing, loads nothing, sends nothing to the pool — the whole point is
that this path stops short of all three, so it is safe on a live box.

    python -m ollama.test_prompt_size
"""

import os
import sys

os.environ["AUTH_ENABLED"] = "0"          # test process only, before import
os.environ.setdefault("SC_TOKEN", "test")

from fastapi.testclient import TestClient

from . import config, prompt_size, server
from .manager import Backend

failures = []
skipped = []


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def skip(msg):
    """An assertion that cannot run here, COUNTED and named.

    Not `check(True, …)` and not silence. Four assertions below need the model
    fixture, and before #920 they ran anyway on the estimate path and failed —
    each naming its own cause ("estimate at 1.5 chars/token") while asserting
    the template path. The obvious fix is to guard them, and the obvious guard
    is the dangerous one: skipping without counting turns four failures into
    four silent passes, and the run still prints "All checks passed". So a skip
    is reported, tallied, and named in the summary, which is the only form in
    which "this box cannot test that" is distinguishable from "that works".
    """
    print(f"  SKIP  {msg}")
    skipped.append(msg)


MODEL = "mlx-community/GLM-4.7-Flash-6bit"


def words(n_tokens):
    """Roughly n_tokens of ordinary text — exactness is the tokenizer's job."""
    return " ".join(["alpha beta gamma delta epsilon zeta"] * (n_tokens // 6 + 1))


def main():
    step(1, "count a prompt with the model's own tokenizer")
    tok, longest, ctx = prompt_size._load(MODEL, config.EXO_MODELS_DIR)
    # ONE flag, derived once, shared by every step that needs the fixture.
    # Step 1 guarded this and five later call sites did not (#920): they took
    # the estimate path and asserted the template path, so a box without the
    # model saw four failures that named the model store nowhere. Deriving it
    # here rather than re-checking per step is deliberate — two guards for one
    # dependency is how they drift apart.
    have_fixture = tok is not None
    if not have_fixture:
        print(f"  SKIP: no tokenizer under {config.EXO_MODELS_DIR} for {MODEL}.")
        print(f"        The template path cannot run on this box at all, so the four")
        print(f"        assertions that pin it are skipped below and named in the summary.")
        print(f"        Steps 2, 6 and 7 still run: the limit, the refusal and the")
        print(f"        estimate fallback do not need the fixture.")
    else:
        n, how = prompt_size.count([{"role": "user", "content": words(1000)}],
                                   MODEL, config.EXO_MODELS_DIR, config.EXO_CHARS_PER_TOKEN)
        check(how == "chat template", f"counted through the model's template ({how})")
        check(800 < n < 1400, f"count is in the right neighbourhood ({n} tokens)")
        check(ctx and ctx > 100000,
              f"model's declared context window read from config.json ({ctx})")

    step(2, "a prompt over the limit raises, under it does not")
    small = [{"role": "user", "content": words(50)}]
    big = [{"role": "user", "content": words(config.EXO_MAX_PROMPT_TOKENS * 2)}]
    ok = prompt_size.check(small, MODEL, limit=config.EXO_MAX_PROMPT_TOKENS,
                           completion_budget=8192, models_dir=config.EXO_MODELS_DIR,
                           chars_per_token=config.EXO_CHARS_PER_TOKEN,
                           resource_id=config.EXO_RESOURCE_ID)
    check(ok[0] > 0, f"a small prompt passes ({ok[0]} tokens, {ok[1]})")
    try:
        prompt_size.check(big, MODEL, limit=config.EXO_MAX_PROMPT_TOKENS,
                          completion_budget=8192, models_dir=config.EXO_MODELS_DIR,
                          chars_per_token=config.EXO_CHARS_PER_TOKEN,
                          resource_id=config.EXO_RESOURCE_ID)
        check(False, "an over-limit prompt raises PromptTooLarge")
    except prompt_size.PromptTooLarge as e:
        check(True, f"an over-limit prompt raises PromptTooLarge ({e.tokens} tokens)")
        body = e.as_dict()
        check(body["limit_tokens"] == config.EXO_MAX_PROMPT_TOKENS,
              "the body names the limit")
        check(body["measured_tokens"] > body["limit_tokens"], "the body names the measurement")
        check("HOST" in body["note"] and "context window" in body["note"].lower(),
              "the body says the limit protects the host, not the context window")
        # The window comes off the fixture's config.json, so it is 0 without it.
        if have_fixture:
            check(body.get("model_context_length", 0) > config.EXO_MAX_PROMPT_TOKENS,
                  "the body names the model's real context window for contrast")
        else:
            skip(f"the body names the model's real context window — needs {MODEL}'s "
                 f"config.json under {config.EXO_MODELS_DIR}")

    step(3, "the 413 comes back over HTTP, and the pool was never touched")
    # Deliberately NOT the app's lifespan: starting it here would reap stray
    # mlx-vlm processes, push stats and open the background loops of a gateway
    # that is already running on this box. The route is what is under test, so
    # the pool and the resolver are stood in for and the spies record whether
    # the refusal reached past them.
    touched = {"state": 0, "lease": 0}

    class SpyPool:
        def state(self, *a, **k):
            touched["state"] += 1
            return {}

    class SpyManager:
        exo = SpyPool()

        def exo_request_defaults(self, model_id):
            return {"max_tokens": config.EXO_MAX_TOKENS}

        def acquire_pool(self, *a, **k):
            touched["lease"] += 1
            raise AssertionError("a refused prompt must not reach the lease")

    class StubModel:
        backend = Backend.EXO
        name = MODEL
        tier = "think"

    real_mgr, real_resolve = server.mgr, server._resolve_model
    server.mgr = SpyManager()

    async def stub_resolve(model_name):
        return StubModel()

    server._resolve_model = stub_resolve
    try:
        client = TestClient(server.app)
        r = client.post("/v1/chat/completions", json={
            "model": "think",
            "messages": [{"role": "user", "content": words(config.EXO_MAX_PROMPT_TOKENS * 2)}],
            "max_tokens": 16,
        })
        check(r.status_code == 413, f"HTTP 413 (got {r.status_code})")
        detail = (r.json() or {}).get("detail", {})
        check(detail.get("error") == "prompt_too_large",
              f"error code is prompt_too_large ({detail.get('error')})")
        check(detail.get("limit_tokens") == config.EXO_MAX_PROMPT_TOKENS,
              "the limit is readable by the client")
        check(detail.get("completion_budget_tokens") == 16,
              f"the completion budget quoted is the one that would have been sent "
              f"({detail.get('completion_budget_tokens')})")
        check(touched["lease"] == 0, "no lease was requested for a refused prompt")
        check(touched["state"] == 0,
              f"the pool's /state was not read for a refused prompt "
              f"({touched['state']} reads)")
    finally:
        server.mgr, server._resolve_model = real_mgr, real_resolve

    step(4, "tool definitions count toward the limit")
    tiny = [{"role": "user", "content": "hi"}]
    fat_tools = [{
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": words(200),
            "parameters": {"type": "object", "properties": {"q": {"type": "string",
                                                                  "description": words(60)}}},
        },
    } for i in range(20)]
    bare, how_bare = prompt_size.count(tiny, MODEL, config.EXO_MODELS_DIR,
                                       config.EXO_CHARS_PER_TOKEN)
    withtools, how_tools = prompt_size.count(tiny, MODEL, config.EXO_MODELS_DIR,
                                             config.EXO_CHARS_PER_TOKEN, tools=fat_tools)
    check(withtools > bare + 1000,
          f"tools move the count ({bare} -> {withtools} tokens for the same messages)")
    # The template path has to survive tools, not just messages. It did not:
    # this template calls `tojson(ensure_ascii=False)` per tool and jinja2's own
    # filter rejects that keyword, so for two days every request carrying tools
    # rendered nothing and fell through to the serialised count — the blunt path,
    # in precisely the case the template was added for. Silent, because a
    # fallback that works looks like success.
    if have_fixture:
        check(how_tools == "chat template",
              f"tools still count through the template, not the fallback ({how_tools})")
    else:
        skip(f"tools still count through the template — needs {MODEL}'s "
             f"chat_template.jinja ({how_tools} without it)")
    try:
        prompt_size.check(tiny, MODEL, limit=config.EXO_MAX_PROMPT_TOKENS,
                          completion_budget=0, models_dir=config.EXO_MODELS_DIR,
                          chars_per_token=config.EXO_CHARS_PER_TOKEN,
                          resource_id=config.EXO_RESOURCE_ID, tools=fat_tools * 3)
        check(False, "a two-word prompt with oversized tools is refused")
    except prompt_size.PromptTooLarge as e:
        check(True, f"a two-word prompt with oversized tools is refused ({e.tokens} tokens)")

    step(5, "a tool-calling conversation still counts through the template")
    # OpenAI specifies `arguments` as a JSON string and hermes sends one. The
    # template does `tc.arguments.items()`, so the string lost the render and
    # the count fell to the serialised fallback — which on this shape measured
    # 126 tokens against the template's 29. A 4x over-count is how a legitimate
    # prompt gets refused, so this asserts the method, not just a number.
    convo = [
        {"role": "user", "content": "what is the weather"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Melbourne"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "17C and raining"},
    ]
    n_convo, how_convo = prompt_size.count(convo, MODEL, config.EXO_MODELS_DIR,
                                           config.EXO_CHARS_PER_TOKEN)
    if have_fixture:
        check(how_convo == "chat template",
              f"tool_calls with string arguments render ({how_convo}, {n_convo} tokens)")
    else:
        skip(f"tool_calls with string arguments render — needs {MODEL}'s "
             f"chat_template.jinja ({how_convo} without it)")
    # Unparseable arguments are left alone and the fallback takes it: a wrong
    # count in the safe direction beats a fabricated one.
    broken = [dict(convo[1], tool_calls=[{"id": "c1", "type": "function", "function": {
        "name": "f", "arguments": "not json at all"}}])]
    _, how_broken = prompt_size.count(broken, MODEL, config.EXO_MODELS_DIR,
                                      config.EXO_CHARS_PER_TOKEN)
    # "serialised" is the fallback BELOW the template, not the bottom one. Without
    # a tokenizer there is no serialised path to reach — it is "estimate" — so
    # this asserts a distinction the fixture creates.
    if have_fixture:
        check(how_broken == "serialised",
              f"arguments that will not parse fall back rather than guess ({how_broken})")
    else:
        skip(f"unparseable arguments fall back to serialised — needs {MODEL}'s "
             f"tokenizer; without it the fallback is {how_broken!r}, one step further down")

    step(6, "an absurd body is refused without being tokenized")
    huge = [{"role": "user", "content": "x" * (config.EXO_MAX_PROMPT_TOKENS * 600)}]
    try:
        prompt_size.check(huge, MODEL, limit=config.EXO_MAX_PROMPT_TOKENS,
                          completion_budget=0, models_dir=config.EXO_MODELS_DIR,
                          chars_per_token=config.EXO_CHARS_PER_TOKEN,
                          resource_id=config.EXO_RESOURCE_ID)
        check(False, "a body past any possible limit is refused")
    except prompt_size.PromptTooLarge as e:
        check(True, f"a body past any possible limit is refused ({e.method})")

    step(7, "with no tokenizer, the estimate still refuses")
    prompt_size._CACHE.pop("no-such/model", None)
    try:
        prompt_size.check([{"role": "user", "content": words(config.EXO_MAX_PROMPT_TOKENS)}],
                          "no-such/model", limit=config.EXO_MAX_PROMPT_TOKENS,
                          completion_budget=0, models_dir="/nonexistent",
                          chars_per_token=config.EXO_CHARS_PER_TOKEN,
                          resource_id=config.EXO_RESOURCE_ID)
        check(False, "the estimate path refuses rather than failing open")
    except prompt_size.PromptTooLarge as e:
        check("estimate" in e.method, f"refused by estimate ({e.method})")

    step(8, "the drift check reads a streamed usage chunk, not just a JSON body")
    # hermes streams — established from the gateway log, where the only
    # mid-stream events on a day I sent no chat requests sit inside hermes-only
    # windows. So a drift check that lives on the non-streaming path only is a
    # check on the client that does not matter.
    lines = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":18118,"completion_tokens":7}}',
        'data: [DONE]',
        ': keep-alive',
        '',
    ]
    found = [server._usage_prompt_tokens(l) for l in lines]
    check(found == [None, 18118, None, None, None],
          f"usage is picked out of the stream and nothing else is ({found})")
    check(server._usage_prompt_tokens('data: {"usage": not json}') is None,
          "a malformed chunk returns None rather than raising into the stream")

    warned = []
    real_warning = server.log.warning
    server.log.warning = lambda msg, *a, **k: warned.append(str(msg))
    try:
        server._check_drift(11742, "chat template", 18118, MODEL)   # 1.54x, the real case
        server._check_drift(11742, "chat template", 12000, MODEL)   # 1.02x, agreement
        server._check_drift(None, "chat template", 18118, MODEL)    # nothing measured
        server._check_drift(11742, "chat template", None, MODEL)    # no usage chunk
    finally:
        server.log.warning = real_warning
    check(len(warned) == 1 and "1.54x" in warned[0],
          f"warns once, on the gap that matters ({len(warned)} warning(s))")

    print(f"\n{'='*60}")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    if skipped:
        # Named, not just counted. "All checks passed" over four silent skips is
        # the failure mode this guard was added to avoid, so the summary says
        # what was not tested and why, and rc stays 0 because a missing fixture
        # is an environment fact and not a defect in the gate (#920).
        print(f"  All runnable checks passed, {len(skipped)} SKIPPED on this box:")
        for m in skipped:
            print(f"    - {m}")
        print(f"  The template path is untested here. A box with {MODEL}")
        print(f"  under EXO_MODELS_DIR runs all of it — wafer does.")
    else:
        print("  All checks passed")


main()
