"""#973: no synchronous client runs on the event loop.

WHY THIS FILE EXISTS

On 2026-10-12 `models-cs` accepted TCP and answered nothing — `/health`
included, which does no I/O at all — at 0.0% CPU, for about twelve minutes.
`sample 848 3` put **2180 of 2180 samples** on the main thread in:

    sock_recv -> sock_recv_guts -> sock_call_ex -> internal_select -> poll

A blocking stdlib `socket.recv()` on the event-loop thread. The cause was one
line in `chat_completions`' non-streaming branch:

    for chunk in _held_sync(mm.name, mgr.ollama.chat(...)):

a sync `for` over a sync client, inside an `async def`, with no
`asyncio.to_thread`. So ONE non-streaming turn held the loop for its whole
generation — against a measured 45,142-token turn, 770.8s wall.

WHAT IT COST BEFORE ANYONE LOOKED AT THE LOOP

    a `000` read as "down" rather than "busy", by every consumer
    `pick_placement` treats unreachable as a FALLTHROUGH reason, so a seat's
      own long turn could route its recreate onto another box entirely
      (claude-containers, #779)
    the background loops starved, so the offer went stale and `think` briefly
      vanished from the model table — the 2026-10-10 23:00-23:05Z window that
      two seats measured and neither could explain

Three distinct symptoms, investigated separately for two days, one cause.

THE DETECTOR DECIDES RATHER THAN COUNTS

A first version of this check listed sync-client call sites and reported a
count. That is not a check: three of the four sites on `main` were on the loop
and one was already safe, and a count cannot tell them apart. So this walks
parents and says WHICH shields each call:

    to_thread              awaited off the loop          safe
    sync def (threadpool)  a plain `def` handed to StreamingResponse, which
                           Starlette iterates in a threadpool               safe
    ON THE LOOP            nothing shields it                               FAIL

And it is exercised against synthetic known-bad sources before it is trusted
on the real one — a detector that reports zero problems is indistinguishable
from one that cannot see.

    python -m ollama.test_event_loop

SCOPE: this is the STRUCTURAL half only. The behavioural half — a blocked
provider, tick counting, cancellation, release — is `test_loop_not_blocked.py`
(#959), which covers it and does it well. The two are not redundant: a
behaviour test must stub the slow call to stay fast, and #959's stubs
`prompt_size.count` to an instant lambda, so the one call still on the loop is
invisible to it. This reads the shape instead, which is what a stub cannot hide.
"""
import ast
import os

if __package__ in (None, ""):
    print("run as:  python -m ollama.test_event_loop")
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


# Clients whose calls block: a sync HTTP request, or CPU-bound tokenisation.
SYNC_CLIENTS = ("mgr.ollama.chat", "mgr.ollama.generate", "prompt_size.count",
                "mgr.exo.chat", "mgr.ollama.chat_collect")

# Project helpers that run their callable in a worker. `_run_claimed` is
# #959's: it claims on the loop, starts a shielded task, and releases from a
# done callback so the claim outlives a cancelled caller.
_SHIELDS = ("_run_claimed",)


def classify(source: str) -> list:
    """(fn, line, callee, verdict) for every sync-client call in an async def."""
    tree = ast.parse(source)
    parent = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parent[c] = n
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]:
        for x in ast.walk(fn):
            if not (isinstance(x, ast.Call)
                    and ast.unparse(x.func) in SYNC_CLIENTS):
                continue
            verdict, node = "ON THE LOOP", x
            while node in parent and node is not fn:
                p = parent[node]
                if isinstance(p, ast.Call):
                    callee = ast.unparse(p.func)
                    if "to_thread" in callee:
                        verdict = "to_thread"
                        break
                    # #959's own shield. Omitting it made this report
                    # `completions`' FIXED non-streaming branch as ON THE LOOP
                    # — a false alarm on correct code, which is the failure
                    # direction that gets investigated rather than banked, but
                    # a defect in the detector all the same. A shield is
                    # whatever actually runs the call in a worker, not only the
                    # stdlib spelling of it.
                    if callee in _SHIELDS:
                        verdict = callee
                        break
                if isinstance(p, ast.FunctionDef):
                    verdict = "sync def (threadpool)"
                    break
                if isinstance(p, ast.AsyncFunctionDef) and p is not fn:
                    verdict = "nested async"
                    break
                node = p
            out.append((fn.name, x.lineno, ast.unparse(x.func), verdict))
    return out


# ------------------------------------------- 1. the detector discriminates
section("1. the detector fires on known-bad and is silent on known-good")

BAD = {
    "the exact shape that hung the gateway":
        "async def h():\n"
        "    for chunk in _held_sync(m, mgr.ollama.chat(m, msgs)):\n"
        "        pass\n",
    "a bare assignment":
        "async def h():\n    chunks = list(mgr.ollama.generate(m, p))\n",
    "CPU-bound tokenisation on the loop":
        "async def h():\n    n, how = prompt_size.count(msgs, m)\n",
    "inside a try, which shields nothing":
        "async def h():\n    try:\n        x = mgr.ollama.chat(m, msgs)\n"
        "    except Exception:\n        pass\n",
    "inside an if":
        "async def h():\n    if cond:\n        x = mgr.ollama.chat(m, msgs)\n",
    "inside an async with":
        "async def h():\n    async with lock:\n        x = mgr.ollama.chat(m, msgs)\n",
}
for label, src in BAD.items():
    v = [r[3] for r in classify(src)]
    check(v == ["ON THE LOOP"], f"fires: {label} (got {v})")

GOOD = {
    "awaited via to_thread":
        "async def h():\n"
        "    c = await asyncio.to_thread(lambda: list(mgr.ollama.chat(m, msgs)))\n",
    "to_thread with the callable passed positionally":
        "async def h():\n"
        "    n, how = await asyncio.to_thread(prompt_size.count, msgs, m)\n",
    "a plain def handed to StreamingResponse — Starlette threadpools it":
        "async def h():\n"
        "    def stream():\n"
        "        for c in mgr.ollama.generate(m, p):\n"
        "            yield c\n"
        "    return StreamingResponse(stream())\n",
    "no sync client at all":
        "async def h():\n    async for c in mgr.ollama.chat_stream(m, msgs):\n"
        "        pass\n",
}
for label, src in GOOD.items():
    v = [r[3] for r in classify(src)]
    check("ON THE LOOP" not in v, f"silent: {label} (got {v or ['none']})")

check(classify(GOOD["a plain def handed to StreamingResponse — Starlette "
                    "threadpools it"])[0][3] == "sync def (threadpool)",
      "and it NAMES the threadpool case rather than lumping it with to_thread "
      "— they are safe for different reasons and only one is ours")

# ------------------------------------------- 2. the real file
section("2. server.py runs no sync client on the loop")

here = os.path.dirname(__file__)
rows = classify(open(os.path.join(here, "server.py")).read())
check(len(rows) >= 3,
      f"found {len(rows)} sync-client call sites to judge — a detector with "
      f"nothing to judge would pass vacuously")

onloop = [r for r in rows if r[3] == "ON THE LOOP"]
for fn, ln, callee, verdict in rows:
    check(verdict != "ON THE LOOP",
          f"{fn}:{ln} {callee} -> {verdict}")
check(not onloop,
      f"nothing blocks the loop ({len(onloop)} offender(s))"
      + (f": {onloop}" if onloop else ""))

# A BLIND SPOT, stated rather than left for "0 offenders" to imply coverage.
#
# This only sees CALLS. `await asyncio.to_thread(prompt_size.count, msgs, m)`
# passes the callable BARE, so there is no `prompt_size.count(...)` Call node
# and the site is invisible here — not classified safe, *unseen*. That is how
# `server.py` went from four judgeable sites to three when the fix landed:
# coverage fell as a side effect of fixing one of them.
#
# Harmless in that direction, because the bare-callable form is the safe one.
# It would NOT be harmless for a future blocking call introduced the same way,
# so the count is asserted: if it drops again, something became invisible
# rather than correct.
_names = {ast.unparse(n.func) for n in ast.walk(ast.parse(
    open(os.path.join(here, "server.py")).read()))
    if isinstance(n, ast.Call)}
check(len(rows) >= 3,
      f"at least 3 judgeable sites ({len(rows)}) — a FALL means a call became "
      f"a bare callable and left this detector's view rather than being fixed. "
      f"Deliberately a floor and not an equality: a new correctly-wrapped call "
      f"site raises this, and an equality check would fire on the good case "
      f"too, which is a false alarm and those get banked")
check("asyncio.to_thread" in _names,
      "and to_thread is in use in server.py, so the bare-callable form above "
      "is a real pattern here rather than hypothetical")


