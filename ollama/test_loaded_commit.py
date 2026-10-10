"""#905: the gateway reports the commit it imported, or reports nothing.

WHY THIS FILE EXISTS

#904 left this box healthy for an hour while the next restart would have 422'd
every chat request: the process had imported its modules on Oct 5, the clone was
reset on Oct 8, and no instrument could compare the two facts. `/health` was 200
the whole time. This asserts the two properties that make the comparison
possible, and it asserts them by DRIVING the code rather than reading its source.

The source-reading trap is specific and this file was written to avoid it: three
assertions in `test_stream_deadlines.py` were satisfied by their own explanatory
comments, because a file that documents its own history contains every string it
ever got wrong. So nothing here greps `server.py`. Every check either calls a
function or drives the ASGI app.

    python -m ollama.test_loaded_commit
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    print("run as:  python -m ollama.test_loaded_commit")
    raise SystemExit(1)

# BEFORE importing server, because config reads this at ITS import and a later
# setdefault is too late — the first version of this file set it after the
# import and got 401 on every /status check, which is the same import-ordering
# mistake the thing under test exists to expose.
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


# ---------------------------------------------------------------- the reader
section("1. _read_loaded_commit against real git layouts")

from . import server  # noqa: E402

_read = server._read_loaded_commit


def _in(tmp, body):
    """Run the reader with __file__ pointed at a synthetic repo root."""
    pkg = Path(tmp) / "ollama"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "server.py").write_text("")
    real = server.__file__
    try:
        server.__file__ = str(pkg / "server.py")
        return body(Path(tmp))
    finally:
        server.__file__ = real


SHA = "a" * 40
OTHER = "b" * 40

with tempfile.TemporaryDirectory() as t:
    def detached(root):
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text(SHA + "\n")
        return _read()
    check(_in(t, detached) == SHA, "detached HEAD (a deploy clone) -> the sha")

with tempfile.TemporaryDirectory() as t:
    def on_branch(root):
        g = root / ".git"
        (g / "refs" / "heads").mkdir(parents=True)
        (g / "HEAD").write_text("ref: refs/heads/main\n")
        (g / "refs" / "heads" / "main").write_text(OTHER + "\n")
        return _read()
    check(_in(t, on_branch) == OTHER, "branch checkout -> follows the ref")

with tempfile.TemporaryDirectory() as t:
    def packed(root):
        g = root / ".git"
        g.mkdir()
        (g / "HEAD").write_text("ref: refs/heads/main\n")
        (g / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted \n"
            f"{OTHER} refs/heads/main\n")
        return _read()
    check(_in(t, packed) == OTHER, "packed-refs -> found when no loose ref file")

with tempfile.TemporaryDirectory() as t:
    def worktree(root):
        real = root / "realgit"
        real.mkdir()
        (real / "HEAD").write_text(SHA + "\n")
        (root / ".git").write_text(f"gitdir: {real}\n")
        return _read()
    check(_in(t, worktree) == SHA, "worktree (.git is a file) -> follows gitdir")

# A LINKED WORKTREE ON A BRANCH keeps HEAD per-worktree and refs/heads in the
# COMMON dir, named by the gitdir's `commondir` file. The case above passes with
# a detached HEAD (the sha is in HEAD itself), which is what a deploy clone has
# and is why this gap stayed hidden: production worked, and the reader was blind
# in exactly the layout contributors use. Measured 2026-10-09 in
# samcloud-services-wt/inflight-907.
with tempfile.TemporaryDirectory() as t:
    def wt_branch_loose(root):
        common = root / "maingit"
        (common / "refs" / "heads" / "feature").parent.mkdir(parents=True)
        (common / "refs" / "heads" / "feature").write_text(OTHER + "\n")
        wt = common / "worktrees" / "wt1"
        wt.mkdir(parents=True)
        (wt / "HEAD").write_text("ref: refs/heads/feature\n")
        (wt / "commondir").write_text("../..\n")
        (root / ".git").write_text(f"gitdir: {wt}\n")
        return _read()
    check(_in(t, wt_branch_loose) == OTHER,
          "worktree on a branch -> ref resolved from commondir, not the gitdir")

with tempfile.TemporaryDirectory() as t:
    def wt_branch_packed(root):
        common = root / "maingit"
        common.mkdir()
        (common / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled sorted \n"
            f"{OTHER} refs/heads/feature\n")
        wt = common / "worktrees" / "wt1"
        wt.mkdir(parents=True)
        (wt / "HEAD").write_text("ref: refs/heads/feature\n")
        (wt / "commondir").write_text("../..\n")
        (root / ".git").write_text(f"gitdir: {wt}\n")
        return _read()
    check(_in(t, wt_branch_packed) == OTHER,
          "...and from the commondir's packed-refs when there is no loose file")

with tempfile.TemporaryDirectory() as t:
    def wt_branch_missing(root):
        common = root / "maingit"
        common.mkdir()
        wt = common / "worktrees" / "wt1"
        wt.mkdir(parents=True)
        (wt / "HEAD").write_text("ref: refs/heads/gone\n")
        (wt / "commondir").write_text("../..\n")
        (root / ".git").write_text(f"gitdir: {wt}\n")
        return _read()
    check(_in(t, wt_branch_missing) is None,
          "...and a ref in neither place is still None, not a guess")

# A per-worktree ref must WIN over a same-named one in the common dir, which is
# the order git resolves in. Getting this backwards would report another
# worktree's commit as this process's own.
with tempfile.TemporaryDirectory() as t:
    def wt_local_wins(root):
        common = root / "maingit"
        (common / "refs" / "heads").mkdir(parents=True)
        (common / "refs" / "heads" / "feature").write_text(SHA + "\n")
        wt = common / "worktrees" / "wt1"
        (wt / "refs" / "heads").mkdir(parents=True)
        (wt / "refs" / "heads" / "feature").write_text(OTHER + "\n")
        (wt / "HEAD").write_text("ref: refs/heads/feature\n")
        (wt / "commondir").write_text("../..\n")
        (root / ".git").write_text(f"gitdir: {wt}\n")
        return _read()
    check(_in(t, wt_local_wins) == OTHER,
          "a per-worktree ref wins over the common dir's same-named ref")

section("2. absent rather than guessed — every failure returns None")

with tempfile.TemporaryDirectory() as t:
    check(_in(t, lambda root: _read()) is None,
          "no .git at all (tarball deploy) -> None")

with tempfile.TemporaryDirectory() as t:
    def truncated(root):
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text("deadbeef\n")   # short
        return _read()
    check(_in(t, truncated) is None, "a short sha is refused, not returned")

with tempfile.TemporaryDirectory() as t:
    def notahex(root):
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text("z" * 40 + "\n")
        return _read()
    check(_in(t, notahex) is None, "40 non-hex chars refused — length is not enough")

with tempfile.TemporaryDirectory() as t:
    def dangling(root):
        g = root / ".git"
        g.mkdir()
        (g / "HEAD").write_text("ref: refs/heads/gone\n")
        (g / "packed-refs").write_text("")
        return _read()
    check(_in(t, dangling) is None, "a ref pointing nowhere -> None, not the ref text")

section("3. this very process reports its own commit")

here = _read()
if (Path(server.__file__).resolve().parent.parent / ".git").exists():
    want = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=Path(server.__file__).resolve().parent.parent
                          ).stdout.strip()
    check(here == want, f"reader agrees with `git rev-parse HEAD` ({str(want)[:8]})")
    check(server.LOADED_COMMIT == want,
          "LOADED_COMMIT captured the same sha AT IMPORT")
else:
    check(here is None, "not a checkout -> None (and LOADED_COMMIT is None)")
    check(server.LOADED_COMMIT is None, "LOADED_COMMIT is None, not a guess")

# -------------------------------------------------------------- the modules
section("4. loaded_modules reports what was imported, dotted")

mods = server.loaded_modules()
check(all(m == "ollama" or m.startswith("ollama.") for m in mods),
      "every entry is `ollama` or `ollama.*`")
check(mods == sorted(mods), "sorted, so a diff between two reads is readable")
check("ollama.server" in mods, "ollama.server present — the module answering")
check("ollama.config" in mods, "ollama.config present")

# whisper_server is the case three hand-picked sets got wrong: on disk,
# non-test, and never imported — it is exec'd as a script per transcription.
check("ollama.whisper_server" not in mods,
      "ollama.whisper_server ABSENT — exec'd as a script, never imported")
check(not any(m.startswith("ollama.test_") for m in mods),
      "no test module leaks into the reported set")

section("5. DOTTED, not bare — the seam that failed in #52")

# `drift-report.py --imported` wants what /status emits. The help text once
# asked for bare names while /status gave dotted, so every module read
# "NOT imported" — a false refusal. Assert the shape, not the help text.
check(all("." in m for m in mods if m != "ollama"),
      "names carry their package, so --imported needs no translation")

# ------------------------------------------------------------- the response
section("6. /status over the real ASGI app")

from fastapi.testclient import TestClient  # noqa: E402

# NO `with`, AND THE OMISSION IS LOAD-BEARING. TestClient runs the app's
# lifespan only when used as a context manager, and this app's startup ADOPTS
# whatever Ollama currently holds while its shutdown UNLOADS it. So
# `with TestClient(server.app)` on a live box EVICTS THE PRODUCTION MODEL.
#
# Measured by doing it, 2026-10-09 15:22 on slice: the first version of this
# file took the resident 27b from `keep_alive=-1` to an `expires_at` three
# minutes in the past, mid-service, while the real gateway was holding a
# 118,272-token request. A gate that cannot be run on the box it describes is
# not a gate. Every other TestClient in this package is already constructed
# bare; this file was the only one that was not.
#
# `mgr` is built in that same startup, so bare construction leaves it None and
# the handler raises. It is stubbed rather than started: what this section
# asserts is the HANDLER's contract — that `loaded_commit` is omitted-not-null
# and `loaded_modules` is a dotted list — and that contract does not involve a
# manager. Stubbing keeps the two failures separable, so a real manager fault
# cannot show up here as a drift-reporting fault.
class _StubMgr:
    def status(self):
        return {"models": [], "stub": True}


body = None
code = None
real_mgr = server.mgr
try:
    server.mgr = _StubMgr()
    c = TestClient(server.app)
    r = c.get("/status")
    code = r.status_code
    check(code == 200, f"GET /status -> 200 (got {code})")
    body = r.json()
except Exception as e:
    check(False, f"GET /status raised {e!r}")
finally:
    server.mgr = real_mgr

check(body is not None and body.get("stub") is True,
      "the stub manager's own status is passed through, so the handler MERGES "
      "rather than replaces")

# GUARDED ON THE 200. The first version checked the body unconditionally and a
# 401 error payload satisfied `body.get("loaded_commit", "absent") is not None`
# — a pass that asserted nothing, on the response of a request that failed.
if code == 200 and isinstance(body, dict):
    check("loaded_modules" in body, "loaded_modules present in the response")
    if server.LOADED_COMMIT is None:
        check("loaded_commit" not in body,
              "loaded_commit OMITTED when unknown — not null")
    else:
        check(body.get("loaded_commit") == server.LOADED_COMMIT,
              "loaded_commit is the import-time value, not a re-read")
    # A null would compare as a value and read as drift. Absent cannot. Assert
    # on the KEY, not on a sentinel default that a missing key also satisfies.
    check(("loaded_commit" not in body) or (body["loaded_commit"] is not None),
          "loaded_commit is never null when the key exists")
    check(isinstance(body.get("loaded_modules"), list),
          "loaded_modules is a list, joinable for --imported")
else:
    check(False, "response checks skipped — /status did not return 200")

section("7. the comparison the ticket asked for actually runs")

# Drive deploy/drift-report.py with this process's own figures. Same commit
# both sides is the no-drift case and must exit 0 saying so.
root = Path(server.__file__).resolve().parent.parent
tool = root / "deploy" / "drift-report.py"
if tool.exists() and server.LOADED_COMMIT:
    p = subprocess.run([sys.executable, str(tool),
                        server.LOADED_COMMIT, server.LOADED_COMMIT,
                        "--imported", ",".join(mods)],
                       capture_output=True, text=True, cwd=root)
    check(p.returncode == 0, f"drift-report same/same -> exit 0 (got {p.returncode})")
    check("no drift" in p.stdout.lower(),
          "and it says so rather than printing an empty diff")
else:
    print("  skip drift-report (tool or commit absent)")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
