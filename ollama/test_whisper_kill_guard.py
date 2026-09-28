#!/usr/bin/env python3
"""`_kill_stray_whisper` must only SIGTERM a real whisper child of ours.

The pids come from `pgrep -f whisper_server.py`, a SUBSTRING match over the whole
command line, and the loop SIGTERMs what it returns. The predicate guarding that
kill is the only thing between the gateway and signalling something unrelated on
a box that also runs the operator's own shells, editors and greps.

`whisper_server.py` is a worse pattern to match on than `mlx_vlm.server` was,
which is why this test exists rather than being assumed from the VLM one: it is
a plain filename, so it appears unquoted in any command line that opens, greps,
tails or edits the file — including this repo's own tooling. Checking uid plus
"the script name is in argv" is therefore not enough on its own, and cases [2]
and [5] are the two shapes that pass that weaker check and must still survive.

Fakes, and why they are shaped this way:
  - `ps -o uid=,command= -p <pid>` really emits a leading-space-padded uid, a
    space, then the command line, and emits NOTHING for a pid that has gone.
    Both are reproduced, because a fake that cannot refuse cannot test a
    contract.
  - `pgrep -f` really emits newline-separated pids, which the caller `.split()`s.
  - The manager is stood in for by a holder carrying only the two attributes
    `_kill_stray_whisper` touches. Both real methods run unmodified.

Run: python3 ollama/test_whisper_kill_guard.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama import manager as mgr  # noqa: E402

passed = failed = 0

OUR_PY = "/Users/someone/code/mlx-whisper-server/.venv/bin/python"
OUR_SCRIPT = "/Users/someone/code/samcloud-services/ollama/whisper_server.py"
OUR_UID = os.getuid()
OTHER_UID = OUR_UID + 1

CHILD = (
    f"{OUR_PY} {OUR_SCRIPT} --model mlx-community/whisper-large-v3-turbo "
    f"--host 127.0.0.1 --port 8803 --spool /Users/someone/var/spool/whisper "
    f"--ffmpeg /opt/homebrew/bin/ffmpeg"
)


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  \033[32m✓\033[0m {label}")
    else:
        failed += 1
        print(f"  \033[31m✗\033[0m {label}  {detail}")


class FakeCompleted:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.returncode = 0 if stdout else 1


class FakeProcTable:
    """Answers `pgrep -f` and `ps -o uid=,command= -p` from a pid -> (uid, cmd) map."""

    def __init__(self, table):
        self.table = table

    def run(self, argv, **kw):
        if argv[0] == "pgrep":
            if not self.table:
                # real pgrep prints nothing and exits 1 when it matches nothing
                return FakeCompleted("")
            return FakeCompleted("\n".join(str(p) for p in self.table) + "\n")
        if argv[0] == "ps":
            pid = int(argv[-1])
            if pid not in self.table:
                # real ps prints nothing for a pid that has exited
                return FakeCompleted("")
            uid, cmd = self.table[pid]
            # real ps right-pads the uid column, hence the leading spaces
            return FakeCompleted(f"  {uid} {cmd}\n")
        raise AssertionError(f"unexpected command in test: {argv!r}")


class Holder:
    """Carries only what _kill_stray_whisper touches; both real methods run."""

    _is_whisper_server = staticmethod(mgr.ModelManager._is_whisper_server)
    _kill_stray_whisper = mgr.ModelManager._kill_stray_whisper

    def __init__(self, models=None):
        self.models = models or {}


def run_killer(table, models=None):
    """Run the real killer over a faked process table. Returns the pids SIGTERMed."""
    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    saved = (mgr.subprocess, os.kill, mgr.WHISPER_PYTHON, mgr.WHISPER_SCRIPT)
    try:
        mgr.subprocess = FakeProcTable(table)
        os.kill = fake_kill
        mgr.WHISPER_PYTHON = OUR_PY
        mgr.WHISPER_SCRIPT = OUR_SCRIPT
        Holder(models)._kill_stray_whisper()
    finally:
        (mgr.subprocess, os.kill, mgr.WHISPER_PYTHON, mgr.WHISPER_SCRIPT) = saved
    return [p for p, _ in killed]


def main():
    import signal

    print("stray-whisper kill guard (#858)")
    print(f"  our uid {OUR_UID}")
    print(f"  WHISPER_PYTHON {OUR_PY}")
    print(f"  WHISPER_SCRIPT {OUR_SCRIPT}")
    print()

    print("  [1] control: a real whisper child of ours IS killed")
    killed = run_killer({5001: (OUR_UID, CHILD)})
    check("the real child is SIGTERMed", killed == [5001], f"killed={killed}")

    print("  [2] a grep for the script path is NOT killed")
    print("      (argv[0] is grep, and this is the case that check exists for)")
    killed = run_killer({5002: (OUR_UID, f"grep -rn -- {OUR_SCRIPT} /Users/someone")})
    check("the grep survives", killed == [], f"killed={killed}")

    print("  [3] an editor holding the file open is NOT killed")
    killed = run_killer({5003: (OUR_UID, f"vim {OUR_SCRIPT}")})
    check("the editor survives", killed == [], f"killed={killed}")

    print("  [4] another user's real-looking child is NOT killed")
    killed = run_killer({5004: (OTHER_UID, CHILD)})
    check("another uid survives", killed == [], f"killed={killed}")

    print("  [5] a DIFFERENT interpreter running the same script is NOT killed")
    print("      (a hand-started copy from another venv is not ours to reap;")
    print("       we cannot tear it down and must not pretend we started it)")
    killed = run_killer({5005: (OUR_UID, f"/usr/bin/python3 {OUR_SCRIPT} --port 8803")})
    check("a foreign interpreter survives", killed == [], f"killed={killed}")

    print("  [5b] our interpreter running a DIFFERENT script is NOT killed")
    print("       (`$WHISPER_PYTHON ollama/test_whisper_child.py` is this shape)")
    killed = run_killer({
        5006: (OUR_UID, f"{OUR_PY} /Users/someone/code/samcloud-services/ollama/test_whisper_child.py"),
    })
    check("a sibling script survives", killed == [], f"killed={killed}")

    print("  [6] our own tracked child is skipped before the predicate is reached")

    class FakeMM:
        backend = mgr.Backend.WHISPER

        class whisper_process:
            pid = 5007

    killed = run_killer({5007: (OUR_UID, CHILD)}, models={"owned": FakeMM()})
    check("an owned pid is left alone", killed == [], f"killed={killed}")

    print("  [6b] a tracked VLM does not make the whisper child look owned")

    class FakeVLM:
        backend = mgr.Backend.VLM

        class vlm_process:
            pid = 5008

    killed = run_killer({5008: (OUR_UID, CHILD)}, models={"vlm": FakeVLM()})
    check("a stray with a VLM's pid is still killed", killed == [5008], f"killed={killed}")

    print("  [7] a pid that exited between pgrep and ps is NOT killed")
    killed = run_killer({})
    check("a vanished pid is not signalled", killed == [], f"killed={killed}")

    print("  [8] mixed table: only the real one dies")
    killed = run_killer({
        5010: (OUR_UID, f"grep -rn -- {OUR_SCRIPT} ."),
        5011: (OUR_UID, CHILD),
        5012: (OUR_UID, f"tail -f /Users/someone/var/logs/whisper_server.py.log"),
        5013: (OTHER_UID, CHILD),
    })
    check("exactly the child is killed", killed == [5011], f"killed={killed}")

    print("  [9] the signal sent is SIGTERM, not SIGKILL")
    sent = []

    def spy(pid, sig):
        sent.append(sig)

    saved = (mgr.subprocess, os.kill, mgr.WHISPER_PYTHON, mgr.WHISPER_SCRIPT)
    try:
        mgr.subprocess = FakeProcTable({5014: (OUR_UID, CHILD)})
        os.kill = spy
        mgr.WHISPER_PYTHON = OUR_PY
        mgr.WHISPER_SCRIPT = OUR_SCRIPT
        Holder()._kill_stray_whisper()
    finally:
        (mgr.subprocess, os.kill, mgr.WHISPER_PYTHON, mgr.WHISPER_SCRIPT) = saved
    check("SIGTERM", sent == [signal.SIGTERM], f"sent={sent}")

    print()
    if failed:
        print(f"  FAILED: {failed} failed, {passed} passed")
        sys.exit(1)
    print(f"  ALL PASSED: stray-whisper kill guard {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
