#!/usr/bin/env python3
"""`_kill_stray_vlm` must only SIGTERM a real `python -m mlx_vlm.server` of ours.

The pids come from `pgrep -f mlx_vlm.server`, which is a SUBSTRING match over the
whole command line, and the loop sends SIGTERM to what it returns. So the predicate
guarding that kill is the only thing standing between the gateway and signalling an
unrelated process on the same box.

Checking uid plus adjacent `-m mlx_vlm.server` argv tokens is not enough, because
`ps` output carries no quoting: a single quoted argument comes back as separate
whitespace tokens. So `grep -r -- "-m mlx_vlm.server" /tree`, one argv element,
prints as adjacent `-m` and `mlx_vlm.server` tokens under our own uid. That is
case [2b], and it is the realistic long-lived form.

Case [2] keeps the unquoted two-argument spelling for completeness. It has the same
argv shape but is not reproducible as a live process: `--` ends option parsing, so
`-m` is the pattern and `mlx_vlm.server` a filename that does not exist, and grep
exits at once. Both are killed without the argv[0] check.

Fakes, and why they are shaped this way:
  - `ps -o uid=,command= -p <pid>` really emits a leading-space-padded uid, a
    space, then the command line ("  504 /bin/zsh -c ..."), and emits NOTHING for
    a pid that has gone. Both are reproduced, including the empty case, because a
    fake that cannot refuse cannot test a contract.
  - `pgrep -f` really emits newline-separated pids, which the caller `.split()`s.
  - The manager is stood in for by a holder carrying the two attributes
    `_kill_stray_vlm` actually touches — `self.models` and `self._is_vlm_server`.
    Both real methods are exercised unmodified; only the model registry is faked,
    and in the real class that is a plain dict too.

Run: python3 ollama/test_vlm_kill_guard.py
"""
import os
import subprocess as real_subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama import manager as mgr  # noqa: E402

passed = failed = 0

OUR_PY = "/Users/someone/code/mlx-vlm-server/.venv/bin/python"
OUR_UID = os.getuid()
OTHER_UID = OUR_UID + 1


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
            # real pgrep emits newline-separated pids
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
    """Carries only what _kill_stray_vlm touches; both real methods run unmodified."""

    _is_vlm_server = staticmethod(mgr.ModelManager._is_vlm_server)
    _kill_stray_vlm = mgr.ModelManager._kill_stray_vlm

    def __init__(self, models=None):
        self.models = models or {}


def run_killer(table, models=None):
    """Run the real killer over a faked process table. Returns the pids SIGTERMed."""
    killed = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    saved_sub, saved_kill, saved_py = mgr.subprocess, os.kill, mgr.VLM_PYTHON
    try:
        mgr.subprocess = FakeProcTable(table)
        os.kill = fake_kill
        mgr.VLM_PYTHON = OUR_PY
        Holder(models)._kill_stray_vlm()
    finally:
        mgr.subprocess, os.kill, mgr.VLM_PYTHON = saved_sub, saved_kill, saved_py
    return [p for p, _ in killed]


def main():
    import signal

    print("stray-VLM kill guard (#845)")
    print(f"  our uid {OUR_UID}, VLM_PYTHON {OUR_PY}")
    print()

    print("  [1] control: a real `python -m mlx_vlm.server` of ours IS killed")
    killed = run_killer({
        4001: (OUR_UID, f"{OUR_PY} -m mlx_vlm.server --model foo --port 8801"),
    })
    check("the real server is SIGTERMed", killed == [4001], f"killed={killed}")

    print("  [2] a grep carrying the same adjacent tokens is NOT killed")
    print("      (this is the case the argv[0] check exists for)")
    killed = run_killer({
        4002: (OUR_UID, "grep -- -m mlx_vlm.server ."),
    })
    check("the grep survives", killed == [], f"killed={killed}")

    print("  [2b] the REALISTIC long-lived shape: a quoted pattern, unquoted by ps")
    print("      `grep -r -- \"-m mlx_vlm.server\" /tree` is ONE argv element;")
    print("      ps prints it unquoted, so cmd.split() sees adjacent tokens")
    killed = run_killer({
        4021: (OUR_UID, "grep -r -- -m mlx_vlm.server /Users/someone/code"),
    })
    check("the recursive grep survives", killed == [], f"killed={killed}")

    print("  [3] an editor whose file path merely mentions it is NOT killed")
    killed = run_killer({
        4003: (OUR_UID, "vim /tmp/notes-about-mlx_vlm.server.md"),
    })
    check("the editor survives", killed == [], f"killed={killed}")

    print("  [4] another user's real-looking server is NOT killed")
    killed = run_killer({
        4004: (OTHER_UID, f"{OUR_PY} -m mlx_vlm.server --model foo"),
    })
    check("another uid survives", killed == [], f"killed={killed}")

    print("  [5] an unrelated interpreter with the adjacent tokens is NOT killed")
    print("      (a different venv's python running a different -m target)")
    killed = run_killer({
        4005: (OUR_UID, "/usr/bin/python3 -m mlx_vlm.server --model foo"),
    })
    check("a foreign interpreter survives", killed == [], f"killed={killed}")

    print("  [6] our own tracked VLM is skipped before the predicate is reached")

    class FakeMM:
        backend = mgr.Backend.VLM

        class vlm_process:
            pid = 4006

    killed = run_killer(
        {4006: (OUR_UID, f"{OUR_PY} -m mlx_vlm.server --model foo")},
        models={"owned": FakeMM()},
    )
    check("an owned pid is left alone", killed == [], f"killed={killed}")

    print("  [7] a pid that exited between pgrep and ps is NOT killed")
    killed = run_killer({})
    check("a vanished pid is not signalled", killed == [], f"killed={killed}")

    print("  [8] mixed table: only the real one dies")
    killed = run_killer({
        4008: (OUR_UID, "grep -- -m mlx_vlm.server ."),
        4009: (OUR_UID, f"{OUR_PY} -m mlx_vlm.server --model foo --port 8801"),
        4010: (OUR_UID, "vim /tmp/mlx_vlm.server.log"),
    })
    check("exactly the server is killed", killed == [4009], f"killed={killed}")

    print("  [9] the signal sent is SIGTERM, not SIGKILL")
    sent = []

    def spy(pid, sig):
        sent.append(sig)

    saved_sub, saved_kill, saved_py = mgr.subprocess, os.kill, mgr.VLM_PYTHON
    try:
        mgr.subprocess = FakeProcTable(
            {4011: (OUR_UID, f"{OUR_PY} -m mlx_vlm.server --model foo")}
        )
        os.kill = spy
        mgr.VLM_PYTHON = OUR_PY
        Holder()._kill_stray_vlm()
    finally:
        mgr.subprocess, os.kill, mgr.VLM_PYTHON = saved_sub, saved_kill, saved_py
    check("SIGTERM", sent == [signal.SIGTERM], f"sent={sent}")

    print()
    if failed:
        print(f"  FAILED: {failed} failed, {passed} passed")
        sys.exit(1)
    print(f"  ALL PASSED: stray-VLM kill guard {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
