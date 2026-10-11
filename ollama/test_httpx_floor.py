"""The runner states the httpx version it needs, instead of failing opaquely (#801).

On 2026-10-12 `test_exo_lease` was run under httpx 0.23.3 (python 3.10, no venv,
`-I` against the interpreter's site-packages) and produced **three assertion
failures** in its streaming cases. Those cases feed `data: …` frames through
`chat_collect`/`chat_stream`, so what they assert is partly how httpx splits an
SSE body into lines — behaviour that changed between 0.23 and 0.24.

The failures read as "main is broken" rather than "this environment is below the
floor", and two of us held different beliefs about whether main was green for
twenty minutes. `requirements.txt` already said `httpx>=0.24.0`; the floor was
not a new fact, it was simply not checked where it mattered.

Run from the repo ROOT: python -m ollama.test_httpx_floor
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama.test_exo_lease import (_httpx_floor_from_requirements,   # noqa: E402
                                   _version_tuple)

PASS = FAIL = 0
ROOT = Path(__file__).resolve().parent.parent


def check(desc, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[0;32mok\033[0m   {desc}")
    else:
        FAIL += 1
        print(f"  \033[0;31mFAIL\033[0m {desc}")
        if detail:
            print(f"       {detail}")


def main():
    print("\n  the runner states its httpx floor\n")

    # --- the floor is READ, not duplicated ------------------------------
    floor = _httpx_floor_from_requirements()
    check("the floor is read from requirements.txt", floor is not None, repr(floor))
    declared = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
                if l.strip().lower().startswith("httpx")]
    check("...and matches what requirements.txt actually declares",
          declared and floor in declared[0],
          f"floor={floor!r} line={declared[:1]}")
    # The property is that the LOGIC reads the floor, not that the number never
    # appears in prose: the docstring citing `httpx>=0.24.0` is documentation and
    # should say it. An earlier version of this check grepped the whole file and
    # failed on that docstring — the test was imprecise, not the code.
    import inspect
    from ollama import test_exo_lease as tel
    logic = (inspect.getsource(tel._httpx_floor_from_requirements)
             + inspect.getsource(tel._check_httpx_floor)
             + inspect.getsource(tel._version_tuple))
    check("...and no version literal is hardcoded in the checking code",
          floor not in logic,
          "a hardcoded copy is how the check and the install drift apart")
    check("...with requirements.txt named as where it comes from",
          "requirements.txt" in logic)

    # --- version comparison, including the pair that caused this --------
    check("0.23.3 < 0.24.0", _version_tuple("0.23.3") < _version_tuple("0.24.0"))
    check("0.28.1 >= 0.24.0", _version_tuple("0.28.1") >= _version_tuple("0.24.0"))
    check("equal is not below", not (_version_tuple("0.24.0") < _version_tuple("0.24.0")))
    check("two-part versions compare", _version_tuple("0.24") == (0, 24))
    check("a pre-release suffix does not crash the parse",
          _version_tuple("0.24.0rc1") == (0, 24, 0), str(_version_tuple("0.24.0rc1")))
    check("garbage parses to something comparable rather than raising",
          isinstance(_version_tuple("not-a-version"), tuple))

    # --- the behaviour: refuse below, run above -------------------------
    # A subprocess, because the preflight calls sys.exit at import.
    prog = (
        "import sys, httpx, runpy\n"
        "httpx.__version__ = %r\n"
        "sys.argv = ['test_exo_lease.py']\n"
        "sys.path.insert(0, '.')\n"
        "try:\n"
        "    runpy.run_path('ollama/test_exo_lease.py', run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    print('EXIT:', e.code)\n"
    )
    r = subprocess.run([sys.executable, "-c", prog % "0.23.3"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    check("below the floor it REFUSES rather than running the cases",
          "REFUSING TO RUN" in out, out[-400:])
    check("...and exits non-zero, so a wrapper cannot read it as a pass",
          "EXIT: 3" in out,
          "a skip printing `passed` would be the answer-shaped non-answer this "
          "repo keeps paying for: " + out[-300:])
    check("...and names the version it found and the floor it wanted",
          "0.23.3" in out and "0.24.0" in out, out[-300:])
    check("...and says what to do about it",
          "venv" in out or "requirements" in out, out[-300:])
    check("...and does NOT print a passing summary",
          "all pool lease tests passed" not in out,
          "refusing must not also claim success")

    print(f"\n  {'all checks passed' if not FAIL else 'FAILED'}: "
          f"{PASS} passed, {FAIL} failed\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
