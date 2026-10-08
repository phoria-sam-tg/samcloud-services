#!/usr/bin/env python3
"""Does a restart change anything? Three named comparisons, and the residue (#905).

A restart moves the gateway from the commit its process imported to the commit
the deploy clone now holds. Whether that is a no-op or a deployment is the
question #905 exists for, and `git diff --stat` cannot answer it: on slice it
said "a runtime module changed, 76 lines in server.py" and every executable
statement was identical -- all 58 insertions were one docstring. That nearly
cost a restart of a healthy gateway (samclaude-services).

So this reports three things, each measured by the instrument that answers it,
and NAMES WHAT IT DID NOT COMPARE:

  executable drift   will behaviour change      AST compare, docstrings stripped
  published drift    will the served schema move  app.openapi() diff
  residue            changed and NOT compared   listed by name

THE LEFT OPERAND IS THE LOADED COMMIT, NOT `$PREV`. This is the one thing to
get right (samclaude-admin). `rollout.sh` holds `$PREV` -- the previously
STAGED commit -- and that equals what is running only when the process was
restarted after the previous staging, i.e. only when there is no drift. Drift
existing is the ticket's premise, so `$PREV` is the correct operand exactly
when these fields are not needed and the wrong one exactly when they are.

It happened twice in the window #905 was filed about. The deploy clone's
reflog, with the process running from Oct 5 on the Sep-30 tree `bf9fef4`:

    09:43:41   bf9fef4 -> ff20347    $PREV == running    (agreed)
    09:50:15   ff20347 -> 6691982    $PREV != running
    10:07:15   6691982 -> e621513    $PREV != running

Three stagings, no restart between them. At 09:50 and 10:07 a `$PREV`-based
report would have described two commits the running process was never party
to -- one of them the misbinding being rolled back.

So pass the loaded commit explicitly. It is the only operand no external tool
can derive, because only the process knows what it imported.

WHY A RESIDUE RATHER THAN A LIST OF SURFACES. `/service-docs` serves
`ollama/README.md`, read from disk at request time, which no AST compare can
see and which `app.openapi()` does not contain -- the schema describes that
route but not the file it returns (verified: the schema holds none of the
README's text). The first fix proposed for this was a third hardcoded line,
and that is an enumeration of known surfaces, which goes stale the moment
someone adds one. Deriving the changed set from git and listing whatever is
not classified means a new served file shows up as "changed, not compared"
without anyone remembering to extend anything (samclaude-admin).

Note `dirname(__file__)`: the served README is `ollama/README.md`, not the
repo-root `README.md`. Two files, one name, one of them a surface.

THIS REPORTS, IT DOES NOT REFUSE. Drift is information, not an alert
(samclaude-admin): a check that fails on `running != staged` would page on a
box where drift is the plan -- wafer right now has old working code running
and verified-good code staged, and its restart IS the deployment of #904.
The alerts live in `rollout.sh`'s gates: does the staged commit route (#41),
and are its failure frames visible (#48).

So exit 0 means "a report was produced", whatever it says. A non-zero exit
means the report could NOT be produced, which must be loud rather than read
as "no drift" -- absence is not equality, the same reason `foreign_mb` returns
None rather than 0 (#861).

    deploy/drift-report.py <loaded_commit> <staged_commit> [--python PATH]

    0  report produced
    4  a ref could not be resolved, or an instrument could not run
"""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile

RUNTIME_PKG = "ollama"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, check=True).stdout


def resolve(ref):
    """A ref, or exit 4. Never a silent fallback -- an unresolvable operand
    must not be reported as 'no difference'."""
    try:
        return git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        print(f"CANNOT RESOLVE '{ref}' -- no report produced.")
        print("  Not 'no drift'. An operand that cannot be read is not an")
        print("  operand that matches.")
        sys.exit(4)


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def executable_form(ref, path):
    """The file's executable statements, or None if absent at that ref."""
    try:
        src = git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None
    return ast.dump(strip_docstrings(ast.parse(src)))


def materialise(ref, dest):
    os.makedirs(dest, exist_ok=True)
    archive = subprocess.run(["git", "archive", ref], capture_output=True,
                             check=True).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=archive, check=True)


SCHEMA_PROBE = """
import json, os, sys
os.environ.setdefault("SC_TOKEN", "drift-report")
import logging; logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from ollama.server import app
json.dump(app.openapi(), sys.stdout, sort_keys=True)
"""


def schema_at(ref, python):
    """The schema FastAPI would serve, or None if it could not be generated.
    Generated by calling `app.openapi()` -- the same call the app serves, so
    it cannot be wrong about which handlers are routed (samclaude-services).
    """
    tmp = tempfile.mkdtemp(prefix="drift-")
    try:
        materialise(ref, tmp)
        # cwd=tmp is why `python` MUST be absolute -- see the caller. Any
        # failure here is reported, never raised: a half-printed report
        # followed by a traceback is worse than a field saying what it could
        # not check.
        try:
            r = subprocess.run([python, "-c", SCHEMA_PROBE], cwd=tmp,
                               capture_output=True, text=True)
        except Exception as e:
            return None, [f"{type(e).__name__}: {e}"]
        if r.returncode != 0:
            return None, (r.stderr or "").strip().splitlines()[-1:] or ["?"]
        return r.stdout, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("loaded", help="the commit the RUNNING process imported")
    p.add_argument("staged", help="the commit a restart would load")
    p.add_argument("--request-time", default=None, dest="request_time",
                   help="comma list (or @file) of repo paths the service "
                        "resolves from the clone PER REQUEST (not at import): "
                        "a staged change to one of these is ALREADY LIVE and a "
                        "restart deploys nothing. ollama/whisper_server.py and "
                        "ollama/README.md are both in this class (#905)")
    p.add_argument("--imported", default=None,
                   help="comma list (or @file) of ollama.* modules the running "
                        "process imported; from /status loaded_modules (#905)")
    p.add_argument("--python", default=None,
                   help="interpreter with the app's deps (for the schema diff)")
    a = p.parse_args()

    left, right = resolve(a.loaded), resolve(a.staged)
    print(f"loaded (running): {left[:8]}")
    print(f"staged (restart): {right[:8]}")
    if left == right:
        print("\nno drift — the running process imported the staged commit.")
        return 0

    changed = [f for f in git("diff", "--name-only", left, right).splitlines() if f]
    pys = [f for f in changed if f.endswith(".py")]

    # WHICH .py FILES THE PROCESS ACTUALLY IMPORTED IS NOT DERIVABLE HERE.
    #
    # The first version of this filtered `ollama/*.py` minus `test_*` and
    # called the rest "runtime modules". That is a hand-chosen heuristic
    # standing in for a process fact -- the error this whole tool exists to
    # report -- and it was wrong immediately, not eventually:
    #
    #   imported by the gateway   capacity config exo_client llama_client
    #                             manager ollama_client prompt_size samcloud
    #                             server whisper_client
    #   on disk, non-test, NOT    __init__  whisper_server
    #
    # `whisper_server.py` is never imported: it is exec'd as a SCRIPT under
    # WHISPER_PYTHON from `Path(__file__).resolve().parent`, i.e. out of the
    # deploy clone, fresh for every transcription (manager.py:81). So the
    # heuristic would have reported "a restart changes behaviour" for a file a
    # restart does not deploy.
    #
    # claude-containers hit the same thing with a hand-picked FRAME_MODULES
    # tuple that omitted `ollama_client.py`, and samclaude-services found a
    # changed `test_stream_deadlines.py` reported as an executable difference
    # that the process never imports. Three of us, three hand-chosen sets.
    #
    # So: pass the set in, from the process that knows it (`loaded_modules` on
    # /status, #905). Absent, this says UNKNOWN rather than guessing -- the
    # same rule as a missing left operand.
    def listarg(v):
        if not v:
            return None
        raw = open(v[1:]).read() if v.startswith("@") else v
        return {x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()}

    imported = listarg(a.imported)

    # READ-AT-REQUEST-TIME IS ONE CLASS, AND IT IS NOT A FILE TYPE.
    #
    # The first version of this put `ollama/README.md` in the residue and
    # `ollama/whisper_server.py` under "not imported" -- two sections for one
    # class, split on the extension. samclaude-services found the unification
    # and it is theirs:
    #
    #   artifact                  how                resolved from         when
    #   ollama/whisper_server.py  exec'd as a script Path(__file__).parent per transcription
    #   ollama/README.md          read as a file     dirname(__file__)     per /service-docs
    #
    # Both resolve from the deploy clone at REQUEST time, so a staged change
    # to either is already serving and a restart deploys nothing. Classifying
    # them apart because one is .py and one is .md is classifying by extension
    # when the property is WHEN THE CLONE IS READ.
    #
    # Like the import set, this cannot be derived here -- "which paths does
    # the service resolve per request" is a fact about the code, not about the
    # diff. So it is passed in, and absent, these files are reported without
    # the claim rather than with a guessed one.
    request_time = listarg(a.request_time)

    def module_name(path):
        if not (path.startswith(f"{RUNTIME_PKG}/") and path.endswith(".py")):
            return None
        return os.path.basename(path)[:-3]
    # The residue: derived, not enumerated. Anything git says changed that no
    # instrument below examines is listed by name rather than passed over.
    residue = [f for f in changed if f not in pys]

    print(f"\nchanged files: {len(changed)}  ({len(pys)} .py)")

    print("\nexecutable drift   (python AST, docstrings stripped)")
    if not pys:
        print("    none — no .py file changed")
    for f in sorted(pys):
        lf, rf = executable_form(left, f), executable_form(right, f)
        if lf is None or rf is None:
            verdict = "ADDED" if lf is None else "REMOVED"
        elif lf == rf:
            verdict = "identical as executed"
        else:
            verdict = "** DIFFERS AS EXECUTED **"
        mod = module_name(f)
        if request_time and f in request_time:
            scope = "read per REQUEST -> ALREADY LIVE; a restart deploys nothing"
        elif imported is None:
            scope = "import status UNKNOWN"
        elif mod in imported:
            scope = "imported -> a restart deploys it"
        elif request_time is None:
            # Do not call this harmless. whisper_server.py sits here and is
            # exec'd per request from the clone, so for it the true statement
            # is stronger than "a restart does not deploy it" -- it is already
            # serving. Without --request-time, say which it could be.
            scope = ("NOT imported -> a restart does not deploy it; if it is "
                     "read from the clone per request it is ALREADY LIVE")
        else:
            scope = "NOT imported, not read per request -> no effect"
        print(f"    {f:34s} {verdict}")
        print(f"    {'':34s}   {scope}")
    if imported is None:
        print("    import status unknown: pass --imported from /status's")
        print("    loaded_modules. Without it this cannot say whether a")
        print("    difference is one a restart would deploy.")

    print("\npublished drift    (app.openapi(), the call FastAPI serves)")
    # ABSOLUTE, AND CHECKED AS THE THING THAT WILL BE RUN.
    #
    # This read `os.path.join(os.environ.get("DEPLOY_DIR", ""), ...)`, so with
    # DEPLOY_DIR unset the path was the RELATIVE `.venv/bin/python`.
    # `os.path.exists()` then resolved it against the caller's cwd -- where a
    # working checkout does have a .venv -- so the guard passed, and
    # `schema_at` ran it with `cwd=<tempdir>` where it does not exist:
    # FileNotFoundError over a half-printed report.
    #
    # The check and the use resolved the same relative path against different
    # directories, which is tonight's shape inside the guard written to stop
    # a crash. Found by running it without --python rather than with.
    python = a.python or (
        os.path.join(os.environ["DEPLOY_DIR"], ".venv", "bin", "python")
        if os.environ.get("DEPLOY_DIR") else None)
    if python:
        python = os.path.abspath(python)
    if not pys:
        print("    none — no .py changed, so the schema cannot move")
    elif not (python and os.path.exists(python)):
        # NOT 'identical'. Say which instrument did not run.
        print(f"    NOT CHECKED — no interpreter with the app's deps")
        print(f"                  (tried {python or '<unset>'}; pass --python)")
        print("    This is the field that catches a routed handler's docstring")
        print("    changing, which the AST compare above deliberately strips.")
        return 4
    else:
        ls, lerr = schema_at(left, python)
        rs, rerr = schema_at(right, python)
        if ls is None or rs is None:
            print(f"    COULD NOT GENERATE — {(lerr or rerr)[0][:90]}")
            return 4
        print("    identical" if ls == rs else "    ** SCHEMA DIFFERS **")

    print("\nresidue            (changed, and NOT compared by anything above)")
    if not residue:
        print("    none")
    else:
        for f in sorted(residue):
            if request_time and f in request_time:
                note = "  <- read per REQUEST: ALREADY LIVE, restart deploys nothing"
            elif request_time is None:
                note = "  (request-time status unknown)"
            else:
                note = ""
            print(f"    {f}{note}")
        print("    Listed rather than classified on purpose: a surface this")
        print("    script does not know about appears here instead of being")
        print("    silently absent from a report that says 'identical'.")

    print("\nreport produced. Drift is information, not a fault — the alerts")
    print("are rollout.sh's gates (#41 routes, #48 frame visibility).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
