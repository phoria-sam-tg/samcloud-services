#!/usr/bin/env bash
# rollout — put a named commit into the clone the gateway actually runs from.
#
# WHY THIS EXISTS
# The service used to run from `~/code/samcloud-services`, which is also the
# obvious place to work on it. Those are the same path and nothing prevented
# the collision, so three times in two days a live deploy checkout held
# somebody's in-progress work (#861):
#
#   2026-09-29  slice   #861's own branch, with capacity.py modified
#   2026-09-29  wafer   57 commits behind, with a hand-edited manager.py
#   2026-09-30  slice   #870's branch, 277 uncommitted lines
#
# None of those was carelessness. The gateway keeps serving either way — its
# modules are already loaded — so nothing reports it, and the damage only
# arrives on the next restart, which happens for reasons nobody chooses: a
# crash, launchd's KeepAlive, a deploy, `restart-when-idle`.
#
# THE COLLISION IS UNDETECTABLE FROM EITHER SIDE, which is the part that makes
# a convention insufficient (claude-wafer-services). The person working in the
# tree has no signal that a service runs from it; the service has no signal
# that its tree is being edited. wafer's instance ran for at least a week
# through several of its own restarts and was found by `lsof` on the
# gateway's pid while chasing an unrelated question — not by a check, and not
# by `git status`, which was perfectly happy.
#
# So this does not detect the failure, it removes it: a deploy that resets to a
# named commit CANNOT silently serve a working tree, because there is no
# working tree to serve. That is a stronger fix than reporting (#866 asks for
# the reporting, and still should have it) and it is why the path is separate
# rather than merely watched.
#
# So the deploy clone is not a working checkout and is not meant to be
# readable as one. It is left on a DETACHED HEAD at an explicit commit,
# because a detached HEAD is the cheapest way to tell a reader — human or
# otherwise — that this is not a place to do work. `reset --hard` means
# anything found here that is not in the commit is discarded without asking.
#
# The same move samcloud made with `wake-repo` for the same reason
# (`~/.local/samcloud/wake-repo`, rail-owned, hook-synced, never the place
# anyone edits).
#
#   deploy/rollout.sh 9d002c9              # what a cutover should use: a NAMED commit
#   deploy/rollout.sh                      # origin/main as of now — warns, for scratch use
#   deploy/rollout.sh 9d002c9              # a named commit
#   DEPLOY_DIR=/tmp/x deploy/rollout.sh    # somewhere else, for a dry run
#
# It does NOT restart anything. Restarting is `restart-when-idle`'s job and
# stays a separate, idle-gated step — see ada/README.md.
#
# It DOES refuse to leave a clone staged that cannot serve: after the venv
# syncs, the route bindings are verified and a failure restores the previous
# commit and exits 7 (#904). Skipped with a note on commits predating #40.
set -euo pipefail

DEPLOY_DIR=${DEPLOY_DIR:-$HOME/var/samcloud-services-deploy}
REMOTE=${DEPLOY_REMOTE:-https://github.com/phoria-sam-tg/samcloud-services.git}
TARGET=${1:-origin/main}

# Floor, in GiB, below which this refuses to do anything at all. Borrowed from
# the sibling on the same boxes: `exo-run.sh`'s `require_space` refuses to
# start below FLOOR_GIB and says why, and on 2026-09-30 it was firing on wafer
# — `com.samcloud.exo runs = 19`, four of them "under 5G free — refusing to
# start into the wall", while a staged macOS update drained the volume at
# ~2 MB/s behind APFS snapshots that `du` cannot see
# (claude-wafer-services, reviewing this PR).
#
# So one script on that box treats <5 GiB as "refuse and say so" and this one
# treated it as "proceed". The asymmetry is the defect rather than the byte
# count: the venv is only ~62 MB and is not what fills a disk.
#
# THE CHECK IS BEFORE EVERYTHING, not just before the venv, which is further
# than the review asked. A rollout that resets the code and then fails
# installing its dependencies leaves the clone at the new commit with the old
# or a partial environment — a state that is neither the version you left nor
# the one you asked for, and the gateway would start into it. Refusing the
# whole rollout leaves the box exactly where it was, which is always a
# recoverable place.
FLOOR_GIB=${ROLLOUT_FLOOR_GIB:-5}

say() { printf 'rollout: %s\n' "$*" >&2; }

free_gib() {
  # The DATA volume, via the deploy path itself — not `df /`, which on macOS
  # reports the sealed system snapshot and reads reassuringly while the volume
  # that matters is full (claude-wafer-services measured 13Gi vs 12Gi at 100%
  # on the same box).
  #
  # `df -g` is BSD-only. On Linux (ada) there is no -g flag: it prints
  # nothing, free_gib returns empty, and the caller's `[ -z "${free:-}" ]`
  # check reads that as "could not read free space" — REFUSING every rollout
  # unconditionally (exit 4), not a wrong number but a hard block on the one
  # Linux box (#861, claude-ada). `-Pk` is POSIX and correct on both; do the
  # KiB-to-GiB arithmetic here instead of trusting a platform-specific unit
  # flag.
  df -Pk "$1" 2>/dev/null | awk 'NR==2 {print int($4/1024/1024)}'
}

free=$(free_gib "$(dirname "$DEPLOY_DIR")")
if [ -z "${free:-}" ]; then
  say "REFUSING: could not read free space for $(dirname "$DEPLOY_DIR")"
  exit 4
fi
if [ "$free" -lt "$FLOOR_GIB" ]; then
  say "REFUSING: ${free} GiB free, floor is ${FLOOR_GIB} GiB."
  say "Nothing has been changed — the clone is still on whatever it was."
  say "A rollout that resets the code and then cannot install its dependencies"
  say "leaves a state that is neither the old version nor the new one, and the"
  say "gateway would start into it. Free space and re-run."
  exit 4
fi
say "preflight: ${free} GiB free, floor ${FLOOR_GIB}"

if [ ! -d "$DEPLOY_DIR/.git" ]; then
  say "no deploy clone at $DEPLOY_DIR — creating one"
  mkdir -p "$(dirname "$DEPLOY_DIR")"
  git clone --quiet "$REMOTE" "$DEPLOY_DIR"
fi

# Refuse to run against a checkout that has a branch and local work. That is
# somebody's tree, not a deploy clone, and `reset --hard` would eat it. The
# check is deliberately on BOTH conditions: a detached HEAD with stray files
# is a previous rollout that raced something, and is ours to clean; a branch
# with a clean tree is recoverable and worth naming rather than destroying.
branch=$(git -C "$DEPLOY_DIR" symbolic-ref --quiet --short HEAD || true)
dirty=$(git -C "$DEPLOY_DIR" status --porcelain)
if [ -n "$branch" ] && [ -n "$dirty" ]; then
  say "REFUSING: $DEPLOY_DIR is on branch '$branch' with uncommitted changes."
  say "That is a working checkout, not a deploy clone. Move the work to a"
  say "worktree and re-run, or point DEPLOY_DIR somewhere else."
  exit 3
fi
[ -n "$branch" ] && say "note: was on branch '$branch' (clean) — detaching"

git -C "$DEPLOY_DIR" fetch --quiet --prune origin

# Resolve BEFORE anything is touched, and fail with a code of our own. Left to
# `set -e`, a typo'd SHA exits 128 — git's raw code — so the caller gets a git
# error where it expected a rollout one, and has to know that 128 means "could
# not resolve" (claude-wafer-services, reviewing this PR).
if ! SHA=$(git -C "$DEPLOY_DIR" rev-parse --verify --quiet "${TARGET}^{commit}"); then
  say "REFUSING: cannot resolve '${TARGET}' to a commit."
  say "Nothing has been changed. Check the SHA, or fetch a branch that has it."
  exit 6
fi
PREV=$(git -C "$DEPLOY_DIR" rev-parse --verify HEAD 2>/dev/null || true)

# A no-argument run is convenient and is not what a cutover should use. The
# design is "reset to a NAMED commit"; `origin/main` is whatever it happens to
# be at this moment, so two no-argument runs a week apart deploy different code
# and both report success — and the command history stops answering "what is
# this box running?" (claude-wafer-services). Warn rather than refuse: the
# convenience is real for a scratch clone, the ambiguity only matters for the
# record.
if [ $# -eq 0 ]; then
  say "note: no commit named, so this is origin/main as of now ($(git -C "$DEPLOY_DIR" rev-parse --short "$SHA"))."
  say "      For a cutover, pass the SHA — it is what the audit trail rests on."
fi

# SAY WHAT IS BEING THROWN AWAY. `reset --hard` + `clean -fd` on a detached
# clone is correct by design — nobody works here — but discarding silently is
# how wafer's hand-edited `manager.py` would have vanished without anyone ever
# learning it had existed (samclaude-admin, reviewing this PR). The listing is
# cheap and it is the only record.
# `status --porcelain` alone: it already lists untracked entries as `??`,
# including directories, so adding `clean -nd` printed every stray file twice.
doomed=$(git -C "$DEPLOY_DIR" status --porcelain)
if [ -n "$doomed" ]; then
  say "discarding:"
  printf '%s\n' "$doomed" | sed 's/^/rollout:   /' >&2
fi

git -C "$DEPLOY_DIR" checkout --quiet --detach "$SHA"
git -C "$DEPLOY_DIR" reset --hard --quiet "$SHA"
git -C "$DEPLOY_DIR" clean -qfd

# The venv lives in the deploy clone, so the running gateway's dependencies
# move with its code rather than with whatever the working checkout last
# installed. `--upgrade` because requirements.txt is the record of what this
# commit needs — jinja2 arrived that way (#862) and a stale venv is how that
# defect stayed invisible on one box and not another.
if [ ! -x "$DEPLOY_DIR/.venv/bin/python" ]; then
  say "creating $DEPLOY_DIR/.venv"
  "${DEPLOY_PYTHON:-python3}" -m venv "$DEPLOY_DIR/.venv"
fi
if ! "$DEPLOY_DIR/.venv/bin/pip" install --quiet --upgrade -r "$DEPLOY_DIR/requirements.txt"; then
  # The code is already at the new commit and its dependencies are not. The
  # next restart would serve that. The disk floor makes this less likely and
  # does not close it — and wafer, which sits nearest the floor, is the box
  # where it is most reachable (claude-wafer-services).
  say "pip install FAILED at $SHA"
  if [ -n "${PREV:-}" ] && [ "$PREV" != "$SHA" ]; then
    git -C "$DEPLOY_DIR" checkout --quiet --detach "$PREV"
    git -C "$DEPLOY_DIR" reset --hard --quiet "$PREV"
    say "code restored to $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (where it was)"
  else
    say "code left at $SHA — there was no earlier commit to restore"
  fi
  say "THE VENV MAY STILL BE PARTIAL. Restoring the code cannot undo a"
  say "half-finished install, so verify before any restart:"
  # NOT `import ollama.server`, which is what this said until #904. A clone
  # that imports is not a clone that routes: on ff20347 the import succeeded
  # perfectly while every POST to /v1/chat/completions 422'd, because a helper
  # had landed under the route decorator. Recommending the weaker check here
  # was the worst place for it — this is the one path where the reader is
  # already in trouble and reaching for the script's own guidance
  # (claude-wafer-services, reviewing #40).
  say "  (cd $DEPLOY_DIR && .venv/bin/python -m ollama.test_route_bindings)"
  exit 5
fi

# GATE THE SURFACE, don't just document it. #904 reached main and this script
# staged it; the running gateway kept serving only because it had loaded the
# old module three days earlier. `deploy/README.md` now tells the operator to
# verify the bindings before restarting, and documentation is not a gate — the
# absence of this check IS the ticket, so writing down what to check and still
# leaving it to whoever remembers repeats the defect one turn later.
#
# A staged commit that cannot route is exactly as fatal as a staged commit
# whose dependencies failed to install, and until now only one of the two was
# gated. Same `$PREV` restore as the pip failure above, distinct exit code.
#
# Only the route bindings, not all three suites from the README: this gates the
# failure mode "the clone imports but does not serve", which is invisible until
# something restarts and is not caught by review. The deadline and num_ctx
# suites assert logic, where a failure is a bug to read rather than a reason to
# refuse a deploy, and they stay in the README as pre-restart checks.
#
# THE CHECK IS INLINE, NOT `-m ollama.test_route_bindings`, and that is the
# whole point. The first draft of this gate ran the suite from the deployed
# commit and skipped when it was absent — which meant it could not catch
# ff20347, the actual defect, because the suite arrived in #40 and the defect
# shipped in #39. Staging the broken commit printed "bindings NOT verified"
# and then "route bindings verified" on the next line, and exited 0. Found by
# running it against ff203471 instead of reasoning about it.
#
# A gate that depends on the deployed commit shipping its own test cannot
# verify any commit older than the test. This asserts the property directly,
# so it holds on every commit that has the route — including a rollback target
# predating #40, which is precisely the move #904 needed in a hurry (the
# deploy clone went back to 6691982 before anything was fixed forward).
#
# `setdefault`, not assignment: a box with a real SC_TOKEN keeps it. The value
# only has to let the module import — nothing here serves a request.
#
# The suite is still run when present, because it asserts four properties this
# cannot (the written-out table, coverage, double-binding, a live POST). Both
# must pass. Only this one is required.
if ! (cd "$DEPLOY_DIR" && .venv/bin/python - <<'ROUTECHECK' >/dev/null 2>&1
import os, sys
os.environ.setdefault("SC_TOKEN", "rollout-route-check")
from ollama.server import app
bound = {}
for r in app.routes:
    ep = getattr(r, "endpoint", None)
    if ep is not None:
        bound.setdefault(r.path, set()).add(ep.__name__)
# A helper landing under a decorator is the failure (#904). It shows up two
# ways: a private name serving a path, or the chat route not on its handler.
private = sorted(p for p, n in bound.items() if any(x.startswith("_") for x in n))
chat = bound.get("/v1/chat/completions", set())
sys.exit(1 if private or chat != {"chat_completions"} else 0)
ROUTECHECK
); then
  say "ROUTE BINDINGS FAILED at $SHA — the clone imports but does not route (#904)."
  if [ -n "${PREV:-}" ] && [ "$PREV" != "$SHA" ]; then
    git -C "$DEPLOY_DIR" checkout --quiet --detach "$PREV"
    git -C "$DEPLOY_DIR" reset --hard --quiet "$PREV"
    say "code restored to $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (where it was)"
  else
    say "code left at $SHA — there was no earlier commit to restore"
  fi
  # ONE LINE, pasteable. Split across three `say` calls it reads fine and
  # cannot be copied into a shell, which is the only thing the reader wants to
  # do with it (samclaude-admin, reviewing #41).
  say "DO NOT RESTART. See which path resolves where with:"
  say "  (cd $DEPLOY_DIR && .venv/bin/python -c 'from ollama.server import app; print({r.path: r.endpoint.__name__ for r in app.routes})')"
  exit 7
fi
verified="route bindings verified"
if [ -f "$DEPLOY_DIR/ollama/test_route_bindings.py" ]; then
  if ! (cd "$DEPLOY_DIR" && .venv/bin/python -m ollama.test_route_bindings >/dev/null 2>&1); then
    say "test_route_bindings FAILED at $SHA — the chat route binds, something else does not."
    if [ -n "${PREV:-}" ] && [ "$PREV" != "$SHA" ]; then
      git -C "$DEPLOY_DIR" checkout --quiet --detach "$PREV"
      git -C "$DEPLOY_DIR" reset --hard --quiet "$PREV"
      say "code restored to $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (where it was)"
    else
      say "code left at $SHA — there was no earlier commit to restore"
    fi
    say "DO NOT RESTART. Read the failure in full with:"
    say "  (cd $DEPLOY_DIR && .venv/bin/python -m ollama.test_route_bindings)"
    exit 7
  fi
  verified="$verified, test_route_bindings passed"
else
  # Say what was NOT checked. The first draft claimed "verified" on this path.
  verified="$verified (inline only — $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) predates ollama/test_route_bindings)"
fi

# A CLONE THAT ROUTES IS NOT A CLONE WHOSE FAILURES ARE VISIBLE (#904).
#
# The gate above proves a request reaches the handler. It says nothing about
# what the handler emits when a deadline fires, and that is a separate way to
# be broken: #904's whole second half was a gateway that aborted a generation
# and sent a terminal frame the caller read as a normal completion. Detection
# needs, PER FRAME, either
#
#   choices: []  plus top-level error_type/error_message        (consumed in
#                                                                the chunk loop)
#   non-empty delta.content  AND  a finish_reason a client
#                cannot read as success                          (the other
#                                                                detector)
#
# and the conjunction in the second is the part that bites. `main` carried a
# frame with `finish_reason: "error"` and `delta: {}` from #39 until #43 and it
# was silent the whole time: the right reason, no text, invisible. Found by
# claude-containers executing the frames against the real client; it had been
# sitting in a posted grep output unnoticed.
#
# SO THIS CANNOT BE A COUNT. The repo's suite asserts the two halves as
# separate totals, and two counts of three do not state that the three are the
# same three (claude-wafer-services). Nor can it be a grep: a comment quoting
# the old shape reads as the defect surviving, which this file has already been
# defeated by once. It is an AST walk that evaluates both detectors on each
# emitted frame and passes only if every frame is seen by at least one.
#
# THIS IS A PROXY, NOT THE AUTHORITY, and the distinction is load-bearing.
# claude-containers' gate EXECUTES the frames against Hermes in the container;
# that is the only check that can show a well-formed frame is actually
# DETECTED. This one runs where that cannot — on every box, with no network and
# no client — and it can only show a frame is MALFORMED. If the two disagree,
# the executable one is right.
#
# THAT IS NOT HYPOTHETICAL — IT HAPPENED TO THIS CHECK, within two hours of the
# sentence above being written, and in the direction the sentence predicted.
# The `d2` test below was `reason not in SUCCESSY`, a blocklist of success
# values. claude-containers drove the sentinel through Hermes' own function and
# found FOUR values — "timeout", "stalled", "aborted", "cancelled" — that this
# proxy passed and Hermes' second detector ignored. samclaude-admin found it by
# reading the citation against the code twelve lines apart.
#
# Neither the citation nor a check that the citation is PRESENT would have
# caught it: a citation that is not asserted against the implementation beneath
# it is the same class as a comment explaining code it contradicts
# (samclaude-admin). What caught it was the authority, which holds no copy of
# the set because it calls Hermes directly. So read the hierarchy above as a
# fact about this file rather than as a disclaimer.
#
# AND IT WORKS BY REIMPLEMENTING SOMEBODY ELSE'S GUARDS, which is the risk in
# it (samclaude-admin). The two conditions below are a COPY of Hermes' logic as
# read on 2026-10-08, not a property of our own code:
#
#   chat_completion_helpers.py:3219  _choiceless_chunk
#       reached when `not chunk.choices`; raises from the top-level
#       error_type / error_message pair. Needs NO text.
#   chat_completion_helpers.py:338   _provider_stream_error_from_text
#       reached via :3428; needs NON-EMPTY text AND finish_reason in the set
#       at :60 — which is HERMES_ERROR_REASONS below, NOT restated here.
#       Two copies of a set is two things to drift; the constant is the only
#       copy and the citation says where it came from (samclaude-admin asked
#       for the constant and an assertion that it matches the comment; one
#       copy removes the disagreement rather than detecting it).
#
# So a Hermes upgrade or a vendored-client bump can move them, and this check
# would keep passing while asserting conditions that no longer exist — a green
# check naming a cause it no longer measures, which is the exact defect class
# #904 is about. Treat divergence as expected drift: when Hermes moves, re-read
# those two functions and update the citation above with the new date. The
# executable gate is what notices; this is what runs in between.
#
# SKIPPED WITH A NOTE when the commit predates `_FAILURE_FINISH_REASON` (#43) —
# every commit before it either has no failure frames at all or has them in the
# known-silent shape, so refusing would block a rollback to any of them. That
# is #41's lesson and I have walked into it once already: a gate that cannot be
# rolled back past is worse than the hole it closes.
#
# THIS DELIBERATELY DISAGREES WITH claude-containers' GATE, AND BOTH ARE RIGHT.
# On a pre-#43 ref — `ebe28b5e`, say — theirs reports ALL SILENT and exits 1;
# this one skips with a note and exits 0. Measured, not assumed.
#
# The gates answer different questions. Theirs is a go/no-go on handing the
# Hermes seat's reporting to the gateway, so a ref whose frames are silent is
# exactly what it must refuse. This one decides whether a deploy may be left
# staged, and the operator reaching for a pre-#43 commit is usually rolling
# back from something worse — refusing would be the #41 failure again.
#
# So do not "harmonise" them. Aligning the two would make one of them wrong,
# and which one depends on a question neither gate asks. Same inputs, different
# verdicts, because they are bounding different things — which is this evening's
# whole lesson, applied to the gates rather than to the numbers.
if ! grep -q '_FAILURE_FINISH_REASON' "$DEPLOY_DIR/ollama/server.py" 2>/dev/null; then
  verified="$verified; failure frames NOT checked ($(git -C "$DEPLOY_DIR" rev-parse --short HEAD) predates #43)"
else
  # `frames_out=$(...)` takes the substitution's exit status as its own, so
  # under `set -e` a FAILING check kills the script before `$?` can be read:
  # the gate would die at exit 1 with the bad commit still staged and no
  # restore — the precise outcome it exists to prevent. Caught by running the
  # refusal path rather than only the passing one. `|| frames_rc=$?` puts the
  # assignment in a list, which `set -e` does not act on.
  frames_rc=0
  frames_out=$(cd "$DEPLOY_DIR" && .venv/bin/python - <<'FRAMECHECK' 2>&1
import ast, sys

# The date the two Hermes guards below were read, cited in the comment above.
# It is PRINTED on every rollout rather than only written here: a provenance
# note that nothing reads can rot to a stale comment silently, and this file
# has no test harness to assert it (samclaude-services guards theirs with two
# checks in test_stream_deadlines). Putting it in the operator's output makes
# staleness visible where someone is acting on it.
GUARDS_READ = "2026-10-08"

src = open("ollama/server.py").read()
tree = ast.parse(src)

# The sentinel's VALUE from the source, never assumed — and checked against
# the client's set below, not against a notion of what "looks like" success.
# Rebinding it to "timeout" is the case that matters and the one the first
# version of this check passed.
reason_name, reason_val = "_FAILURE_FINISH_REASON", None
for n in ast.walk(tree):
    if isinstance(n, ast.Assign) and any(
            getattr(t, "id", "") == reason_name for t in n.targets):
        try:
            reason_val = ast.literal_eval(n.value)
        except Exception:
            pass

# THE CLIENT'S SET, and `in` it — not `not in` a list of success values.
#
# This was `reason not in SUCCESSY` (a success-value blocklist) and that was
# WRONG, found by samclaude-admin and measured through Hermes by
# claude-containers. The two tests diverge on everything outside both sets:
#
#   sentinel   Hermes detector 2   old check   new check
#   "error"    detects             passes      passes
#   "timeout"  SILENT              passes      refuses
#   "stalled"  SILENT              passes      refuses
#   "aborted"  SILENT              passes      refuses
#   "stop"     SILENT              refuses     refuses
#
# `"timeout"` is not hypothetical: it is the literal value main shipped in #39
# and the reason this ticket exists. The old check would have put
# `2 failure frames detectable` in the rollout output for it.
#
# Why it was wrong is worth keeping, because the code was not careless — it was
# a faithful implementation of a rule that had been superseded hours earlier:
#
#   v1  never stamp a finish_reason on a failure        (too strong)
#   v2  never stamp one a client could read as SUCCESS  (necessary, insufficient)
#   v3  a reason the client's set contains, AND text    (measured; this)
#
# `not in SUCCESSY` is v2. v2 was disproved by main's own third frame, which
# carried "error" — not success-y — with an empty delta and was silent for
# weeks. The rule moved and the code did not.
#
# The gate's question is "will a caller see this failure", and only the
# client's own condition answers that. A superset cannot.
HERMES_ERROR_REASONS = {"error", "error_finish"}   # chat_completion_helpers.py:60

frames = []
for n in ast.walk(tree):
    if not isinstance(n, ast.Dict):
        continue
    keys = [k.value for k in n.keys if isinstance(k, ast.Constant)]
    if "choices" not in keys:
        continue
    # An error key is what marks this as a FAILURE frame rather than a
    # success one. The success path legitimately carries `delta: {}`.
    if not ({"error", "error_type", "error_message"} & set(keys)):
        continue
    ch = n.values[keys.index("choices")]
    flat = "error_type" in keys and "error_message" in keys
    choiceless = isinstance(ch, ast.List) and not ch.elts

    has_text = False
    reason = "<none>"
    if isinstance(ch, ast.List) and ch.elts and isinstance(ch.elts[0], ast.Dict):
        d = ch.elts[0]
        dk = [k.value for k in d.keys if isinstance(k, ast.Constant)]
        if "delta" in dk:
            dv = d.values[dk.index("delta")]
            if isinstance(dv, ast.Dict):
                dvk = [k.value for k in dv.keys if isinstance(k, ast.Constant)]
                if "content" in dvk:
                    cv = dv.values[dvk.index("content")]
                    # A literal empty string is not text. An f-string or a
                    # name is, and `f"{type(e).__name__}: {e}"` cannot be
                    # empty even for a bare TimeoutError.
                    has_text = not (isinstance(cv, ast.Constant)
                                    and not cv.value)
        if "finish_reason" in dk:
            fv = d.values[dk.index("finish_reason")]
            if isinstance(fv, ast.Name):
                reason = reason_val if fv.id == reason_name else f"<{fv.id}>"
            elif isinstance(fv, ast.Constant):
                reason = fv.value

    d1 = choiceless and flat
    d2 = has_text and reason in HERMES_ERROR_REASONS
    frames.append((n.lineno, choiceless, flat, has_text, reason, d1, d2))

# ZERO FRAMES IS A FAILURE, NOT A PASS. If the emitter is restructured so this
# walk stops finding its subject, that must be loud — a check that cannot see
# what it checks has to say so rather than report health (claude-containers).
if not frames:
    print("found NO failure frames to check — the extractor is blind, "
          "not the code clean")
    sys.exit(2)

bad = []
for ln, cl, fl, tx, rs, d1, d2 in frames:
    if not (d1 or d2):
        bad.append(f"line {ln}: choiceless={cl} flat_pair={fl} "
                   f"has_text={tx} finish_reason={rs!r} -> SILENT")
if bad:
    print(f"{len(bad)} of {len(frames)} failure frames are invisible to a caller:")
    for b in bad:
        print("  " + b)
    sys.exit(1)

print(f"{len(frames)} failure frames detectable "
      f"(Hermes guards as read {GUARDS_READ})")
sys.exit(0)
FRAMECHECK
) || frames_rc=$?
  if [ "$frames_rc" -ne 0 ]; then
    if [ "$frames_rc" -eq 2 ]; then
      say "FAILURE-FRAME CHECK COULD NOT RUN at $SHA — $frames_out"
    else
      say "FAILURE FRAMES NOT DETECTABLE at $SHA (#904):"
      printf '%s\n' "$frames_out" | sed 's/^/rollout:   /' >&2
    fi
    if [ -n "${PREV:-}" ] && [ "$PREV" != "$SHA" ]; then
      git -C "$DEPLOY_DIR" checkout --quiet --detach "$PREV"
      git -C "$DEPLOY_DIR" reset --hard --quiet "$PREV"
      say "code restored to $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (where it was)"
    else
      say "code left at $SHA — there was no earlier commit to restore"
    fi
    # SCOPE, stated precisely (claude-containers). Since #46 the emitter sends
    # the choiceless frame FIRST, and its detection does not depend on the
    # sentinel — so one silent frame does not mean Hermes is blind today. What
    # it means is that the frame is invisible to any consumer that does not
    # read choiceless chunks, which is the whole reason the second frame
    # exists. Saying "the caller would see a completed answer" would be the
    # thread's own error: a claim about one frame stated as a claim about the
    # client.
    say "DO NOT RESTART. A frame listed above is invisible to any consumer that"
    say "does not read choiceless chunks — which is what that frame is for."
    say "Re-check with:"
    say "  (cd $DEPLOY_DIR && .venv/bin/python -m ollama.test_stream_deadlines)"
    exit 8
  fi
  verified="$verified; $frames_out"
fi

say "$DEPLOY_DIR is at $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (detached), venv synced"
say "$verified"
say "nothing restarted — that is restart-when-idle's step"
