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

say "$DEPLOY_DIR is at $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (detached), venv synced"
say "$verified"
say "nothing restarted — that is restart-when-idle's step"
