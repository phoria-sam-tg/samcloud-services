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
  say "  $DEPLOY_DIR/.venv/bin/python -c 'import ollama.server'"
  exit 5
fi

say "$DEPLOY_DIR is at $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (detached), venv synced"
say "nothing restarted — that is restart-when-idle's step"
