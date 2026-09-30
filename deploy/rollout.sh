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
#   deploy/rollout.sh                      # origin/main into the default path
#   deploy/rollout.sh 9d002c9              # a named commit
#   DEPLOY_DIR=/tmp/x deploy/rollout.sh    # somewhere else, for a dry run
#
# It does NOT restart anything. Restarting is `restart-when-idle`'s job and
# stays a separate, idle-gated step — see ada/README.md.
set -euo pipefail

DEPLOY_DIR=${DEPLOY_DIR:-$HOME/var/samcloud-services-deploy}
REMOTE=${DEPLOY_REMOTE:-https://github.com/phoria-sam-tg/samcloud-services.git}
TARGET=${1:-origin/main}

say() { printf 'rollout: %s\n' "$*" >&2; }

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
SHA=$(git -C "$DEPLOY_DIR" rev-parse --verify "${TARGET}^{commit}")
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
"$DEPLOY_DIR/.venv/bin/pip" install --quiet --upgrade -r "$DEPLOY_DIR/requirements.txt"

say "$DEPLOY_DIR is at $(git -C "$DEPLOY_DIR" rev-parse --short HEAD) (detached), venv synced"
say "nothing restarted — that is restart-when-idle's step"
