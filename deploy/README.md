# Deploying the gateway

**The service does not run from your working checkout.** It runs from a deploy
clone that only `deploy/rollout.sh` touches, left on a detached HEAD at an
explicit commit.

```sh
deploy/rollout.sh                 # origin/main -> ~/var/samcloud-services-deploy
deploy/rollout.sh 9d002c9         # a named commit
```

Then, separately and idle-gated:

```sh
restart-when-idle.sh --check gateway --resource <this box's resource> \
  --action '<the box's restart>'
```

`rollout.sh` never restarts anything. Putting code in place and interrupting
service are different decisions with different risks, and the second one has a
script of its own for a reason (#823, #827).

## Why the separate clone

Three times in two days a live deploy checkout held somebody's in-progress
work (#861):

| | | |
|---|---|---|
| 2026-09-29 | slice | #861's own branch, `capacity.py` modified |
| 2026-09-29 | wafer | 57 commits behind, hand-edited `manager.py` |
| 2026-09-30 | slice | #870's branch, 277 uncommitted lines |

None was carelessness. `~/code/samcloud-services` was both the path the
LaunchDaemon ran from and the obvious place to work, and nothing prevented the
collision.

**It is undetectable from either side.** The person working in the tree has no
signal that a service runs from it; the service has no signal that its tree is
being edited. wafer's instance ran for at least a week through several of its
own restarts and was found by `lsof` on the gateway's pid during an unrelated
investigation — not by a check, and not by `git status`, which was content.

A deploy that resets to a named commit **cannot** silently serve a working
tree, because there is no working tree to serve. That removes the failure
rather than detecting it.

## What the script refuses to do

```
free space < ROLLOUT_FLOOR_GIB (5)   REFUSES, exit 4, before anything
on a branch AND dirty                REFUSES, exit 3, touches nothing
on a branch AND clean                detaches, with a note
detached                             resets, as intended
```

The disk check runs **before** the reset, not just before the venv. A rollout
that updates the code and then cannot install its dependencies leaves the
clone at the new commit with a partial environment — neither the version you
left nor the one you asked for — and the gateway would start into it.

`free_gib` reads the **data volume via the deploy path**, not `df /`: on macOS
that reports the sealed system snapshot and reads reassuringly while the
volume that matters is full (13Gi vs 12Gi at 100% capacity on the same box).

The floor matches `exo-run.sh`'s, which refuses below 5 GiB on these same
boxes and was firing on wafer while this PR was in review.



It `reset --hard`s and `clean -fd`s, so anything in the deploy clone that is
not in the commit is discarded. That is safe only because nobody works there —
so it checks:

A detached HEAD is deliberate and is the signal: if you find yourself on one,
you are in the deploy clone and should not be editing.

## The venv moves with the code

`rollout.sh` creates `.venv` inside the deploy clone and installs
`requirements.txt` into it on every run. So the running gateway's dependencies
travel with its commit rather than with whatever the working checkout last
installed.

**This has a consequence for testing.** Your working checkout's venv is no
longer the one production uses. A suite that passes in your checkout can fail
in the deploy clone and vice versa — that is exactly #862, where `jinja2` was
undeclared and one box's venv happened to have it. Verify against the deploy
clone's interpreter before restarting:

```sh
cd ~/var/samcloud-services-deploy && .venv/bin/python -m ollama.test_renew_ordering
```

## Moving work out of a deploy path — the order matters

If you find your own work in a path a service runs from, the sequence is:

```sh
git -C <deploy path> add -A && git commit -m "wip: #NNN"
git -C <deploy path> push -u origin <branch>          # off this laptop's disk
git -C <deploy path> checkout main                     # FREE THE BRANCH FIRST
git -C <deploy path> worktree add ~/code/<repo>-wt/<branch> <branch>
```

**`worktree add` before `checkout main` cannot work**, and the failure does not
say so:

```
fatal: 'samclaude-services/ring-short-offering' is already checked out at
       ~/code/samcloud-services
```

A branch cannot be checked out in two trees. That message reads like a path or
a permissions problem, and it arrives at the moment someone is already being
asked to get out of the way — so it costs twenty minutes at the worst time
(#861, the #870 holder hit it exactly).

The **push** matters independently of the worktree. Uncommitted work in a
directory a restart can surprise exists in one place only; a pushed branch
survives the box.

## This is the pattern, not a fix for one box

Apply it anywhere the gateway could run, not only where it has already bitten.
`claude-wafer-services` has **four** `samcloud-services` directories and none
is a deploy path today — so if the service is ever installed there, "the
obvious place to work" is ambiguous four ways before anyone starts, and the
collision is available without anyone doing anything unusual.

## Cutting a box over

**Merging this changes nothing about what runs.** `rollout.sh` fills the deploy
clone; each box's launcher still `cd`s into its working checkout until you
change it. The cutover is per box and in this order: **slice, then wafer, then
ada — and ada only in `claude-ada`'s quiet window, never during a render.**

The env file stays where it is (`~/.config/samcloud-services/env`, outside the
clone), so no identity or token changes.

**1. Fill the clone.**

```sh
deploy/rollout.sh <the commit this box should run>
```

Expect `exit 4` sometimes and **retry rather than debug it.** wafer's free
space oscillates across the 5 GiB floor — 4, 14, 5.70, 6.43, 6.78 GiB within
one hour — so a refusal there after a clean run on slice is the guard working,
not the script breaking (`claude-wafer-services`).

**2. Point the launcher at it.** This is **not one line, and not the same file
on each box.** Find the `cd` *and the variable that feeds it*:

| box | launcher | what to change |
|---|---|---|
| slice | `~/.local/bin/cs-model-service-run.sh` | `cd "$HOME/code/samcloud-services"` |
| wafer | `~/.local/bin/start-model-service.sh` | `REPO=…` (:6) **and** `cd "$REPO"` (:25) — two lines |
| ada | systemd unit | `ExecStart` → the deploy clone's `ada/run.sh`; the script itself needs no edit, it derives the repo from its own location |

**3. Restart through the gate**, never by hand:

```sh
restart-when-idle.sh --check gateway --resource <this box's resource> \
  --action '<the box's restart>'
```

**4. Verify the cwd — this is the step that proves the cutover, not step 2.**

```sh
lsof -p <gateway pid> | grep cwd
```

It has to name the deploy clone. This is not a formality: wafer has **four**
`samcloud-services` paths (`~/.config/…` env, `~/code/…` the working checkout,
`~/var/…` logs, `~/work/845/…` an old task checkout), so *"the deploy clone"*
and *"a samcloud-services directory"* are different claims and only `lsof`
separates them. It is also how wafer's 57-commit drift was found in the first
place.

**5. Then delete the Run/Test warning** in `CLAUDE.md` for that box — see the
note there. A warning that has stopped being true is worse than none.

## Per-box setup afterwards

Nothing. The working checkout goes back to being only a working checkout, and
`rollout.sh` is the only thing that writes to the deploy clone.

## What this does not fix

Nothing here reports a box running something other than what it claims (#866).
It makes one class of that impossible; the general ask stands.
