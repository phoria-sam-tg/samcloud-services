# Deploying the gateway

**The service does not run from your working checkout.** It runs from a deploy
clone that only `deploy/rollout.sh` touches, left on a detached HEAD at an
explicit commit.

```sh
cd ~/code/samcloud-services && git pull    # the WORKING CHECKOUT, pulled first
deploy/rollout.sh 9d002c9                  # a named commit
```

**Always from the working checkout, never from the deploy clone.** The clone
contains a `deploy/` directory too — and it is a snapshot of whatever commit
it serves, so its `rollout.sh` and this README are exactly as old as the code
it is running. Running the clone's copy gets you the tooling from the version
you are replacing: on 2026-09-30 that meant a script without the named exit
code for an unresolvable commit, beside a README whose verification step
aborts a `set -e` script on the passing case — the two fixes that existed
*because* the next box was about to run them.

A snapshot's tooling is a snapshot (`claude-wafer-services`). Version-bumping
cannot fix it; only running from the tree that tracks a branch can.

It also resets the clone out from under the script it is executing. Tried
once and it completed — bash had buffered enough of the file — but it
re-reads from a byte offset, so a script that changes size under itself can
execute garbage. Not a property to rely on.

**Then verify the clone, from inside the clone, before you restart anything.**
Not your checkout — the thing that is about to be served:

```sh
cd ~/var/samcloud-services-deploy
.venv/bin/python -m ollama.test_route_bindings    # every path -> its own handler
.venv/bin/python -m ollama.test_num_ctx
.venv/bin/python -m ollama.test_stream_deadlines
```

`test_route_bindings` is first because of #904, where a helper inserted between
`@app.post("/v1/chat/completions")` and its handler bound the **helper**. The
route count was unchanged, 51 other checks passed, `main` merged it and
`rollout.sh` staged it — and every POST to that path would have 422'd on the
next restart. The running process was serving a module it had loaded three days
earlier, which is the only thing that hid it. **A clone that imports is not a
clone that routes**, and the difference is invisible until something restarts.

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
deploy/rollout.sh 9d002c9        # name the SHA — see below
```

**Name the commit.** `deploy/rollout.sh` with no argument takes `origin/main`
as of that moment, which is convenient for a scratch clone and wrong for a
cutover: two no-argument runs a week apart deploy different code and both
report success, so the command history stops answering *"what is this box
running?"*. The script warns when you omit it.

Exit codes, so a refusal is not mistaken for a broken script:

```
0    done
3    the clone is a working checkout (branch + uncommitted changes)
4    below the disk floor — nothing touched
5    pip install failed; code rolled back, venv may be partial
6    the commit could not be resolved — nothing touched
```

Expect `exit 4` sometimes and **retry rather than debug it.** wafer's free
space oscillates across the 5 GiB floor — 4, 14, 5.70, 6.43, 6.78 GiB within
one hour — so a refusal there after a clean run on slice is the guard working,
not the script breaking (`claude-wafer-services`).

**2. Point the launcher at it.** This is **not one line, and not the same file
on each box.** Find the `cd` *and the variable that feeds it*:

| box | launcher | what to change |
|---|---|---|
| slice | `~/.local/bin/cs-model-service-run.sh` | the `cd`, which names the path directly |
| wafer | `~/.local/bin/start-model-service.sh` | the `REPO=` assignment; the `cd "$REPO"` below it then follows — **change the variable, check the `cd` uses it, do not edit both** |
| ada | systemd unit | `ExecStart` → the deploy clone's `ada/run.sh`; the script itself needs no edit, it derives the repo from its own location |

**No line numbers here on purpose.** An earlier draft cited wafer's as `:6`
and `:25`; adding the explanatory comment moved them to `:11` and `:30` before
anyone else read it. A coordinate into a file someone is about to edit is
stale by the time it is used — describe the thing, not where it sat.

**3. Restart through the gate**, never by hand:

```sh
restart-when-idle.sh --check gateway --resource <this box's resource> \
  --action '<the box's restart>'
```

**4. Verify — this is the step that proves the cutover, not step 2.** Three
checks, and the third is the one people skip:

```sh
PID=<gateway pid>
lsof -p $PID | grep cwd                                    # 1. runs there
lsof -p $PID | grep "$HOME/var/samcloud-services-deploy"   # 2. loads from there
! lsof -p $PID | grep -q "$HOME/code/samcloud-services"    # 3. and nowhere else
```

**Check 3 is written with `!` and `-q` on purpose.** The obvious form,
`grep -c …`, prints `0` and **exits 1** when it matches nothing — so under
`set -e` the *passing* case is the one that aborts the script, and by exit
code alone a clean cutover is indistinguishable from a broken command. Read
by hand it is fine; wrapped in anything it inverts. The `!` form exits 0 on
success and 1 when the old path really is still open, which is the way round
a caller expects.

**Name the full path, never just `samcloud-services`.** On a box with several
such directories the bare word also matches the log directory
(`~/var/samcloud-services/logs/server.log`, open as fd 1 and 2), which is
correctly neither checkout — so `grep -c samcloud-services` reads 4 and tells
you nothing.

**Check 2 needs an open file under `.venv`, not the cwd.** A launcher can
`cd` into the deploy clone and still `exec` an interpreter from somewhere
else, and check 1 would pass. What proves the venv is a loaded extension:

```
…/samcloud-services-deploy/.venv/lib/python3.14/site-packages/pydantic_core/….so
```

**`ps -o command=` cannot do this.** It shows the base interpreter
(`/opt/homebrew/.../Python.app/.../Python`) rather than `.venv/bin/python`,
because a venv's python resolves to the interpreter it was built from. Read
as proof of the venv it is misleading.

This is not a formality: wafer has four `samcloud-services` paths — env,
working checkout, logs, and an old task checkout — so *"the deploy clone"* and
*"a samcloud-services directory"* are different claims and only the full-path
form separates them. It is also how wafer's 57-commit drift was found.

**5. Leave the `CLAUDE.md` warning alone until the LAST box is over.** The
condition is per machine and `CLAUDE.md` is one shared file, so it cannot be
deleted per box — removing it when the first box cuts over would take away a
warning that is still true for the others. Its own wording is per-box on
purpose (*"until THIS BOX's launcher…"*), so it stays correct throughout; it
is only the deletion that waits.

## Per-box setup afterwards

Nothing. The working checkout goes back to being only a working checkout, and
`rollout.sh` is the only thing that writes to the deploy clone.

## What this does not fix

Nothing here reports a box running something other than what it claims (#866).
It makes one class of that impossible; the general ask stands.
