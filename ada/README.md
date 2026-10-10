# ada-wsl — the CUDA node

`ada-wsl` (RTX 6000 Ada, 48 GB, Ollama under WSL2) runs the same `main` as
slice and wafer. What is different is not the code but the situation: **ada is
somebody's work machine.** Windows-side renders on the `Phoria - RN02` account
hold VRAM this gateway never allocated and holds no lease on, and their work
takes precedence over ours.

## The profile

`ada/env` is the whole of it — an `EnvironmentFile` for the systemd unit, or
sourced by `ada/run.sh`. Every value there is either an identity or a number
measured on this box, and each carries the measurement in a comment. Two
things to know before changing any of them:

- **There is no `SC_TOKEN` in the environment, deliberately.** `SC_TOKEN_FILE`
  names a 0600 file and the process reads it itself, so the bearer is not
  inherited by anything the gateway spawns (#845).
- **`FOREIGN_IDLE_MB` being set is what turns the work gate on.** slice and
  wafer leave it unset and are unaffected by any of this.

## Deploying

`qwen3-coder:30b` has a candidate `num_ctx` pin of 65,536 in `ada/env`.
Read `/api/ps` after load to verify the served window; native model metadata
is not an allocation measurement. Hermes must budget against that same window.
Before inference, require `work_in_progress: false` and cross-check hardware
memory. Sample memory during generation, not only after loading. The pin does
not certify maximum-window fit or permit co-residency with a render.

The gateway runs from the **deploy clone**, not from a working checkout — see
`deploy/README.md` for why and for ada's one-time setup. The rollout is two
steps and they are deliberately separate:

```sh
deploy/rollout.sh <commit>        # put the code in place; restarts nothing
# then the idle-gated restart below
```


`restart-when-idle --check lease` takes an **exclusive** lease on
`--resource`, and on a `shared` resource an exclusive take is refused while
*any* share is held — including the gateway's own, for every model it has
resident. Wait for active requests to finish, then unload idle residents before
the restart. Proved on wafer 2026-09-29
(claude-wafer-services): exclusive take → 200 with nothing resident, 409 with
one model loaded, 200 again after unloading it.

A restart drops the gateway's view of what is resident anyway, so this costs
a cold load and nothing else.

```sh
# A header file, never the token on a command line — #582. Anything in argv is
# readable in the process table for the lifetime of the process.
install -m 600 /dev/null ~/.samcloud/ada.hdr
printf 'Authorization: Bearer %s\n' "$(cat ~/.samcloud/token)" > ~/.samcloud/ada.hdr

# 1. Read authenticated state. Wait for active serving to finish; missing
# in_flight is unknown, not zero. Do not force-unload active work.
curl -fsS -H @"$HOME/.samcloud/ada.hdr" http://localhost:8800/status
# When a named model is confirmed idle, unload it through /models/unload.
# If traffic resumes, wait again rather than racing it.

# 2. Restart once no models remain and the lease check can take the resource.
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
restart-when-idle.sh --check lease --resource ada-wsl/gpu-0 \
  --action 'systemctl --user restart samcloud-model-gateway'
```

Run step 2 with a header file whose identity can see `ada-wsl` — claude-ada's
seat token can. It is **not** `ada-model-service`'s token: that one belongs to
the gateway and should be read by nothing else.

A refusal from step 2 after step 1 means *another tenant* holds a share of the
GPU, which is the correct answer and not a reason to force the restart.

## Checking it worked

```sh
curl -s http://localhost:8800/warm | python3 -m json.tool      # auth-exempt
curl -s -H @"$HOME/.samcloud/ada.hdr" http://localhost:8800/status
```

`/warm`'s `capacity` block is the thing to read:

| field | idle | during a render |
|---|---|---|
| `available_mb` | ~45,300 | 188–2,963 |
| `device_inuse_mb` | ~3,100–5,500 | ~46,000 |
| `foreign_mb` | same as `device_inuse_mb` when nothing is loaded | ~46,000 |
| `work_in_progress` | `false` | `true` |

`work_in_progress: null` means the box has not declared `FOREIGN_IDLE_MB` and
the gate is not running — **not** that nobody is working. If ada shows `null`,
the profile is not loaded.

The one line that proves the identity took, which has never appeared in this
gateway's log:

```
[model-manager] INFO: Lease for <model>: lease_… (NNNN MB, TTL=3600s)
```

Before #861 every lease request 404'd — `service:` scopes grant no device
visibility — and `_request_lease` swallows the error and loads anyway, so the
box served happily for months without ever holding a lease. A `Lease request
failed` line at WARNING is the failure; absence of `Lease for` is the same
failure seen from the other side.

## What this node still does not do

**C1 makes ada safe for the next load, not for the current one.** A render
starting while a model is already resident still collides: ada's own
measurements show a render going from 3.1 GB to ~47 GB *between two
consecutive 5s samples*, so there is no ramp for any reconcile to catch. The
gateway withdraws its offer and evicts an idle model, which protects the next
caller; a model already loaded is in the render's way before anything can
react.

Closing that needs a measurement nobody has yet: what WDDM actually does when
they collide — page our allocation out to system memory (slow for both,
survivable) or fail the render's allocation (the outcome Sam ruled out). That
measurement and the policy it selects are C2.

**Do not load a model onto ada to find out.** The renders are real work. Use a
synthetic CUDA allocation from a separate process, sized like a render, while
the `ada-ps` counter (`\GPU Process Memory(*)\Dedicated Usage`) shows no
UnrealEditor.
