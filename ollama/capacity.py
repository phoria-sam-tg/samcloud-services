"""Canonical capacity signal for inference delivery to samcloud.

ONE collector, with the same contract on every inference box, so that an
`offering:` a service advertises means the same thing on slice as on wafer as
on ada. There are two backends because the hardware differs; there is one
question, and both arms answer it:

    memory_available_mb  what this device can hand to a model RIGHT NOW
                         without swapping, counting EVERY tenant — the OS,
                         other people's work, and our own resident models.

"Counting every tenant" is the load-bearing half, and it is why the reading is
taken from the device rather than from the registry. The registry's
`available_memory_mb` is `total - leased`: it sees only memory somebody asked
it for. On 2026-09-28 it reported 49,140 MB available on `ada-wsl/gpu-0` while
that card's own `memory.used` was 48,016 MB of 49,140 — nothing had taken a
lease, so as far as the registry was concerned the card was empty. Any offer
sized off that number promises memory that is not there.

METAL (Apple silicon, unified memory)
-------------------------------------
macOS "used" is not a fit signal. `active + wired + compressor` counts pages the
compressor is merely *sitting on* — long-idle app pages it has no reason to
release until something asks. Measured on wafer 2026-08-30: it read 22.7 GiB
used of 36 GiB while the entire top-20 process RSS on the box was ~3.9 GiB, and
free jumped 0.2 GiB -> 15.3 GiB the instant a model actually demanded memory.

Sizing off "used" therefore refuses models that would have fitted, while telling
you nothing about whether a load will swap. The number that answers "can I serve
this model" is what the kernel can hand over WITHOUT swapping:

    available = free + inactive + speculative

Validated against a real GPU allocation on wafer 2026-09-29
(claude-wafer-services), loading and unloading a 3336 MiB model:

                     available   used    IOAccel_inuse
    idle                 13967  21988            1167
    resident             10641  25309            4295
    unloaded             13398  22554             947

`available` moved -3326 for a 3336 MiB model and returned to baseline. Metal
buffers are wired and the compressor cannot touch them, so for GPU-resident
bytes this is a ledger, not an estimate — which is the case the whole feature
turns on.

Its bound, from the same run, because someone will otherwise "fix" the
discrepancy: CPU-side anonymous pages do NOT track. Stepping 4096 MiB of
incompressible `os.urandom` into RSS moved `available` by only -1009, and
non-monotonically (it ROSE 175 while RSS rose 1025), because the kernel refills
`inactive` by compressing and purging about as fast as a consumer drains it.
So a render's GPU residency we see exactly and its CPU-side staging we
under-see by up to ~4x. Faithful where it matters, lossy where it does not.

And the trap the CUDA arm below makes MORE likely, not less: `free` ALONE read
477 MiB on that box while `available` read 13165. `total - used` is correct on
a discrete card and catastrophic here — carrying that intuition across
collapses the offer by 27x. The two arms sit one function apart in this file
on purpose; they are not interchangeable.

CUDA (discrete NVIDIA card)
---------------------------
A discrete card has no compressor and no reclaimable-page distinction: an
allocation either holds board memory or it does not. So the same question has
the direct answer

    available = memory.total - memory.used

and `memory.used` is already the whole-board figure. That matters most on ada,
which is a WORK machine: Windows-side renders (Unreal, etc.) hold VRAM this
gateway never allocated and holds no lease on, and `memory.used` counts them.
Measured inside ada-wsl 2026-09-28 22:5x UTC: `memory.used` read 5,515 MiB of
49,140 while `--query-compute-apps` listed ZERO processes — so the total
includes usage WSL cannot attribute, and per-process attribution is empty.
The inference, kept separate: under WDDM the Windows driver owns the
allocations and does not expose them to the Linux-side nvidia-smi as compute
apps. What would disconfirm it: a render whose VRAM does NOT move `memory.used`
inside WSL, or a `--query-compute-apps` that starts listing Windows pids.

That measurement is why `foreign_mb()` takes what the gateway knows it holds
and subtracts, rather than summing a process table: there is no process table
to sum.

THE SECOND NUMBER: memory_device_inuse_mb
----------------------------------------
`available` answers "do I fit". It cannot answer "is anyone working", and on a
work machine that is the question Sam actually asked. So the reading carries a
second figure — what the ACCELERATOR holds, as opposed to what the machine has
committed:

    metal   IOAccelerator -> PerformanceStatistics -> "In use system memory"
    cuda    nvidia-smi memory.used  (the same number as memory_used_mb there,
            because on a discrete card the board is the accelerator)

On Metal the two are nothing alike: `memory_used_mb` is ~22 GiB of OS on an
idle 36 GB box, while the accelerator figure idles at ~1.0-1.2 GiB
(WindowServer and friends) and tracked the model load above +3128 MiB. That is
why `foreign_mb` subtracts from THIS number and not from `used` — an earlier
draft of this module used `used`, which would have been a garbage signal on
exactly one of the two arms while every test stayed green.

Neither arm can attribute per process (see `foreign_mb`), so both get one
aggregate device figure, subtract what the gateway knows it holds, and call
the rest somebody else's. One contract, two collectors — not a CUDA feature
with a Metal stub.

Neither of these numbers is a decision. This module measures; `manager` and
the endpoints decide, and both of them need hysteresis that is not here:
`available` was measured swinging 2950 MiB on an IDLE wafer and 188 -> 2963
MiB inside a single continuous ada render, so any threshold placed inside
those bands flaps with nothing having changed.

Both arms report total and used alongside available so the registry keeps every
view. Units are MiB throughout, on both arms, labelled `_mb` — `vm_stat` pages
are divided by 1024*1024, `ioreg` bytes likewise, and `nvidia-smi
--format=nounits` emits MiB.
"""

import logging
import os
import re
import subprocess
import sys
from typing import Optional

log = logging.getLogger("model-capacity")

# Absolute paths. The gateway runs under a launchd PATH that omits /usr/sbin,
# so a bare "sysctl" raised FileNotFoundError inside the load path and every
# request came back 404 "not available" — a capacity read must never be able
# to fail for an environment reason.
#
# /usr/lib/wsl/lib is the same hazard on the CUDA arm: under WSL the
# driver-provided nvidia-smi lives there and nowhere else, and it is on the
# interactive PATH only because /etc/profile puts it there. A systemd unit does
# not read /etc/profile, so a bare "nvidia-smi" resolves for a human in a shell
# and not for the service — the failure mode this function already exists to
# stop, on a box where it would take the whole capacity signal down.
def _tool(name: str) -> str:
    for d in ("/usr/bin", "/usr/sbin", "/bin", "/sbin", "/usr/lib/wsl/lib"):
        c = os.path.join(d, name)
        if os.path.exists(c):
            return c
    return name


_VM_STAT = _tool("vm_stat")
_SYSCTL = _tool("sysctl")
_IOREG = _tool("ioreg")
_NVIDIA_SMI = _tool("nvidia-smi")

# Which arm of collect() this box uses. Detected, but overridable by env,
# because detection is a guess about hardware and every other identity on this
# service is declared (see config.py) — a box that has nvidia-smi installed and
# serves models off the CPU should be able to say so without us sniffing.
BACKEND_METAL = "metal"
BACKEND_CUDA = "cuda"


def _detect_backend() -> str:
    declared = os.environ.get("SC_CAPACITY_BACKEND", "").strip().lower()
    if declared in (BACKEND_METAL, BACKEND_CUDA):
        return declared
    if declared:
        log.warning(
            f"SC_CAPACITY_BACKEND={declared!r} is not one of "
            f"{BACKEND_METAL}/{BACKEND_CUDA} — detecting instead"
        )
    if sys.platform == "darwin":
        return BACKEND_METAL
    # Not a Mac. CUDA is the only other arm there is, so name it even when
    # nvidia-smi is missing: the resulting failure says "nvidia-smi not found",
    # which is true and fixable, where falling back to METAL would run vm_stat
    # on Linux and fail with something that sends the reader nowhere useful.
    if not os.path.isabs(_NVIDIA_SMI):
        log.error(
            "capacity backend is cuda (not darwin) but nvidia-smi was not found "
            "in /usr/bin, /usr/sbin, /bin, /sbin or /usr/lib/wsl/lib. Every "
            "capacity read will fail until it is installed or SC_CAPACITY_BACKEND "
            "is set."
        )
    return BACKEND_CUDA


BACKEND = _detect_backend()

# Which card, on a box with more than one. The resource this gateway leases is
# named `<device>/gpu-0`, so the reading has to describe that one card and not
# a sum over the box — an offer sized off two cards' free memory fits in
# neither. Metal ignores it: unified memory is the machine.
CUDA_INDEX = os.environ.get("SC_CUDA_INDEX", "0").strip() or "0"

# Ceiling on how much of what is currently available we will commit to a
# model. Sam's call, and the honest one: an earlier version invented a "25%
# runtime overhead" from two measurements and applied it as if it were a law.
# This makes no claim about model internals — it just declines to fill the
# machine. 90% of whatever is free right now, recomputed every request, so a
# busy desktop shrinks the offer automatically.
#
# THE FORMULA IS SHARED; THE TWO NUMBERS ARE PER-BACKEND, because the cost of
# getting them wrong is not symmetric (claude-wafer-services, #861):
#
#   Metal over-commit degrades to swap. Slow, recoverable, and it lands on US —
#   the pool is unified memory shared with an OS that reclaims elastically.
#   0.9 and a 1 GiB floor are defensible and stay exactly as they were.
#
#   CUDA over-commit is a hard allocation failure, and under WDDM a competing
#   render can GROW after we have committed. So the cost lands on someone
#   else's work, which is the thing this ticket exists to stop. The floor
#   matters more than the fraction there, and it wants to be materially larger.
_USABLE_FRACTION_BY_BACKEND = {
    BACKEND_METAL: 0.9,
    BACKEND_CUDA: 0.9,
}

# Floor, so a nearly-full box never offers a sliver it cannot honour.
#
# THE CUDA FLOOR BELOW IS NOT MEASURED YET. It is the Metal number because a
# placeholder has to be something, and 1 GiB is the value that changes nothing
# rather than a claim about a discrete card. The real one is a TIME-DERIVED
# quantity with no Metal analogue: how much VRAM a render can claim in the
# window between our reading and the next reconcile — i.e. peak minus the first
# post-launch sample, over one reconcile interval. It is being measured on ada
# and lands with the ada profile, and until it does `SC_MIN_HEADROOM_MB`
# overrides it per box.
#
# When you set it: do not round it to 4096 because 4 GB is a nice number. The
# whole reason it is a separate constant is that it comes from a measurement
# on one card under one kind of work.
_MIN_HEADROOM_MB_BY_BACKEND = {
    BACKEND_METAL: 1024,
    BACKEND_CUDA: 1024,
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning(f"{name}={raw!r} is not a number — using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning(f"{name}={raw!r} is not an integer — using {default}")
        return default


# Resolved for THIS box. Kept as module-level names because the refusal message
# in manager.py quotes them ("90% of the NNNN MB free right now"), and a caller
# told a percentage that is not the one actually applied is worse off than one
# told nothing.
USABLE_FRACTION = _env_float(
    "SC_USABLE_FRACTION", _USABLE_FRACTION_BY_BACKEND[BACKEND])
MIN_HEADROOM_MB = _env_int(
    "SC_MIN_HEADROOM_MB", _MIN_HEADROOM_MB_BY_BACKEND[BACKEND])


class InsufficientCapacity(Exception):
    """A load refused because it does not fit right now — not a broken gateway.

    `manager` has raised this by name and `server` has caught it by name since
    the capacity migration on 2026-08-30, but nothing ever defined it. Every
    refusal therefore raised AttributeError instead; worse, evaluating the
    `except capacity.InsufficientCapacity` clause raised it a second time, so
    the error escaped the whole `try` without reaching the fallback below it
    and every capacity answer surfaced as a bare 500. The gate held the whole
    time — it just could not say why, which is the one thing a refusal is for.

    Carries the reading it was refused against so the caller gets told what
    does fit, instead of being told the box is broken.
    """

    def __init__(
        self,
        detail: str,
        *,
        need_mb: Optional[int] = None,
        usable_mb: Optional[int] = None,
        available_mb: Optional[int] = None,
        fits_now: Optional[list] = None,
    ):
        super().__init__(detail)
        self.detail = detail
        self.need_mb = need_mb
        self.usable_mb = usable_mb
        self.available_mb = available_mb
        self.fits_now = list(fits_now or [])

    def as_dict(self) -> dict:
        """Body for a 503, with the numbers a caller needs to choose again."""
        return {
            "error": "insufficient_capacity",
            "message": self.detail,
            "need_mb": self.need_mb,
            "usable_mb": self.usable_mb,
            "available_mb": self.available_mb,
            "fits_now": self.fits_now,
        }


class PoolBusy(Exception):
    """An exclusive resource is held by someone else right now.

    The sibling of `InsufficientCapacity`, and deliberately the same *kind* of
    answer: a refusal that carries the numbers to retry on, not a fault. The
    two are distinct because they mean different things and a caller should act
    on them differently — `insufficient_capacity` says "this box cannot fit
    that model, here is what does fit", and there is no point retrying in ten
    seconds. `resource_busy` says "the thing you want exists and works, but
    someone else has it", and retrying is exactly the right move.

    Only exclusive resources can raise it. On `claude-services-slice/exo-pool`
    a lease means the pool is TAKEN, not that bytes are reserved, so a second
    consumer that starts work while a lease is held is not merely slow — it
    rebuilds the 20-minute collision of 2026-09-20.

    `queue_position` is carried for shape-compatibility with the
    `insufficient_capacity` body and is honestly `None` on an exclusive
    conflict: the registry maintains no queue for that path. Only the
    memory-oversubscription branch assigns queue positions, and a resource
    that leases no bytes never enters it. `retry_after_s` is the field doing
    the real work, derived from the holder's `expires_at`.
    """

    def __init__(
        self,
        detail: str,
        *,
        resource_id: Optional[str] = None,
        retry_after_s: Optional[int] = None,
        expires_at: Optional[str] = None,
        queue_position: Optional[int] = None,
        error: str = "resource_busy",
    ):
        super().__init__(detail)
        self.detail = detail
        self.resource_id = resource_id
        self.retry_after_s = retry_after_s
        self.expires_at = expires_at
        self.queue_position = queue_position
        # Lets a caller tell "someone holds the lease, come back" from "the
        # pool says it is generating but holds no lease", which need different
        # reactions: the first resolves itself, the second usually needs a
        # human. Same body shape either way.
        self.error = error

    def as_dict(self) -> dict:
        """Body for a 503, in the same shape as an insufficient_capacity one."""
        return {
            "error": self.error,
            "message": self.detail,
            "resource_id": self.resource_id,
            "queue_position": self.queue_position,
            "retry_after_s": self.retry_after_s,
            "expires_at": self.expires_at,
        }


def usable_mb(available_mb: int) -> int:
    """How much of `available_mb` we are willing to commit."""
    return max(0, min(int(available_mb * USABLE_FRACTION),
                      available_mb - MIN_HEADROOM_MB))


def _vm_stat() -> tuple[dict, int]:
    """Return (pages_by_name, page_size_bytes) from vm_stat.

    Page size comes from the header, never assumed — Apple silicon is 16 KiB and
    hardcoding 4 KiB under-reported slice's memory by 4x for weeks.
    """
    out = subprocess.run(
        [_VM_STAT], capture_output=True, text=True, check=True, timeout=10
    ).stdout
    page_size = 4096
    header = re.search(r"page size of (\d+) bytes", out)
    if header:
        page_size = int(header.group(1))
    pages = {}
    for line in out.splitlines():
        m = re.match(r"Pages (\w[\w ]*\w):\s+(\d+)\.", line)
        if m:
            pages[m.group(1)] = int(m.group(2))
    return pages, page_size


def _total_mb() -> int:
    out = subprocess.run(
        [_SYSCTL, "-n", "hw.memsize"], capture_output=True, text=True,
        check=True, timeout=10,
    ).stdout.strip()
    return round(int(out) / 1024 / 1024)


# Note the closing quote inside each pattern. IOAccelerator publishes BOTH
# `"In use system memory"` and `"In use system memory (driver)"`, and the
# second reads 0 on an idle Mac — an unanchored search matches whichever comes
# first in the dump and would silently return the wrong counter.
_ACCEL_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_ACCEL_INUSE_RE = re.compile(r'"In use system memory"\s*=\s*(\d+)')


def _accel_stats() -> tuple[float, Optional[int]]:
    """(compute_pct, device_inuse_mb) from ONE `ioreg` read.

    One read rather than two because the two numbers are compared against each
    other downstream, and sampling them 30ms apart on a box whose GPU
    utilisation is genuinely volatile would put a gap between them for no
    reason. See `_compute_pct`'s note on how volatile.

    `device_inuse_mb` is None, never 0, when the counter cannot be read.
    Zero would mean "the accelerator holds nothing", which is the reading that
    makes `foreign_mb` say nobody is working — precisely the wrong direction to
    fail in, since the consequence is loading a model on top of someone's
    render. A consumer that treats None as 0 has reintroduced the bug.
    """
    try:
        out = subprocess.run(
            [_IOREG, "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except Exception as e:
        log.warning(f"ioreg IOAccelerator read failed: {e}")
        return 0.0, None
    m = _ACCEL_UTIL_RE.search(out)
    pct = float(m.group(1)) if m else 0.0
    m = _ACCEL_INUSE_RE.search(out)
    inuse = round(int(m.group(1)) / 1024 / 1024) if m else None
    if inuse is None:
        log.warning('ioreg gave no "In use system memory" counter')
    return pct, inuse


def _compute_pct() -> float:
    """GPU utilisation on the Metal arm. The CUDA arm reads it from nvidia-smi
    (`utilization.gpu`) in the same pass as memory, so it never calls this.

    DO NOT GATE ON THIS, on either arm. Measured on wafer 2026-09-29
    (claude-wafer-services), 24 samples over 2 minutes with a Unity editor
    session active: min 10, median 32, max 49, hitting 15 distinct values. A
    single instantaneous read cannot distinguish "busy" from "between frames",
    so a withdrawal triggered off one sample flaps.

    ada reached the same conclusion from the other arm, and more sharply
    (claude-ada, 2026-09-29). Across 53 samples of one continuous render
    `utilization.gpu` ranged 27-73 (median 60) — and then at 23:12:41-23:12:53
    UTC it fell to 0-4% for twelve seconds, one read hitting 0, while
    `memory.used` held flat at 43,634-43,674 MiB and never dropped. A
    utilisation of ZERO during an active render is a real reading, not a
    glitch. Gate on it and the gateway offers the whole card to the next
    caller at a frame boundary.

    Only the memory figures say whether work is present. This one is reported
    because the registry displays it; a decision that wants a utilisation term
    needs a sampled window, and neither B nor C uses one in its first cut.
    """
    return _accel_stats()[0]


def _load_avg() -> float:
    """1-minute load average — the 'general compute' half of the offering.

    ON THE CUDA ARM THIS IS DECORATION, and measured to be so. Under WSL it is
    the Linux VM's load and cannot see the Windows side, which on a work
    machine is where the competing work runs. claude-ada, 2026-09-29: loadavg
    read 7.44 during a burst and then 0.13 a few minutes later while GPU
    utilisation was still ~61% and a render still held ~46 GB — it moved
    independently of GPU state rather than merely lagging it. So it is not a
    weak signal on that arm, it is not a signal at all. Reported because the
    contract is shared and Metal uses it; never gated on.
    """
    try:
        import os
        return round(os.getloadavg()[0], 2)
    except Exception:
        return 0.0


# The four keys every arm must return. Anything reading a capacity reading may
# rely on these and nothing else; an arm that can measure more (a discrete card
# knows its temperature and power draw, a Mac does not) adds keys on top.
CONTRACT_KEYS = (
    "memory_total_mb",
    "memory_used_mb",
    "memory_available_mb",
    "memory_device_inuse_mb",
    "load_avg_1m",
)


def _collect_metal() -> dict:
    pages, page_size = _vm_stat()
    mb = lambda n: round(n * page_size / 1024 / 1024)

    used = mb(
        pages.get("active", 0)
        + pages.get("wired down", 0)
        + pages.get("occupied by compressor", 0)
    )
    # Reclaimable without swapping. Inactive and speculative are both pages the
    # kernel will hand over on demand; free is already ours.
    available = mb(
        pages.get("free", 0)
        + pages.get("inactive", 0)
        + pages.get("speculative", 0)
    )
    compute_pct, device_inuse = _accel_stats()
    return {
        "memory_total_mb": _total_mb(),
        "memory_used_mb": used,
        "memory_available_mb": available,
        "memory_device_inuse_mb": device_inuse,
        "compute_pct": compute_pct,
        "load_avg_1m": _load_avg(),
    }


# What we ask nvidia-smi for, in order. Kept beside the parser so the two
# cannot drift: the CSV has no header (--format=csv,noheader), so a field added
# here and not there silently shifts every column after it.
_CUDA_FIELDS = (
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
)


def _cuda_number(raw: str) -> Optional[float]:
    """One nvidia-smi CSV cell as a number, or None if the card won't say.

    nvidia-smi fills a field it cannot read with `[N/A]`, `[Not Supported]` or
    `[Unknown Error]` rather than omitting it. Those are not zero — a card
    reporting no power draw is not drawing no power — so they become None and
    `registry_payload` drops them, leaving the registry's last good value in
    place instead of overwriting it with a fiction.
    """
    cell = raw.strip()
    if not cell or cell.startswith("["):
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _parse_cuda_csv(line: str) -> dict:
    """One `--format=csv,noheader,nounits` row into a capacity reading.

    Split out from the subprocess call so the arm is testable on a box with no
    NVIDIA card in it — which is every box that reviews this code.

    `memory.used` is the WHOLE BOARD: every tenant, including ones this Linux
    side cannot even enumerate (see the module docstring). Subtracting it from
    total is therefore the same question the Metal arm answers, not a narrower
    one.
    """
    cells = line.split(",")
    if len(cells) < len(_CUDA_FIELDS):
        raise ValueError(
            f"nvidia-smi returned {len(cells)} fields, expected "
            f"{len(_CUDA_FIELDS)} ({', '.join(_CUDA_FIELDS)}): {line!r}"
        )
    total, used, util, temp, power = (_cuda_number(c) for c in cells[:5])
    if total is None or used is None:
        raise ValueError(f"nvidia-smi gave no memory reading: {line!r}")
    total_mb = round(total)
    used_mb = round(used)
    return {
        "memory_total_mb": total_mb,
        # Clamped at 0 rather than trusted: used > total is not physically
        # meaningful, and a negative "available" would read as a huge unsigned
        # number to anything downstream that is not expecting one.
        "memory_available_mb": max(0, total_mb - used_mb),
        "memory_used_mb": used_mb,
        # The same number as memory_used_mb, deliberately, and not a shortcut:
        # on a discrete card the board IS the accelerator, so "memory the
        # machine has committed" and "memory the accelerator holds" are one
        # quantity. On Metal they are very different (one is ~22 GiB of OS,
        # the other ~1 GiB of WindowServer), which is why the reading carries
        # both keys on both arms rather than making callers know which is
        # which.
        "memory_device_inuse_mb": used_mb,
        "compute_pct": util,
        "temperature_c": temp,
        "power_draw_w": power,
        "load_avg_1m": _load_avg(),
    }


def _collect_cuda() -> dict:
    out = subprocess.run(
        [
            _NVIDIA_SMI,
            f"--id={CUDA_INDEX}",
            f"--query-gpu={','.join(_CUDA_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    rows = [r for r in out.splitlines() if r.strip()]
    if not rows:
        raise ValueError(f"nvidia-smi --id={CUDA_INDEX} returned no rows")
    return _parse_cuda_csv(rows[0])


def collect() -> dict:
    """The canonical capacity reading. Same contract on every box.

    CONTRACT_KEYS on both arms; the CUDA arm adds temperature_c and
    power_draw_w, and both add compute_pct (which is None on a CUDA card that
    declines to report utilisation).
    """
    return _collect_cuda() if BACKEND == BACKEND_CUDA else _collect_metal()


def foreign_mb(device_inuse_mb: Optional[int], own_mb: int) -> Optional[int]:
    """Accelerator memory held by someone other than this gateway.

    THE SECOND TERM, and the one that answers the question a fit check cannot.
    `available` answers "do I fit". This answers "is anyone working", and they
    are not the same question — they only look the same when the other tenant
    happens to take the whole device.

    Worked from ada's own numbers (claude-ada, 53 samples, 2026-09-29): during
    one continuous 46 GB Unreal render, free memory swung 188 -> 2963 MiB. Run
    the fit check alone over that band and at the render's MEDIAN free (1406)
    the box offers 382 MB, and at its peak free (2963) it offers 1939 MB —
    enough for a small model, loaded into a gap inside a running render, which
    every log we keep would record as a success. `foreign` over the same window
    reads ~46.5 GB throughout and is never ambiguous. (Cross-checked from the
    Windows side, which WSL cannot see: UnrealEditor pid 15240, 46,331 MB
    dedicated; dwm 188 MB; nothing else over 100 MB.)

    The converse matters as much and has NOT been measured: a lighter render
    holding 20 GB of a 48 GB board leaves ~29 GB free, clears any floor we
    could reasonably set, and only this term would notice it. Every render
    observed so far takes the whole card, so the two questions have coincided
    in all the evidence we have. Do not read that coincidence as a licence to
    keep only one of them.

    `own_mb` is what the gateway knows it has loaded — NOT a sum over a process
    table, because on neither arm is there one to sum. Measured inside WSL
    2026-09-28/29: `nvidia-smi --query-compute-apps` returns zero rows against
    46 GB of real usage, every time, because under WDDM the Windows driver owns
    the allocations. IOAccelerator is the same shape for a different reason —
    it publishes one aggregate figure with no per-process split at all. Two
    backends, one constraint, so attribution comes from our own bookkeeping and
    everything we cannot account for is, by definition, somebody else's.

    The error direction is deliberate. Under-counting what we hold invents a
    stranger and costs us an offer; over-counting would hide one and let us
    load on top of somebody's work. Only the second is a failure Sam asked us
    to prevent.

    Returns None when the device figure is unavailable — see `_accel_stats`.
    None is not zero and must not be coerced to it by a caller: "we cannot tell
    who is using the GPU" has to be handled as its own case.

    Clamped at 0 otherwise: `own_mb` can legitimately exceed the device figure
    for a moment (the gateway records a model's estimated size before the
    allocation lands, and an unload frees board memory before we drop our
    record of it), and "negative foreign memory" is not a state to propagate.
    """
    if device_inuse_mb is None:
        return None
    return max(0, int(device_inuse_mb) - max(0, int(own_mb)))


def fits(need_mb: int, available_mb: int) -> bool:
    """Would loading `need_mb` stay inside our share of what is free now?"""
    return need_mb <= usable_mb(available_mb)


def servable(catalogue: dict, available_mb: int) -> list[str]:
    """Which of `catalogue` ({name: size_mb}) fits right now, largest first."""
    return [
        name
        for name, size_mb in sorted(
            catalogue.items(), key=lambda kv: kv[1], reverse=True
        )
        if fits(size_mb, available_mb)
    ]


def offering_tier(catalogue: dict, available_mb: int) -> str:
    """Coarse tier derived from what actually fits, not from fixed MB bands.

    Fixed bands go stale the moment the catalogue changes — wafer advertised
    `offering:full` on ~14 GiB available because the top band (>=10 GiB) was set
    when the biggest model was ~6 GB, long before a 17.5 GB model existed.
    Deriving the tier from the catalogue keeps `full` meaning "the big one fits".
    """
    if not catalogue:
        return "none"
    can = servable(catalogue, available_mb)
    if not can:
        return "none"
    if len(can) == len(catalogue):
        return "full"
    return "mini" if len(can) == 1 else "degraded"


# The samcloud registry's ResourceStats is a strict model — unknown keys 422.
# It does not yet carry memory_available_mb, which is the whole point of this
# module, so until the schema gains it we send what the wire accepts and keep
# the richer reading local for fit and offering decisions.
REGISTRY_STATS_KEYS = (
    "memory_used_mb",
    "memory_total_mb",
    "compute_pct",
    "temperature_c",
    "power_draw_w",
    "processes",
)


def registry_payload(stats: Optional[dict] = None) -> dict:
    """Subset of a reading that the registry's stats endpoint will accept."""
    s = collect() if stats is None else stats
    return {k: v for k, v in s.items() if k in REGISTRY_STATS_KEYS and v is not None}
