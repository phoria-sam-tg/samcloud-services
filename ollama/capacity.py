"""Canonical capacity signal for inference delivery to samcloud.

ONE collector, byte-identical on every Apple-silicon inference box, so that an
`offering:` a service advertises means the same thing on slice as on wafer.

WHY THIS EXISTS
---------------
macOS "used" is not a fit signal. `active + wired + compressor` counts pages the
compressor is merely *sitting on* — long-idle app pages it has no reason to
release until something asks. Measured on wafer 2026-08-30: it read 22.7 GiB
used of 36 GiB while the entire top-20 process RSS on the box was ~3.9 GiB, and
free jumped 0.2 GiB -> 15.3 GiB the instant a model actually demanded memory.

Sizing off "used" therefore refuses models that would have fitted, while telling
you nothing about whether a load will swap. The number that answers "can I serve
this model" is what the kernel can hand over WITHOUT swapping:

    available = free + inactive + speculative

Reported alongside total and used so the registry keeps both views.
"""

import os
import re
import subprocess
from typing import Optional

# Absolute paths. The gateway runs under a launchd PATH that omits /usr/sbin,
# so a bare "sysctl" raised FileNotFoundError inside the load path and every
# request came back 404 "not available" — a capacity read must never be able
# to fail for an environment reason.
def _tool(name: str) -> str:
    for d in ("/usr/bin", "/usr/sbin", "/bin", "/sbin"):
        c = os.path.join(d, name)
        if os.path.exists(c):
            return c
    return name


_VM_STAT = _tool("vm_stat")
_SYSCTL = _tool("sysctl")
_IOREG = _tool("ioreg")

# Ceiling on how much of what is currently available we will commit to a
# model. Sam's call, and the honest one: an earlier version invented a "25%
# runtime overhead" from two measurements and applied it as if it were a law.
# This makes no claim about model internals — it just declines to fill the
# machine. 90% of whatever is free right now, recomputed every request, so a
# busy desktop shrinks the offer automatically.
USABLE_FRACTION = 0.9

# Floor, so a nearly-full box never offers a sliver it cannot honour.
MIN_HEADROOM_MB = 1024


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


def _compute_pct() -> float:
    try:
        out = subprocess.run(
            [_IOREG, "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def _load_avg() -> float:
    """1-minute load average — the 'general compute' half of the offering."""
    try:
        import os
        return round(os.getloadavg()[0], 2)
    except Exception:
        return 0.0


def collect() -> dict:
    """The canonical capacity reading. Same keys on every box."""
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
    return {
        "memory_total_mb": _total_mb(),
        "memory_used_mb": used,
        "memory_available_mb": available,
        "compute_pct": _compute_pct(),
        "load_avg_1m": _load_avg(),
    }


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
