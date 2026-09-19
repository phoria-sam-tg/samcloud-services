"""Pushes claude-services-slice/gpu-0 stats to SAMcloud (~15s cadence).

Apple Silicon has no root-free power/temperature read, so this reports what
the /resources/<id>/stats schema actually accepts: memory_used_mb + compute_pct
(temperature_c/power_draw_w are accepted too but we have no non-root source).
"""

import re
import subprocess
import time

from . import capacity
from . import config
from .samcloud import SamcloudClient

PUSH_INTERVAL_SECONDS = 15

# vm_stat reports PAGES, not bytes. Apple Silicon uses 16 KiB pages, not the
# 4 KiB this used to assume — so every memory_used_mb pushed to the registry
# was 4x too low, and slice read as far roomier than it is. Ask the kernel.
_PAGE_SIZE = int(
    subprocess.run(
        ["sysctl", "-n", "hw.pagesize"], capture_output=True, text=True, check=True
    ).stdout.strip()
)


def _memory_used_mb() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    pages = {}
    for line in out.splitlines():
        m = re.match(r"Pages (\w[\w ]*\w):\s+(\d+)\.", line)
        if m:
            pages[m.group(1)] = int(m.group(2))
    used_pages = (
        pages.get("active", 0)
        + pages.get("wired down", 0)
        + pages.get("occupied by compressor", 0)
    )
    return round(used_pages * _PAGE_SIZE / 1024 / 1024)


def _compute_pct() -> float:
    out = subprocess.run(
        ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, check=True
    ).stdout
    m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
    return float(m.group(1)) if m else 0.0


def collect_stats() -> dict:
    """Canonical capacity reading — see ollama/capacity.py.

    Identical collector to the one wafer's gateway uses, so an offering means
    the same thing on both boxes. Adds memory_available_mb (free + inactive +
    speculative), which is the number that answers "will this model load
    without swapping"; memory_used_mb alone does not.
    """
    return capacity.collect()


def main() -> None:
    client = SamcloudClient(token=config.SC_TOKEN)
    while True:
        try:
            stats = collect_stats()
            client.push_stats(config.SC_RESOURCE_ID, capacity.registry_payload(stats))
            print(f"[{time.strftime('%X')}] pushed {stats}")
        except Exception as e:
            print(f"[{time.strftime('%X')}] push failed: {e}")
        time.sleep(PUSH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
