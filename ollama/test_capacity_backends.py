#!/usr/bin/env python3
"""The capacity contract holds on both arms — Metal and CUDA.

`capacity.py` was Apple-silicon only: `vm_stat`, `sysctl`, `ioreg`. Bringing ada
(RTX 6000 Ada, 48 GB, under WSL) onto the same `main` needs a second arm that
answers the SAME question, because `offering:` and every fit decision fleet-wide
are read as if they mean one thing (#861).

Runs anywhere. The CUDA arm is exercised through `_parse_cuda_csv` on rows
recorded from the real card, which is the point of having the parser separate
from the subprocess call — every box that reviews this code has no NVIDIA GPU in
it, and an arm that can only be tested on the one box it runs on is an arm
nobody checks.

    python -m ollama.test_capacity_backends
"""

import os
import sys

from . import capacity

# Recorded from `ada-wsl/gpu-0` via the registry's own utilisation snapshot,
# 2026-09-28 23:03 UTC: 49,140 MiB card, 48,016 MiB used, 62% compute, 72 C,
# 115.25 W. Field order is `capacity._CUDA_FIELDS`.
ADA_ROW = "49140, 48016, 62, 72, 115.25"

# The same card with the two optional sensors unavailable. nvidia-smi fills a
# field it cannot read rather than omitting it, so the row still has five cells.
ADA_ROW_NA = "49140, 5515, [N/A], [Not Supported], [N/A]"

failures = []
checks = 0


def step(n, msg):
    print(f"\n{'='*60}\n  Step {n}: {msg}\n{'='*60}")


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def main():
    step(1, "this box's live reading satisfies the contract")
    print(f"  BACKEND={capacity.BACKEND}")
    live = capacity.collect()
    print(f"  {live}")
    missing = [k for k in capacity.CONTRACT_KEYS if k not in live]
    check(not missing, f"every contract key present (missing: {missing})")
    check(live["memory_available_mb"] <= live["memory_total_mb"],
          "available <= total")
    check(live["memory_used_mb"] >= 0, "used is not negative")

    step(2, "the CUDA arm reads the whole board")
    r = capacity._parse_cuda_csv(ADA_ROW)
    print(f"  {r}")
    check(r["memory_total_mb"] == 49140, f"total 49140 ({r['memory_total_mb']})")
    check(r["memory_used_mb"] == 48016, f"used 48016 ({r['memory_used_mb']})")
    # The whole feature in one number: 48,016 MiB of that card was held by
    # tenants the registry knew nothing about (it reported the card 100% free
    # at the same instant), so available has to be 1,124 and not 49,140.
    check(r["memory_available_mb"] == 1124,
          f"available = total - used = 1124 ({r['memory_available_mb']})")
    # On a discrete card the board IS the accelerator, so these coincide.
    check(r["memory_device_inuse_mb"] == 48016,
          f"device_inuse == used on CUDA ({r['memory_device_inuse_mb']})")
    check(r["compute_pct"] == 62.0, f"compute_pct 62 ({r['compute_pct']})")
    check(r["temperature_c"] == 72.0, f"temperature_c 72 ({r['temperature_c']})")
    check(r["power_draw_w"] == 115.25, f"power_draw_w 115.25 ({r['power_draw_w']})")
    missing = [k for k in capacity.CONTRACT_KEYS if k not in r]
    check(not missing, f"CUDA arm meets the same contract (missing: {missing})")

    step(3, "an unreadable sensor is None, never zero")
    r = capacity._parse_cuda_csv(ADA_ROW_NA)
    print(f"  {r}")
    check(r["compute_pct"] is None, f"[N/A] -> None ({r['compute_pct']})")
    check(r["temperature_c"] is None,
          f"[Not Supported] -> None ({r['temperature_c']})")
    check(r["power_draw_w"] is None, f"[N/A] -> None ({r['power_draw_w']})")
    check(r["memory_available_mb"] == 49140 - 5515,
          "memory still read when the sensors are not")
    # A None must not reach the registry as a value: its ResourceStats is
    # strict, and overwriting a last-good temperature with null is worse than
    # sending nothing.
    payload = capacity.registry_payload(r)
    check("temperature_c" not in payload and "power_draw_w" not in payload,
          f"registry_payload drops the None sensors ({sorted(payload)})")
    check(payload.get("memory_used_mb") == 5515,
          f"registry_payload keeps the memory reading ({payload})")

    step(4, "a short row raises rather than shifting every column")
    # --format=csv,noheader has no names in it, so a field dropped from the
    # query and not from the parser would silently read power draw as a
    # temperature. Fail loudly instead.
    try:
        capacity._parse_cuda_csv("49140, 48016, 62")
        check(False, "a 3-field row was rejected")
    except ValueError as e:
        check("expected 5" in str(e) or "expected" in str(e),
              f"a short row raises ValueError naming the count ({e})")
    try:
        capacity._parse_cuda_csv("[N/A], [N/A], 0, 0, 0")
        check(False, "a row with no memory reading was rejected")
    except ValueError as e:
        check(True, f"no-memory row raises ValueError ({e})")

    step(5, "foreign memory is what we cannot account for")
    check(capacity.foreign_mb(48016, 8000) == 40016,
          "in use 48016 - ours 8000 = 40016")
    check(capacity.foreign_mb(5515, 0) == 5515,
          "holding nothing, all of it is someone else's")
    # Both of these happen in normal operation: the gateway records an
    # estimated size before the allocation lands, and frees board memory
    # before it drops its own record.
    check(capacity.foreign_mb(1000, 8000) == 0,
          "ours > in use clamps to 0, never negative")
    check(capacity.foreign_mb(5515, -1) == 5515,
          "a negative own_mb cannot inflate the foreign figure")
    # "We cannot tell" is a third state and has to survive as one. Coerced to
    # 0 it reads as "nobody is working", which is the direction that loads a
    # model on top of a render.
    check(capacity.foreign_mb(None, 8000) is None,
          "an unreadable device figure gives None, not 0")

    step(6, "the two questions are different, on ada's own numbers")
    # claude-wafer-services' finding over claude-ada's 53 samples: inside ONE
    # continuous 46 GB Unreal render, free memory swung 188 -> 2963 MiB. The
    # fit check alone is satisfiable in the upper half of that band, so a
    # gateway with only a fit check loads into a gap inside a live render and
    # logs it as a success. `foreign` is unambiguous across the whole band.
    RENDER_FREE = (188, 460, 1406, 2207, 2963)   # MiB, one render
    offers = [capacity.usable_mb(f) for f in RENDER_FREE]
    print(f"  usable() across one render's free-memory swing: {offers}")
    check(max(offers) > 1024,
          f"fit ALONE would offer {max(offers)}MB mid-render — the failure")
    # Same five moments, seen through the second term. Board is 49140 MiB and
    # the gateway holds nothing, so in-use is total - free.
    foreign = [capacity.foreign_mb(49140 - f, 0) for f in RENDER_FREE]
    print(f"  foreign_mb across the same swing: {foreign}")
    check(min(foreign) > 40000,
          f"foreign never drops below {min(foreign)}MB — never ambiguous")
    check(len([f for f in foreign if f is None]) == 0,
          "and it is readable at every one of them")

    step(7, "the backend is declared before it is detected")
    prior = os.environ.get("SC_CAPACITY_BACKEND")
    try:
        os.environ["SC_CAPACITY_BACKEND"] = "cuda"
        check(capacity._detect_backend() == capacity.BACKEND_CUDA,
              "SC_CAPACITY_BACKEND=cuda is honoured on a Mac")
        os.environ["SC_CAPACITY_BACKEND"] = "metal"
        check(capacity._detect_backend() == capacity.BACKEND_METAL,
              "SC_CAPACITY_BACKEND=metal is honoured")
        os.environ["SC_CAPACITY_BACKEND"] = "nonsense"
        detected = capacity._detect_backend()
        check(detected in (capacity.BACKEND_METAL, capacity.BACKEND_CUDA),
              f"an unreadable value falls back to detection ({detected})")
        os.environ.pop("SC_CAPACITY_BACKEND")
        expected = (capacity.BACKEND_METAL if sys.platform == "darwin"
                    else capacity.BACKEND_CUDA)
        check(capacity._detect_backend() == expected,
              f"undeclared detects {expected} on {sys.platform}")
    finally:
        os.environ.pop("SC_CAPACITY_BACKEND", None)
        if prior is not None:
            os.environ["SC_CAPACITY_BACKEND"] = prior

    step(8, "nvidia-smi is resolved where WSL actually keeps it")
    # Under WSL the driver's nvidia-smi is only in /usr/lib/wsl/lib, and only on
    # the interactive PATH because /etc/profile adds it. A systemd unit does not
    # read /etc/profile. Without that directory in the search list the gateway
    # resolves a bare name that works for a human in a shell and not for the
    # service — the exact failure `_tool` exists to prevent.
    WSL = "/usr/lib/wsl/lib/nvidia-smi"
    real_exists = os.path.exists
    os.path.exists = lambda p: p == WSL
    try:
        found = capacity._tool("nvidia-smi")
    finally:
        os.path.exists = real_exists
    check(found == WSL, f"_tool finds the WSL nvidia-smi ({found})")

    step(9, "the Metal arm reads the counter, not the one beside it")
    # IOAccelerator publishes "In use system memory" AND "In use system memory
    # (driver)". Recorded from slice 2026-09-29, idle with no model resident —
    # note the (driver) one reads 0, and that it comes FIRST in the dump, so an
    # unanchored search returns it and `foreign_mb` then reports that nobody is
    # using the GPU no matter who is.
    DUMP = ('      "PerformanceStatistics" = {"In use system memory (driver)"=0,'
            '"Alloc system memory"=1519484928,"Device Utilization %"=37,'
            '"Allocated PB Size"=29491200,"In use system memory"=249692160}')
    m = capacity._ACCEL_INUSE_RE.search(DUMP)
    check(m is not None, "the in-use counter is found")
    check(m and int(m.group(1)) == 249692160,
          f"it is the real one, not the (driver) 0 ({m and m.group(1)})")
    m = capacity._ACCEL_UTIL_RE.search(DUMP)
    check(m and int(m.group(1)) == 37, f"utilisation parses ({m and m.group(1)})")
    # And on this box, live: the two numbers must not be confused for each
    # other. `used` is dominated by the OS; the accelerator figure is not.
    if capacity.BACKEND == capacity.BACKEND_METAL:
        pct, inuse = capacity._accel_stats()
        print(f"  live: compute_pct={pct} device_inuse_mb={inuse}")
        check(inuse is not None, "the live counter reads")
        check(inuse is None or inuse < live["memory_used_mb"],
              f"accelerator in-use ({inuse}) < machine used "
              f"({live['memory_used_mb']}) — they are different numbers")

    step(10, "the CUDA arm asks for one card, and for what it parses")
    # Two failures this catches. `<device>/gpu-0` names ONE card, and an offer
    # sized off two cards' combined free memory fits in neither — so the query
    # must carry an --id. And the CSV has no header, so a field added to
    # _CUDA_FIELDS and not to the parser (or the reverse) shifts every column
    # after it: power draw silently read as a temperature, no error anywhere.
    recorded = {}

    class _Done:
        stdout = ADA_ROW + "\n"

    real_run = capacity.subprocess.run

    def _fake_run(argv, **kw):
        recorded["argv"] = argv
        recorded["kw"] = kw
        return _Done()

    capacity.subprocess.run = _fake_run
    try:
        reading = capacity._collect_cuda()
    finally:
        capacity.subprocess.run = real_run
    argv = recorded["argv"]
    print(f"  {argv}")
    check(f"--id={capacity.CUDA_INDEX}" in argv,
          f"the query names one card (--id={capacity.CUDA_INDEX})")
    check(capacity.CUDA_INDEX == "0",
          f"which defaults to card 0 ({capacity.CUDA_INDEX})")
    check("--format=csv,noheader,nounits" in argv,
          "MiB, no units, no header — the shape the parser expects")
    queried = [a for a in argv if a.startswith("--query-gpu=")]
    check(len(queried) == 1, f"exactly one --query-gpu ({len(queried)})")
    fields = queried[0].split("=", 1)[1].split(",")
    check(fields == list(capacity._CUDA_FIELDS),
          f"the query asks for _CUDA_FIELDS in order ({fields})")
    check(len(fields) == len(capacity._CUDA_FIELDS) == 5,
          f"five fields, which is what the parser unpacks ({len(fields)})")
    check(reading["memory_available_mb"] == 1124,
          f"and the row comes back parsed ({reading['memory_available_mb']})")
    check(recorded["kw"].get("timeout") == 10,
          f"the call is bounded ({recorded['kw'].get('timeout')})")

    print(f"\n{'='*60}")
    print(f"  {checks} checks run")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
