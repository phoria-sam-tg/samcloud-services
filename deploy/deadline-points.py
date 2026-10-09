#!/usr/bin/env python3
"""Regenerate #908's point sets from a gateway log. Reads, never writes.

    deploy/deadline-points.py ~/var/samcloud-services/logs/server.log
    deploy/deadline-points.py <log> --model qwen3.8:27b-mlx

`ollama/deadline_points.py` holds the published points as literal data with
provenance, because a service must not depend on a log file being present in a
deploy clone. This script is how that literal was produced and how it is
checked, so the curation is reproducible rather than asserted.

It prints the pairs too (`k = 0.375 x our_count / prompt_eval_count`), because
the same log lines carry both measurements and the density question is asked
against the same population as the timing one.

THE ONE JUDGEMENT, AND WHY IT IS NOT A THRESHOLD GUESS

A prefix-cache hit prefills at thousands of tok/s and a partial reuse at
115-142, against a cold band that is tight and self-identifying. Publishing a
cache-assisted observation as the slowest-at-a-size would be optimistic in the
direction that cuts requests, so they are excluded — and COUNTED in the output
rather than silently dropped, so the filter's effect is visible.

`--cold-max` exists so the cutoff is an argument rather than a constant. Its
default sits in the empty region between the cold band's top (104.9) and the
slowest partial reuse (115.7); nothing lies between, which is why the
classification is stable rather than a tuned number.
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict

LINE = re.compile(r"^(\S+ \S+).*Ollama timings for (\S+): (.*)$")
PREFILL = re.compile(r"prefill (\d+) tok in ([\d.]+)s")
DECODE = re.compile(r"decode (\d+) tok in ([\d.]+)s")
COUNTS = re.compile(r"our count (\d+) vs ollama (\d+)")

# The gateway's estimate divides serialised JSON by 1.5; a consumer's structural
# walk divides rendered content by 4. `k` is the consumer's over-count factor,
# so converting between them carries 1.5/4 and BOTH estimators' shapes. The
# ratio `est/real` is exact and ranks requests correctly on its own; `k` is
# reached only through this constant and is therefore a proxy, good to ~1.3%
# (`samclaude-admin`, #906). Printed as both so the exact one is available.
K_CONST = 1.5 / 4.0

BANDS = [(5000, 10000), (10000, 30000), (30000, 35000), (35000, 40000),
         (40000, 45000), (45000, 52000), (52000, 60000), (60000, 10 ** 9)]


def parse(path):
    rows = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            m = LINE.match(ln.strip())
            if not m:
                continue
            ts, model, rest = m.groups()
            pf, co = PREFILL.search(rest), COUNTS.search(rest)
            if not (pf and co):
                continue
            dec = DECODE.search(rest)
            real = int(pf.group(1))
            rows.append(dict(
                ts=ts, model=model, real=real, secs=float(pf.group(2)),
                est=int(co.group(1)), tps=real / float(pf.group(2)),
                reply=int(dec.group(1)) if dec else None,
                dsec=float(dec.group(2)) if dec else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--model", default="qwen3.8:27b-mlx")
    ap.add_argument("--cold-max", type=float, default=110.0,
                    help="tok/s above which a prefill is treated as "
                         "cache-assisted (default 110)")
    a = ap.parse_args()

    rows = [r for r in parse(a.log) if r["model"] == a.model]
    if not rows:
        print(f"no timings lines for {a.model} in {a.log}", file=sys.stderr)
        return 1
    print(f"{len(rows)} pairs for {a.model}\n")

    sized = [r for r in rows if r["real"] >= 5000]
    cold = [r for r in sized if r["tps"] < a.cold_max]
    warm = [r for r in sized if a.cold_max <= r["tps"] < 300]
    hot = [r for r in sized if r["tps"] >= 300]
    print(f"prefill classification at >= 5,000 real tokens")
    print(f"  cold                      n={len(cold):<3} "
          f"{min(r['tps'] for r in cold):.1f}-{max(r['tps'] for r in cold):.1f} tok/s"
          if cold else "  cold                      n=0")
    print(f"  partial prefix reuse      n={len(warm):<3} "
          + ", ".join(f"{r['real']}@{r['tps']:.0f}" for r in warm))
    print(f"  full/near-full cache hit  n={len(hot):<3} "
          + ", ".join(f"{r['real']}@{r['tps']:.0f}" for r in hot))

    print("\nPREFILL POINTS — slowest per band, at the N where it occurred")
    print(f"  {'N':>8} {'secs':>8} {'tok/s':>7} {'n':>3} {'band spread':>15}  dates")
    for lo, hi in BANDS:
        s = [r for r in cold if lo <= r["real"] < hi]
        if not s:
            # Printed rather than skipped: a band with no cold observation is a
            # HOLE in the point set and a consumer interpolating across it
            # should be able to see that it is doing so.
            print(f"  {lo:>8} {'-':>8} {'-':>7} {0:>3}   (no cold observation)")
            continue
        w = max(s, key=lambda r: r["secs"] / r["real"])
        t = [r["tps"] for r in s]
        print(f"  {w['real']:>8} {w['secs']:>8.1f} {w['real'] / w['secs']:>7.1f} "
              f"{len(s):>3} {min(t):>6.1f}-{max(t):<7.1f} "
              f"{','.join(sorted({r['ts'][:10] for r in s}))}")

    print("\nDECODE POINTS — by reply-length class, slowest per context band")
    print("  (a decode rate without its reply length is not comparable to "
          "another one)")
    dec = [r for r in rows if r["reply"] and r["dsec"]]
    for r in dec:
        r["dtps"] = r["reply"] / r["dsec"]
    for lab, pred in [("reply>=2000", lambda r: r["reply"] >= 2000),
                      ("reply<300 (overhead-dominated)",
                       lambda r: r["reply"] < 300)]:
        cls = [r for r in dec if pred(r)]
        if not cls:
            continue
        print(f"  {lab}")
        for lo, hi in [(0, 20000), (20000, 10 ** 9)]:
            s = [r for r in cls if lo <= r["real"] < hi]
            if not s:
                continue
            w = min(s, key=lambda r: r["dtps"])
            print(f"    ctx {w['real']:>7}  {w['dtps']:>5.1f} tok/s  "
                  f"reply {w['reply']:>5}  n={len(s):<3} "
                  f"reply range {min(r['reply'] for r in s)}-"
                  f"{max(r['reply'] for r in s)}")

    print("\nDENSITY — the same population, est against Ollama's own count")
    big = [r for r in rows if r["real"] >= 29000]
    if big:
        ks = sorted(K_CONST * r["est"] / r["real"] for r in big)
        print(f"  real >= 29,000: n={len(ks)}  k {ks[0]:.4f} - {ks[-1]:.4f}  "
              f"median {statistics.median(ks):.4f}")
        print(f"  the floor is a RUNNING MINIMUM and descends with n — it is a "
              f"property of n,\n  not of the box. Do not publish it as a bound "
              f"(#906).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
