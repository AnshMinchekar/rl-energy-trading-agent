"""Summarise the per-solve diagnostics HNOptimizer writes to output/solver_diagnostics/.

Answers the question the 2026-08-16 smoke run could not: are the bus LPs actually
being solved to optimality, or are they stopping at the time limit with an optimality
gap wider than the arbitrage spread they are supposed to be optimising?

The reference numbers to judge against:
  * a full 146.7 kWh cycle at a 1-3 ct/kWh spread is worth ~EUR 1.5-4.4
  * node energy cost over a horizon is EUR 10-50
so a gap of even 0.05 on a EUR 30 objective (EUR 1.50 of slack) can swallow the entire
signal. A gap that is small in *absolute EUR* terms is what matters, not the ratio --
both are reported.

Usage:
    .venv/Scripts/python.exe analysis/summarize_solver_log.py
    .venv/Scripts/python.exe analysis/summarize_solver_log.py --dir output/solver_diagnostics
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def load(directory):
    records = []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn append from a killed run
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", default="output/solver_diagnostics")
    p.add_argument("--time-limit", type=float, default=300.0,
                   help="solver TimeLimit in seconds, for the 'hit the limit' count")
    p.add_argument("--target-gap", type=float, default=0.01,
                   help="configured MIPGap, for the 'exceeded tolerance' count")
    args = p.parse_args()

    recs = load(args.dir)
    if not recs:
        raise SystemExit(f"No solve records under {args.dir}/*.jsonl")

    print(f"{len(recs)} solves across {len({r['bus'] for r in recs})} buses\n")

    print("Termination conditions")
    for tc, k in Counter(r["termination"] for r in recs).most_common():
        print(f"  {tc:<20} {k:>6}  ({100 * k / len(recs):5.1f}%)")

    walls = sorted(r["wall_seconds"] for r in recs)
    print("\nWall time per solve (s)")
    for label, q in (("median", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0)):
        print(f"  {label:<8} {fmt(percentile(walls, q), 1)}")
    at_limit = sum(1 for w in walls if w >= args.time_limit * 0.98)
    print(f"  hit the {args.time_limit:g}s limit: {at_limit} "
          f"({100 * at_limit / len(walls):.1f}%)")

    gaps = sorted(r["mip_gap"] for r in recs if r.get("mip_gap") is not None)
    missing = len(recs) - len(gaps)
    print(f"\nRelative MIP gap  ({len(gaps)} reported, {missing} missing)")
    for label, q in (("median", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0)):
        print(f"  {label:<8} {fmt(percentile(gaps, q), 4)}")
    over = sum(1 for g in gaps if g > args.target_gap * 1.5)
    print(f"  above the {args.target_gap:g} tolerance: {over} "
          f"({100 * over / len(gaps):.1f}%)" if gaps else "  n/a")

    # The decisive number: gap in EUR, against a ~EUR 1.5-4.4 arbitrage cycle.
    abs_gaps = sorted(abs(r["objective"] - r["best_bound"]) for r in recs
                      if r.get("objective") is not None and r.get("best_bound") is not None)
    if abs_gaps:
        print("\nAbsolute gap (EUR of objective slack) -- compare to ~EUR 1.5-4.4 per cycle")
        for label, q in (("median", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0)):
            print(f"  {label:<8} {fmt(percentile(abs_gaps, q), 3)}")
        swamped = sum(1 for g in abs_gaps if g > 1.5)
        print(f"  slack exceeding a full cycle's value (EUR 1.5): {swamped} "
              f"({100 * swamped / len(abs_gaps):.1f}%)")

    print("\nPer bus")
    by_bus = defaultdict(list)
    for r in recs:
        by_bus[r["bus"]].append(r)
    print(f"  {'bus':>4} {'solves':>7} {'med s':>8} {'max s':>8} {'med gap':>9} "
          f"{'max gap':>9} {'non-optimal':>12}")
    for bus in sorted(by_bus):
        rs = by_bus[bus]
        w = sorted(x["wall_seconds"] for x in rs)
        g = sorted(x["mip_gap"] for x in rs if x.get("mip_gap") is not None)
        bad = sum(1 for x in rs if x["termination"] != "optimal")
        print(f"  {bus:>4} {len(rs):>7} {fmt(percentile(w, 0.5), 1):>8} "
              f"{fmt(percentile(w, 1.0), 1):>8} {fmt(percentile(g, 0.5), 4):>9} "
              f"{fmt(percentile(g, 1.0), 4):>9} {bad:>12}")

    # Older records (before the buy_flag disjunction was replaced by the convex
    # form) carry big_m/big_mm instead; report whichever the log actually has so a
    # pre- and post-change log can both be summarised.
    big_m = [r["big_m"] for r in recs if r.get("big_m") is not None]
    if big_m:
        big_mm = [r["big_mm"] for r in recs if r.get("big_mm") is not None]
        print(f"\nBig-M actually used: bigM {min(big_m):.1f}-{max(big_m):.1f}, "
              f"bigMM {min(big_mm):.1f}-{max(big_mm):.1f}  (was 1e7 / 1e6)")

    bins = [r["n_binaries"] for r in recs if r.get("n_binaries") is not None]
    if bins:
        pure_lp = sum(1 for b in bins if b == 0)
        print(f"\nBinaries per solve: {min(bins)}-{max(bins)}  "
              f"({pure_lp} of {len(bins)} solves are pure LPs)")


if __name__ == "__main__":
    main()
