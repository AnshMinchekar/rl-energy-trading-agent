"""Analyse the frozen-policy EVAL run in the full Mesa market.

output/sac/episode_logs.jsonl may contain several concatenated runs (training
passes + the eval pass). The eval pass is the contiguous block of records with
buffer_size == 0 (no learning). This script isolates that block, prints a
warm-up vs steady-state breakdown, and plots the learning-free performance.

Usage:
    python analysis/analyze_eval_run.py
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def eval_block(recs):
    """Return the last contiguous run of buffer_size==0 records (the eval pass)."""
    idx = [i for i, r in enumerate(recs) if r.get("buffer_size", 1) == 0]
    if not idx:
        return []
    # take the last contiguous stretch
    end = idx[-1]
    start = end
    s = set(idx)
    while start - 1 in s:
        start -= 1
    return recs[start:end + 1]


def col(ev, k):
    return np.array([r.get(k, np.nan) for r in ev], dtype=float)


def rolling(x, w=7):
    if len(x) < w:
        return np.array([]), np.array([])
    k = np.ones(w) / w
    return np.arange(w - 1, len(x)), np.convolve(x, k, mode="valid")


def main():
    recs = load("output/sac/episode_logs.jsonl")
    ev = eval_block(recs)
    if not ev:
        print("No eval (buffer_size==0) records found.")
        return

    pr = col(ev, "actual_profit_eur")
    soc = col(ev, "soc_avg") * 100
    spread = col(ev, "avg_sell_price") - col(ev, "avg_buy_price")
    days = np.arange(1, len(ev) + 1)
    WARMUP = 21

    print("=" * 64)
    print(f"FROZEN-POLICY EVAL — {len(ev)} days (real Mesa market)")
    print("=" * 64)
    def line(name, m):
        print(f"  {name:16s} profit/day {pr[m].mean():+.3f}  total {pr[m].sum():+7.2f}  "
              f"SOC {soc[m].mean():4.1f}%  spread {spread[m].mean():+.2f}ct  "
              f"profitable {int((pr[m] > 0).sum())}/{int(m.sum())}")
    allm = np.ones(len(ev), bool)
    line("ALL", allm)
    line(f"warm-up (1-{WARMUP})", days <= WARMUP)
    line(f"steady ({WARMUP+1}-{len(ev)})", days > WARMUP)
    line("last 30", days > len(ev) - 30)
    print("=" * 64)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle("Frozen SAC policy — evaluation in the full market (warm-up shaded)",
                 fontweight="bold", fontsize=14)

    ax[0].bar(days, pr, color=np.where(pr >= 0, "C2", "C3"), alpha=0.5, width=0.9)
    xs, ys = rolling(pr, 7)
    if len(xs):
        ax[0].plot(days[xs], ys, "k", lw=2, label="7-day mean")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axvspan(0, WARMUP, color="grey", alpha=0.12, label="warm-up")
    ax[0].set_title("Profit / day (EUR)"); ax[0].set_xlabel("day"); ax[0].legend()

    ax[1].plot(days, soc, "C0", lw=1.5)
    ax[1].axhspan(20, 85, color="green", alpha=0.06, label="target band")
    ax[1].axvspan(0, WARMUP, color="grey", alpha=0.12)
    ax[1].set_ylim(0, 100); ax[1].set_title("Average SOC (%)")
    ax[1].set_xlabel("day"); ax[1].legend()

    ax[2].plot(days, spread, "C4", lw=1.5)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].axvspan(0, WARMUP, color="grey", alpha=0.12)
    ax[2].set_title("Captured spread (sell - buy, ct/kWh)"); ax[2].set_xlabel("day")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("output/sac", exist_ok=True)
    out = "output/sac/eval_analysis.png"
    fig.savefig(out, dpi=120)
    print(f"Saved plot -> {out}")


if __name__ == "__main__":
    main()
