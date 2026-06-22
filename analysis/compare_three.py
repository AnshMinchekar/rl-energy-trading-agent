"""Compare SAC vs MC vs TD across four metrics in a single PNG.

Episodes are NOT directly comparable across algorithms: a SAC episode is 24 h
(update_frequency=96) while MC/TD episodes are 12 h (update_frequency=48). All
three, however, cover the same calendar window (Jan 1 - Mar 30 2021). So we
aggregate every algorithm to a common PER-DAY basis before plotting:

  * trades/day  = sum of (buys + sells) over the day
  * profit/day  = sum of realised profit (EUR) over the day
  * reward/day  = sum of shaped reward over the day
  * SOC         = mean soc_avg over the day (+ daily min/max envelope)

Each series is shown raw (faint) with a 7-day rolling mean (bold).

Note: the reward panel is only loosely comparable across algorithms — each uses
a different shaped reward (SAC mark-to-market x10; MC/TD their own). Profit and
SOC are the fair head-to-head comparisons.

Usage:
    python analysis/compare_three.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGOS = [
    ("SAC", "output/sac/episode_logs.jsonl", "C0"),
    ("MC",  "output/mc/episode_logs.jsonl",  "C1"),
    ("TD",  "output/td/episode_logs.jsonl",  "C2"),
]


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def per_day(recs):
    """Aggregate episode records to one row per calendar day."""
    days = defaultdict(list)
    for r in recs:
        day = r["timestamp"][:10]          # 'YYYY-MM-DD'
        days[day].append(r)
    rows = []
    for day in sorted(days):
        group = days[day]
        def s(k):
            return float(np.nansum([g.get(k, np.nan) for g in group]))
        def m(k):
            return float(np.nanmean([g.get(k, np.nan) for g in group]))
        rows.append({
            "date": datetime.strptime(day, "%Y-%m-%d"),
            "trades": s("trade_count_buy") + s("trade_count_sell"),
            "profit": s("actual_profit_eur"),
            "reward": s("cumulative_reward"),
            "soc_avg": m("soc_avg") * 100,
            "soc_min": float(np.nanmin([g.get("soc_min", np.nan) for g in group])) * 100,
            "soc_max": float(np.nanmax([g.get("soc_max", np.nan) for g in group])) * 100,
        })
    return rows


def rolling(x, w=7):
    if len(x) < w:
        return None, None
    k = np.ones(w) / w
    return np.arange(w - 1, len(x)), np.convolve(x, k, mode="valid")


def select_clean(name, recs, ref_window=None):
    """The SAC log accumulates multiple appended runs (from-scratch training +
    several frozen-eval passes over different years), so summing it raw triple-
    counts dates. Keep only frozen-eval records (buffer_size == 0), restrict to
    the same calendar window as MC/TD, and dedupe by date keeping the latest
    appended run. MC/TD are single clean runs and pass through unchanged."""
    if name != "SAC":
        return recs
    recs = [r for r in recs if r.get("buffer_size", 1) == 0]
    if ref_window:
        lo, hi = ref_window
        recs = [r for r in recs if lo <= r["timestamp"][:10] <= hi]
    by_date = {}
    for r in recs:                         # later record overwrites earlier
        by_date[r["timestamp"][:10]] = r
    return [by_date[d] for d in sorted(by_date)]


def main():
    raw = {name: load(path) for name, path, _ in ALGOS}
    ref = next((r for r in (raw.get("MC"), raw.get("TD")) if r), None)
    ref_window = None
    if ref:
        ds = sorted(r["timestamp"][:10] for r in ref)
        ref_window = (ds[0], ds[-1])
    data = {name: per_day(select_clean(name, raw[name], ref_window))
            for name in raw}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    fig.suptitle("Storage RL algorithm comparison (per-day, same Jan-Mar 2021 window)",
                 fontsize=15, fontweight="bold")

    def panel(ax, key, title, ylabel, band=False):
        for name, _, c in ALGOS:
            rows = data[name]
            if not rows:
                continue
            dates = [r["date"] for r in rows]
            y = np.array([r[key] for r in rows], dtype=float)
            ax.plot(dates, y, color=c, alpha=0.18, lw=0.8)
            xs, ys = rolling(y, 7)
            if xs is not None:
                ax.plot([dates[i] for i in xs], ys, color=c, lw=2.0, label=name)
            if band:
                lo = np.array([r["soc_min"] for r in rows])
                hi = np.array([r["soc_max"] for r in rows])
                ax.fill_between(dates, lo, hi, color=c, alpha=0.06)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("date")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=30)

    panel(axes[0], "trades", "Trade frequency (trades / day)", "trades / day")
    panel(axes[1], "soc_avg", "SOC behaviour (daily mean, shaded = min-max)", "SOC (%)", band=True)
    axes[1].axhspan(20, 85, color="grey", alpha=0.07)
    axes[1].set_ylim(0, 100)
    panel(axes[2], "profit", "Profit / day (realised, EUR)", "EUR / day")
    axes[2].axhline(0, color="k", lw=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "analysis/algorithm_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}\n")

    print("            days  trades/day  avg_SOC%  profit/day  total_profit")
    for name, _, _ in ALGOS:
        rows = data[name]
        if not rows:
            continue
        tr = np.array([r["trades"] for r in rows])
        pr = np.array([r["profit"] for r in rows])
        so = np.array([r["soc_avg"] for r in rows])
        print(f"  {name:4s}    {len(rows):4d}    {np.mean(tr):8.1f}   {np.mean(so):6.1f}   "
              f"{np.mean(pr):8.3f}   {np.sum(pr):9.3f}")


if __name__ == "__main__":
    main()
