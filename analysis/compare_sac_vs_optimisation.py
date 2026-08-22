"""Compare the SAC battery against an optimisation baseline on the same
physical unit (storage row 0, bus 12) over the same window.

Baselines (--baseline, default 'optimisation' for back-compat):
  * optimisation — the whole-node HNOptimizer/LP path (rows 1-4's method)
  * arbitrage    — the battery-only level-field LP with SAC's information
                   set and objective (optimization/arbitrage_optimizer.py)

Inputs are the two logs written by analysis/run_comparison_eval.py:

    output/comparison/eval_sac.jsonl
    output/comparison/eval_<baseline>.jsonl

Outputs:
  * output/comparison/sac_vs_optimisation.png — per-day profit / SOC / trades
  * stdout summary:
      - raw and terminal-SOC-adjusted totals (adjusted = cash profit +
        liquidation value of the terminal charge; both runs start at the same
        SOC, so the start term cancels in the comparison)
      - SAC / optimisation profit ratio vs the >75% success criterion
      - rows 1-4 cross-run sanity check (those units run 'optimisation' in
        BOTH runs, so their totals should nearly match; a large gap means the
        two runs' market outcomes diverged too much to compare row 0 fairly)

Guards: refuses to compare logs from different commits or dirty trees (the
meta records carry git_sha/git_dirty), and warns on a window mismatch.

Caveats that remain even with the guards passing (see
analysis/codebase_state_2026-08-21.md): the LP battery is co-optimised with
its node's load/EV/heat-pump while SAC is a standalone arbitrageur, and the
LP re-plans with ~12 h of actual future prices vs SAC's 6 h of DA forwards.
The historical fee-handicap (phantom 16.7 ct/kWh charging tax in the LP
objective) was fixed 2026-08-16.

Usage:
    python analysis/compare_sac_vs_optimisation.py [--baseline arbitrage]
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

SUCCESS_CRITERION = 0.75


def load(path):
    if not os.path.exists(path):
        sys.exit(f"Missing {path} — run analysis/run_comparison_eval.py first "
                 f"(--mode sac and the baseline mode).")
    meta, days, summaries = None, [], {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "meta":
                meta = rec
            elif rec["type"] == "day":
                days.append(rec)
            elif rec["type"] == "summary":
                summaries[rec["agent_id"]] = rec
    if meta is None:
        sys.exit(f"{path} has no meta record — incomplete run?")
    return meta, days, summaries


def series(days, agent_id):
    rows = sorted((r for r in days if r["agent_id"] == agent_id),
                  key=lambda r: r["date"])
    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in rows]
    return dates, rows


def rolling(y, w=7):
    if len(y) < w:
        return None, None
    k = np.ones(w) / w
    return np.arange(w - 1, len(y)), np.convolve(y, k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["optimisation", "arbitrage"],
                    default="optimisation",
                    help="which eval_<baseline>.jsonl to compare SAC against "
                         "(default: %(default)s)")
    baseline = ap.parse_args().baseline
    runs = [("SAC", "output/comparison/eval_sac.jsonl", "C0"),
            (baseline, f"output/comparison/eval_{baseline}.jsonl", "C1")]

    data = {}
    for name, path, color in runs:
        meta, days, summaries = load(path)
        data[name] = {"meta": meta, "days": days, "summaries": summaries,
                      "color": color}

    id0 = data["SAC"]["meta"]["storage0_id"]
    if data[baseline]["meta"]["storage0_id"] != id0:
        sys.exit("storage0_id differs between the two runs — different grid?")

    # --- Provenance guard ----------------------------------------------------
    # Both runs must come from the SAME commit, or rows 1-4 run different
    # optimizers and every cross-run number is uninterpretable (this exact
    # failure produced the invalid 2026-08 comparison: eval_sac.jsonl predated
    # the five LP fixes by a week and its per-agent margins were off by up to
    # 5x while the aggregate cancelled to +0.9%).
    shas = {name: data[name]["meta"].get("git_sha") for name, _, _ in runs}
    for name, sha in shas.items():
        if sha is None:
            sys.exit(f"The {name} run has no git_sha in its meta record — it "
                     f"predates provenance stamping and cannot be verified as "
                     f"same-code. Re-run analysis/run_comparison_eval.py for it.")
        if data[name]["meta"].get("git_dirty"):
            sys.exit(f"The {name} run was made from a dirty working tree "
                     f"({sha[:10]}+local edits) — commit first and re-run so "
                     f"the comparison is reproducible.")
    if shas["SAC"] != shas[baseline]:
        sys.exit(f"The two runs come from different commits "
                 f"(SAC {shas['SAC'][:10]} vs {baseline} "
                 f"{shas[baseline][:10]}) — rows 1-4 ran different "
                 f"optimizer code, so no cross-run number is meaningful. "
                 f"Re-run the stale side on the current commit.")
    print(f"Provenance OK: both runs from commit {shas['SAC'][:10]}, clean tree.")

    # --- Window guard --------------------------------------------------------
    n_days = {name: len(series(data[name]["days"], id0)[1]) for name, _, _ in runs}
    if n_days["SAC"] != n_days[baseline]:
        print(f"WARNING: window mismatch — SAC has {n_days['SAC']} day records "
              f"for row 0, {baseline} has {n_days[baseline]}. Totals "
              f"and the ratio below are NOT comparable; margins (ct/kWh) are "
              f"the only defensible cross-run numbers, and only on the "
              f"overlapping days.")

    # --- Plot: per-day profit / SOC / trades for the row-0 battery ----------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    baseline_label = ("HNOptimizer 'optimisation'" if baseline == "optimisation"
                      else "arbitrage-only LP")
    fig.suptitle(f"Storage row 0 (bus 12): SAC vs {baseline_label}, "
                 f"same window", fontsize=15, fontweight="bold")

    for name, _, _ in runs:
        d = data[name]
        dates, rows = series(d["days"], id0)
        if not rows:
            sys.exit(f"No day records for agent {id0} in the {name} run.")
        c = d["color"]

        profit = np.array([r["actual_profit_eur"] for r in rows])
        xs, ys = rolling(profit)
        axes[0].plot(dates, profit, color=c, alpha=0.18, lw=0.8,
                     label=name if xs is None else None)
        if xs is not None:
            axes[0].plot([dates[i] for i in xs], ys, color=c, lw=2.0, label=name)

        soc = np.array([r["soc_avg"] for r in rows]) * 100
        lo = np.array([r["soc_min"] for r in rows]) * 100
        hi = np.array([r["soc_max"] for r in rows]) * 100
        axes[1].plot(dates, soc, color=c, lw=1.5, label=name)
        axes[1].fill_between(dates, lo, hi, color=c, alpha=0.08)

        trades = np.array([r["trade_count_buy"] + r["trade_count_sell"]
                           for r in rows], dtype=float)
        xs, ys = rolling(trades)
        axes[2].plot(dates, trades, color=c, alpha=0.18, lw=0.8,
                     label=name if xs is None else None)
        if xs is not None:
            axes[2].plot([dates[i] for i in xs], ys, color=c, lw=2.0, label=name)

    axes[0].set_title("Profit / day (EUR, 7-day mean)", fontweight="bold")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[1].set_title("SOC (daily mean, shaded = min-max)", fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[2].set_title("Trades / day (7-day mean)", fontweight="bold")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = f"output/comparison/sac_vs_{baseline}.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}\n")

    # --- Head-to-head summary for row 0 -------------------------------------
    # Margin (ct per kWh sold) is the honest headline post-fee-fix: with the
    # LP's phantom charging tax gone the two sides trade near-equal volume,
    # so the contest is margin-vs-margin, and margin is also the number that
    # survives a window mismatch.
    print(f"{'':16s}{'days':>6s}{'profit EUR':>12s}{'sold kWh':>10s}"
          f"{'ct/kWh':>8s}{'term. SOC':>11s}{'liq. EUR':>10s}{'adjusted EUR':>14s}")
    adjusted = {}
    for name, _, _ in runs:
        d = data[name]
        _, rows = series(d["days"], id0)
        total = sum(r["actual_profit_eur"] for r in rows)
        sold = sum(r["energy_sold_kwh"] for r in rows)
        margin = total / sold * 100 if sold else float("nan")
        s = d["summaries"].get(id0, {})
        liq = s.get("terminal_liquidation_eur", 0.0)
        adjusted[name] = total + liq
        print(f"  {name:14s}{len(rows):6d}{total:12.2f}{sold:10.1f}"
              f"{margin:8.3f}{s.get('terminal_soc', float('nan')):11.3f}"
              f"{liq:10.2f}{adjusted[name]:14.2f}")

    opt = adjusted[baseline]
    sac = adjusted["SAC"]
    print()
    if opt > 1.0:
        ratio = sac / opt
        verdict = "MET" if ratio >= SUCCESS_CRITERION else "NOT met"
        print(f"SAC / {baseline} (terminal-SOC-adjusted): {ratio:.1%} "
              f"-> >{SUCCESS_CRITERION:.0%} criterion {verdict}")
    else:
        print(f"{baseline} adjusted profit is {opt:+.2f} EUR (near zero or "
              f"negative) — the ratio criterion is not meaningful; compare "
              f"absolute profits above. Note the fee-handicap caveat in the "
              f"plan file.")

    # --- Rows 1-4 cross-run sanity check ------------------------------------
    # Margins are compared alongside totals: per-agent MARGIN divergence is
    # what exposed the 2026-08 stale-run mismatch (margins off up to 5x while
    # the profit totals cancelled to within 1%).
    print("\nCross-run sanity check — always-'optimisation' units "
          "(totals AND margins should nearly match):")
    print(f"{'':4s}{'agent':>6s}{'bus':>5s}{'SAC-run EUR':>13s}"
          f"{'base-run EUR':>13s}{'diff':>9s}"
          f"{'SAC ct/kWh':>12s}{'base ct/kWh':>12s}")
    worst = 0.0
    for st in data["SAC"]["meta"]["storages"]:
        aid = st["agent_id"]
        if aid == id0:
            continue
        t, mg = {}, {}
        for name, _, _ in runs:
            _, rows = series(data[name]["days"], aid)
            t[name] = sum(r["actual_profit_eur"] for r in rows)
            sold = sum(r["energy_sold_kwh"] for r in rows)
            mg[name] = t[name] / sold * 100 if sold else float("nan")
        diff = t["SAC"] - t[baseline]
        denom = max(abs(t[baseline]), 1.0)
        worst = max(worst, abs(diff) / denom)
        # Margin divergence, floored at 0.5 ct/kWh so near-zero margins don't
        # explode the ratio.
        if np.isfinite(mg["SAC"]) and np.isfinite(mg[baseline]):
            m_denom = max(abs(mg[baseline]), 0.5)
            worst = max(worst, abs(mg["SAC"] - mg[baseline]) / m_denom)
        print(f"    {aid:6d}{st['bus']:5d}{t['SAC']:13.2f}"
              f"{t[baseline]:13.2f}{diff:+9.2f}"
              f"{mg['SAC']:12.3f}{mg[baseline]:12.3f}")
    if worst > 0.10:
        print(f"  WARNING: up to {worst:.0%} divergence (worst of per-agent "
              f"profit and margin) — the two runs' market outcomes differ "
              f"noticeably; treat the row-0 comparison with care.")
    else:
        print(f"  OK: max divergence {worst:.0%} — interaction effects are small.")


if __name__ == "__main__":
    main()
