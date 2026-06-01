"""Analyse the SAC storage agent's training logs.

Reads output/sac/episode_logs.jsonl (and optionally output/mc/ and output/td/
for comparison) and produces:
  * learning curves (profit, reward, alpha, entropy)
  * SOC behaviour band (avg with min/max envelope)
  * arbitrage quality (avg buy vs sell price over episodes)
  * a printed summary table

Safe to run while training is still in progress — it just plots what exists.

Usage:
    python analysis/analyze_sac.py
"""
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def col(records, key, default=np.nan):
    return np.array([r.get(key, default) for r in records], dtype=float)


def rolling(x, w=7):
    if len(x) < 1:
        return x
    w = min(w, len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def summarise(name, records):
    if not records:
        print(f"  [{name}] no records")
        return
    profit = col(records, "actual_profit_eur")
    soc_avg = col(records, "soc_avg")
    soc_min = col(records, "soc_min")
    buy = col(records, "avg_buy_price")
    sell = col(records, "avg_sell_price")
    n_buy = col(records, "trade_count_buy")
    n_sell = col(records, "trade_count_sell")
    last = profit[-30:] if len(profit) >= 30 else profit
    traded = (sell > 0) & (buy > 0)
    spread = float(np.mean((sell - buy)[traded])) if np.any(traded) else float("nan")
    time_at_floor = float(np.mean(soc_min < 0.15)) * 100

    print(f"\n  [{name}] episodes: {len(records)}")
    print(f"    Total profit (EUR):        {np.nansum(profit):>10.3f}")
    print(f"    Best episode profit (EUR): {np.nanmax(profit):>10.3f}")
    print(f"    Avg profit last-30 (EUR):  {np.nanmean(last):>10.3f}")
    print(f"    Avg trades/ep (buy/sell):  {np.nanmean(n_buy):>5.1f} / {np.nanmean(n_sell):.1f}")
    print(f"    Avg captured spread (ct):  {spread:>10.3f}")
    print(f"    Avg SOC:                   {np.nanmean(soc_avg)*100:>9.1f}%")
    print(f"    Time at floor (<15%):      {time_at_floor:>9.1f}%")


def main():
    sac = load_jsonl("output/sac/episode_logs.jsonl")
    mc = load_jsonl("output/mc/episode_logs.jsonl")
    td = load_jsonl("output/td/episode_logs.jsonl")

    print("=" * 60)
    print("STORAGE RL TRAINING SUMMARY")
    print("=" * 60)
    summarise("SAC", sac)
    summarise("MC", mc)
    summarise("TD", td)
    print("=" * 60)

    if not HAVE_MPL or not sac:
        if not sac:
            print("No SAC logs yet — skipping plots.")
        return

    ep = col(sac, "episode")
    profit = col(sac, "actual_profit_eur")
    reward = col(sac, "cumulative_reward")
    alpha = col(sac, "alpha")
    entropy = col(sac, "entropy")
    soc_avg = col(sac, "soc_avg")
    soc_min = col(sac, "soc_min")
    soc_max = col(sac, "soc_max")
    buy = col(sac, "avg_buy_price")
    sell = col(sac, "avg_sell_price")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.plot(ep, profit, alpha=0.3, label="profit/ep")
    if len(profit) >= 7:
        ax.plot(ep[6:], rolling(profit, 7), color="C0", label="7-ep mean")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Episode profit (EUR)"); ax.set_xlabel("episode"); ax.legend()

    ax = axes[0, 1]
    ax.plot(ep, reward, alpha=0.3)
    if len(reward) >= 7:
        ax.plot(ep[6:], rolling(reward, 7), color="C1")
    ax.set_title("Episode reward (shaped)"); ax.set_xlabel("episode")

    ax = axes[1, 0]
    ax.plot(ep, soc_avg * 100, color="C2", label="avg SOC")
    ax.fill_between(ep, soc_min * 100, soc_max * 100, alpha=0.2, color="C2", label="min-max")
    ax.axhspan(20, 85, color="green", alpha=0.05, label="soft band")
    ax.set_ylim(0, 100); ax.set_title("SOC behaviour (%)"); ax.set_xlabel("episode"); ax.legend()

    ax = axes[1, 1]
    ax.plot(ep, buy, color="C3", label="avg buy price")
    ax.plot(ep, sell, color="C0", label="avg sell price")
    ax.set_title("Arbitrage: buy vs sell price (ct/kWh)"); ax.set_xlabel("episode"); ax.legend()

    ax = axes[2, 0]
    ax.plot(ep, alpha, color="C4")
    ax.set_title("Entropy temperature alpha"); ax.set_xlabel("episode")

    ax = axes[2, 1]
    ax.plot(ep, entropy, color="C5")
    ax.set_title("Policy entropy"); ax.set_xlabel("episode")

    fig.tight_layout()
    out = "output/sac/analysis.png"
    fig.savefig(out, dpi=110)
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
