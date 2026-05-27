"""
Comparison analysis: Monte Carlo vs TD Actor-Critic
Usage: python analysis/compare_algorithms.py
Reads:  output/mc/episode_logs.jsonl
        output/td/episode_logs.jsonl
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
MC_LOG = ROOT / "output" / "mc" / "episode_logs.jsonl"
TD_LOG = ROOT / "output" / "td" / "episode_logs.jsonl"

WINDOW = 7  # rolling mean window (episodes)
LAST_N = 30  # late-stage performance window


def load_log(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Drop large weight tensors — not needed for analysis
            rec.pop("theta", None)
            rec.pop("value_theta", None)
            records.append(rec)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("episode").reset_index(drop=True)
    return df


def rolling_mean(series: pd.Series, window: int = WINDOW) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def convergence_episode(profit: pd.Series, threshold: float = 0.5) -> int:
    """First episode where rolling profit exceeds threshold * max rolling profit."""
    rolled = rolling_mean(profit)
    target = rolled.max() * threshold
    above = rolled[rolled >= target]
    return int(above.index[0]) + 1 if not above.empty else -1


def price_spread(df: pd.DataFrame) -> float:
    traded = df[(df["avg_buy_price"] > 0) & (df["avg_sell_price"] > 0)]
    if traded.empty:
        return float("nan")
    return (traded["avg_sell_price"] - traded["avg_buy_price"]).mean()


def print_summary(mc: pd.DataFrame, td: pd.DataFrame) -> None:
    def stat(df, col, fn):
        try:
            return fn(df[col])
        except Exception:
            return float("nan")

    rows = [
        ("Total profit (EUR)",
         f"{stat(mc, 'actual_profit_eur', sum):.4f}",
         f"{stat(td, 'actual_profit_eur', sum):.4f}"),
        ("Best episode profit (EUR)",
         f"{stat(mc, 'actual_profit_eur', max):.4f}",
         f"{stat(td, 'actual_profit_eur', max):.4f}"),
        (f"Avg profit - last {LAST_N} ep (EUR)",
         f"{mc['actual_profit_eur'].tail(LAST_N).mean():.4f}",
         f"{td['actual_profit_eur'].tail(LAST_N).mean():.4f}"),
        ("Convergence episode",
         str(convergence_episode(mc["actual_profit_eur"])),
         str(convergence_episode(td["actual_profit_eur"]))),
        ("Avg trades/episode (buy)",
         f"{mc['trade_count_buy'].mean():.1f}",
         f"{td['trade_count_buy'].mean():.1f}"),
        ("Avg trades/episode (sell)",
         f"{mc['trade_count_sell'].mean():.1f}",
         f"{td['trade_count_sell'].mean():.1f}"),
        ("Price spread captured (ct/kWh)",
         f"{price_spread(mc):.4f}",
         f"{price_spread(td):.4f}"),
        ("Avg SOC",
         f"{mc['soc_avg'].mean():.4f}",
         f"{td['soc_avg'].mean():.4f}"),
        ("Time at floor - SOC<0.15 (% ep w/ soc_min<0.15)",
         f"{(mc['soc_min'] < 0.15).mean() * 100:.1f}%",
         f"{(td['soc_min'] < 0.15).mean() * 100:.1f}%"),
        ("Episodes available",
         str(len(mc)),
         str(len(td))),
    ]

    col_w = [40, 18, 18]
    header = f"{'Metric':<{col_w[0]}} {'Monte Carlo':>{col_w[1]}} {'TD Actor-Critic':>{col_w[2]}}"
    sep = "-" * sum(col_w)
    print("\n" + sep)
    print(header)
    print(sep)
    for label, mc_val, td_val in rows:
        print(f"{label:<{col_w[0]}} {mc_val:>{col_w[1]}} {td_val:>{col_w[2]}}")
    print(sep + "\n")


def plot_learning_curves(mc: pd.DataFrame, td: pd.DataFrame, ax_profit, ax_reward) -> None:
    for df, label, color in [(mc, "MC", "steelblue"), (td, "TD", "tomato")]:
        ep = df["episode"]
        ax_profit.plot(ep, df["actual_profit_eur"], alpha=0.25, color=color, linewidth=0.8)
        ax_profit.plot(ep, rolling_mean(df["actual_profit_eur"]),
                       label=f"{label} ({WINDOW}-ep mean)", color=color, linewidth=1.8)

        ax_reward.plot(ep, df["cumulative_reward"], alpha=0.25, color=color, linewidth=0.8)
        ax_reward.plot(ep, rolling_mean(df["cumulative_reward"]),
                       label=f"{label} ({WINDOW}-ep mean)", color=color, linewidth=1.8)

    ax_profit.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_profit.set_ylabel("Profit (€)")
    ax_profit.set_title("Episode Profit")
    ax_profit.legend(fontsize=8)

    ax_reward.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_reward.set_ylabel("Cumulative Reward")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_title("Episode Reward")
    ax_reward.legend(fontsize=8)


def plot_soc(mc: pd.DataFrame, td: pd.DataFrame, ax_mc, ax_td) -> None:
    y_min = min(mc["soc_min"].min(), td["soc_min"].min()) - 0.02
    y_max = max(mc["soc_max"].max(), td["soc_max"].max()) + 0.02

    for df, label, color, ax in [(mc, "Monte Carlo", "steelblue", ax_mc),
                                  (td, "TD Actor-Critic", "tomato", ax_td)]:
        ep = df["episode"]
        ax.fill_between(ep, df["soc_min"], df["soc_max"], alpha=0.2, color=color, label="min–max range")
        ax.plot(ep, df["soc_avg"], color=color, linewidth=1.5, label="avg SOC")
        ax.axhline(0.15, color="gray", linestyle=":", linewidth=0.8, label="floor (0.15)")
        ax.axhline(0.85, color="gray", linestyle="--", linewidth=0.8, label="ceiling (0.85)")
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"SOC Behaviour — {label}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("State of Charge")
        ax.legend(fontsize=8)


def plot_arbitrage(mc: pd.DataFrame, td: pd.DataFrame, ax_mc, ax_td) -> None:
    for df, label, cmap, ax in [(mc, "Monte Carlo", "Blues", ax_mc),
                                  (td, "TD Actor-Critic", "Reds", ax_td)]:
        traded = df[(df["avg_buy_price"] > 0) & (df["avg_sell_price"] > 0)]
        if traded.empty:
            ax.set_title(f"Arbitrage Quality — {label}\n(no episodes with both buy & sell)")
            continue

        sc = ax.scatter(
            traded["avg_buy_price"],
            traded["avg_sell_price"],
            c=traded["episode"],
            cmap=cmap,
            s=30,
            alpha=0.8,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="Episode")

        # Diagonal: sell == buy (zero spread)
        lo = min(traded["avg_buy_price"].min(), traded["avg_sell_price"].min())
        hi = max(traded["avg_buy_price"].max(), traded["avg_sell_price"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="break-even")

        ax.set_xlabel("Avg Buy Price (ct/kWh)")
        ax.set_ylabel("Avg Sell Price (ct/kWh)")
        ax.set_title(f"Arbitrage Quality — {label}\n(blue=early, dark=late)")
        ax.legend(fontsize=8)


def main() -> None:
    missing = [p for p in (MC_LOG, TD_LOG) if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing log file: {p}", file=sys.stderr)
        sys.exit(1)

    mc = load_log(MC_LOG)
    td = load_log(TD_LOG)

    print(f"Loaded MC: {len(mc)} episodes | TD: {len(td)} episodes")
    print_summary(mc, td)

    fig = plt.figure(figsize=(14, 16))
    fig.suptitle("Monte Carlo vs TD Actor-Critic — Algorithm Comparison", fontsize=14, y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Plot 1 — Learning curves (spans full width, 2 rows)
    ax_profit = fig.add_subplot(gs[0, :])
    ax_reward = fig.add_subplot(gs[1, :])
    plot_learning_curves(mc, td, ax_profit, ax_reward)

    # Plot 2 — SOC behaviour
    ax_soc_mc = fig.add_subplot(gs[2, 0])
    ax_soc_td = fig.add_subplot(gs[2, 1])
    plot_soc(mc, td, ax_soc_mc, ax_soc_td)

    # Plot 3 — Arbitrage quality
    ax_arb_mc = fig.add_subplot(gs[3, 0])
    ax_arb_td = fig.add_subplot(gs[3, 1])
    plot_arbitrage(mc, td, ax_arb_mc, ax_arb_td)

    out_path = ROOT / "analysis" / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
