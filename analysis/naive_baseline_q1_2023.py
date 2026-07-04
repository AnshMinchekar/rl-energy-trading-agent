# -*- coding: utf-8 -*-
"""Offline arbitrage baselines on the held-out Q1 2023 spot series.

Context: the live SAC_EVAL run (main.py) already produces the RL result
(learning agent) and the LP-optimiser result (optimisation agents) on this same
window. This script adds the two reference points that run doesn't:
  * a perfect-foresight daily-arbitrage CEILING (independent of the market LP), and
  * a simple causal threshold FLOOR (what a non-foresight rule achieves).

Battery physics are imported from mesa_model.storage_logic so they match the
agent exactly (corrected 0.13%/day self-discharge, 95% one-way efficiency). Every
trade is charged the same 0.2 ct/kWh bid-ask spread the live agent crosses, and
profit is realized cashflow -- directly comparable to the agents' actual_profit.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesa_model.storage_logic import provide_power_kwh, soc_transition

# --- battery params (identical to the 147 kWh learning agent) ---
CAP = 147.0          # kWh
PMAX = 73.0          # kW
ETA = 0.95           # one-way efficiency
DT_H = 0.25          # 15-min step
SELF_DISCH_DAY = 0.13 / 100          # corrected: 0.13%/day
DISCHARGE = (1 - SELF_DISCH_DAY) ** (DT_H / 24.0)   # per-step retention
SPREAD = 0.2         # ct/kWh, each side (matches action_to_bid)
SOC0 = 0.40

START, END = "01.01.2023 00:00", "30.03.2023 23:45"


def load_prices():
    path = os.path.join("data", "config", "scenario_data", "spot_price.csv")
    df = pd.read_csv(path, sep=";", decimal=",")
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M", errors="coerce")
    df["price"] = df["price (Ct/kWh)"].astype(float)   # sref=100 -> scale 1.0
    lo, hi = pd.to_datetime(START, format="%d.%m.%Y %H:%M"), pd.to_datetime(END, format="%d.%m.%Y %H:%M")
    df = df[(df["time"] >= lo) & (df["time"] <= hi)].dropna(subset=["time", "price"]).sort_values("time")
    return df["price"].to_numpy(float), list(df["time"])


def simulate(prices, action_fn, **kw):
    """Run a policy (action in [-1,1], +charge/-discharge) over the series.
    Returns (profit_eur, n_cycles_equiv, soc_series). Cashflow charged the spread."""
    soc = SOC0
    cash = 0.0
    bought_tot = sold_tot = 0.0
    socs = []
    for t, p in enumerate(prices):
        socs.append(soc)
        a = float(np.clip(action_fn(t, p, soc, prices, **kw), -1.0, 1.0))
        max_sold, max_bought = provide_power_kwh(soc, CAP, PMAX, DT_H, ETA)
        bought = a * max_bought if a > 0 else 0.0
        sold = -a * max_sold if a < 0 else 0.0
        cash += (sold * (p - SPREAD) - bought * (p + SPREAD)) / 100.0
        bought_tot += bought; sold_tot += sold
        soc = soc_transition(soc, bought, sold, CAP, ETA, DISCHARGE)
    cycles = (bought_tot + sold_tot) / (2 * CAP)
    return cash, cycles, np.array(socs)


# --- causal FLOOR: trailing-window percentile threshold ---
def causal_threshold(t, p, soc, prices, win=96, lo_pct=33, hi_pct=67):
    if t < win:
        return 0.0
    hist = prices[t - win:t]
    if p <= np.percentile(hist, lo_pct):
        return 1.0     # cheap -> charge
    if p >= np.percentile(hist, hi_pct):
        return -1.0    # expensive -> discharge
    return 0.0


# --- foresight CEILING: per-day greedy threshold, best split searched ---
def foresight_daily(prices, timestamps):
    """Greedy: each calendar day, charge in the cheapest slots / discharge in the
    dearest, choosing the split fraction that maximizes that day's cashflow.
    Independent of the market LP; a clean single-battery arbitrage ceiling."""
    days = pd.Series(timestamps).dt.date.to_numpy()
    best_total = 0.0
    # reuse simulate() with a precomputed per-step action map
    actions = np.zeros(len(prices))
    for d in np.unique(days):
        idx = np.where(days == d)[0]
        dp = prices[idx]
        best_day, best_act = -1e9, None
        for frac in (0.10, 0.15, 0.20, 0.25, 0.33):
            lo_thr = np.quantile(dp, frac)
            hi_thr = np.quantile(dp, 1 - frac)
            act = np.where(dp <= lo_thr, 1.0, np.where(dp >= hi_thr, -1.0, 0.0))
            # cheap daily cashflow estimate (ignores SOC coupling) to pick frac
            val = np.sum(np.where(act < 0, dp - SPREAD, 0) - np.where(act > 0, dp + SPREAD, 0))
            if val > best_day:
                best_day, best_act = val, act
        actions[idx] = best_act
    return simulate(prices, lambda t, p, soc, pr: actions[t])


def main():
    prices, ts = load_prices()
    print(f"Q1 2023: {len(prices)} steps  {ts[0]} .. {ts[-1]}")
    print(f"price ct/kWh: mean {prices.mean():.2f}  min {prices.min():.2f}  max {prices.max():.2f}")
    print(f"per-step retention {DISCHARGE:.8f} | round-trip eff {ETA**2:.3f} | spread {SPREAD} ct/side\n")

    cash, cyc, socs = simulate(prices, causal_threshold)
    print(f"CAUSAL threshold (floor):    {cash:+8.2f} EUR | {cyc:5.1f} cycles | SOC mean {socs.mean():.2f} [{socs.min():.2f}-{socs.max():.2f}]")

    cash, cyc, socs = foresight_daily(prices, ts)
    print(f"FORESIGHT daily (ceiling):   {cash:+8.2f} EUR | {cyc:5.1f} cycles | SOC mean {socs.mean():.2f} [{socs.min():.2f}-{socs.max():.2f}]")

    print("\n(RL and LP-optimiser numbers come from the live SAC_EVAL run, same window.)")


if __name__ == "__main__":
    main()
