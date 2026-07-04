# -*- coding: utf-8 -*-
"""Sanity check: run a saved SAC policy deterministically on the held-out
Q1 2023 spot series *inside the surrogate env*. Isolates policy quality from
live-market fills -- if this is positive/mid-band but the live eval floor-hugs,
the problem is the market (fills), not the policy. See memory
self-discharge-bug-and-fill-fix.

Usage:
    python analysis/eval_policy_on_2023_surrogate.py [path/to/policy.pt]
    (default: output/sac/surrogate_policy.pt)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from mesa_model.storage_env import StorageArbitrageEnv
from mesa_model.storage_logic import STATE_DIM
from mesa_model.sac import SACLearner

POLICY = sys.argv[1] if len(sys.argv) > 1 else "output/sac/surrogate_policy.pt"
CAP, PMAX, ETA = 147.0, 73.0, 0.95
DISCH = (1 - 0.13 / 100) ** (0.25 / 24.0)     # corrected self-discharge
MARGIN_BUY, MARGIN_SELL = 1.0, 0.3            # real grid margins


def load_prices():
    path = os.path.join("data", "config", "scenario_data", "spot_price.csv")
    df = pd.read_csv(path, sep=";", decimal=",")
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M", errors="coerce")
    df["price"] = df["price (Ct/kWh)"].astype(float)     # sref=100 -> scale 1.0
    lo, hi = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-03-30 23:45")
    df = df[(df.time >= lo) & (df.time <= hi)].dropna(subset=["time", "price"]).sort_values("time")
    return df["price"].to_numpy(float), list(df["time"])


def main():
    prices, ts = load_prices()
    env = StorageArbitrageEnv(
        prices=prices, timestamps=ts, capacity=CAP, max_power_kw=PMAX, efficiency=ETA,
        discharge_per_step=DISCH, dt_h=0.25, soc_start=0.40,
        margin_buy=MARGIN_BUY, margin_sell=MARGIN_SELL, soc_shaping_weight=0.0,
        episode_len=len(prices) - 2, random_start=False, random_soc=False, seed=0)
    learner = SACLearner(state_dim=STATE_DIM, gamma=0.996, tau=0.005, lr=3e-4, target_entropy=-1.0,
                         buffer_size=1000, batch_size=256, warmup_steps=0, actor_update_every=2,
                         updates_per_step=1, reward_scale=10.0, alpha_min=0.05, seed=0)
    learner.load(POLICY)

    s = env.reset()
    profit = 0.0
    socs, acts = [], []
    for _ in range(len(prices) - 3):
        a = learner.select_action(s, deterministic=True)
        s, r, done, info = env.step(a, learner.total_env_steps)
        profit += info["profit_eur"]
        socs.append(info["soc"])
        acts.append(float(a))
        if done:
            break
    socs, acts = np.array(socs), np.array(acts)
    print(f"policy: {POLICY}")
    print(f"Q1 2023 surrogate eval (real grid margins {MARGIN_BUY}/{MARGIN_SELL}):")
    print(f"  profit    = {profit:+.2f} EUR")
    print(f"  SOC mean  = {socs.mean():.3f}  range [{socs.min():.3f}-{socs.max():.3f}]")
    print(f"  actions   = mean {acts.mean():+.3f} | charge {np.mean(acts>0.05):.0%} "
          f"discharge {np.mean(acts<-0.05):.0%} idle {np.mean(np.abs(acts)<=0.05):.0%}")
    print(f"  reference: causal floor +358, foresight ceiling +577 EUR")


if __name__ == "__main__":
    main()
