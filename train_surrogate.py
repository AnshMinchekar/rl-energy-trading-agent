# -*- coding: utf-8 -*-
"""
Train the storage SAC policy on the fast surrogate environment, then save it.

This is the scalable training path: the surrogate (mesa_model/storage_env.py) has
no Gurobi market solve, so it runs ~10^5-10^6 steps in minutes instead of the
weeks the live market would take. Physical parameters and the price series are
read from a constructed LEM model so they are byte-identical to the live agent;
state and reward come from the shared mesa_model/storage_logic.py.

Train/eval split: the policy trains on a wide spot-price window (default the two
full years 2021-2022, env vars SAC_TRAIN_START/SAC_TRAIN_END) read straight from
spot_price.csv, bypassing the config window. The held-out evaluation quarter is
config.yaml's simulation window (default Q1 2023) and must not overlap the
training range, so the live eval is genuinely out-of-sample.

Usage:
    # 1. Train the brain on 2 years of prices (fast, no Gurobi):
    SAC_SURROGATE_STEPS=300000 python train_surrogate.py

    # 2. Evaluate the frozen policy on the held-out quarter in the full Mesa
    #    market (config.yaml window, live Gurobi solve, deterministic):
    SAC_LOAD_POLICY=output/sac/surrogate_policy.pt SAC_EVAL=1 python main.py
"""

import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import data.config.config as _cfgmod          # for the scenario_data directory
from mesa_model.model import model           # construction only — no market solve
from mesa_model.sac import SACLearner
from mesa_model.storage_env import StorageArbitrageEnv

TOTAL_STEPS = int(os.environ.get("SAC_SURROGATE_STEPS", "300000"))
LOG_EVERY = int(os.environ.get("SAC_LOG_EVERY", "5000"))
OUT_PATH = os.environ.get("SAC_POLICY_OUT", "output/sac/surrogate_policy.pt")
SEED = 42

# Train the policy on a wide spot-price window (default: the two full years
# 2021-2022). The held-out evaluation quarter lives in config.yaml's simulation
# window (default Q1 2023) and must NOT overlap this range — see the live eval
# in main.py with SAC_EVAL=1.
TRAIN_START = os.environ.get("SAC_TRAIN_START", "01.01.2021 00:00")
TRAIN_END = os.environ.get("SAC_TRAIN_END", "31.12.2022 23:45")


def _learning_agent(m):
    for a in m.agents:
        if getattr(a, "method", None) == "learning" and hasattr(a, "learner"):
            return a
    raise RuntimeError("No learning storage agent found in the model.")


def _price_series(m, start=TRAIN_START, end=TRAIN_END):
    """Spot series over an arbitrary [start, end] window, byte-identical to the
    ext-grid agent's energy_price (mesa_model/agents.py) but NOT limited to the
    config simulation window — config windows every time-stamped CSV at load time
    (data/config/config.py), so the model's own energy_price only spans the eval
    quarter. Reading the raw CSV here lets the surrogate train on years of prices
    while the live eval runs on a held-out window. Returns (prices, timestamps)."""
    path = os.path.join(os.path.dirname(_cfgmod.__file__),
                        "scenario_data", "spot_price.csv")
    df = pd.read_csv(path, sep=";", decimal=",")
    df["time"] = pd.to_datetime(df["time"], format="%d.%m.%Y %H:%M", errors="coerce")
    # Same unit scaling the ext-grid agent applies (agents.py:170).
    df["price"] = df["price (Ct/kWh)"].astype(float) / (100 / m.sref)
    lo = pd.to_datetime(start, format="%d.%m.%Y %H:%M")
    hi = pd.to_datetime(end, format="%d.%m.%Y %H:%M")
    df = df[(df["time"] >= lo) & (df["time"] <= hi)]
    df = df.dropna(subset=["time", "price"]).sort_values("time")
    if df.empty:
        raise RuntimeError(f"No spot prices in training window {start} .. {end}")
    return df["price"].to_numpy(dtype=float), list(df["time"])


def build_env(m):
    agent = _learning_agent(m)
    prices, timestamps = _price_series(m)
    dt_h = m.timestep.seconds / 3600.0
    env = StorageArbitrageEnv(
        prices=prices, timestamps=timestamps,
        capacity=agent.capacity, max_power_kw=agent.max_power,
        efficiency=agent.efficiency, discharge_per_step=agent.discharge, dt_h=dt_h,
        soc_start=agent.soc, soc_floor=agent.soc_floor, soc_ceiling=agent.soc_ceiling,
        margin_buy=getattr(m, "market_price_margin_buy", 1.0),
        margin_sell=getattr(m, "market_price_margin_sell", 0.3),
        soc_shaping_weight=agent.soc_shaping_weight,
        soc_shaping_anneal=agent.soc_shaping_anneal,
        episode_len=96, random_start=True, random_soc=True, seed=SEED,
    )
    return env, agent


def make_learner(agent):
    return SACLearner(
        state_dim=agent.state_dim, gamma=agent.gamma, tau=0.005, lr=3e-4,
        target_entropy=-1.0, buffer_size=200_000, batch_size=256,
        warmup_steps=2_000, actor_update_every=2, updates_per_step=1,
        reward_scale=agent.reward_scale, alpha_min=0.05, seed=SEED,
    )


def evaluate(env, learner, n_steps=8000):
    """Deterministic profit over a contiguous window (true performance, no
    exploration). Uses a fresh fixed-start env pass."""
    eval_env = StorageArbitrageEnv(
        prices=env.prices, timestamps=env.timestamps, capacity=env.capacity,
        max_power_kw=env.max_power_kw, efficiency=env.efficiency,
        discharge_per_step=env.discharge, dt_h=env.dt_h, soc_start=env.soc_start,
        soc_floor=env.soc_floor, soc_ceiling=env.soc_ceiling,
        margin_buy=env.margin_buy, margin_sell=env.margin_sell,
        soc_shaping_weight=0.0, soc_shaping_anneal=env.soc_shaping_anneal,
        episode_len=n_steps, random_start=False, random_soc=False, seed=0,
    )
    s = eval_env.reset()
    profit = 0.0
    socs = []
    for _ in range(n_steps):
        a = learner.select_action(s, deterministic=True)
        s, _, done, info = eval_env.step(a, learner.total_env_steps)
        profit += info["profit_eur"]
        socs.append(info["soc"])
        if done:
            break
    return profit, float(np.mean(socs)), float(np.min(socs)), float(np.max(socs))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    env, agent = build_env(model)
    learner = make_learner(agent)
    print(f"Surrogate: {len(env.prices)} price points "
          f"({env.timestamps[0]} .. {env.timestamps[-1]}) "
          f"| capacity={env.capacity:.0f} kWh "
          f"| max_power={env.max_power_kw:.0f} kW | eff={env.efficiency:.3f} "
          f"| discharge/step={env.discharge:.5f}")
    print(f"Training for {TOTAL_STEPS} steps...\n", flush=True)

    state = env.reset()
    ep_profit = ep_reward = 0.0
    recent_profit, recent_reward = [], []
    t0 = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        a = learner.select_action(state)
        next_state, reward, done, info = env.step(a, learner.total_env_steps)
        learner.push(state, a, reward, next_state, 0.0)
        learner.learn()
        state = next_state
        ep_profit += info["profit_eur"]
        ep_reward += reward

        if done:
            recent_profit.append(ep_profit)
            recent_reward.append(ep_reward)
            recent_profit = recent_profit[-200:]
            recent_reward = recent_reward[-200:]
            ep_profit = ep_reward = 0.0
            state = env.reset()

        if step % LOG_EVERY == 0:
            rate = step / (time.time() - t0)
            ev_profit, ev_soc, ev_lo, ev_hi = evaluate(env, learner)
            print(f"step {step:>7d} | {rate:6.0f} st/s | alpha {learner.alpha.item():.3f} "
                  f"| train profit/day {np.mean(recent_profit) if recent_profit else 0:+.2f} "
                  f"| EVAL profit {ev_profit:+8.2f} EUR  SOC {ev_soc*100:4.1f}% "
                  f"[{ev_lo*100:.0f}-{ev_hi*100:.0f}]", flush=True)

    learner.save(OUT_PATH)
    ev_profit, ev_soc, ev_lo, ev_hi = evaluate(env, learner)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved policy -> {OUT_PATH}")
    print(f"Final deterministic eval: profit {ev_profit:+.2f} EUR | "
          f"SOC mean {ev_soc*100:.1f}% range [{ev_lo*100:.0f}-{ev_hi*100:.0f}]")


if __name__ == "__main__":
    main()
