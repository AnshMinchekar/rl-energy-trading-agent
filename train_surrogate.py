# -*- coding: utf-8 -*-
"""
Train the storage SAC policy on the fast surrogate environment, then save it.

This is the scalable training path: the surrogate (mesa_model/storage_env.py) has
no Gurobi market solve, so it runs ~10^5-10^6 steps in minutes instead of the
weeks the live market would take. Physical parameters and the price series are
read from a constructed LEM model so they are byte-identical to the live agent;
state and reward come from the shared mesa_model/storage_logic.py.

Train/validation/test split:
  * TRAIN      (SAC_TRAIN_START/END, default 2021-01-01 .. 2022-09-30) — SAC
    interacts and learns here (random episode starts).
  * VALIDATION (SAC_VAL_START/END, default 2022-10-01 .. 2022-12-31) — never
    trained on; a deterministic pass runs every LOG_EVERY steps and the policy
    with the BEST validation profit is checkpointed (SAC can degrade late in
    training, so "last" is not "best").
  * TEST       config.yaml's simulation window (default Q1 2023) — the live
    Mesa-market eval; must not overlap the other two.

Usage:
    # 1. Train the brain (fast, no Gurobi):
    SAC_SURROGATE_STEPS=300000 python train_surrogate.py

    # 2. Evaluate the frozen best-validation policy on the held-out quarter in
    #    the full Mesa market (config.yaml window, live Gurobi solve):
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
SEED = int(os.environ.get("SAC_SEED", "42"))

TRAIN_START = os.environ.get("SAC_TRAIN_START", "01.01.2021 00:00")
TRAIN_END = os.environ.get("SAC_TRAIN_END", "30.09.2022 23:45")
VAL_START = os.environ.get("SAC_VAL_START", "01.10.2022 00:00")
VAL_END = os.environ.get("SAC_VAL_END", "31.12.2022 23:45")


def _learning_agent(m):
    for a in m.agents:
        if getattr(a, "method", None) == "learning" and hasattr(a, "learner"):
            return a
    raise RuntimeError("No learning storage agent found in the model.")


def _price_series(m, start, end):
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
    # Same unit scaling the ext-grid agent applies (agents.py).
    df["price"] = df["price (Ct/kWh)"].astype(float) / (100 / m.sref)
    lo = pd.to_datetime(start, format="%d.%m.%Y %H:%M")
    hi = pd.to_datetime(end, format="%d.%m.%Y %H:%M")
    df = df[(df["time"] >= lo) & (df["time"] <= hi)]
    df = df.dropna(subset=["time", "price"]).sort_values("time")
    if df.empty:
        raise RuntimeError(f"No spot prices in window {start} .. {end}")
    return df["price"].to_numpy(dtype=float), list(df["time"])


def _make_env(m, agent, prices, timestamps, *, shaping, episode_len,
              random_start, random_soc, seed):
    dt_h = m.timestep.seconds / 3600.0
    return StorageArbitrageEnv(
        prices=prices, timestamps=timestamps,
        capacity=agent.capacity, max_power_kw=agent.max_power,
        efficiency=agent.efficiency, discharge_per_step=agent.discharge, dt_h=dt_h,
        soc_start=agent.soc, soc_floor=agent.soc_floor, soc_ceiling=agent.soc_ceiling,
        margin_buy=getattr(m, "market_price_margin_buy", 1.0),
        margin_sell=getattr(m, "market_price_margin_sell", 0.3),
        gamma=agent.gamma,
        soc_shaping_weight=agent.soc_shaping_weight if shaping else 0.0,
        soc_shaping_anneal=agent.soc_shaping_anneal,
        episode_len=episode_len, random_start=random_start,
        random_soc=random_soc, seed=seed,
    )


def make_learner(agent):
    return SACLearner(
        state_dim=agent.state_dim, gamma=agent.gamma, tau=0.005, lr=3e-4,
        target_entropy=-1.0, buffer_size=200_000, batch_size=256,
        warmup_steps=2_000, actor_update_every=2, updates_per_step=1,
        reward_scale=agent.reward_scale, alpha_min=0.05, seed=SEED,
    )


def evaluate(env, learner):
    """Deterministic profit over one contiguous pass of `env` (no exploration,
    no shaping). Resets the env; runs to the end of its price series."""
    s = env.reset()
    profit = 0.0
    socs = []
    while True:
        a = learner.select_action(s, deterministic=True)
        s, _, done, info = env.step(a, learner.total_env_steps)
        profit += info["profit_eur"]
        socs.append(info["soc"])
        if done:
            break
    return profit, float(np.mean(socs)), float(np.min(socs)), float(np.max(socs))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    agent = _learning_agent(model)

    train_prices, train_ts = _price_series(model, TRAIN_START, TRAIN_END)
    val_prices, val_ts = _price_series(model, VAL_START, VAL_END)

    env = _make_env(model, agent, train_prices, train_ts, shaping=True,
                    episode_len=96, random_start=True, random_soc=True, seed=SEED)
    # Validation: one deterministic pass over the whole held-out window.
    val_env = _make_env(model, agent, val_prices, val_ts, shaping=False,
                        episode_len=len(val_prices), random_start=False,
                        random_soc=False, seed=0)
    learner = make_learner(agent)

    print(f"Surrogate: train {len(train_prices)} pts ({train_ts[0]} .. {train_ts[-1]}) "
          f"| val {len(val_prices)} pts ({val_ts[0]} .. {val_ts[-1]})")
    print(f"Battery: capacity={env.capacity:.0f} kWh | max_power={env.max_power_kw:.0f} kW "
          f"| eff={env.efficiency:.3f} | discharge/step={env.discharge:.5f} "
          f"| state_dim={env.state_dim}")
    print(f"Training for {TOTAL_STEPS} steps...\n", flush=True)

    best_path = OUT_PATH
    last_path = OUT_PATH.replace(".pt", "_last.pt")
    best_val = -np.inf

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
            v_profit, v_soc, v_lo, v_hi = evaluate(val_env, learner)
            star = ""
            if v_profit > best_val:
                best_val = v_profit
                learner.save(best_path)
                star = "  *best*"
            print(f"step {step:>7d} | {rate:6.0f} st/s | alpha {learner.alpha.item():.3f} "
                  f"| train profit/day {np.mean(recent_profit) if recent_profit else 0:+.2f} "
                  f"| VAL profit {v_profit:+8.2f} EUR  SOC {v_soc*100:4.1f}% "
                  f"[{v_lo*100:.0f}-{v_hi*100:.0f}]{star}", flush=True)

    learner.save(last_path)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print(f"Best validation policy -> {best_path} ({best_val:+.2f} EUR over val window)")
    print(f"Last policy            -> {last_path}")


if __name__ == "__main__":
    main()
