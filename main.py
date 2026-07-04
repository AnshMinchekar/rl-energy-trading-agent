# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:44:41 2023
@autor: mjulschm

Multi-epoch training driver.

The simulation window (config.yaml) is a single chronological pass of ~8.5k
15-min steps. That is far too little experience for SAC to converge — the agent
only reaches break-even just as the data runs out. Set the SAC_EPOCHS env var to
replay that same window N times, carrying the trained SAC learner (networks +
replay buffer + step counters) across epochs so experience accumulates:

    SAC_EPOCHS=20 python main.py        # 20 passes over the calendar window

Each epoch re-instantiates a fresh market/calendar (clean SOC, prices, etc.) but
re-uses the learner from the previous epoch. The warm-up only happens once
(total_env_steps persists). CSV output is written only on the final epoch (the
evaluation pass); intermediate epochs are training-only for speed.

NOTE: each step runs a Gurobi market solve (~2.25 s), so one epoch ≈ 5 h of
wall-clock. Many epochs over the live market is expensive — train on the fast
surrogate instead (train_surrogate.py) and use this script for evaluation.
"""

import gc
import os
import shutil
from datetime import datetime

import numpy as np
import random
np.random.seed(42)
random.seed(42)

from data.config.config import config
from mesa_model.model import model, LEM
from data.csv_writer import EntryWriter

EPOCHS = int(os.environ.get("SAC_EPOCHS", "1"))
EPISODE_LOG = "output/sac/episode_logs.jsonl"


def _find_learner_agent(m):
    """Return the storage agent running the 'learning' (SAC) method, or None."""
    for a in m.agents:
        if getattr(a, "method", None) == "learning" and hasattr(a, "learner"):
            return a
    return None


def main():
    # For a fresh multi-epoch run, archive any prior episode log so the learning
    # curve in output/sac/episode_logs.jsonl is clean and continuous.
    if EPOCHS > 1 and os.path.exists(EPISODE_LOG):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(EPISODE_LOG, EPISODE_LOG.replace(".jsonl", f"_{stamp}.jsonl"))

    writer = EntryWriter()

    saved_learner = None
    saved_episode = 0

    for epoch in range(EPOCHS):
        # Epoch 0 re-uses the model already built at import; later epochs get a
        # fresh market/calendar so each pass starts from clean physical state.
        m = model if epoch == 0 else LEM(config.main["sb_grid"],
                                         config.main["simulation_start_time"])

        # Carry the trained learner (and the running episode counter) forward.
        agent = _find_learner_agent(m)
        if agent is not None:
            agent.current_epoch = epoch
            if saved_learner is not None:
                agent.learner = saved_learner
                agent.episode_counter = saved_episode

        is_final = (epoch == EPOCHS - 1)
        if EPOCHS > 1:
            print(f"\n{'#'*60}\n# EPOCH {epoch + 1}/{EPOCHS}"
                  f"{'  (final / evaluation pass)' if is_final else '  (training pass)'}"
                  f"\n{'#'*60}\n", flush=True)

        hn_node = int(agent.bus) if agent is not None else 5
        for n in range(m.total_steps):
            m.step()
            # Write per-step CSV only on the final (evaluation) epoch.
            if is_final:
                result = m.results[int(m.stepcount)]
                price = m.market_price[m.market_price["time"] == m.current_date]["price"].values[0]
                writer.write_entry(result, price, m.market_price_margin_sell,
                                   m.market_price_margin_buy, m.current_date,
                                   hn_node=hn_node)
            if n % 96 == 0:
                gc.collect()
                print(f"Simulation completed a day")

        # Persist the learner state for the next epoch.
        agent = _find_learner_agent(m)
        if agent is not None:
            saved_learner = agent.learner
            saved_episode = agent.episode_counter

    writer.close()
    print("Simulation complete - files closed")


if __name__ == "__main__":
    main()
