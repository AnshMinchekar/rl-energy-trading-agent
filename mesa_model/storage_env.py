# -*- coding: utf-8 -*-
"""
Fast surrogate environment for training the storage SAC policy.

The live Mesa market solves a Gurobi LP every 15-min step (~2.25 s), which caps
training at one slow chronological pass — far too little for SAC to converge.
This environment removes the market solve: the storage agent is a *price-taker*
(it fills against the external grid at the market price ± a small spread), so its
arbitrage MDP can be driven directly from the historical price series. It runs at
~10^5-10^6 steps in minutes instead of weeks.

Faithfulness: state (state_features), reward (mark_to_market_reward) and SOC/power
dynamics (provide_power_kwh / soc_transition) are imported from
mesa_model/storage_logic.py — the *same* functions the live agent uses — and the
step/observe ordering mirrors storage.step()/update_status() exactly. Trades are
charged the same bid-ask spread the live agent crosses (action_to_bid posts
p±spread), so training sees the real round-trip transaction cost. The one
remaining simplification is fill *quantity*: we assume the agent always clears its
requested volume against the grid (close to true, since the live agent bids
aggressively enough to clear against the price-of-last-resort external grid).
Validate the trained policy in the full Mesa market before drawing conclusions.

After training, drop the frozen policy into the real agent for evaluation:
    SAC_LOAD_POLICY=output/sac/surrogate_policy.pt SAC_EVAL=1 python main.py
"""

from collections import deque

import numpy as np

from mesa_model.storage_logic import (
    state_features, mark_to_market_reward, provide_power_kwh, soc_transition,
)

_EPS = 1e-6


class StorageArbitrageEnv:
    def __init__(self, prices, timestamps, capacity, max_power_kw, efficiency,
                 discharge_per_step, dt_h, soc_start=0.40, soc_floor=0.20,
                 soc_ceiling=0.85, margin_buy=1.0, margin_sell=0.3,
                 soc_shaping_weight=1.0, soc_shaping_anneal=100_000.0,
                 episode_len=96, random_start=True, random_soc=True, seed=42,
                 spread=0.2):
        self.prices = np.asarray(prices, dtype=float)
        self.timestamps = list(timestamps)
        assert len(self.prices) == len(self.timestamps)

        self.capacity = float(capacity)
        self.max_power_kw = float(abs(max_power_kw))
        self.efficiency = float(efficiency)
        self.discharge = float(discharge_per_step)
        self.dt_h = float(dt_h)

        self.soc_start = soc_start
        self.soc_floor = soc_floor
        self.soc_ceiling = soc_ceiling
        self.margin_buy = margin_buy
        self.margin_sell = margin_sell
        self.soc_shaping_weight = soc_shaping_weight
        self.soc_shaping_anneal = soc_shaping_anneal
        # Bid-ask cost the live agent pays: action_to_bid crosses the market by
        # `spread` ct/kWh on every buy (p+spread) and sell (p-spread). Charge it
        # here so training sees the same round-trip cost as the live eval.
        self.spread = float(spread)

        self.episode_len = episode_len
        self.random_start = random_start
        self.random_soc = random_soc
        self.rng = np.random.RandomState(seed)

        self.ewma_beta = 0.05
        self.state_dim = 16
        self._warmup_window = 96

        self.soc = soc_start
        self.t = self._warmup_window
        self.steps_in_ep = 0
        self._reset_buffers()

    # -- internal ----------------------------------------------------------
    def _reset_buffers(self):
        self.price_history = deque(maxlen=96)
        self._lag_1h = deque(maxlen=4)
        self._lag_4h = deque(maxlen=16)
        self.hourly_ewma = {}

    def _observe(self, idx, price):
        """Append an observed price to the causal buffers (mirrors the buffer
        update at the top of storage.update_status)."""
        self.price_history.append(price)
        self._lag_1h.append(price)
        self._lag_4h.append(price)
        h = self.timestamps[idx].hour
        self.hourly_ewma[h] = ((1 - self.ewma_beta) * self.hourly_ewma.get(h, price)
                               + self.ewma_beta * price)

    def _state(self, idx):
        t = self.timestamps[idx]
        return np.asarray(state_features(
            soc=self.soc, soc_floor=self.soc_floor, soc_ceiling=self.soc_ceiling,
            price=self.prices[idx], price_history=self.price_history,
            lag_1h=self._lag_1h, lag_4h=self._lag_4h, hourly_ewma=self.hourly_ewma,
            hour=t.hour, weekday=t.weekday(), month=t.month,
            margin_buy=self.margin_buy, margin_sell=self.margin_sell,
        ), dtype=np.float32)

    # -- API ---------------------------------------------------------------
    def reset(self):
        last_valid = len(self.prices) - self.episode_len - 1
        if self.random_start and last_valid > self._warmup_window:
            start = self.rng.randint(self._warmup_window, last_valid)
        else:
            start = self._warmup_window
        self.soc = (float(self.rng.uniform(0.10, 0.90))
                    if self.random_soc else self.soc_start)
        self.steps_in_ep = 0
        self._reset_buffers()
        # Warm the causal buffers with the prices leading up to `start`,
        # ending with prices[start] (matches the agent observing the current
        # price in update_status before its decision state is built).
        for i in range(max(0, start - self._warmup_window + 1), start + 1):
            self._observe(i, self.prices[i])
        self.t = start
        return self._state(start)

    def step(self, action, total_env_steps):
        t = self.t
        p_dec = self.prices[t]
        soc_old = self.soc

        # Hard safety overrides (identical to storage.step()).
        if soc_old < 0.10:
            action = 1.0
        elif soc_old > 0.95:
            action = -1.0
        action = float(np.clip(action, -1.0, 1.0))

        max_sold, max_bought = provide_power_kwh(
            soc_old, self.capacity, self.max_power_kw, self.dt_h, self.efficiency)
        bought = sold = 0.0
        if action > _EPS:
            bought = action * max_bought
        elif action < -_EPS:
            sold = -action * max_sold

        soc_new = soc_transition(soc_old, bought, sold, self.capacity,
                                 self.efficiency, self.discharge)

        # Advance one step and observe the now-settled price (causal).
        t2 = t + 1
        p_now = self.prices[t2]
        self._observe(t2, p_now)

        reward = mark_to_market_reward(
            bought=bought, sold=sold, p_decision=p_dec, p_now=p_now,
            soc_old=soc_old, soc_new=soc_new, capacity=self.capacity,
            soc_floor=self.soc_floor, soc_ceiling=self.soc_ceiling,
            shaping_weight=self.soc_shaping_weight,
            total_env_steps=total_env_steps, shaping_anneal=self.soc_shaping_anneal,
        )

        # Bid-ask cost: buying pays p_dec+spread, selling receives p_dec-spread,
        # i.e. a cost of (bought+sold)*spread/100 € on top of the mid-price fill
        # that mark_to_market_reward assumes. Inventory revaluation stays at the
        # mid price, so only the cashflow leg is charged.
        spread_cost = (bought + sold) * self.spread / 100.0
        reward -= spread_cost

        self.soc = soc_new
        self.t = t2
        self.steps_in_ep += 1
        done = (self.steps_in_ep >= self.episode_len) or (t2 >= len(self.prices) - 1)

        info = {
            "bought": bought, "sold": sold, "p_decision": p_dec, "p_now": p_now,
            "soc": soc_new,
            "profit_eur": (sold - bought) * p_dec / 100.0 - spread_cost,
        }
        return self._state(t2), float(reward), done, info
