# -*- coding: utf-8 -*-
"""
Shared storage-agent logic — the single source of truth for the state vector,
the mark-to-market reward, and the SOC/power dynamics.

Both the live Mesa ``storage`` agent (mesa_model/agents.py) and the fast
surrogate training environment (mesa_model/storage_env.py) import from here.
Keeping these as pure functions guarantees the surrogate-trained policy sees an
*identical* state representation and is optimised for an *identical* reward, so
a policy trained on the surrogate transfers faithfully to the full market.

Units: prices in ct/kWh, energy in kWh, capacity in kWh, power in kW.
"""

import numpy as np

# Actions with |a| below this are a deliberate "hold" — no order is placed.
# A tanh-squashed Gaussian almost never outputs exactly 0, so without a
# deadband the agent trades (and pays spread) every single step.
ACTION_DEADBAND = 0.05

# Live asks are floored at this price (a sell below it cannot clear), so the
# surrogate applies the same no-fill rule for sells when p - margin_sell is
# under the floor (e.g. negative prices).
ASK_PRICE_FLOOR = 0.01

# How many future known day-ahead prices the state can see (24 x 15 min = 6 h).
FORECAST_STEPS = 24


def _clip(x, lo=-5.0, hi=5.0):
    return float(np.clip(x, lo, hi))


def state_features(soc, soc_floor, soc_ceiling, price, price_history,
                   lag_1h, lag_4h, hourly_ewma, hour, weekday, month,
                   margin_buy, margin_sell, future_prices=None):
    """Build the 20-D state vector.

    Backward-looking features use only the observed-price buffers passed in.
    ``future_prices`` are the next known *day-ahead* prices (published 12-36 h
    ahead in reality, so using them is not leakage); pass the next
    ``FORECAST_STEPS`` prices, or None/short — missing entries are padded with
    the current price (features -> 0).

    All price comparisons are differences normalised by the rolling std, not
    ratios: the price series contains thousands of negative / near-zero points
    where ratio features blow up or flip sign.
    """
    eps = 1e-6

    p = float(price)
    hist = (np.asarray(price_history, dtype=float)
            if len(price_history) > 0 else np.array([p], dtype=float))
    mean = float(hist.mean())
    std = float(hist.std())
    denom = std + eps

    soc = float(soc)
    span = (soc_ceiling - soc_floor) + eps
    headroom_ceiling = _clip((soc_ceiling - soc) / span)
    headroom_floor = _clip((soc - soc_floor) / span)

    price_norm = _clip((p - mean) / denom) if std > eps else 0.0
    percentile = float(np.mean(hist < p)) if len(hist) > 1 else 0.5

    base_h = hourly_ewma.get(hour, mean)
    price_vs_base = _clip((p - base_h) / denom)

    p1 = lag_1h[0] if len(lag_1h) == lag_1h.maxlen else p
    p4 = lag_4h[0] if len(lag_4h) == lag_4h.maxlen else p
    mom_1h = _clip((p - p1) / denom)
    mom_4h = _clip((p - p4) / denom)

    vol = _clip(std / (abs(mean) + eps), 0.0, 5.0)

    # Round-trip transaction spread relative to the price volatility the agent
    # could capture — the trade-or-hold economics signal.
    spread_norm = _clip((float(margin_buy) + float(margin_sell)) / denom, 0.0, 5.0)

    # Known future day-ahead prices (pad with p so missing data -> 0 features).
    fut = np.full(FORECAST_STEPS, p, dtype=float)
    if future_prices is not None and len(future_prices) > 0:
        f = np.asarray(future_prices, dtype=float)[:FORECAST_STEPS]
        fut[:len(f)] = f
    fwd_1h = _clip((float(fut[:4].mean()) - p) / denom)
    fwd_6h_mean = _clip((float(fut.mean()) - p) / denom)
    fwd_6h_min = _clip((float(fut.min()) - p) / denom)
    fwd_6h_max = _clip((float(fut.max()) - p) / denom)

    hour_rad = 2 * np.pi * hour / 24.0
    dow_rad = 2 * np.pi * weekday / 7.0
    mon_rad = 2 * np.pi * (month - 1) / 12.0

    return [soc, headroom_ceiling, headroom_floor, price_norm, percentile,
            price_vs_base, mom_1h, mom_4h, vol, spread_norm,
            float(np.sin(hour_rad)), float(np.cos(hour_rad)),
            float(np.sin(dow_rad)), float(np.cos(dow_rad)),
            float(np.sin(mon_rad)), float(np.cos(mon_rad)),
            fwd_1h, fwd_6h_mean, fwd_6h_min, fwd_6h_max]


STATE_DIM = 20


def liquidation_value(price, soc, capacity, efficiency, margin_sell):
    """€ the stored energy would realise if sold now: grid-side energy is
    soc·capacity·efficiency, sold no better than price − margin_sell (and a
    sell below the ask floor cannot clear, hence the max with 0)."""
    return soc * capacity * efficiency * max(price - margin_sell, 0.0) / 100.0


def mark_to_market_reward(bought, sold, p_buy, p_sell, p_decision, p_now,
                          soc_old, soc_new, capacity, efficiency, margin_sell,
                          gamma, soc_floor, soc_ceiling,
                          shaping_weight, total_env_steps, shaping_anneal):
    """Mark-to-market wealth change in € plus the annealed SOC-band shaping.

    cashflow values the trade at the prices actually paid/received (p_buy /
    p_sell — the settled price in the live market, price ± margin in the
    surrogate), NOT the raw spot price, so the reward sees the true round-trip
    transaction cost. Storage is fee-exempt (§118 EnWG), so no gridfee/levies
    appear here or in the live settlement.

    The inventory term is potential-based shaping Φ with the γ-correction
    (γ·Φ(s') − Φ(s)), which leaves the optimal policy exactly equal to the
    pure-realised-cash optimum while fixing the credit-assignment bias against
    buying. Φ values stored energy at its *liquidation* value
    (η·(p − margin_sell)·soc·capacity), not the mid price — the mid-price
    version over-rewarded hoarding by the efficiency loss + sell margin.

    Returned raw (in €); the SACLearner applies reward_scale.
    """
    cashflow = (sold * p_sell - bought * p_buy) / 100.0
    phi_old = liquidation_value(p_decision, soc_old, capacity, efficiency, margin_sell)
    phi_new = liquidation_value(p_now, soc_new, capacity, efficiency, margin_sell)
    reward = cashflow + gamma * phi_new - phi_old

    # Hard physical bounds — always on.
    if soc_new < 0.05 or soc_new > 0.97:
        reward -= 0.5

    # Annealed soft-band guidance (decays to 0 → final policy unbiased).
    progress = min(total_env_steps / shaping_anneal, 1.0) if shaping_anneal > 0 else 1.0
    w = shaping_weight * (1.0 - progress)
    if w > 0.0:
        if soc_new < soc_floor:
            reward -= w * (soc_floor - soc_new)
        elif soc_new > soc_ceiling:
            reward -= w * (soc_new - soc_ceiling)

    return reward


# ---------------------------------------------------------------------------
# kWh-space dynamics — surrogate equivalents of storage.provide_a_power and the
# SOC update in storage.update_status. The live agent computes these in per-unit
# (÷ sref) market space; sref cancels between bidding and settlement, so the
# physics below are equivalent. Cross-checked against the agent's formulas:
#   * charge limited to bring SOC up to 0.95, discharge down to 0.05
#   * efficiency applied to the power cap AND inversely in the SOC update,
#     so a non-power-bound charge of headroom/eff stores exactly the headroom.
# ---------------------------------------------------------------------------

def provide_power_kwh(soc, capacity, max_power_kw, dt_h, efficiency):
    """Return (max_sold_kwh, max_bought_kwh) available this timestep — the
    grid-side energy the battery can discharge / charge."""
    e_cap = max_power_kw * dt_h                       # energy at rated power
    max_sold = min(max(soc - 0.05, 0.0) * capacity, e_cap) * efficiency
    max_bought = min(max(0.95 - soc, 0.0) * capacity, e_cap) / efficiency
    return max_sold, max_bought


def soc_transition(soc, bought, sold, capacity, efficiency, discharge):
    """SOC after a (bought, sold) grid-side trade — identical to the update in
    storage.update_status: soc·discharge + (bought·η − sold/η)/capacity."""
    energy_delta = bought * efficiency - sold / efficiency
    return min(max(soc * discharge + energy_delta / capacity, 0.0), 1.0)
