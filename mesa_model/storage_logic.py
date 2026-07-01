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


def _clip(x, lo=-5.0, hi=5.0):
    return float(np.clip(x, lo, hi))


def state_features(soc, soc_floor, soc_ceiling, price, price_history,
                   lag_1h, lag_4h, hourly_ewma, hour, weekday, month,
                   margin_buy, margin_sell):
    """Build the 16-D state vector (see docs/redesign-plan.md §4).

    All price-derived features use only the observed-price buffers passed in —
    no future data. ``lag_1h`` / ``lag_4h`` are deques with a ``maxlen`` (4 and
    16 respectively); ``hourly_ewma`` is a {hour: ewma_price} dict.
    """
    eps = 1e-6

    p = float(price)
    hist = (np.asarray(price_history, dtype=float)
            if len(price_history) > 0 else np.array([p], dtype=float))
    mean = float(hist.mean())
    std = float(hist.std())

    soc = float(soc)
    span = (soc_ceiling - soc_floor) + eps
    headroom_ceiling = _clip((soc_ceiling - soc) / span)
    headroom_floor = _clip((soc - soc_floor) / span)

    price_norm = _clip((p - mean) / (std + eps)) if std > eps else 0.0
    percentile = float(np.mean(hist < p)) if len(hist) > 1 else 0.5

    base_h = hourly_ewma.get(hour, mean)
    price_vs_base = _clip((p - base_h) / (base_h + eps))

    p1 = lag_1h[0] if len(lag_1h) == lag_1h.maxlen else p
    p4 = lag_4h[0] if len(lag_4h) == lag_4h.maxlen else p
    mom_1h = _clip(p / (p1 + eps) - 1.0)
    mom_4h = _clip(p / (p4 + eps) - 1.0)

    vol = _clip(std / (mean + eps), 0.0, 5.0)

    mb = float(margin_buy)
    ms = float(margin_sell)
    spread_norm = _clip((mb + ms) / (p + mb + eps), 0.0, 5.0)

    hour_rad = 2 * np.pi * hour / 24.0
    dow_rad = 2 * np.pi * weekday / 7.0
    mon_rad = 2 * np.pi * (month - 1) / 12.0

    return [soc, headroom_ceiling, headroom_floor, price_norm, percentile,
            price_vs_base, mom_1h, mom_4h, vol, spread_norm,
            float(np.sin(hour_rad)), float(np.cos(hour_rad)),
            float(np.sin(dow_rad)), float(np.cos(dow_rad)),
            float(np.sin(mon_rad)), float(np.cos(mon_rad))]


def mark_to_market_reward(bought, sold, p_decision, p_now, soc_old, soc_new,
                          capacity, soc_floor, soc_ceiling,
                          shaping_weight, total_env_steps, shaping_anneal):
    """Mark-to-market wealth change in € (see docs/redesign-plan.md §3) plus the
    annealed SOC-band shaping. Returned raw; the SACLearner applies reward_scale.
    """
    cashflow = (sold - bought) * p_decision / 100.0
    inventory_delta = (p_now * soc_new - p_decision * soc_old) * capacity / 100.0
    reward = cashflow + inventory_delta

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
