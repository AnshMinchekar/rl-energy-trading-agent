# -*- coding: utf-8 -*-
"""Battery-only arbitrage LP — the level-field optimisation baseline for SAC.

Plans storage row 0 over exactly the information SAC's state encodes (the
current price plus the known day-ahead forward window, FORECAST_STEPS = 24
x 15 min = 6 h) and maximises exactly the objective SAC's reward expresses:
trading cash flow at price +/- margin, plus the terminal charge valued at
liquidation (storage_logic.liquidation_value). Receding horizon: only the
first step of each solve is acted on, via the agent's normal
action_to_bid() path — bid pricing, deadband and emergency overrides are
the shared live-agent code, not re-implemented here.

Dynamics and caps mirror storage_logic exactly:
  * soc_t = soc_{t-1}·discharge + (b_t·eta − s_t/eta)/C, 0.05 <= soc_t <= 0.95
    (soc_transition)
  * per-step grid-side energy caps b_t <= P·dt/eta, s_t <= P·dt·eta
    (provide_power_kwh at rated power)

Units match the rest of the codebase: prices ct/kWh, energy kWh, power kW,
objective in EUR. Solved with scipy's HiGHS (~ms per solve) — no Gurobi
license contention with the HNOptimizer runs.
"""
import numpy as np
from scipy.optimize import linprog

from mesa_model.storage_logic import ASK_PRICE_FLOOR

# Hard SOC limits — the same bounds provide_power_kwh charges/discharges
# toward and the reward penalises beyond.
SOC_MIN = 0.05
SOC_MAX = 0.95


def solve_arbitrage(soc, prices, capacity, max_power_kw, dt_h, efficiency,
                    discharge, margin_buy, margin_sell):
    """Solve the full-horizon LP.

    Returns a dict with grid-side kWh plans ``buy``/``sell`` (length T),
    the implied ``soc`` path, and the ``objective`` in EUR — or None when
    there are no prices or the solve fails/is infeasible.
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    if T == 0 or capacity <= 0:
        return None
    eta = float(efficiency)
    d = float(discharge)
    C = float(capacity)
    e_cap = float(max_power_kw) * float(dt_h)   # stored-side energy at rated power

    # Variables x = [b_0..b_{T-1}, s_0..s_{T-1}], grid-side kWh.
    # soc after step t = alpha_t + sum_{k<=t} d^{t-k}·(b_k·eta − s_k/eta)/C
    idx = np.arange(T)
    expo = idx[:, None] - idx[None, :]
    decay = np.where(expo >= 0, d ** np.maximum(expo, 0), 0.0)
    b_soc = decay * (eta / C)
    s_soc = decay / (eta * C)
    alpha = soc * d ** (idx + 1)

    # Maximise sum_t [s_t·(p_t − margin_sell) − b_t·(p_t + margin_buy)]/100
    #          + kappa·soc_T,  kappa = liquidation value of a full battery at
    # the (known) end-of-horizon price — linear because p_T is a constant.
    kappa = C * eta * max(prices[-1] - margin_sell, 0.0) / 100.0
    d_term = d ** (T - 1 - idx)                 # d soc_T / d(trade at step k)
    c_buy = (prices + margin_buy) / 100.0 - kappa * eta * d_term / C
    c_sell = -(prices - margin_sell) / 100.0 + kappa * d_term / (eta * C)
    c = np.concatenate([c_buy, c_sell])

    A_ub = np.vstack([np.hstack([b_soc, -s_soc]),      # soc_t <= SOC_MAX
                      np.hstack([-b_soc, s_soc])])     # soc_t >= SOC_MIN
    b_ub = np.concatenate([SOC_MAX - alpha, alpha - SOC_MIN])

    # Sells whose ask price (p − margin_sell − 0.01) would sit below the live
    # ask floor can never clear (action_to_bid refuses them) — the same
    # no-fill rule the surrogate env applies. Besides parity, this removes
    # the LP's only degenerate money pump: at strongly negative prices a
    # simultaneous buy+sell that burns energy in the efficiency loss would
    # otherwise be "profitable".
    sellable = prices - margin_sell - 0.01 >= ASK_PRICE_FLOOR
    bounds = ([(0.0, e_cap / eta)] * T
              + [(0.0, e_cap * eta if sellable[t] else 0.0) for t in range(T)])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    buy, sell = res.x[:T], res.x[T:]
    return {"buy": buy, "sell": sell,
            "soc": alpha + b_soc @ buy - s_soc @ sell,
            "objective": float(-res.fun + kappa * alpha[-1])}


def plan_arbitrage(soc, prices, capacity, max_power_kw, dt_h, efficiency,
                   discharge, margin_buy, margin_sell):
    """First-step plan as grid-side (buy_kwh, sell_kwh).

    (0, 0) when there are no prices or the LP fails — the agent then simply
    holds (and action_to_bid's emergency overrides still cover SOC extremes).
    """
    sol = solve_arbitrage(soc, prices, capacity, max_power_kw, dt_h,
                          efficiency, discharge, margin_buy, margin_sell)
    if sol is None:
        return 0.0, 0.0
    return float(sol["buy"][0]), float(sol["sell"][0])
