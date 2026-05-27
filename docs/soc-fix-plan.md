# SOC Stagnation Fix Plan

## Observed Symptoms (first 13 training days)

- Agent starts at 80% SOC, burns through stored energy in episodes 1–2 (sold 30 + 44 kWh, zero buying), then collapses to ~12% and stays there.
- 11 of 24 episodes show zero activity (0 kWh bought/sold, 0 profit, 0 reward).
- Several episodes show negative profit — agent bought high and sold low.
- Output weights (W2) change only in the 3rd decimal place per episode — learning is stalled.

---

## Root Causes & Fixes

### Fix 1 — Remove SOC gate on exploration (`agents.py:704`)

**Problem:** Exploration noise is only added when `0.20 < soc < 0.90`. Below 20% SOC the agent can't discover that charging is rewarding — it's frozen with whatever policy it has already learned, which outputs near-zero or slightly negative actions.

**Change:** Remove the SOC bounds on the noise injection. Let the agent explore at any SOC (or at minimum extend the lower bound to `soc > 0.05`).

```python
# Before
if 0.20 < soc < 0.90:
    noise = np.random.normal(...)
    action += noise

# After
if soc < 0.95:  # only suppress exploration at true ceiling emergency
    noise = np.random.normal(...)
    action += noise
```

---

### Fix 2 — Dynamic soft bias blend (`agents.py:701`)

**Problem:** The bias blend is `0.9 × network + 0.1 × soc_bias`. If the network outputs even −0.1, the soc_bias of 0.8 contributes only `0.1 × 0.8 = +0.08`, giving a net action of −0.01 → effectively zero. The safety bias cannot override a network that has learned to do nothing at 12% SOC.

**Change:** Make the blend SOC-dependent. At `soc < 0.20`, shift weight toward the safety bias so it can actually force a charge.

```python
# Before (flat 90/10 blend)
action = 0.9 * action + 0.1 * (soc_bias + price_bias)

# After (dynamic blend)
if soc < 0.15:
    bias_weight = 0.50
elif soc < 0.20:
    bias_weight = 0.30
else:
    bias_weight = 0.10
action = (1 - bias_weight) * action + bias_weight * (soc_bias + price_bias)
```

---

### Fix 3 — Extend SOC penalty into the 10–30% range (`agents.py:841–845`)

**Problem:** The SOC penalty is zero for `soc` in the range 10–92%. The agent learns that sitting at 12% is fine — there is no signal telling it this is a bad state.

**Change:** Add graded penalties for the 10–30% band.

```python
# Before
if self.soc < 0.10:
    soc_penalty = -2.0 * (0.10 - self.soc) / 0.10
elif self.soc > 0.92:
    soc_penalty = -2.0 * (self.soc - 0.92) / 0.08

# After
if self.soc < 0.10:
    soc_penalty = -2.0 * (0.10 - self.soc) / 0.10   # max -2.0 at soc=0
elif self.soc < 0.20:
    soc_penalty = -0.8 * (0.20 - self.soc) / 0.10   # max -0.8 at soc=0.10
elif self.soc < 0.30:
    soc_penalty = -0.3 * (0.30 - self.soc) / 0.10   # max -0.3 at soc=0.20
elif self.soc > 0.92:
    soc_penalty = -2.0 * (self.soc - 0.92) / 0.08
```

---

### Fix 4 — Recharge allowance in profit reward (`agents.py:813`)

**Problem:** `profit_reward = -5.0 × (price × bought / 100)`. All buying looks bad immediately. The arbitrage bonus compensates, but only when `price_ratio < 0.95` — and early in training, `price_history` is too short for `avg_price` to be reliable, so the ratio is noisy. Net result: the agent develops a strong aversion to buying.

**Change:** When SOC is low (`< 0.35`) and the current price is not above average (`price_ratio < 1.1`), halve the profit penalty on buying to reduce the disincentive for recovery charging.

```python
# Before
energy_cost_eur = price * (bought - sold) / 100
profit_reward = -energy_cost_eur * 5.0

# After
energy_cost_eur = price * (bought - sold) / 100
penalty_scale = 5.0
if self.soc < 0.35 and bought > 0 and price_ratio < 1.1:
    penalty_scale = 2.5   # halve the buying penalty during low-SOC recovery
profit_reward = -energy_cost_eur * penalty_scale
```

---

## Priority Order

| # | Change | Location | Impact |
|---|--------|----------|--------|
| 1 | Remove SOC gate on exploration | `agents.py:704` | Unblocks learning at low SOC |
| 2 | Dynamic soft bias blend | `agents.py:701` | Forces charging when stuck |
| 3 | Extend SOC penalty to 10–30% | `agents.py:841–845` | Discourages staying low |
| 4 | Recharge allowance in profit reward | `agents.py:813` | Removes buying aversion at low SOC |

All changes are confined to `mesa_model/agents.py`.
