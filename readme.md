# Reinforcement Learning Storage Agent — Soft Actor-Critic (SAC)

> **Branch:** `algo/soft-actor-critic`. Each `algo/*` branch implements a
> different RL algorithm in `mesa_model/agents.py` (class `storage`) while the
> rest of the simulation is unchanged. This branch uses **Soft Actor-Critic**,
> implemented with PyTorch in `mesa_model/sac.py`.

### The Goal

**Maximize profit through energy arbitrage:**

```
Buy Low  →  Store  →  Sell High  →  Profit
```

The agent must learn to *buy low and sell high* — not simply discharge its
starting energy. The reward (Section 5) is designed so that holding charge into
a rising price is rewarded and draining the battery is not.

---

## 1. Architecture

SAC is an **off-policy, maximum-entropy actor-critic**. Three things make it a
good fit for battery arbitrage:

- **Off-policy replay** — every transition (especially rare price spikes/troughs,
  the highest-value events) is stored and reused many times.
- **Twin critics** — two Q-networks and a `min` target prevent value
  overestimation, which would otherwise inflate the apparent value of selling.
- **Automatic entropy temperature** — exploration is a principled, self-tuning
  objective, not a hand-scheduled noise that decays before convergence.

### Data Flow

```
                          ┌──────────────────────────────────────┐
                          │            Replay Buffer             │
                          │     (s, a, r, s′) × up to 100k        │
                          └──────────────────────────────────────┘
                              ▲ push                  │ sample 256
   Market price/SOC           │                       ▼
        │              ┌──────────────┐      ┌───────────────────────────┐
        ▼              │  update_     │      │  SAC update × 4 / step    │
  build_state (16-D) → │  status()    │      │  • twin critics (min)     │
        │              │  mark-to-    │      │  • actor (reparam.)       │
        ▼              │  market      │      │  • entropy temp α         │
   actor.select_action │  reward      │      │  • Polyak target update   │
        │              └──────────────┘      └───────────────────────────┘
        ▼
  fixed-spread bid/ask → market clears → realised bought/sold
```

Acting happens in `storage.step()`; learning happens in `storage.update_status()`
**every timestep** (4 gradient updates per step — see Section 6).

---

## 2. State Representation

The agent builds a **16-dimensional state vector** (`storage.build_state`). All
price-derived features come from rolling buffers of *observed* prices only —
there is **no future-price leakage**.

```python
state = [soc, headroom_to_ceiling, headroom_to_floor,
         price_norm, price_percentile, price_vs_hourly_baseline,
         price_momentum_1h, price_momentum_4h, price_vol_24h, spread_norm,
         sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month]
```

| Idx | Feature | Description |
|-----|---------|-------------|
| 0 | `soc` | Battery charge level [0, 1] |
| 1 | `headroom_to_ceiling` | Room left to charge: `(ceiling − soc)/(ceiling − floor)` |
| 2 | `headroom_to_floor` | Room left to discharge: `(soc − floor)/(ceiling − floor)` |
| 3 | `price_norm` | `(price − mean₂₄ₕ)/std₂₄ₕ` — cheap/expensive right now |
| 4 | `price_percentile` | Fraction of the last 96 observed prices below the current one |
| 5 | `price_vs_hourly_baseline` | Price vs a **causal** EWMA of price *for this hour of day* |
| 6 | `price_momentum_1h` | `price / price₁ₕ_ago − 1` |
| 7 | `price_momentum_4h` | `price / price₄ₕ_ago − 1` |
| 8 | `price_vol_24h` | `std₂₄ₕ / mean₂₄ₕ` — is arbitrage worth it now? |
| 9 | `spread_norm` | External-grid buy/sell spread — opportunity cost of trading |
| 10–11 | `sin/cos_hour` | Time of day (cyclic) |
| 12–13 | `sin/cos_dow` | Day of week (cyclic) |
| 14–15 | `sin/cos_month` | Season (cyclic) |

**The hourly baseline (idx 5) is the key arbitrage feature.** A causal EWMA keyed
by hour-of-day tells the agent whether the current price is unusually cheap or
dear *for this time of day*, so it can charge ahead of the predictable evening
peak — using only prices it has already observed.

---

## 3. Networks

### Actor — squashed Gaussian policy

```
state(16) → Linear(16→64) → LayerNorm → ReLU
          → Linear(64→64) → LayerNorm → ReLU
          → Linear(64→2)   →  (μ, log σ)

u      = μ + σ · ε ,   ε ~ N(0, 1)        # reparameterised sample
action = tanh(u) ∈ (−1, +1)               # +1 = max charge, −1 = max discharge
log π  = log N(u; μ, σ) − Σ log(1 − tanh(u)² + 1e-6)   # tanh correction
```

The reparameterisation trick makes the entropy term differentiable. At
evaluation, the deterministic mean `tanh(μ)` is used.

### Twin critics — Q(s, a)

Two independent networks `(state ⊕ action) ∈ ℝ¹⁷ → 64 → 64 → 1`, each with a
slow-moving **target** copy. The bootstrap target takes the **minimum** of the
two target critics to curb overestimation:

```
a′, log π′ = actor(s′)
y = r + γ · ( min(Q1_target(s′, a′), Q2_target(s′, a′)) − α · log π′ )
```

---

## 4. Action → Market Bid/Ask

The action maps directly to power with a **fixed small spread** — the agent
learns *when* and *how much* to trade, not how to shade prices
(`storage.action_to_bid`):

```python
spread = 0.2  # ct/kWh, just enough to clear vs the external-grid margin

if action > 0:   # charge
    power     = action × max_charge
    bid_price = price_now + spread
elif action < 0: # discharge
    power     = |action| × max_discharge
    ask_price = max(price_now − spread, 0.01)
```

Filled volumes are read back from the market clearing, so the reward always uses
*actually traded* energy, not the requested amount.

### Safety overrides

```python
if soc < 0.10:  action = +1.0   # emergency charge (aggressive bid to guarantee fill)
if soc > 0.95:  action = -1.0   # emergency discharge
```

The soft operating band (`floor 0.20`, `ceiling 0.85`) is **not** hard-enforced —
the policy learns to respect it because leaving it forfeits future profit. The
overrides are physical safety only. Override transitions are still stored in
replay so the critic learns that hitting the floor triggers a costly recharge.

---

## 5. Reward — Mark-to-Market Wealth Change

This is the heart of the design. A naive per-step cash-flow reward
(`(sold − bought)·price`) makes every sale an instant gain and every purchase an
instant loss, so the agent learns to drain the battery and sit empty. We instead
reward the change in **total economic wealth = cash + value of stored energy**
(`storage.compute_reward`):

```python
cashflow        = (sold − bought) · p_decision / 100                    # €
inventory_delta = (p_now · soc_new − p_decision · soc_old) · capacity / 100   # €
reward          = cashflow + inventory_delta
```

- `p_decision` — the price the agent saw when it acted (the settlement price)
- `p_now` — the price now observed, one step later

Both are observed prices, so there is no future leakage.

**Why this fixes the draining trap:**

| Situation | Reward | Effect |
|-----------|--------|--------|
| Buy `E` at price `p` | ≈ 0 (cash out balanced by inventory gained) | buying is **not** punished |
| Hold charge while price rises `p → p′` | `+E·(p′−p)/100` | **rewarded for being charged into a peak** |
| Sell at a high price | realises the gain already credited | no double counting |
| Sell at a low price / churn | slightly negative (efficiency loss) | discourages pointless cycling |

`reward = cashflow + (Φ_t − Φ_{t−1})` where `Φ = price·soc·capacity/100` is a
state potential, so this is **potential-based reward shaping** — it densifies the
learning signal without changing the optimal policy. A small `−0.5` penalty is
added only at the hard SOC limits (`<0.05` or `>0.97`).

Rewards are scaled by `reward_scale = 10` before learning so the tiny per-step
euro amounts are not swamped by the entropy term.

---

## 6. Learning Algorithm — SAC

Unlike Monte Carlo or on-policy TD, SAC learns **off-policy from a replay buffer
every timestep**, reusing past experience many times.

### Per-timestep cycle

```python
# storage.step()  — act at time τ
state  = build_state()
action = learner.select_action(state)        # random during warm-up, else actor sample
place_market_bid_or_ask(action)

# storage.update_status()  — at time τ+1, after the market clears
reward     = compute_reward(bought, sold, p_decision, p_now, soc_old, soc_new)
next_state = build_state()
learner.push(state, action, reward, next_state)   # → replay buffer
learner.learn()                                   # 4 gradient updates (see below)
```

### The SAC updates (`SACLearner.update`, run 4× per step)

```python
# 1. Critic: regress both Q-nets onto the min-target Bellman backup
y          = r + γ · (min(Q1ᵗ(s′,a′), Q2ᵗ(s′,a′)) − α · log π(a′|s′))
critic_loss = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

# 2. Actor (every 2nd update): maximise Q while staying stochastic
actor_loss = E[ α · log π(a|s) − min(Q1(s,a), Q2(s,a)) ]      # a reparameterised

# 3. Temperature α: drive entropy toward the target
alpha_loss = −E[ α · (log π(a|s) + target_entropy) ]          # target_entropy = −1

# 4. Polyak update of the target critics
θ_target ← (1 − τ)·θ_target + τ·θ
```

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `γ` | Discount factor (0.99 → ≈25 h horizon, spans the daily price cycle) |
| `α` | Entropy temperature — auto-tuned so policy entropy ≈ `target_entropy` |
| `log π(a\|s)` | Log-probability of the action under the current policy |
| `Q1, Q2` | The twin critics; `Q1ᵗ, Q2ᵗ` their slow target copies |
| `τ` | Polyak averaging coefficient (0.005) for target updates |
| `min(Q1,Q2)` | Clipped double-Q — curbs value overestimation |

### Update-to-data ratio (UTD)

The gurobi market solve (~2.25 s/step) dominates wall-clock, so SAC's tiny
networks are nearly free to update. We therefore run **4 gradient updates per
environment step** (`updates_per_step = 4`), extracting ~4× the learning from the
same simulation — the single most effective knob for sample efficiency here.

### Warm-up

The first **500** steps use a uniform-random policy to seed the replay buffer
before any gradient update, so learning starts from a diverse batch.

---

## 7. Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `state_dim` | 16 | State vector dimension |
| hidden width | 64 | Per layer, actor and critics |
| `gamma` | 0.99 | Discount (~25 h horizon) |
| `actor_lr` / `critic_lr` / `alpha_lr` | 3e-4 | Adam learning rates |
| `tau` | 0.005 | Polyak target-update coefficient |
| `target_entropy` | −1.0 | Entropy target for a scalar action |
| `buffer_size` | 100,000 | Replay capacity |
| `batch_size` | 256 | Minibatch per gradient update |
| `warmup_steps` | 500 | Random-policy steps before learning |
| `updates_per_step` | 4 | Gradient updates per environment step (UTD) |
| `actor_update_every` | 2 | Actor + α updated every other gradient step |
| `reward_scale` | 10.0 | Scales the per-step € reward |
| `soc_floor` / `soc_ceiling` | 0.20 / 0.85 | Soft band (policy-enforced) |

---

## 8. Running

```bash
conda activate Diss_clean
pip install torch --index-url https://download.pytorch.org/whl/cpu   # one-time: SAC dependency
python main.py
```

Per-episode metrics (profit, captured spread, SOC band, α, entropy, critic loss)
are written to `output/sac/episode_logs.jsonl`. Run
`python analysis/analyze_sac.py` to produce learning-curve plots and a summary
table comparing SAC against the MC and TD branches.

> **Dependency note:** this branch requires **PyTorch** (CPU build is sufficient),
> which the MC/TD branches do not. SAC also sets `KMP_DUPLICATE_LIB_OK=TRUE` and
> pins torch to a single thread so its OpenMP runtime coexists with gurobi/MKL on
> Windows.
