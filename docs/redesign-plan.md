# Storage RL Agent Redesign Plan
## From TD Actor-Critic to Soft Actor-Critic (SAC) for Maximum Arbitrage

---

## 1. Executive Summary

The current TD actor-critic implementation has several structural limitations that cap its arbitrage performance: a tiny 8-neuron network, a completely on-policy training loop that discards experience after 48 steps, hand-crafted reward shaping that competes with the profit signal, and a two-stage bid/ask translation layer that introduces non-differentiable noise between the policy output and market outcome.

The proposed replacement is a **Soft Actor-Critic (SAC)** agent with a proper experience replay buffer, twin Q-critics, automatic entropy tuning, and a clean direct-volume action interface. These changes address the root causes rather than tuning around them.

---

## 2. Diagnosis: What Is Wrong With the Current Agent

### 2.1 The Network Is Too Small

The actor has 8 hidden neurons mapping a 9-dimensional state to a scalar action. The arbitrage problem requires the agent to simultaneously represent:

- High-SOC + high-price → sell
- Low-SOC + low-price → buy
- Mid-SOC + uncertain price → wait
- Near-midnight + high-price + mid-SOC → sell to recharge cheap overnight

These are four distinct modes, and each has continuous gradations. An 8-neuron hidden layer almost certainly cannot partition this space cleanly. Empirically, the W2 output weights printed per episode are noisy and rarely stabilise.

### 2.2 On-Policy Learning Throws Away Information

Every 48 steps (12 simulation hours), `self.trajectory` is cleared. This means the agent **never revisits** any transition. If a genuine price spike occurs at step 10 of an episode and the agent reacts poorly, that experience is gone within 12 hours of simulation time. Price extremes — the highest-value events for arbitrage — are also the rarest, so discarding them is disproportionately harmful.

The current buffer holds 48 transitions and is wiped every update. For context, a well-tuned SAC agent for a similar continuous control problem typically draws from a replay buffer of 50,000–1,000,000 transitions, repeatedly.

### 2.3 The Reward Function Is a Competing Supervisor

The reward has three components: profit reward (scale ×5), an arbitrage timing bonus (±2.0, ±1.0, ±0.3 at fixed percentile thresholds), and a SOC penalty. The arbitrary multipliers were hand-tuned and their relative magnitudes are not derivable from any financial principle.

The arbitrage bonus in particular is double-counting: it fires whenever the agent buys below 85% or sells above 115% of the 24-hour average. But the profit reward already penalises buying expensive and rewards buying cheap. The bonus creates a secondary implicit objective that the policy is jointly optimising, and the two objectives can conflict.

The SOC penalty range (`> 0.92 → full penalty`) is far tighter than the operating range (`soc_ceiling = 0.85`). These inconsistencies create a dead zone where the policy gets penalised for states that are ostensibly allowed.

### 2.4 The Bid/Ask Translation Layer Breaks the Policy Gradient

After the network produces an action ∈ [-1, +1], `action_to_bid()` applies a second decision layer: it selects a bid premium or ask discount based on hard-coded price-percentile thresholds. This layer is not differentiable and not learned. The market may or may not fill the resulting bid.

This means the gradient of actual profit with respect to network weights is not the gradient of the action with respect to network weights: it includes an unlearned heuristic in the middle. The policy can improve the first gradient but has no visibility into the second.

### 2.5 Exploration Decays Too Fast

`exploration_rate` starts at 0.35 and decays by 0.997 each episode (every 48 steps). After 400 episodes (~200 simulated days), the rate reaches its floor of 0.08. However, the agent's policy often hasn't converged after 400 episodes — the weight restore mechanism (10 consecutive below-best episodes) actively interrupts learning. The result is an agent that stops exploring before it has learned a reliable policy.

### 2.6 The "Best Weights" Restore Is Anti-Learning

When profit falls below the best seen so far for 10 consecutive episodes, the agent restores the best weights. In a financial time series with non-stationarity (weather, seasonal spot prices), the best historical weights may not generalise to future price regimes. This mechanism is greedy in a way that is incompatible with long-run optimisation: it optimises for the past episode window, not expected future profit.

### 2.7 Lag-Based "Forecasts" Are Low-Information

The "1-hour forecast" is actually the price from 1 hour ago used as a proxy for the price in 1 hour. For a stationary mean-reverting process this has a small correlation with the future, but German day-ahead spot prices have intraday structure (morning/evening peaks) that a lag-based proxy systematically misrepresents.

---

## 3. Proposed Architecture: Soft Actor-Critic (SAC)

### Why SAC Over the Alternatives

| Algorithm | Off-policy? | Handles continuous actions? | Built-in exploration? | Stability |
|-----------|-------------|-----------------------------|-----------------------|-----------|
| TD Actor-Critic (current) | No | Via tanh | ε-greedy noise, decays | Low |
| PPO | No | Yes | Entropy bonus, but still on-policy | High |
| DQN | Yes | No (needs discretisation) | ε-greedy | Medium |
| **SAC** | **Yes** | **Yes** | **Entropy maximisation** | **High** |
| DDPG | Yes | Yes | Action noise only | Low |

SAC is the right choice because:

1. **Off-policy replay** means every observed transition, including rare price spikes, is used repeatedly until the agent extracts all available signal.
2. **Maximum entropy objective** (`J = E[r] + α·H(π)`) builds exploration in as a principled objective, not an ad-hoc schedule. The agent stays uncertain in situations it genuinely hasn't learned, and becomes confident only where evidence is strong.
3. **Twin critics** (double-Q) prevent Q-overestimation, which is the principal failure mode of single-critic actor-critic in continuous control.
4. **Automatic temperature tuning** for `α` removes a critical hyperparameter. The temperature self-adjusts so entropy stays near a target value, adapting as the policy improves.
5. Proven in energy storage arbitrage literature (see Cao et al. 2020, Fang et al. 2022, numerous BESS RL studies since 2021).

---

## 4. State Representation (15-D)

Replace the 9-D state with a 15-D state that separates battery state, price features, and temporal features more cleanly.

```
Index  Feature               Computation
─────────────────────────────────────────────────────────────────
0      soc                   self.soc  (already normalised [0,1])
1      energy_headroom       (soc_ceiling - soc) / soc_ceiling
2      energy_floor          (soc - soc_floor) / (1 - soc_floor)
3      price_norm            (p_now - μ_24h) / (σ_24h + ε)
4      price_percentile      fraction of last-96 prices below p_now
5      price_momentum_1h     p_now / p_1h_ago - 1
6      price_momentum_4h     p_now / p_4h_ago - 1
7      price_vol_24h         σ_24h / (μ_24h + ε)
8      spread_norm           (ask_ext - bid_ext) / (ask_ext + ε)
9      sin_hour              sin(2π·h/24)
10     cos_hour              cos(2π·h/24)
11     sin_dow               sin(2π·d/7)
12     cos_dow               cos(2π·d/7)
13     sin_month             sin(2π·m/12)
14     cos_month             cos(2π·m/12)
```

**Why better than the current 9-D state:**

- `energy_headroom` and `energy_floor` make the soft operating bounds explicit. The current state only has raw SOC; the policy has to infer how close it is to the ceiling/floor by comparing SOC to implicit internal thresholds. Giving it both distances directly simplifies the learning task.
- `price_vol_24h` gives the agent signal about whether the current period is worth trading actively. In low-volatility periods, arbitrage spread is small; in high-volatility periods (storms, cold snaps) it is large. The current state has no volatility signal.
- `spread_norm` (external grid buy/sell spread) tells the agent the opportunity cost of trading externally vs. internally. This directly informs whether LEC participation is profitable.
- Month cyclical encoding captures seasonal patterns in spot price (winter heating peaks, summer solar surpluses) that are invisible to hour and day-of-week alone.
- Removing `max_discharge` and `max_charge` from state (already done in current) is correct — they are deterministic from SOC and kept that way.

---

## 5. Network Architecture

### Actor (Gaussian Policy)

```
Input(15) → Linear(15→64) → LayerNorm → ReLU
          → Linear(64→64) → LayerNorm → ReLU
          → Linear(64→2)    # outputs [μ, log_σ]

μ         = tanh(raw_μ)                  # mean action ∈ (-1, 1)
log_σ     = clamp(raw_log_σ, -5, 2)     # log std ∈ (-5, 2)
action    = tanh(μ + σ · ε),  ε~N(0,1)  # reparameterisation trick
```

The actor outputs a **stochastic Gaussian policy**. The reparameterisation trick makes the entropy term differentiable through the sampling operation. During inference (after training), set `σ=0` (deterministic exploitation).

**Why 64 neurons instead of 8:**

For reference, the current 8-neuron layer has 8×9 + 8×1 = 80 learnable parameters in the hidden layer. The 64-neuron design has 64×15 + 64×64 + 64×2 = 5,248 parameters across all three layers — 65× more capacity. The arbitrage problem, which requires the agent to learn distinct behaviours across the joint space of SOC, price level, price momentum, and time-of-day, genuinely requires this capacity.

### Twin Critics (Q-functions)

Two identical networks, trained on independent mini-batches, each computing Q(s, a):

```
Input = concat(state(15), action(1)) = 16-D
16 → Linear(16→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1)
```

The target Q-value used in the critic loss is:

```
y = r + γ · (min(Q1_target(s', a'), Q2_target(s', a')) - α · log_π(a'|s'))
```

Taking the minimum of two Q-estimates reduces the systematic overestimation that plagues single-critic methods in continuous action spaces. Overestimated Q-values cause the actor to pursue actions that look good to an over-optimistic critic but don't actually yield profit.

---

## 6. Reward Function: Pure Financial Signal

Replace all hand-tuned multipliers with a single pure financial reward:

```python
def compute_reward(self, bought_kwh, sold_kwh, price_ct_per_kwh):
    revenue_eur = (sold_kwh * price_ct_per_kwh) / 100
    cost_eur    = (bought_kwh * price_ct_per_kwh) / 100
    profit_eur  = revenue_eur - cost_eur

    # Hard constraint only — never violated under normal operation
    if self.soc < 0.05:
        return profit_eur - 1.0   # penalty for hitting hard floor
    if self.soc > 0.97:
        return profit_eur - 1.0   # penalty for hitting hard ceiling

    return profit_eur
```

**Arguments for removing the timing bonus:**

The current arbitrage bonus (`+2.0` for buying below 70% avg, etc.) is a dense shaping signal overlaid on a sparse financial one. Dense shaping is generally beneficial, but these thresholds are **hyperparameters with no financial derivation**. Why 70%? Why not 65% or 75%? The bonus can teach the agent to maximise threshold-crossings rather than profit — a subtle reward-hacking behaviour.

SAC's entropy term already provides persistent exploration pressure. The agent will discover that buying at low price percentiles produces better long-run Q-values without needing an explicit hint. In the long run, a clean financial signal is more robust because it doesn't create a second implicit objective.

**SOC penalty only at hard limits:**

The current penalty fires at SOC < 0.10, < 0.20, < 0.30, and > 0.92 with different magnitudes. This creates a gradient that pushes the agent away from the operating range edges even when the financial opportunity is good. Setting the penalty only at the true hard limits (5% and 97%) allows the soft limits (20%–85%) to be enforced by the policy learning that going out of range reduces future profit opportunity.

---

## 7. Training Protocol

### Experience Replay Buffer

```
capacity      : 100,000 transitions
sampling      : uniform random mini-batch (could upgrade to prioritised later)
batch_size    : 256
starts        : after 2,000 warm-up steps (random policy to seed the buffer)
```

The buffer holds 100,000 transitions. At 96 steps/day, this is ~1,040 days of simulation experience. Every rare event (price spike, deep price trough) remains available for learning until naturally displaced by newer data. Contrast with the current approach: a 48-step trajectory is cleared every 12 simulated hours.

### Update Schedule

```
critic_updates_per_step : 1
actor_updates_per_step  : 1 (delayed by 2 critic updates, i.e. every other step)
target_update_tau       : 0.005  (Polyak averaging, same as current)
```

Update after every environment step (online SAC). This is computationally heavier per simulation step but converges in far fewer environment steps. The net simulation time to reach good performance is much shorter.

### Hyperparameters

```python
actor_lr    = 3e-4     # Adam
critic_lr   = 3e-4     # Adam
alpha_lr    = 3e-4     # automatic entropy temperature
gamma       = 0.99     # longer-horizon discount than current 0.98
target_entropy = -1.0  # entropy target: H(π) ≈ -dim(A) = -1 for scalar action
```

### Automatic Entropy Temperature (α)

The temperature `α` is updated by minimising:

```
L(α) = E[-α · (log_π(a|s) + target_entropy)]
```

When the policy becomes too deterministic (entropy < target), `α` increases, forcing more exploration. When entropy is high enough, `α` decreases to let the policy exploit. This replaces the entire `exploration_rate` decay schedule with a single principled mechanism.

---

## 8. Market Interface: Simplify the Action Space

### Current Two-Stage Interface (Remove This)

```
RL action → action_to_bid() → bid premium/discount heuristic → market
```

The heuristic in `action_to_bid()` applies different premiums depending on `price_percentile` thresholds (0.15, 0.35, 0.65, 0.85). This is a hand-coded sub-policy running on top of the learned policy. It introduces a non-differentiable transformation that breaks the policy gradient in the same way any reward shaping does.

### Proposed Direct Interface

The action `a ∈ [-1, +1]` maps directly to power with a fixed small price premium/discount:

```python
def action_to_bid(self, action):
    max_discharge, max_charge = self.provide_a_power()
    price_now = self.get_current_price()
    fixed_spread = 0.2  # Ct/kWh — just enough to beat the ext_grid margin

    if action > 0:     # charge
        power = action * max_charge
        bid_price = price_now + fixed_spread
        self.bid = [0, power, self.offer_function(bid_price), "lin"]

    elif action < 0:   # discharge
        power = -action * max_discharge
        ask_price = max(price_now - fixed_spread, 0.01)
        self.ask = [0, power, self.offer_function(ask_price), "lin"]
```

The fixed spread of 0.2 Ct/kWh ensures bids are slightly above the market clearing price (so they fill) and asks slightly below (so they fill), without the current regime of 0.5–1.2 Ct/kWh premiums that vary with percentile. The policy learns *when* to trade; the interface just ensures the bids clear.

---

## 9. Expected Improvements Over Current

| Dimension | Current (TD Actor-Critic) | Proposed (SAC) | Argument |
|-----------|--------------------------|----------------|----------|
| Sample efficiency | Very low — 48 transitions discarded each update | High — 100k replay buffer | Off-policy replay reuses every transition |
| Network capacity | 8 neurons, ~80 hidden params | 64 neurons, ~5k params | Joint SOC×price×time policy requires this |
| Exploration | ε-greedy with decay schedule, floor at 0.08 | Entropy maximisation, auto-temp | Principled, adapts to policy confidence |
| Reward signal | 3 competing components with arbitrary scales | Pure financial profit | Removes double-counting and reward hacking |
| Q-overestimation | Single critic, can over-estimate | Twin critics, take min | Prevents actor from exploiting critic errors |
| Policy gradient | Approximate (heuristic in action_to_bid corrupts it) | Exact (reparameterisation trick) | Direct gradient through action to profit |
| Rare event learning | Lost when trajectory cleared | Retained in replay buffer indefinitely | Price spikes/troughs are most valuable experiences |
| Hyperparameter count | exploration_rate, decay, min, 5× reward scales, SOC bands × 2, bid premium × 3 | alpha (auto-tuned), gamma, lr × 3 | Fewer manual knobs = less overfitting to hand-tuning |

### Quantitative Target

The current optimisation mode (HN_optimizer) provides a deterministic upper bound on achievable LEC arbitrage. The TD agent currently likely achieves 20–50% of this theoretical maximum, based on typical results from comparable 8-neuron AC implementations in energy markets.

A well-trained SAC agent in battery arbitrage literature reaches 70–90% of the deterministic optimum within 100–200 training days. The target for this redesign is to close the gap to >75% of the optimisation baseline, measurable by comparing cumulative profit over the same 90-day simulation window.

---

## 10. Implementation Roadmap

The redesign can be implemented entirely within `mesa_model/agents.py` without changing any other file. The method flag `"learning"` is preserved; the SAC code replaces only the `__init__` internals, `build_state`, `policy`, `action_to_bid`, `compute_reward`, `update_status`, and `update_parameters`.

### Phase 1: Replay Buffer and Critic (1–2 days)

- Implement `ReplayBuffer` class (circular numpy array, `push`, `sample`)
- Implement twin Q-networks (`_critic_forward(state, action, which=1|2)`)
- Implement critic loss: MSE against Bellman target with target networks
- Warm-up: run 2,000 steps of random policy to seed the buffer before learning begins

### Phase 2: SAC Actor and Entropy Tuning (1–2 days)

- Replace the deterministic policy with a Gaussian actor: `policy(state)` returns `(action, log_prob, mean)`
- Implement reparameterisation sampling: `action = tanh(mean + std * eps)`, `eps ~ N(0,1)`
- Implement log-prob correction for the tanh squashing: `log_prob -= log(1 - action² + ε)`
- Implement `alpha` as a learnable scalar with its own Adam optimiser

### Phase 3: State and Reward Cleanup (0.5 days)

- Expand `build_state()` to 15-D
- Replace `compute_reward()` with pure financial reward
- Replace `action_to_bid()` with the fixed-spread version

### Phase 4: Logging and Validation (0.5 days)

- Log Q-values, entropy, alpha, and fill rate alongside existing profit/SOC metrics
- Run a 90-day simulation and compare cumulative profit against the TD baseline and the optimisation baseline
- Tune `gamma` and `target_entropy` if needed

### Phase 5 (Optional): Prioritised Experience Replay

Rare high-value transitions (top 10% price events) can be assigned higher sampling weights proportional to their TD error. This is a well-established improvement over uniform replay, but adds complexity and should only be added if Phase 4 results plateau.

---

## 11. What This Plan Does Not Change

- The Mesa simulation loop (`main.py`, `model.py`) — untouched
- The market clearing mechanism (`optimization/market_optimizer.py`) — untouched
- All other agents (EV, heatpump, ext_grid) — untouched
- The config schema (`config.yaml`) — untouched; `method: "learning"` continues to activate the RL path
- CSV output and logging infrastructure — untouched

The redesign is a surgical replacement of the internals of the `storage` class `__init__`, and five methods, without changing any interface the rest of the codebase depends on.
