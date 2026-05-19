# RL Energy Trading Agent — How It Works

This document explains the full system from scratch: what the simulation is, how the market works, how the storage agent makes decisions, and how it learns. It covers both learning algorithms implemented in this project — Monte Carlo (REINFORCE) and Temporal Difference (Actor-Critic).

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Simulation Environment](#2-the-simulation-environment)
3. [The RL Problem: What Is the Agent Trying to Do?](#3-the-rl-problem-what-is-the-agent-trying-to-do)
4. [One Timestep, Start to Finish](#4-one-timestep-start-to-finish)
5. [What the Agent Observes: The State Vector](#5-what-the-agent-observes-the-state-vector)
6. [How the Agent Decides: The Policy Network](#6-how-the-agent-decides-the-policy-network)
7. [From Decision to Market Bid](#7-from-decision-to-market-bid)
8. [The Reward Signal](#8-the-reward-signal)
9. [Prior Knowledge vs. First Experience](#9-prior-knowledge-vs-first-experience)
10. [Monte Carlo Algorithm (REINFORCE)](#10-monte-carlo-algorithm-reinforce)
11. [Temporal Difference Algorithm (Actor-Critic)](#11-temporal-difference-algorithm-actor-critic)
12. [MC vs TD: Side-by-Side Comparison](#12-mc-vs-td-side-by-side-comparison)
13. [Hyperparameter Reference](#13-hyperparameter-reference)

---

## 1. What Is This Project?

This is a **multi-agent energy market simulation** where a battery storage unit learns to trade electricity profitably.

The setting is a **Local Energy Market (LEM)** — a small community grid (one farm, several households, one industrial consumer, renewable generators, electric vehicles, heat pumps, and one battery). These agents trade energy with each other at prices set by a market clearing mechanism. If there is a surplus or deficit, the community can also buy from or sell to the external grid (think: the national electricity market).

The battery agent's goal is **energy arbitrage**: buy electricity when it is cheap, store it, and sell it back when prices are high. It learns this strategy through reinforcement learning — it is not programmed with explicit rules, but rather discovers the strategy through trial and error.

The same physical simulation infrastructure runs two different RL algorithms:
- **Monte Carlo (main branch)** — learns by looking back at a complete episode of experience
- **Temporal Difference / Actor-Critic (algo/temporal-difference branch)** — learns step-by-step, without waiting for the episode to end

---

## 2. The Simulation Environment

### Physical Setup

The grid is a low-voltage rural network modelled with **PandaPower** and **SimBench**. It has buses (connection points), transmission lines, and the following agents attached to it:

| Agent type | Role | Buys / Sells |
|---|---|---|
| `res` | Renewable generator (solar/wind) | Sells only |
| `ext_grid` | External grid connection | Buys and sells (unlimited) |
| `household` | Residential load | Buys only |
| `farm` | Agricultural load | Buys only |
| `industry` | Industrial load | Buys only |
| `heatpump` | Flexible thermal load | Buys only |
| `EV` | Electric vehicle | Buys only |
| `storage` | Battery — **the RL agent** | Buys and sells |

### Time

The simulation runs at **15-minute intervals** (96 timesteps per day). Each timestep is one round of the market.

### The Market

At each timestep, all agents submit bids (willingness to buy) and asks (willingness to sell). A **market optimizer** (`optimization/market_optimizer.py`) solves a welfare-maximisation problem to find how much each agent trades and at what price. The result is a clearing price and a set of energy exchanges.

The storage agent competes against the `ext_grid`, which acts as the price floor and ceiling: it will always buy or sell, but at a margin above/below the spot price. The storage agent must offer competitive prices to get its trades filled.

---

## 3. The RL Problem: What Is the Agent Trying to Do?

Reinforcement learning frames the agent's situation as a **Markov Decision Process (MDP)**:

- **State** `s`: what the agent observes about the world at this moment (battery charge level, current price, price trend, time of day, etc.)
- **Action** `a ∈ [−1, +1]`: how aggressively to charge (+1 = charge at full power) or discharge (−1 = sell at full power)
- **Reward** `r`: a scalar signal after each action — positive for profitable trades, negative for bad timing or dangerous battery states
- **Policy** `π(s)`: the agent's decision function — a neural network that maps state → action

The agent's goal is to find a policy π that maximises cumulative future reward:

```
Objective: maximise E[∑ γᵗ rₜ]
```

where γ = 0.98 is a **discount factor** that makes near-term rewards more valuable than distant ones.

The agent has **no domain knowledge baked in** (beyond weak initialisation discussed in Section 9). It must discover that buying at low prices and selling at high prices is profitable purely from the reward signal.

---

## 4. One Timestep, Start to Finish

Understanding the execution order is essential. Each call to `model.step()` in `main.py` follows this sequence:

```
┌─────────────────────────────────────────────────────────────┐
│  main.py — outer loop (runs total_steps times)              │
│                                                             │
│  model.step()                                               │
│    │                                                         │
│    ├─ 1. Each agent calls agent.step()                       │
│    │       └─ storage.step():                                │
│    │            a. Build state s from current conditions     │
│    │            b. Pass s through policy network → action a  │
│    │            c. Convert action to a market bid/ask        │
│    │                                                         │
│    ├─ 2. MarketOptimizer solves the market                   │
│    │       └─ Returns: who traded how much, at what price    │
│    │                                                         │
│    └─ 3. Each agent calls agent.update_status()             │
│            └─ storage.update_status():                       │
│                 a. Read actual energy bought/sold            │
│                 b. Update battery charge level (SOC)         │
│                 c. Compute reward for this step              │
│                 d. Store experience in memory                │
│                 e. [Every N steps] call update_parameters()  │
│                      └─ Run the learning algorithm           │
└─────────────────────────────────────────────────────────────┘
```

**Key point**: the agent makes its decision (`step`) before it knows the market outcome. It only finds out what actually happened — and earns its reward — in `update_status`, one step later. This is why the experience tuple stores `last_state` and `last_action`: the agent needs to remember what it did in the previous step to learn from what just happened.

---

## 5. What the Agent Observes: The State Vector

Both algorithms share the same physical battery simulation but build the state slightly differently.

### Monte Carlo state (9 features)

| Index | Feature | Formula | Why |
|---|---|---|---|
| 0 | `soc` | raw value ∈ [0, 1] | Battery charge level |
| 1 | `max_discharge` | kW available to sell | Physical capacity (from SOC) |
| 2 | `max_charge` | kW available to buy | Physical capacity (from SOC) |
| 3 | `price_norm` | `(p_now − p̄) / p̄` | Is now cheap or expensive vs. recent average? |
| 4 | `price_diff_1h` | `(p_1h − p_now) / p_now` | Is price rising in the next hour? |
| 5 | `price_diff_4h` | `(p_4h − p_now) / p_now` | Is price rising in the next 4 hours? |
| 6 | `hour_norm` | `hour / 23` | Linear time of day (0 to 1) |
| 7 | `dow_norm` | `weekday / 6` | Linear day of week (0 to 1) |
| 8 | `price_percentile` | fraction of recent prices below p_now | How extreme is the current price? |

### Temporal Difference state (9 features)

| Index | Feature | Formula | Why |
|---|---|---|---|
| 0 | `soc` | raw value ∈ [0, 1] | Battery charge level |
| 1 | `price_norm` | `(p_now − p̄) / p̄` | Price deviation from recent average |
| 2 | `price_diff_1h` | `(p_1h − p_now) / p_now` | 1-hour price trend |
| 3 | `price_diff_4h` | `(p_4h − p_now) / p_now` | 4-hour price trend |
| 4 | `sin_hour` | `sin(2π × hour / 24)` | Continuous circular encoding of time |
| 5 | `cos_hour` | `cos(2π × hour / 24)` | Ensures midnight is continuous with 23:45 |
| 6 | `sin_dow` | `sin(2π × weekday / 7)` | Continuous day-of-week |
| 7 | `cos_dow` | `cos(2π × weekday / 7)` | Ensures Sunday is continuous with Monday |
| 8 | `price_percentile` | fraction of recent prices below p_now | Price extremity |

**Key differences:**
- TD drops `max_discharge` and `max_charge` because they are deterministic functions of SOC — including them would add redundant information.
- TD uses sin/cos pairs for time instead of linear normalisation. This is important: with linear encoding, 23:00 and 00:00 look maximally different to the network, even though they are adjacent. Sin/cos encoding wraps continuously around.
- Both use a rolling 96-step (24-hour) deque of **observed** prices for the average and percentile — not the full dataset. This prevents the agent from accidentally using future price information.

### Where do the price forecasts come from?

The 1-hour and 4-hour "forecasts" are simply looked up from the external spot price dataset (`spot_price.csv`). They are **ground-truth future prices**, not learned predictions. This gives the agent perfect short-term foresight — a deliberate simplification that lets the study focus on the learning dynamics rather than forecasting.

---

## 6. How the Agent Decides: The Policy Network

Both algorithms use a two-layer neural network to map state → action. The architectures differ.

### Monte Carlo policy (scalar hidden unit)

```
Input:  9 features × named weights  →  h1_input  (scalar dot product)
                                         ↓
                                    LeakyReLU
                                         ↓
                                        h1        (scalar)
                                         ↓
                              h1 × hidden_w + bias
                                         ↓
                                       tanh
                                         ↓
                                    action ∈ [−1, +1]
```

Layer 1 is a manually-named weight dictionary (e.g., `theta["layer1"]["soc_w"]`). There is one hidden scalar, not a vector. This is a very compact network — effectively a linear model with a nonlinear activation.

**Initial weights (hand-coded priors):**
```
soc_w       = −5.0   # strong pull: low SOC → charge (+), high SOC → sell (−)
price_w     = −1.5   # lower price → more positive h1 → charge
price_diff_w =  0.8  # rising price → charge now before it gets expensive
forecast_1h_w = 0.5
forecast_4h_w = 0.5
time_w      =  0.1
dow_w       =  0.0
bias        = −0.3
hidden_w    =  1.8
```

These are strong domain-knowledge priors. The network begins with a meaningful policy already encoded, and learning refines it.

### Temporal Difference policy (8-neuron hidden layer)

```
Input:  x (9,)
         ↓
    W1 @ x + b1       shape: (8,)     [matrix multiply]
         ↓
    LeakyReLU(h1_in)  shape: (8,)     [element-wise]
    clip to [−5, 5]
         ↓
    W2 · h1 + b2      shape: scalar   [dot product]
         ↓
    tanh(z)
         ↓
    action ∈ [−1, +1]
```

**Initial weights (small random):**
```python
W1 ~ N(0, 0.1)   shape: (8, 9)
W2 ~ N(0, 0.1)   shape: (8,)
b1 = zeros(8)
b2 = 0.0
```

Weights are initialised from a seeded RNG (`seed=42`) so results are reproducible. The small scale (0.1) means early actions are near-neutral, letting the SOC/price biases described below drive early exploration.

**LeakyReLU** is used instead of ReLU to prevent dead neurons: if `h1_in < 0`, the gradient is 0.1 × input rather than 0, so weight updates can still flow through inactive neurons.

### Safety overrides applied after the network output

Both algorithms apply rule-based corrections on top of the raw network action. These are not learned — they are hard-coded safety rails.

**Monte Carlo — graduated SOC overrides (applied before noise):**
```
SOC < 0.10  →  action = 1.0            (emergency: charge at full power)
SOC < 0.15  →  action = max(action, 0.8)
SOC < 0.20  →  action = max(action, 0.5)
SOC < 0.30  →  action = max(action, 0.2)
SOC > 0.90  →  action = −1.0           (emergency: discharge at full power)
SOC > 0.85  →  action = min(action, −0.9)
SOC > 0.70  →  action = min(action, −0.3)
SOC > 0.60  →  action = min(action, 0.0)
```

Price percentile nudges (applied after noise):
```
price_percentile < 0.15 and SOC < 0.60  →  action += 0.25
price_percentile > 0.85 and SOC > 0.35  →  action −= 0.25
```

**Temporal Difference — dynamic soft biases (blended with network output):**

The TD agent uses a softer blending approach. A `soc_bias` and `price_bias` are computed, then mixed with the network action using a weight that depends on how extreme the SOC is:

```
bias_weight = 0.50  if SOC < 0.15     (safety dominates)
            = 0.30  if SOC < 0.20
            = 0.10  otherwise          (network dominates)

action = (1 − bias_weight) × network_action + bias_weight × (soc_bias + price_bias)
```

Hard overrides only at true emergencies:
```
SOC < 0.10  →  action = 1.0   (override everything: charge)
SOC > 0.95  →  action = −1.0  (override everything: discharge)
```

**Exploration noise** is added after all biases. Both algorithms inject Gaussian noise to encourage the agent to try actions it would not otherwise choose:
```
noise ~ N(0, exploration_rate × 0.3)   [MC, only when 0.25 < SOC < 0.75]
noise ~ N(0, exploration_rate × 0.4)   [TD, when SOC < 0.95]
```

The exploration rate starts at 0.35, decays by a factor of 0.997 each episode, and floors at 0.08. As the agent improves, it explores less and exploits its learned policy more.

---

## 7. From Decision to Market Bid

The policy outputs a continuous action `a ∈ [−1, +1]`. This must be converted into a concrete market offer.

### Charge (action > 0): submit a bid to buy

The desired charge power is `action × max_charge_power`. The agent submits a **bid** — a willingness to pay — at a price set slightly above the current market price so the bid is likely to be filled:

```
bid_price = current_price + premium

MC:  premium = current_price × 0.4 × action + 15
TD:  premium = 0.5  if price_percentile < 0.15  (cheap: don't overpay)
             = 0.8  if price_percentile < 0.35
             = 1.2  otherwise
```

### Discharge (action < 0): submit an ask to sell

The agent submits an **ask** at a small discount to the market price to ensure the sale goes through:

```
ask_price = current_price − discount

MC:  ask_price = current_price × (0.7 + 0.3 × (1 + action))
TD:  discount  = 0.10  if price_percentile > 0.85  (expensive: stay near market)
               = 0.25  if price_percentile > 0.65
               = 0.40  otherwise
```

### Battery state of charge (SOC) update

After the market clears and actual `bought` and `sold` energy are known:

```
energy_delta = bought × η − sold / η        (η = efficiency = 0.95)
soc_delta    = energy_delta / capacity
SOC_new      = SOC_old × discharge_factor + soc_delta
```

The `discharge_factor` models self-discharge: the battery loses a small fraction of its stored energy each timestep even if it does nothing. In TD this is modelled as `discharge_per_day ^ (timestep_seconds / 86400)`.

---

## 8. The Reward Signal

The reward function is shared in structure across both algorithms but differs in some parameter values.

### Three components

**1. Profit reward** — the primary financial signal

```
energy_cost = price × (bought − sold) / 100    [in €]
profit_reward = −energy_cost × 5.0
```

Buying costs money (negative reward); selling earns money (positive reward). The ×5 scale amplifies the financial signal relative to the other components.

TD adds a special case: when SOC is low (< 35%) and prices are not high, the penalty for buying is halved to reduce the agent's reluctance to charge when it is critically empty.

**2. Arbitrage bonus** — rewards good timing

This is an additive bonus for making smart trades. It does not penalise average trades — only rewards good ones.

| Condition | MC bonus | TD bonus |
|---|---|---|
| Buy when price < 80% of average | +3.0 | — |
| Buy when price < 70% of average | — | +2.0 |
| Buy when price < 85% of average | — | +1.0 |
| Buy when price < 95% of average | — | +0.3 |
| Sell when price > 120% of average | +3.0 | — |
| Sell when price > 130% of average | — | +2.0 |
| Sell when price > 115% of average | — | +1.0 |
| Sell when price > 105% of average | — | +0.3 |
| Buy when price > 110% of average | −2.0 (MC only) | — |
| Sell when price < 90% of average | −2.0 (MC only) | — |

TD also scales the bonus by trade size, so a full-power trade at a good price earns the full bonus while a small hedge earns a fraction.

**3. SOC penalty** — safety guardrail

| SOC range | MC penalty | TD penalty |
|---|---|---|
| SOC < 0.10 | −100.0 | graded, max −2.0 |
| 0.10–0.20 | −30.0 × fraction | graded, max −0.8 |
| 0.20–0.30 | — | graded, max −0.3 |
| SOC > 0.90 (MC) or > 0.92 (TD) | −30.0 × fraction | graded, max −2.0 |

MC also has a small **SOC centering bonus** of +1.0 when SOC is in [0.40, 0.60]. TD removed this, as it was found to reward inaction (holding SOC at 0.5 without trading) over profitable arbitrage.

---

## 9. Prior Knowledge vs. First Experience

A central question in RL is: what does the agent know before it has seen any data?

### Monte Carlo: strong domain-knowledge priors

The MC network is initialised with **hand-coded weights** that encode domain knowledge:
- `soc_w = −5.0` — a very strong signal: low battery means charge, high battery means discharge
- `price_w = −1.5` — below-average prices are a signal to charge
- `price_diff_w = 0.8` — rising prices favour charging now

The agent does not start from zero. Its first decision will already reflect a reasonable arbitrage policy. Learning then adjusts these weights based on actual market outcomes.

### Temporal Difference: near-random initialisation

The TD network is initialised with **small random weights** (scale 0.1, seed 42). The network output starts near zero, meaning the agent's first actions are near-neutral and dominated by the soft SOC/price biases.

The **critic**, however, has warm-start priors:
```python
value_theta = {
    "soc_w":     0.0,
    "soc_sq_w": −1.0,   # states with extreme SOC are already judged as bad
    "price_w":   0.3,   # states with high prices are good for selling
    ...
}
```

This means even before the TD critic has seen any data, it already knows that extreme battery states are undesirable and that high prices are an opportunity. The actor is blind; the critic has a head start.

### Neither algorithm has visited any state before training

Both algorithms use **function approximation** (neural networks), not lookup tables. There is no state-visit memory. When the agent encounters a situation it has been in before, it does not recall that specific visit — instead, the neural network generalises from all past experience at once.

### What external information is always available

From the very first step, the agent can see:
- **Current market price** — from the spot price dataset
- **1h and 4h price forecasts** — ground-truth future prices from the same dataset (perfect foresight)
- **Current SOC** — tracked internally

The rolling 24-hour price history used for normalisation (`price_history` deque) is empty at step 1 and fills up over the first 96 steps. During this warm-up period, the agent falls back to the current price as its baseline, so early rewards are less well-calibrated.

---

## 10. Monte Carlo Algorithm (REINFORCE)

### Core idea

Monte Carlo RL waits until the end of an episode (one full day, 96 steps) before updating the network. It then computes the **actual total return** each action received — not an estimate, but the true sum of all future rewards from that point. The policy is updated to make good-return actions more likely and bad-return actions less likely.

This is the **REINFORCE** algorithm (Williams, 1992).

### Episode accumulation

During each timestep, the agent appends to an `episode_buffer`:

```python
episode_buffer.append((state, action, reward, h1))
```

This buffer stores the complete trajectory of one episode.

### Computing discounted returns

At the end of the episode, returns are computed **backwards** — from the last step to the first:

```
G_T     = r_T
G_{T-1} = r_{T-1} + γ × G_T
G_{T-2} = r_{T-2} + γ × G_{T-1}
...
G_t     = r_t + γ × G_{t+1}
```

In code:
```python
G = 0.0
for (_, _, reward, _) in reversed(episode_buffer):
    G = reward + gamma * G
    returns.insert(0, G)
```

`G_t` is the total discounted future reward the agent received from step `t` onwards. It is the ground truth of how good being in state `s_t` and taking action `a_t` turned out to be.

### Normalisation

Returns are normalised across the episode before the update:

```
returns_normalised = (G − mean(G)) / (std(G) + ε)
```

This reduces variance — some episodes may have uniformly high or low rewards, and normalisation ensures the gradient signal is always centred. Without this, large reward episodes would dominate and cause instability.

### Policy gradient update (backpropagation)

For each step `t` in the episode, the policy gradient is:

```
∂J/∂θ ∝ G_t × ∂log π(aₜ | sₜ) / ∂θ
```

Since the policy is `a = tanh(z)` and the log-probability derivative of a deterministic tanh policy is `(1 − a²)`:

```
grad_output = G_t × (1 − aₜ²)               # Layer 2 output gradient
grad_hidden = grad_output × hidden_w × leaky_relu_deriv(h1)

# Layer 2 update
Δhidden_w += grad_output × h1
Δbias_l2  += grad_output

# Layer 1 update (chain rule)
Δsoc_w    += grad_hidden × (soc − soc_target)
Δprice_w  += grad_hidden × price_norm
...
```

All gradients are clipped to `[−0.3, +0.3]` before the weight update, then averaged over the episode:

```
θ ← θ + lr × mean(Δθ, over episode)
```

### When does the update fire?

Every 96 steps (one full day). The episode buffer is cleared after each update.

### Learning rate adaptation

```python
if recent_profit > older_profit:
    lr *= 1.05   # Improving: be slightly bolder
else:
    lr *= 0.95   # Regressing: be more conservative
lr = clip(lr, 0.001, 0.015)
```

---

## 11. Temporal Difference Algorithm (Actor-Critic)

### Core idea

TD learning does not wait until the end of an episode. After every step, it estimates how good the current situation is with a **value function** (the critic), and uses this estimate to compute an **advantage signal** — did this step turn out better or worse than expected? The policy (actor) is then updated to make advantageous actions more likely.

The advantage is called the **TD error** (δ):

```
δ = r + γ × V(s') − V(s)
```

- If `δ > 0`: the outcome was better than the critic expected → reinforce this action
- If `δ < 0`: the outcome was worse than expected → suppress this action
- If `δ ≈ 0`: nothing surprising happened → small update

### The critic: a linear value function

The critic estimates the value (expected future reward) of a state using a linear model:

```
V(s) = w_soc × (soc − 0.5)
     + w_soc² × (soc − 0.5)²
     + w_price × price_norm
     + w_pd × price_diff
     + w_hour × sin(hour)
     + bias
```

This is a hand-designed feature set chosen to capture the most important structure: SOC deviation from target (linear and quadratic terms), price level, price trend, and time of day.

The quadratic SOC term `(soc − 0.5)²` is important: it makes the value function a bowl shape in SOC space — states near the centre (0.5) are estimated as more valuable than extremes, regardless of direction.

### Experience replay

Rather than learning from the most recent step only, the agent stores experience in a **replay memory**:

```
memory.append((state, action, raw_action, reward, next_state, h1, h1_in, soc))
```

The memory holds up to 4,000 transitions. At each update, a random mini-batch of 32 is sampled. This breaks temporal correlations and makes the gradient updates more stable.

### Update frequency and critic-first training

The update fires every **48 steps** (every 12 hours of simulation). At each update:

1. **Run 5 independent critic updates** (each on a fresh random batch)
2. **Run 1 actor update** (on another fresh random batch, using the now-improved critic)

Running the critic more often than the actor ensures the value function is a good baseline before the actor's gradient uses it. A poor critic produces noisy advantage estimates, which destabilise the actor.

### Critic update

For each transition in the batch:

```
δ = r + γ × V(s') − V(s)

grad["soc_w"]      += δ × (soc − 0.5)
grad["soc_sq_w"]   += δ × (soc − 0.5)²
grad["price_w"]    += δ × price_norm
grad["price_diff_w"] += δ × price_diff
grad["hour_w"]     += δ × sin(hour)
grad["bias"]       += δ
```

Averaged over the batch, clipped to `[−0.5, +0.5]`, then applied:

```
θ_critic ← θ_critic + critic_lr × mean(grad)
```

### Actor update

**Step 1 — compute advantages across the batch:**

```
advantages = [δᵢ for each transition i]
Â = (advantages − mean) / (std + ε)   # normalise
```

Normalisation here serves the same purpose as in MC: large TD error spikes would otherwise dominate the gradient.

**Step 2 — backpropagate through the network:**

For each transition `i`:

```
dz = Âᵢ × (1 − raw_action²)       # tanh derivative

grad_W2 += dz × h1                 # output layer
grad_b2 += dz

leaky_deriv = 1.0 if h1_in > 0 else 0.1
delta_h = dz × W2 × leaky_deriv    # hidden layer gradient (chain rule)

grad_W1 += outer(delta_h, state)   # input-to-hidden weights
grad_b1 += delta_h
```

Note that `raw_action` is the `tanh(z)` value **before** any bias blending or clipping. Using the clean network output for the gradient is essential: the biases and noise are intentional exploration tools, not part of the learned policy, so they should not be credited or blamed for outcomes.

**Step 3 — apply the update:**

```
θ_actor ← θ_actor + actor_lr × clip(mean(grad) / batch_size, −0.5, +0.5)
```

Weight decay is applied to prevent saturation:
```
W1 *= 0.999
W2 *= 0.999
```

### Adaptive actor learning rate

```python
if mean(recent 2 episodes profit) > mean(older 2 episodes profit):
    actor_lr *= 1.1   # trending up: exploit faster
else:
    actor_lr *= 0.9   # trending down: be careful
actor_lr = clip(actor_lr, 0.01, 0.08)
```

---

## 12. MC vs TD: Side-by-Side Comparison

| Property | Monte Carlo (main) | Temporal Difference (algo/temporal-difference) |
|---|---|---|
| **When it learns** | End of episode (96 steps / 1 day) | Every 48 steps (twice per day) |
| **What it uses to learn** | Actual full-episode returns G_t | Bootstrap estimate: r + γ×V(s') |
| **Has a critic?** | No — pure policy gradient | Yes — linear value function |
| **Network architecture** | 1 scalar hidden unit | 8-neuron hidden layer |
| **Weight initialisation** | Hand-coded domain priors | Small random (seed 42) + critic warm-start |
| **State dimensions** | 9 (includes max_power explicitly) | 9 (power implicit via SOC; sin/cos time) |
| **Time encoding** | Linear (hour/23, weekday/6) | Circular (sin/cos pairs) |
| **Memory capacity** | 15,000 transitions | 4,000 transitions |
| **Batch size** | 64 | 32 |
| **Gradient clip** | 0.3 | 0.5 |
| **SOC overrides** | Graduated hard clamps in policy() | Soft blended biases + 2 hard overrides |
| **Exploration region** | Only when 0.25 < SOC < 0.75 | Always when SOC < 0.95 |
| **Bad trade penalty** | Yes (buy high → −2.0, sell low → −2.0) | No — only rewards good timing |
| **SOC centering bonus** | Yes (+1.0 near 0.5) | Removed (caused inaction) |
| **Output files** | `output/mc/episode_logs.jsonl` | `output/td/episode_logs.jsonl` |

### Key trade-offs

**MC advantages:**
- Returns G_t are unbiased — the agent sees exactly what happened, with no estimation error from the critic
- Simpler algorithm: no critic to tune, no bootstrap bias
- Strong priors mean it has a reasonable policy from the very first episode

**MC disadvantages:**
- Must wait a full episode before any weight update — slow credit assignment
- High variance: a single bad step early in an episode inflates/deflates all G_t values for later steps
- Network has very limited capacity (1 scalar hidden unit)

**TD advantages:**
- Updates every 12 hours — faster credit assignment and more frequent policy improvement
- Critic reduces variance by providing a baseline; the actor only learns from surprises
- More network capacity (8 neurons) enables learning more complex state-action mappings

**TD disadvantages:**
- Bootstrap introduces bias: V(s') is an estimate, so δ inherits the critic's errors
- Two networks to tune (actor + critic), more hyperparameters
- Requires warm-start critic priors to avoid blind early updates

---

## 13. Hyperparameter Reference

### Shared parameters

| Parameter | Value | Meaning |
|---|---|---|
| `gamma` | 0.98 | Discount factor (future reward weight) |
| `exploration_rate` (initial) | 0.35 | Initial noise magnitude |
| `exploration_decay` | 0.997 | Multiplied each episode |
| `min_exploration` | 0.08 | Floor on exploration rate |
| `soc_floor` | 0.10 | Emergency low SOC threshold |
| `soc_ceiling` | 0.85 | Emergency high SOC threshold |
| `soc_target` | 0.50 | Neutral/target SOC |
| `efficiency` (η) | 0.95 | Round-trip battery efficiency |
| `SOC_start` | 0.40 | Battery SOC at simulation start |

### Monte Carlo parameters

| Parameter | Value | Meaning |
|---|---|---|
| `learning_rate` | 0.008 | Base weight update step size |
| `lr_range` | [0.001, 0.015] | Adaptive LR bounds |
| `batch_size` | 64 | Transitions per gradient step |
| `memory_size` | 15,000 | Replay buffer capacity |
| `update_frequency` | 96 steps | Update every 1 simulated day |
| `gradient_clip` | 0.3 | Per-parameter gradient ceiling |
| Hidden units | 1 scalar | Network width |
| Input features | 9 | State dimension |

### Temporal Difference parameters

| Parameter | Value | Meaning |
|---|---|---|
| `actor_lr` | 0.03 | Actor (policy) learning rate |
| `actor_lr_range` | [0.01, 0.08] | Adaptive LR bounds |
| `critic_lr` | 0.05 | Critic (value function) learning rate |
| `batch_size` | 32 | Transitions per gradient step |
| `memory_size` | 4,000 | Replay buffer capacity |
| `update_frequency` | 48 steps | Update every 12 simulated hours |
| `critic_updates_per_step` | 5 | Critic updates before each actor update |
| `gradient_clip` | 0.5 | Per-parameter gradient ceiling |
| `weight_decay` | 0.999 | Multiplied per update to prevent saturation |
| Hidden units | 8 vector | Network width |
| Input features | 9 | State dimension |

### Market / simulation parameters (from `config.yaml`)

| Parameter | Value | Meaning |
|---|---|---|
| `timestep` | 15 min | One simulation step |
| `sref` | 100 kVA | System reference power |
| `gridfee_LEC` | 5.5 ct/kWh | Grid fee for trades within the LEM |
| `gridfee_ext` | 11 ct/kWh | Grid fee for trades with external grid |
| `levies_LEC` | 2.35 ct/kWh | Levies for LEM trades |
| `levies_ext` | 4.7 ct/kWh | Levies for external trades |
| `ext_grid.margin_buy` | 1 ct/kWh | External grid buy markup above spot |
| `ext_grid.margin_sell` | 0.3 ct/kWh | External grid sell discount below spot |
