# Reinforcement Learning Storage Agent

### The Goal

**Maximize profit through energy arbitrage:**

```
Buy Low  →  Store  →  Sell High  →  Profit
```

## 1. Architecture

### Data Flow

```
Market Data → State Builder → Actor (Neural Network) → Action → Market Bid/Ask
                                      ↓
                               Trajectory Buffer
                                      ↓
                    ┌─────────────────────────────────┐
                    │  TD Actor-Critic Update         │
                    │  Critic: learn V(s)             │
                    │  Actor:  improve π using δ      │
                    └─────────────────────────────────┘
                                      ↓
                              Updated Weights
```

---

## 2. State Representation

The agent creates a **9-dimensional state vector**:

```python
state = [soc, price_norm, price_diff_1h, price_diff_4h,
         sin_hour, cos_hour, sin_dow, cos_dow, price_percentile]
```

### State Features Explained

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `soc` | [0, 1] | Current battery charge level |
| 1 | `price_norm` | ~[-1, 1] | Current price relative to 24h rolling average |
| 2 | `price_diff_1h` | ~[-1, 1] | Price 1 hour ago relative to current price (lag-based, no future leakage) |
| 3 | `price_diff_4h` | ~[-1, 1] | Price 4 hours ago relative to current price (lag-based, no future leakage) |
| 4 | `sin_hour` | [-1, 1] | Sine of hour-of-day (circular encoding) |
| 5 | `cos_hour` | [-1, 1] | Cosine of hour-of-day (circular encoding) |
| 6 | `sin_dow` | [-1, 1] | Sine of day-of-week (circular encoding) |
| 7 | `cos_dow` | [-1, 1] | Cosine of day-of-week (circular encoding) |
| 8 | `price_percentile` | [0, 1] | Where the current price sits within the last 24h observed prices |

**Why sin/cos for time?**
Raw hour values (0–23) create a discontinuity at midnight: hour 23 and hour 0 appear far apart numerically but are only 15 minutes apart. Encoding as sin/cos pairs makes the time space continuous and circular — midnight wraps smoothly.

**Why lag-based price diffs?**
`price_diff_1h` uses the price observed 1 hour ago (stored in a rolling buffer), not a future forecast. This prevents data leakage — the agent only uses prices it has already seen.

### Example State Interpretation

```python
state = [0.45, -0.25, +0.15, +0.40, -1.0, 0.0, 0.5, 0.87, 0.12]
```

**Breakdown:**
- Battery is 45% full
- Current price is 25% below the 24h average
- Price was 15% higher 1 hour ago (price falling)
- Price was 40% higher 4 hours ago (price has been falling for a while)
- It's around 6 AM on a weekday
- Price is in the bottom 12% of the last 24 hours

**Optimal decision:** To buy. (Price is cheap and historically low)

---

## 3. Neural Network Policy (Actor)

### Architecture

The actor is a **2-layer neural network** with an 8-neuron hidden layer:

```
Input (9 features)
       │
       ▼
┌──────────────────────────┐
│  Hidden Layer            │
│  W1: [8×9] weight matrix │
│  b1: [8] bias vector     │
│  Activation: LeakyReLU   │
│  8 neurons               │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Output Layer            │
│  W2: [8] weight vector   │
│  b2: [1] bias scalar     │
│  Activation: Tanh        │
│  1 neuron                │
└──────────────────────────┘
       │
       ▼
Action ∈ [-1, +1]
```

**Layer 1 (Hidden):**
```
h1_in = W1 · x + b1        ← matrix multiply: shape [8]

h1 = LeakyReLU(h1_in)

  if h1_in > 0:  h1 = h1_in
  else:          h1 = 0.1 × h1_in
```

**Layer 2 (Output):**
```
z = W2 · h1 + b2            ← dot product: scalar

raw_action = tanh(z) ∈ [-1, +1]
```

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `x` | Input state vector (length 9) |
| `W1` | Weight matrix, shape [8 × 9] — connects inputs to hidden neurons |
| `b1` | Bias vector, shape [8] — one bias per hidden neuron |
| `h1_in` | Pre-activation hidden values (before LeakyReLU) |
| `h1` | Post-activation hidden values (after LeakyReLU), shape [8] |
| `W2` | Output weight vector, shape [8] — connects hidden to output |
| `b2` | Output bias, scalar |
| `z` | Pre-activation output value (before tanh) |
| `raw_action` | Network output tanh(z) before safety overrides — stored separately for clean gradient computation |
| `·` | Dot product / matrix-vector multiply |

### Why These Activations?

| Activation | Location | Purpose |
|------------|----------|---------|
| **LeakyReLU** | Hidden layer | Prevents dead neurons — gradient still flows when h1_in < 0 (slope = 0.1 instead of 0) |
| **Tanh** | Output layer | Bounds action to [-1, +1] — maps naturally to charge/discharge |

### Safety Overrides

After the network computes `raw_action`, hard limits are applied for emergencies:

```python
# Emergency conditions (override neural network)
if soc < 0.10:  action = +1.0   # Must charge immediately
if soc > 0.95:  action = -1.0   # Must discharge immediately
```

`raw_action` is kept unchanged so gradients are computed on the clean network output, not the overridden value.

---

## 4. Action Space

### Continuous Action

The actor outputs a single continuous value:

```
action ∈ [-1, +1]
```

| Action Value | Meaning | Market Behaviour |
|--------------|---------|-----------------|
| +1.0 | Maximum charge | Buy as much as physically possible |
| +0.5 | Moderate charge | Buy at half capacity |
| 0.0 | Hold | No trading |
| -0.5 | Moderate discharge | Sell at half capacity |
| -1.0 | Maximum discharge | Sell as much as safely allowed |

### Action to Market Bid/Ask

```python
if action > 0:  # Charge — place a buy bid
    power = action × max_charge
    bid_price = current_price + bid_premium   # premium scales with price percentile

if action < 0:  # Discharge — place a sell ask
    power = |action| × max_safe_discharge
    ask_price = current_price - ask_discount  # discount scales with price percentile
```

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `max_charge` | Maximum power the battery can absorb this timestep (kW), derived from SOC |
| `max_safe_discharge` | Maximum power the battery can release while staying above `soc_floor` |
| `bid_premium` | Markup above market price to ensure the buy order fills (0.5–1.2 ct/kWh depending on price percentile) |
| `ask_discount` | Markdown below market price to ensure the sell order fills (0.1–0.4 ct/kWh depending on price percentile) |
| `\|action\|` | Absolute value of action |

---

## 5. Reward Function

The reward signal guides learning. It has three components:

### 5.1 Profit Reward (Primary)

```python
energy_cost_eur = price × (bought - sold) / 100   # actual €
profit_reward = -energy_cost_eur × penalty_scale
```

`penalty_scale` is normally 5.0. It is reduced to 2.5 when SOC < 0.35 and the agent is buying at a non-excessive price — to avoid discouraging necessary recovery charging.

| Transaction | Energy Cost | Profit Reward |
|-------------|-------------|---------------|
| Buy 10 kWh @ 5 ct | +€0.50 | -€2.50 |
| Sell 10 kWh @ 8 ct | -€0.80 | +€4.00 |

**Note:** Negative cost = profit = positive reward.

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `price` | Current market price in ct/kWh |
| `bought` | Energy bought this timestep in kWh |
| `sold` | Energy sold this timestep in kWh |
| `penalty_scale` | Reward scaling factor (5.0 normally, 2.5 during low-SOC recovery) |

### 5.2 Arbitrage Bonus (Timing)

```python
price_ratio = price / avg_price

if bought > 0:
    if price_ratio < 0.70:   bonus += 2.0   # Very cheap buy
    elif price_ratio < 0.85: bonus += 1.0   # Cheap buy
    elif price_ratio < 0.95: bonus += 0.3   # Slightly cheap buy
    bonus *= (0.3 + 0.7 × trade_size_norm)  # Scale by trade size

if sold > 0:
    if price_ratio > 1.30:   bonus += 2.0   # Very expensive sell
    elif price_ratio > 1.15: bonus += 1.0   # Expensive sell
    elif price_ratio > 1.05: bonus += 0.3   # Slightly expensive sell
    bonus *= (0.3 + 0.7 × trade_size_norm)
```

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `avg_price` | Rolling 24h average of observed market prices |
| `price_ratio` | Current price divided by avg_price — how expensive this moment is relative to recent history |
| `trade_size_norm` | Trade size as a fraction of max possible energy per step — rewards larger, more committed trades |

### 5.3 SOC Penalty (Safety)

```python
if soc < 0.10:   penalty = -2.0 × (0.10 - soc) / 0.10   # max -2.0 at soc = 0
elif soc < 0.20: penalty = -0.8 × (0.20 - soc) / 0.10
elif soc < 0.30: penalty = -0.3 × (0.30 - soc) / 0.10
elif soc > 0.92: penalty = -2.0 × (soc - 0.92) / 0.08   # max -2.0 at soc = 1
else:            penalty = 0.0
```

The penalty tapers smoothly from the boundary inward, so extreme SOC states are strongly discouraged without creating discontinuous jumps in the reward signal.

### Total Reward

```python
reward = profit_reward + arbitrage_bonus + soc_penalty
```

---

## 6. Learning Algorithm

### TD Actor-Critic

The agent uses **Temporal Difference (TD) Actor-Critic** — a two-network architecture that updates weights every episode (every 48 timesteps = 12 hours) using bootstrapped value estimates rather than waiting for full episode returns.

- The **Actor** (policy network) decides what action to take.
- The **Critic** (value network) estimates how good each state is, providing a training signal for the actor.

### Episode Cycle

Each timestep:

```python
# 1. Act
state = build_state()
action, raw_action, h1, h1_in = policy(state)   # actor forward pass
action += gaussian_noise                          # exploration
place_market_bid_or_ask(action)

# 2. Observe outcome (after market clears)
reward = compute_reward(bought, sold, price)
next_state = build_state()
trajectory.append((state, action, raw_action, reward, next_state, h1, h1_in, soc))
```

After 48 steps, `update_parameters()` runs and the trajectory is cleared.

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `raw_action` | tanh(z) before safety overrides — stored so gradients are computed on the clean network output |
| `h1` | Post-activation hidden vector — stored for backprop |
| `h1_in` | Pre-activation hidden vector — stored for LeakyReLU derivative during backprop |
| `trajectory` | On-policy buffer of transitions collected in this episode; cleared after each update |

### The TD Error

At each step, the TD error `δ` measures how wrong the critic's prediction was:

```
δ = r + γ · V(s') - V(s)
```

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `δ` (delta) | TD error — the surprise signal. Positive means the outcome was better than expected; negative means worse |
| `r` | Reward received at this timestep |
| `γ` (gamma) | Discount factor (0.98) — future rewards are worth slightly less than immediate ones. A reward 1 step away is worth 98% of an immediate reward |
| `V(s)` | Critic's estimate of the value of the current state s |
| `V(s')` | Critic's estimate of the value of the next state s' |
| `γ · V(s')` | Bootstrapped estimate of future value — this is what makes it TD rather than Monte Carlo |

A positive `δ` means the actual outcome was better than the critic predicted → the actor is nudged toward that action. A negative `δ` discourages it.

### Critic Update

The critic is updated 3 times per episode using semi-gradient descent:

```
θ_critic += critic_lr · δ · ∇V(s)
```

`V(s')` is computed from a **frozen target critic** (a slowly-updated copy of the critic) to prevent the training target from moving too fast and destabilising learning.

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `θ_critic` | Critic network weights (W1, b1, W2, b2) |
| `critic_lr` | Critic learning rate (0.05) |
| `∇V(s)` | Gradient of the critic's value estimate with respect to its weights |
| `target critic` | A frozen copy of the critic used only to compute `V(s')`. Updated slowly via Polyak averaging |

### Actor Update

After the critic is updated, the actor is updated once using the now-improved critic's TD errors as advantages:

```
advantage = normalise(δ)

θ_actor += actor_lr · advantage · ∇log π(a|s)
```

Advantages are **normalised** (zero mean, unit std) across the episode before the update, so large TD spikes do not dominate the gradient.

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `θ_actor` | Actor network weights (W1, b1, W2, b2) |
| `actor_lr` | Actor learning rate (~0.03, adaptive) |
| `advantage` | Normalised TD error — how much better/worse this transition was than the episode average |
| `∇log π(a\|s)` | Policy gradient — direction to update weights to make this action more (or less) likely |
| `normalise(δ)` | (δ - mean(δ)) / std(δ) — centres and scales the advantages across the episode |

### Target Critic — Polyak Averaging

After each update, the frozen target critic is softly moved toward the current critic:

```
θ_target = (1 - τ) · θ_target + τ · θ_critic
```

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| `θ_target` | Frozen target critic weights |
| `τ` (tau) | Polyak averaging coefficient (0.005) — small value means the target moves very slowly, giving stable bootstrap targets |

### Weight Decay

After each actor update, the output weights are slightly shrunk:

```
W1 *= 0.999
W2 *= 0.999
```

This prevents weights from growing large enough to push the tanh output into saturation (where gradients vanish).

### Adaptive Actor Learning Rate

```python
if recent_profits > older_profits:
    actor_lr *= 1.1   # Working → speed up
else:
    actor_lr *= 0.9   # Struggling → slow down

actor_lr = clip(actor_lr, 0.01, 0.08)
```

### Best-Weights Restore

If profit does not improve for 10 consecutive episodes, the agent automatically reverts actor and critic weights to the best-performing checkpoint.

---

## 7. Hyperparameters

### Learning Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `actor_lr` | 0.03 | Actor learning rate (adaptive, range 0.01–0.08) |
| `critic_lr` | 0.05 | Critic learning rate |
| `gamma` | 0.98 | Discount factor for TD error: r + γ·V(s') |
| `update_frequency` | 48 steps | Episode length (12 hours at 15-min timesteps) |
| `critic_passes` | 3 | Critic gradient steps per episode |
| `target_update_tau` | 0.005 | Polyak coefficient for target critic soft-update |
| `restore_patience` | 10 episodes | Episodes below best profit before reverting to best weights |

### Exploration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `exploration_rate` | 0.35 | Initial Gaussian noise std added to action |
| `exploration_decay` | 0.997 | Multiplied each episode |
| `min_exploration` | 0.08 | Floor — exploration never drops below this |

### SOC Boundaries

| Parameter | Value | Description |
|-----------|-------|-------------|
| `soc_floor` | 0.10 | Below this: charge-only mode (no discharge allowed) |
| `soc_ceiling` | 0.85 | Operating ceiling for normal discharge planning |
| `soc_target` | 0.50 | Centre of the safe operating range |

---
