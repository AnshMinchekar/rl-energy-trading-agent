# Reinforcement Learning Storage Agent

### The Goal

**Maximize profit through energy arbitrage:**

```
Buy Low  →  Store  →  Sell High  →  Profit
```

## 1. Architecture

### Data Flow

```
Market Data → State Builder → Neural Network → Action → Market Bid/Ask
                                   ↓
                            Experience Memory
                                   ↓
                            Policy Gradient Update
                                   ↓
                            Updated Weights
```

---

## 2. State Representation

The agent creates a **9-dimensional state vector**:

```python
state = [soc, max_discharge, max_charge, price_norm, 
         price_diff_1h, price_diff_4h, hour_norm, dow_norm, price_percentile]
```

### State Features Explained

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `soc` | [0, 1] | Current battery charge level |
| 1 | `max_discharge` | [0, P_max] | Maximum power available to sell |
| 2 | `max_charge` | [0, P_max] | Maximum power available to buy |
| 3 | `price_norm` | ~[-1, 1] | Current price relative to 24h average |
| 4 | `price_diff_1h` | ~[-1, 1] | Expected price change in 1 hour |
| 5 | `price_diff_4h` | ~[-1, 1] | Expected price change in 4 hours |
| 6 | `hour_norm` | [0, 1] | Hour of day |
| 7 | `dow_norm` | [0, 1] | Day of week |
| 8 | `price_percentile` | [0, 1] | Price percentile in last 24 hours |

### Example State Interpretation

```python
state = [0.45, 0.37, 0.37, -0.25, +0.15, +0.40, 0.25, 0.33, 0.12]
```

**Breakdown:**
- Battery is 45% full 
- Can charge or discharge at 37 kW
- Current price is 25% below average 
- Price rising 15% in 1 hour, 40% in 4 hours
- It's 6 AM on Wednesday
- Price is in bottom 12% of last 24 hours

**Optimal decision:** To buy. (Since the prices are low but rising)

---

## 3. Neural Network Policy

### Architecture

The policy is a simple **2-layer neural network**:

```
Input (9 features)
       │
       ▼
┌─────────────────────┐
│  Hidden Layer       │
│  - Weighted sum     │
│  - LeakyReLU        │
│  - 1 neuron         │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Output Layer       │
│  - Weighted sum     │
│  - Tanh activation  │
│  - 1 neuron         │
└─────────────────────┘
       │
       ▼
Action ∈ [-1, +1]
```



**Layer 1 (Hidden):**
```
h1_input = soc_w × (soc - target) + price_w × price_norm + ... + bias₁

h1 = LeakyReLU(h1_input)

Here, if h1_input > 0
  h1 = h1_input 

  or else, 
  h1 = 0.1 × h1_input 
```

**Layer 2 (Output):**
```
z = hidden_w × h1 + bias₂

action = tanh(z) ∈ [-1, +1]
```

### Why These Activations?

| Activation | Location | Purpose |
|------------|----------|---------|
| **LeakyReLU** | Hidden layer | Prevents "dead neurons" during exploration, basically keeping the weights updated |
| **Tanh** | Output layer | Bounds action to [-1, +1] range |

### Network Weights

```python
theta = {
    "layer1": {
        "soc_w": -3.0,        # High SOC → sell (negative action)
        "price_w": -1.2,      # High price → sell
        "price_diff_w": 0.8,  
        "forecast_1h_w": 0.5,
        "forecast_4h_w": 0.5,
        "time_w": 0.2,
        "dow_w": 0.0,
        "bias": 0.0
    },
    "layer2": {
        "hidden_w": 1.5,
        "bias": 0.0
    }
}
```

**Intuition behind `soc_w = -3.0`:**
- When SOC is above target: `(soc - target) > 0`
- Multiplied by negative weight: `-3.0 × positive = negative`
- Negative hidden value → negative action → **SELL**

---

## 4. Action Space

### Continuous Action

The neural network outputs a single continuous value:

```
action ∈ [-1, +1]
```

| Action Value | Meaning | Market Behavior |
|--------------|---------|-----------------|
| +1.0 | Maximum charge | Buy at high price |
| +0.5 | Moderate charge | Buy at reasonable price |
| 0.0 | Hold | No trading |
| -0.5 | Moderate discharge | Sell at reasonable price |
| -1.0 | Maximum discharge | Sell at low price |

### Action to Market Bid/Ask

The continuous action is converted to market orders:

```python
if action > 0:  # Positive = Buy
    power = action × max_charge
    bid_price = current_price × (1 + 0.5 × action) + premium
    
if action < 0:  # Negative = Sell
    power = |action| × max_discharge
    ask_price = current_price × (0.8 + 0.4 × (1 + action))
```

### Safety Overrides

The policy includes hard limits to prevent the sotrage from completely depeleting:

```python
# Emergency conditions (override neural network)
if soc < 0.10:  action = +1.0   # Must charge!
if soc > 0.90:  action = -1.0   # Must discharge!

# Soft boundaries
if soc < 0.20:  action = max(action, +0.6)  # Encourage charging
if soc > 0.80:  action = min(action, -0.6)  # Encourage discharging
```

---

## 5. Reward Function

The reward signal guides learning. Our reward has three components:

### 5.1 Profit Reward (Primary)

```python
energy_cost = price × (bought - sold) / 100  # in €
profit_reward = -energy_cost × 5.0
```

| Transaction | Energy Cost | Profit Reward |
|-------------|-------------|---------------|
| Buy 10 kWh @ 5ct | +€0.50 | -€2.50 |
| Sell 10 kWh @ 8ct | -€0.80 | +€4.00 |

**Note:** Negative cost = profit = positive reward

### 5.2 SOC Penalty (Safety)

```python
if soc < 0.10:       penalty = -100.0   # Critical!
elif soc < 0.20:     penalty = -30.0 × (0.20 - soc) / 0.10
elif soc > 0.90:     penalty = -30.0 × (soc - 0.90) / 0.10
elif soc > 0.80:     penalty = -10.0 × (soc - 0.80) / 0.10
else:                penalty = 0.0      # Safe zone
```

### 5.3 Arbitrage Bonus (Timing)

```python
# Reward buying cheap
if bought > 0 and price < avg_price × 0.80:
    bonus += 3.0  # Good buy!

# Reward selling expensive  
if sold > 0 and price > avg_price × 1.20:
    bonus += 3.0  # Good sell!
```

### Total Reward

```python
reward = profit_reward + soc_penalty + arbitrage_bonus + soc_center_bonus
```

---

## 6. Learning Algorithm

### Policy Gradient Method

We use a variant of a type of RL that utilizes gradient ascent to maximize cumulative reward by adjustic its weights.

### Experience Memory

Each timestep, we store a transition:

```python
memory.append((state, action, reward, next_state, hidden, soc))
```

The memory holds up to 15,000 transitions (~156 days of data).

### Adaptive Learning Rate

```python
if recent_profits > older_profits:
    lr *= 1.05  # Learning is working, speed up
else:
    lr *= 0.95  # Learning is struggling, slow down

lr = clip(lr, 0.001, 0.015)
```

---

## 7. Hyperparameters

### Learning Parameters

Applied directly to the weights.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 0.008 | Step size for weight updates |
| `gamma` | 0.98 | Discount factor (not actively used) |
| `batch_size` | 64 | Samples per update |
| `memory_size` | 15,000 | Maximum stored transitions |
| `update_frequency` | 96 | Timesteps between updates (1 day) |

### Exploration Parameters

Applied to the "buy/sell action"

| Parameter | Value | Description |
|-----------|-------|-------------|
| `exploration_rate` | 0.35 | Initial noise level |
| `exploration_decay` | 0.997 | Decay per episode |
| `min_exploration` | 0.08 | Minimum noise level |

---

## 8. What We Are Contributing — TD Actor-Critic (`algo/temporal-difference`)

The original implementation used a Monte Carlo policy gradient: weights were updated once per episode using the full cumulative return. This branch replaces it with a **Temporal Difference (TD) Actor-Critic**, which bootstraps value estimates at every timestep and produces much faster, lower-variance learning.

### 8.1 From Monte Carlo to TD Actor-Critic

| Property | Monte Carlo (old) | TD Actor-Critic (this branch) |
|----------|------------------|-------------------------------|
| Update frequency | Once per episode (end of day) | Every timestep (online) |
| Return estimate | Full rollout `G_t` | Bootstrapped: `r + γ·V(s')` |
| Variance | High | Low |
| Bias | None | Small (from bootstrapping) |
| Separate critic | No | Yes — linear value function |

### 8.2 The Critic (Value Function)

A linear critic approximates how good each state is. It is trained separately from the actor:

```python
V(s) = soc_w × (soc - target)
     + soc_sq_w × (soc - target)²
     + price_w × price_norm
     + price_diff_w × price_diff
     + hour_w × sin(hour)
     + bias
```

Warm-started with `soc_sq_w = -1.0` so the critic immediately penalises extreme SOC states before any learning has occurred.

### 8.3 The TD Update Rule

At each timestep the TD error `δ` measures how wrong the critic's prediction was:

```
δ = r + γ·V(s') - V(s)        ← TD error (prediction error)

critic: θ_v += critic_lr · δ · ∇V(s)    ← make prediction more accurate
actor:  θ_π += actor_lr · δ · ∇log π    ← reinforce good actions
```

A positive `δ` means the outcome was better than expected → the actor is nudged toward the action that caused it. A negative `δ` discourages that action.

### 8.4 Network Architecture Changes

The hidden layer was widened from **1 neuron to 4 neurons** to allow the policy to learn interactions between features (e.g. "low price AND rising forecast AND low SOC → strong buy"):

```
Input (9) → W1 [4×9] + b1 [4] → LeakyReLU → W2 [1×4] + b2 [1] → Tanh → action
```

Weights are stored as numpy arrays (`W1`, `b1`, `W2`, `b2`) rather than the old named-key dict.

### 8.5 Bug Fixes & Design Improvements

Several bugs were found and fixed during this branch:

| # | Bug | Fix |
|---|-----|-----|
| 1 | `value_theta` missing `"bias"` key → critic crashed every run | Added `"bias": 0.0` with warm-start |
| 2 | Two weights both received `price_diff_1h` → redundant, one weight never learned | Corrected to separate `price_diff` and `forecast_1h` inputs |
| 3 | `get_average_price` sliced a static array → data leakage from future prices | Replaced with a rolling `price_history` deque (only seen prices) |
| 4 | Tanh gradient used the post-processed (clipped) action → wrong gradient | Stored `raw_action = tanh(z)` separately; gradient computed from that |

Additional design improvements:

- **Ask price:** changed from `price × 0.92` (always underselling) to `price × (1.0 + 0.05 × |action|)` — the agent now sells at or above market
- **SOC penalty:** reduced maximum from −20 to −3 and replaced hard steps with a smooth taper, reducing reward spikes that disrupted learning
- **Bias blend in policy:** reduced from 40% to 10% — the network now dominates over the hand-coded prior, improving credit assignment
- **Weight decay:** `W1 × 0.999` and `W2 × 0.999` each update, preventing weights from growing into tanh saturation
- **Adaptive actor learning rate:** scales between 0.01–0.08 based on whether recent profits exceed older profits

---
