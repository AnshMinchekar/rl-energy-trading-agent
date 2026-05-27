# Algorithm Comparison Plan: Monte Carlo vs Temporal Difference

## 1. Current State of Both Branches

### Branch Mapping
| Branch | Algorithm | Status |
|--------|-----------|--------|
| `main` | Monte Carlo (REINFORCE) | Implemented — logging incomplete |
| `algo/temporal-difference` | TD Actor-Critic | Implemented — episode JSON logging exists |

---

## 2. Algorithmic Differences (As-Is)

The table below captures how the two implementations currently differ. These differences matter for interpreting results and deciding what to standardize.

| Dimension | Monte Carlo (`main`) | TD Actor-Critic (`algo/temporal-difference`) |
|-----------|----------------------|----------------------------------------------|
| **Network architecture** | Scalar weights (1 hidden node — dict-based) | Matrix weights (8 hidden neurons — numpy arrays) |
| **Update algorithm** | Policy gradient — batch of immediate rewards, no bootstrapping | Actor-Critic — 5 critic updates + 1 actor update per trigger |
| **Return estimate** | Immediate reward `r_t` with mean baseline | `δ = r + γ·V(s') - V(s)` (TD error) |
| **Critic/baseline** | Scalar mean of batch rewards | Linear value function over 5 state features |
| **Update frequency** | Every 96 steps (1 day) | Every 48 steps (half day) |
| **Memory size** | 15,000 transitions | 4,000 transitions |
| **Batch size** | 64 | 32 |
| **State encoding (time)** | Linear: `hour / 23`, `dow / 6` | Circular: `sin/cos(hour)`, `sin/cos(dow)` |
| **State features** | `[soc, max_discharge, max_charge, price_norm, price_diff_1h, price_diff_4h, hour_norm, dow_norm, price_pct]` | `[soc, price_norm, price_diff_1h, price_diff_4h, sin_hour, cos_hour, sin_dow, cos_dow, price_pct]` |
| **Price history** | Full dataset slice (risk of future data leakage) | Rolling 96-step observed deque (no leakage) |
| **SOC floor / ceiling** | 0.20 / 0.80 | 0.10 / 0.85 |
| **Episode logging** | Console prints only | JSONL file at `output/episode_logs.jsonl` |
| **Algorithm tag in logs** | None | None |

### Critical Issue: MC Is Not True Monte Carlo
The `main` branch `update_parameters()` draws a **random mini-batch of past individual transitions** and uses their immediate rewards as advantages. This is **not** true Monte Carlo, which requires computing discounted returns over complete episodes:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T
```

Before comparing, the MC branch must be corrected to store full episode trajectories and compute proper discounted returns. This is the primary algorithmic fix required.

---

## 3. Standardization Required for a Fair Comparison

To isolate the learning algorithm as the only variable, the following must be aligned between branches **before** running comparison experiments.

### 3.1 Fix MC to Use True Monte Carlo Returns
**File:** `mesa_model/agents.py` (main branch, `storage.update_parameters()`)

- Replace random mini-batch with full current episode trajectory stored in a separate `episode_buffer`
- At episode end (every 96 steps), compute discounted return for each step: `G_t = Σ γ^k · r_{t+k}`
- Normalize returns: `G_t_norm = (G_t - mean) / (std + 1e-8)` to reduce variance
- Update policy using `G_t_norm` as advantage — no bootstrapping, no critic
- Keep existing replay `memory` deque for reference but do not use it for MC updates

### 3.2 Fix Price History Leakage in MC
**File:** `mesa_model/agents.py` (main branch, `get_average_price()` and `get_price_percentile()`)

- Replace `self._price_array[-steps_back:]` (which slices the full dataset) with a rolling `price_history = deque(maxlen=96)` populated in `update_status()` — identical to TD branch
- This ensures MC only uses prices the agent has actually observed

### 3.3 Standardize State Representation
Both branches use 9-D state vectors but with different encodings. For the comparison, **keep each branch's state as-is** — this is a valid algorithmic difference — but document it clearly in the report. Circular encoding (TD) is architecturally superior for time features; this is worth noting as a design choice that accompanies TD.

### 3.4 Standardize SOC Operating Bounds
- Align both to `soc_floor = 0.10`, `soc_ceiling = 0.85` so the trading range is identical
- Emergency hard overrides (soc < 0.10 → charge, soc > 0.95 → discharge) should be identical in both

### 3.5 Align Update Frequency
Set `update_frequency = 96` in both branches so each performs exactly one update per day. This removes update cadence as a confounding variable.

---

## 4. Logging Improvements Required

Both branches need structured per-episode logging to enable quantitative comparison. The TD branch already has a JSONL logger (`_log_episode`); the MC branch needs an equivalent, and both need an **algorithm identifier tag**.

### 4.1 Unified Log Schema
Each episode record written to a `.jsonl` file must include:

```json
{
  "algorithm": "mc" | "td",
  "agent_id": 6,
  "episode": 42,
  "timestamp": "2021-03-15 00:00:00",
  "cumulative_reward": 12.34,
  "actual_profit_eur": 0.1823,
  "energy_bought_kwh": 5.60,
  "energy_sold_kwh": 4.90,
  "exploration_rate": 0.2910,
  "soc_avg": 0.51,
  "soc_min": 0.18,
  "soc_max": 0.83,
  "trade_count_buy": 14,
  "trade_count_sell": 11,
  "avg_buy_price": 28.4,
  "avg_sell_price": 34.1,
  "avg_return": 3.21,
  "return_std": 1.05
}
```

New fields vs the current TD schema:
- `algorithm` — branch identifier
- `trade_count_buy` / `trade_count_sell` — number of non-zero buy/sell timesteps per episode
- `avg_buy_price` / `avg_sell_price` — average price when trades occurred
- `avg_return` / `return_std` — MC-specific: mean and std of episode returns G_t

### 4.2 MC Branch: Add `_log_episode()` and `_reset_episode_counters()`
Mirror the structure already in `algo/temporal-difference`:
- Add `_log_episode()` writing to `output/mc_episode_logs.jsonl`
- Add `_reset_episode_counters()` called at episode end
- Track `trade_count_buy`, `trade_count_sell`, `avg_buy_price`, `avg_sell_price` inside `update_status()`

### 4.3 TD Branch: Add Algorithm Tag and New Fields
- Add `"algorithm": "td"` to existing `_log_episode()` output
- Add `trade_count_buy`, `trade_count_sell`, `avg_buy_price`, `avg_sell_price` tracking in `update_status()`
- Write to `output/td_episode_logs.jsonl` instead of the generic `episode_logs.jsonl`

### 4.4 Separate Output Directories
```
output/
  mc/
    episode_logs.jsonl   ← per-episode MC records
  td/
    episode_logs.jsonl   ← per-episode TD records
```

---

## 5. Comparison Methodology

### 5.1 Experimental Setup
- **Scenario:** Same date range and price data for both runs — use year 2021 (full year = 365 days = 365 episodes)
- **Initial conditions:** Both start with `SOC_start = 0.40` (same as MC config)
- **Random seed:** Fix `np.random.seed(42)` at the top of `main.py` for reproducibility
- **Runs:** 1 run per algorithm (single seed is acceptable for a dissertation comparison; note this as a limitation)

### 5.2 What to Run
```bash
# On main branch (MC)
git checkout main
python main.py
# Output → output/mc/episode_logs.jsonl

# On algo/temporal-difference branch (TD)
git checkout algo/temporal-difference
python main.py
# Output → output/td/episode_logs.jsonl
```

### 5.3 Metrics to Compare

**Primary (financial performance):**
- Cumulative profit (€) per episode — the headline figure
- Total profit over the full run
- Profit per kWh traded — efficiency metric

**Learning dynamics:**
- Reward per episode — convergence curve
- Rolling 7-day average reward — smoothed learning signal
- Exploration rate decay — same schedule in both so this is confirmatory

**Trading behaviour:**
- Buy/sell trade count per episode
- Average buy price vs average sell price — did the agent learn arbitrage timing?
- Price spread captured: `avg_sell_price - avg_buy_price`

**SOC management:**
- Average SOC per episode — should hover near 0.50
- SOC min/max range per episode — narrow range = agent isn't trading; wide but controlled = good
- Time-at-floor (SOC < 0.15) and time-at-ceiling (SOC > 0.85) per episode — safety violations

### 5.4 Statistical Reporting
Since only one run is feasible (simulation runtime), use:
- **Rolling window statistics** (7-episode window) to smooth noise
- **Convergence point** — episode where rolling profit first exceeds 50% of final best profit
- **Late-stage performance** — mean and std of profit over the last 30 episodes

---

## 6. Analysis and Visualisation Script

Create `analysis/compare_algorithms.py` with the following plots:

### Plot 1: Learning Curves (2 subplots, shared x-axis)
- Top: Episode profit (€) — MC vs TD, with 7-episode rolling mean overlay
- Bottom: Episode reward — MC vs TD, with rolling mean

### Plot 2: SOC Behaviour (2 subplots)
- Left: SOC avg ± (min, max) band per episode for MC
- Right: Same for TD
- Common y-axis so scales are directly comparable

### Plot 3: Arbitrage Quality Over Time
- Buy price vs sell price scatter, coloured by episode number (early = blue, late = red)
- Separate panels for MC and TD
- Shows whether the agent learned to buy cheap and sell expensive

### Plot 4: Summary Table
Rendered as a matplotlib table or printed to console:

| Metric | Monte Carlo | TD Actor-Critic |
|--------|------------|-----------------|
| Total profit (€) | | |
| Best episode profit (€) | | |
| Avg profit — last 30 ep (€) | | |
| Convergence episode | | |
| Avg trades/episode | | |
| Price spread captured (ct/kWh) | | |
| Avg SOC | | |
| Time at floor (%) | | |

---

## 7. Implementation Task Order

The tasks below are ordered by dependency. Complete each before starting the next.

### Phase 1 — Fix MC Algorithm (main branch)
1. [ ] Fix `get_average_price()` and `get_price_percentile()` — rolling deque, no future leakage
2. [ ] Fix `update_parameters()` — compute true discounted returns over full episode
3. [ ] Add `episode_buffer` list cleared at each update, storing `(state, action, reward, h1)` per step
4. [ ] Align SOC floor/ceiling to `0.10 / 0.85`
5. [ ] Align `update_frequency = 96`

### Phase 2 — Improve Logging (both branches)
6. [ ] MC: Add `_log_episode()` writing to `output/mc/episode_logs.jsonl` with full schema
7. [ ] MC: Add `_reset_episode_counters()` and trade tracking variables to `update_status()`
8. [ ] TD: Add `"algorithm": "td"` field; rename output to `output/td/episode_logs.jsonl`
9. [ ] TD: Add `trade_count_buy`, `trade_count_sell`, `avg_buy_price`, `avg_sell_price` tracking

### Phase 3 — Reproducibility
10. [ ] Add `np.random.seed(42)` and `random.seed(42)` to `main.py` on both branches
11. [ ] Confirm both use the same scenario date range in `config.yaml`

### Phase 4 — Run Experiments
12. [ ] Run MC for full year → collect `output/mc/episode_logs.jsonl`
13. [ ] Run TD for full year → collect `output/td/episode_logs.jsonl`

### Phase 5 — Analysis
14. [ ] Write `analysis/compare_algorithms.py` — load both log files, produce all plots
15. [ ] Generate summary table for the report

---

## 8. Limitations to Acknowledge in the Report

- **Single run per algorithm** — no statistical significance testing across seeds. Runtime constraints make multi-seed runs impractical.
- **Architecture differs** — MC uses a scalar 1-neuron network; TD uses an 8-neuron hidden layer. The TD network has strictly more capacity. Results reflect algorithm + architecture jointly.
- **Reward function not identical** — SOC centering bonus exists in MC but was removed from TD. This was a deliberate design improvement in the TD branch. The report should note this.
- **Same environment, not identical policy space** — state encoding (linear vs circular time features) differs. This is an algorithmic design choice tied to each method.
- **No held-out test set** — both algorithms are evaluated on their training data. Generalisation to unseen price scenarios is not assessed.
