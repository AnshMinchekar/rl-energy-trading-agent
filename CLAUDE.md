# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-agent energy trading simulation using reinforcement learning. A battery storage agent learns to perform energy arbitrage (buy low, store, sell high) within a local energy market (LEM) simulated with the Mesa framework.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate Diss_clean
```

Requires commercial solver licenses: **Gurobi** (primary) and/or **Mosek**. The license file is at `data/config/gurobi.lic`.

## Running the Simulation

```bash
python main.py
```

Runs 96 timesteps per day (15-minute intervals). Results are written to auto-indexed CSV files via `data/csv_writer.py`.

## Architecture

### Component Relationships

```
main.py → LEM (mesa_model/model.py)
             ├── Config singleton (data/config/config.py + config.yaml)
             ├── PandaPower grid (buses, loads, generators)
             ├── Agents (mesa_model/agents.py)
             │     ├── res, ext_grid          — supply-side
             │     ├── household, farm, industry — fixed loads
             │     ├── heatpump, EV           — flexible loads/storage
             │     └── storage                — PRIMARY RL AGENT
             ├── MarketOptimizer (optimization/market_optimizer.py) — Pyomo+Gurobi
             ├── HNOptimizer (optimization/HN_optimizer.py)         — local opt
             └── CSV output (data/csv_writer.py)
```

### RL Storage Agent (`mesa_model/agents.py`, class `storage`)

This is the primary learning agent. It has two operational modes set in `config.yaml`:
- `"optimisation"` — deterministic bids via HN_optimizer
- `"learning"` — neural network policy with TD learning

**State vector (9-D):**
- SOC, max discharge/charge power
- Normalized current price, 1h and 4h price forecasts
- Hour of day, day of week, price percentile in 24h window

**Policy network (2-layer):**
```
Input(9) → [weights + bias] → LeakyReLU → [weights + bias] → Tanh → action ∈ [-1, +1]
```
Action: +1 = max charge, -1 = max discharge

**Value function (critic):** Linear in `(soc - target)`, `(soc - target)²`, price, price_diff, hour

**Reward (3 components):**
1. Profit: `-5.0 × energy_cost`
2. SOC penalty for extreme states
3. Arbitrage bonus: `+3.0` for buying below 80% avg / selling above 120% avg

**TD Actor-Critic update** (`update_parameters()`):
```
δ = r + γ·V(s') - V(s)
critic: θ_v += critic_lr · δ · ∂V/∂θ_v
actor:  θ_π += actor_lr · δ · ∂log_π/∂θ_π
```

Key hyperparameters: `actor_lr=0.03`, `critic_lr=0.05`, `gamma=0.98`, `batch_size=32`, `memory_size=4000`, `exploration_rate=0.35` (decays by 0.997/day, min 0.08).

Safety overrides: SOC < 10% forces charge; SOC > 90% forces discharge.

### Market Clearing (`optimization/market_optimizer.py`)

Pyomo model with quadratic supply/demand curves. Solves welfare maximization subject to power flow constraints. Supports Gurobi, CPLEX, CBC backends. Computes LEC grid fees and levies.

### Configuration

All market/agent parameters are in `data/config/config.yaml`. The `Config` class in `data/config/config.py` is a singleton that loads this YAML and the scenario data CSVs/Parquets.

Key config sections: `main` (timestep, grid fees, levies), `storage` (SOC_start, capacity, power), `ev`, `heatpump`, agent bus assignments.

### Scenario Data (`data/config/scenario_data/`)

Large files — do not edit directly:
- `load_profile.csv` (140MB) — 15-min load profiles
- `spot_price.csv` — market electricity prices
- `res_profile.csv` — renewable generation profiles
- `temperature.csv` — for heatpump thermal modeling
- `profiles2021-2023.parquet` — hourly demand profiles (3 years)

### Branch Structure

- `main` — base branch
- `algo/temporal-difference` (current) — TD actor-critic implementation
- `algo/monte-carlo` — prior Monte Carlo implementation

Each `algo/*` branch implements a different RL algorithm in `mesa_model/agents.py` while keeping the rest of the infrastructure unchanged.
