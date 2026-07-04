# Codebase Review — RL Energy-Trading Agent (July 2026)

Scope: full pass over `mesa_model/` (agents, model, sac, storage_env, storage_logic),
`optimization/` (market + HN optimizers), `data/` (config, csv_writer), `main.py`,
`train_surrogate.py`, and the analysis scripts. Goal: **maximise arbitrage profit of the
learning storage agent**. Findings are ordered by importance; the storage agent section
comes first. File:line references are to the current working tree
(branch `algo/soft-actor-critic`).

---

## 0. Executive summary

The SAC implementation itself (`mesa_model/sac.py`) is correct and clean. The surrogate
training pipeline is well designed (shared `storage_logic.py` guarantees identical
state/reward/dynamics). **The dominant problem is no longer the RL algorithm — it is that
the reward the agent optimises does not match the money the market actually settles.**

Three numbers from the last frozen-policy live eval (`output/sac/episode_logs.jsonl`,
30 eval days) tell the story:

| metric | value |
|---|---|
| energy bought / sold per day | 255 / 232 kWh (on a ~147 kWh battery) |
| captured spot spread (avg sell − avg buy price) | **0.28 ct/kWh** |
| reported profit | **−2.76 €/day** (and real cash is worse — see §1.1) |

The agent churns ~1.7 full cycles/day while capturing a spread that is an order of
magnitude below breakeven. It does this because **its reward says round trips are almost
free**, while the live market charges efficiency losses (~5%), grid margins
(1.0 + 0.3 ct/kWh) and — crucially — **grid fees + levies (7.85 ct/kWh LEC,
15.7 ct/kWh external) on every kWh bought**. Fix the economics first; every RL tweak is
secondary to that.

---

## 1. Storage agent (`mesa_model/agents.py:402-892`) — most important section

### 1.1 BUG (critical): reward and profit tracking ignore transaction costs

* `compute_reward` → `mark_to_market_reward` (`storage_logic.py:72-95`) values the cash
  leg at the raw spot price `p_decision`. No margins, no fees.
* The surrogate env at least charges the grid margins (`storage_env.py:169`,
  1.3 ct round trip) — **the live agent's reward charges nothing**, so a policy
  fine-tuned in the live market (`SAC_EPOCHS` path) is trained on an even more
  optimistic reward than the surrogate one. Train and live rewards disagree.
* Actual settlement in the market is different from both: every agent settles at the
  blended `slack_price` (`market_optimizer.py:610-636`), and buyers additionally pay
  `(gridfee + levies)` per kWh — 5.5+2.35 = **7.85 ct/kWh** for LEC energy,
  11+4.7 = **15.7 ct/kWh** for external energy
  (`market_optimizer.py:635-636`, `config.yaml:570-573`). The storage agent is not
  exempt (see the "Fees and Levies … Storage" columns in `data/csv_writer.py:103-104`).
* `cumulative_profit` (`agents.py:717`) and the episode-log `actual_profit_eur` use
  `(sold − bought) · p_decision / 100` — so even the reported −2.76 €/day understates
  the true loss; it excludes ~7.85–15.7 ct/kWh of fees on the 255 kWh bought daily
  (≈ another −20 €/day or more of real cash).

**Breakeven arithmetic** (why this matters for "maximise arbitrage"): buying 1 kWh at
price `p_buy` with LEC fees and selling the stored energy later requires roughly
`p_sell ≥ (p_buy + 7.85)/0.9025 + 0.3` (η²=0.9025 round trip, 0.3 sell margin).
At `p_buy = 10 ct` that means `p_sell ≈ 20 ct` — an ~10 ct/kWh spread with LEC fees,
~19 ct with external fees. Mean 2021–23 price is 13.8 ct/kWh; daily spreads that large
are rare outside 2022. **Two consequences:**

1. Decide the modelling question explicitly: *should storage pay grid fees?* (In
   Germany, BESS is largely exempt under §118 EnWG.) If exempt, remove the fee charge
   for storage in the market accounting. If not exempt, arbitrage may be structurally
   near-unprofitable in this LEM and no RL algorithm will fix that.
2. Whatever the answer, make **reward = surrogate cost model = live settlement**. Add
   the actual per-kWh cost (margins + applicable fees, and value cash at the settled
   price, available from `result["Revenue Energy LEC [€]"] + result["Revenue Energy
   External [€]"] − result["Fees and Levies …"]` instead of `p_decision`). An agent
   whose reward includes the true round-trip cost will learn to stop churning on
   0.28 ct spreads by itself.

### 1.2 BUG (high): `action_to_bid` prices tie with the grid and lose to fee terms

`action_to_bid` (`agents.py:585-637`) bids `p + margin_buy` / asks `p − margin_sell` —
*exactly* the external grid's coefficients (`agents.py:197-198`). In the clearing
objective (`market_optimizer.py:304-307`) that trade contributes **zero** welfare, and
external purchases additionally add `gridfee_levies_ext = P_ext_buy·15.7/4` to the
minimised objective (`market_optimizer.py:269-271`). A welfare-minimising LP therefore
**strictly prefers not to fill a battery buy against the external grid** — the docstring
claim "guaranteed to clear against the price-of-last-resort grid" is wrong.

What actually happens (consistent with the eval logs: 14.6 buys/day vs 28.8 sells/day):

* **Buys** clear only when local RES surplus exists — RES agents ask at price 0
  (`agents.py:138` sets `self.price = 0`, never updated), so a daytime PV-hours buy has
  positive welfare. **Night charging from the grid essentially cannot clear.** The
  agent cannot execute the classic buy-cheap-at-night strategy at all.
* **Sells** clear readily (battery undercuts the grid by 1.3 ct and saves ext fees).

Fixes: bid `p + margin_buy + gridfee_ext + levies_ext + ε` if you truly want a
price-taking fill guarantee (and charge that same cost in the reward), or make the
storage fee-exempt in the market model. Also verify empirically: log the storage row of
`model.results[...]["agents"]` per step for one live day and check whether any nighttime
buy clears.

### 1.3 BUG (high): surrogate fill model diverges from live fills

`storage_env.py` assumes every requested volume fills at `p ± margin`. Live, per §1.2,
buys fill only against daytime PV and sells are welfare ties. Two concrete divergences:

* The surrogate lets the agent **sell at negative prices** (fills at `p − margin` even
  when negative). Live, the ask floor of 0.01 (`agents.py:635`, and the emergency ask
  at `agents.py:621`) will not cross a negative grid bid — so trained behaviour around
  negative prices never transfers. The price file has **2,036 negative and 5,280
  |p|<0.5 ct points** (min −39.9 ct in 2023), so this is not a corner case.
* The surrogate has no notion of "buy fails at night". After fixing §1.2 pricing this
  gap closes; otherwise consider a fill probability / RES-availability signal in the
  surrogate.

### 1.4 State features are unstable exactly when prices are most interesting

`state_features` (`storage_logic.py:50-59`) uses ratio features:
`mom_1h = p/p_lag − 1`, `price_vs_base = (p − ewma)/(ewma + eps)`,
`spread_norm = (mb+ms)/(p + mb + eps)`. With 2k negative and 5k near-zero prices in the
training data, these divide by ~0 or flip sign (a price rise from −1 → +2 gives
momentum −3), then saturate at the ±5 clip. Volatile/negative-price periods are where
arbitrage money lives. Replace ratios with **difference features normalised by the
rolling std** (you already compute `std`): `mom_1h = (p − p_lag1h)/(std+eps)`, same for
4h and vs-baseline. This is a small change in one shared function, and both the live
agent and surrogate pick it up automatically.

### 1.5 Biggest profit lever: the agent is blind to known future prices

The state is entirely backward-looking, but the price series *is* the day-ahead spot
price — in reality it is public 12–36 h ahead. Forcing the agent to *infer* tomorrow's
prices from lags throws away the single most valuable, perfectly legitimate signal for
arbitrage. Add to the state (both env and live agent read the same price series, so no
leakage in the simulation sense either):

* next 4–8 raw DA prices (normalised like `price_norm`), and/or
* aggregates: `(min, max, mean)` of the next 6 h and next 24 h relative to `p_now`.

With known future prices this problem is close to an LP (see the perfect-foresight
bound in `analysis/naive_baseline_q1_2023.py`); an RL policy with those features should
approach it much faster than one that must learn a forecaster implicitly in a 64-unit
MLP. This is the highest-upside improvement after fixing the economics in §1.1/1.2.

### 1.6 No "hold" deadband — the policy pays costs every single step

`action_to_bid` trades for any `|action| > 1e-6` (`agents.py:604`), and a
tanh-squashed Gaussian essentially never outputs 0. So the agent *always* trades a
little, paying margins/fees each step (part of the churn in the eval). Add a deadband —
e.g. `|action| < 0.05 → no order` — identically in `action_to_bid` and
`storage_env.step` (best: put the mapping in `storage_logic.py` so it stays shared).

### 1.7 Inventory valuation biases toward hoarding

The mark-to-market inventory term values stored energy at the full mid price
(`p·soc·capacity`), but its liquidation value is `(p − margin_sell − fees)·η·soc·cap` —
~6 %+ lower. The agent gets paid "paper appreciation" it can never realise, biasing it
to hold high SOC into the end of horizon. Also, with γ = 0.99 the telescoping term is
not exactly policy-invariant (potential-based shaping requires `γ·Φ(s′) − Φ(s)`). Both
fixes are cheap: value inventory at `η·(p − margin_sell)` (and, if fees stay, subtract
them) and multiply the `Φ(s′)` leg by γ.

### 1.8 Smaller storage-agent items

* **Discount horizon**: γ = 0.99 at 15-min steps ⇒ half-life ≈ 17 h — barely one price
  cycle. Consider γ = 0.995–0.997 or n-step (3–5) targets so cross-day arbitrage is
  visible in the value function.
* **Dropped transitions**: if the agent's row is missing from `results`,
  `update_status` returns early (`agents.py:669-679`) and the pending
  `(last_state, last_action)` is silently overwritten next step. Rare, but log it.
* **Emergency buy is a must-run constraint**: `bid = [max_charge, max_charge, …]` at
  price 1000 (`agents.py:613`) forces the LP to fill exactly `max_charge`; combined
  with other constraints this can make a step infeasible. A `[0, max_charge]` range at
  a high price is safer.
* `p_decision` fallback `30.0` in `get_current_price` (`agents.py:566`) — fine, but log
  when hit; a silent 30 ct placeholder inside a reward is hard to debug.
* Seeds are fixed to 42 everywhere; for any conclusion about profit, run ≥3 seeds
  (surrogate training is cheap now).

---

## 2. SAC learner (`mesa_model/sac.py`) — correct; minor notes

Verified against the SAC paper conventions: twin critics + min-target, reparameterised
tanh-Gaussian with the log-prob correction, automatic α with a sensible floor
(`alpha_min=0.05`), delayed actor/target updates, Polyak averaging, warm-up, UTD ratio.
The two self-tests are a nice touch. Only small observations:

* Polyak target updates ride the `actor_update_every` branch (`sac.py:282-284`), so the
  effective τ is 0.0025/update. Intentional (TD3-style) but worth a comment since τ is
  documented as 0.005.
* `updates_per_step` is 4 in the live agent but 1 in `train_surrogate.py:103`. In the
  surrogate, gradient updates are no longer "nearly free" — data is. UTD 1 is right
  there; the inconsistency is fine, just deliberate-looking rather than accidental.
* Critics have no LayerNorm while the actor does; with reward_scale=10 and fee-corrected
  (larger-magnitude) rewards, consider LayerNorm in critics for stability.
* `hidden=64` is small. Fine for 16-D; if you add forecast features (§1.5) go to 128.
* Replay `done` is always 0.0 — correct for a continuing task with time-limit
  bootstrapping; keep it that way.

---

## 3. Surrogate env & training pipeline

* `storage_logic.py` as a single source of truth is the best design decision in the
  repo — keep extending it (deadband, txn costs, forecast features) rather than
  editing the two call sites.
* **`evaluate()` is in-sample** (`train_surrogate.py:108-130`): fixed start at the
  beginning of the *training* window (Jan 2021, 8000 steps). It's a fine progress
  signal but says nothing about generalisation, and the final `learner.save()` keeps
  the *last* policy, not the best one. SAC can and does degrade late in training. Fix:
  hold out a validation slice (e.g. H2-2022), evaluate on it at each `LOG_EVERY`, and
  **checkpoint the best validation policy**.
* Train window 2021–22 vs eval Q1-2023 is a regime shift (2022 energy crisis: huge
  spreads; 2023: negative-price era). The month sin/cos feature learned on 2021-22
  months has never seen "January with 2023 dynamics". Consider training on rolling
  windows that include part of 2023 outside Q1, or dropping the month encoding.
* `StorageArbitrageEnv.step` charges margins only on the cash leg (correct given its
  fill assumption) — when §1.1 is resolved, add the fee term here too so all three
  (reward, surrogate, settlement) agree.
* Off-by-one nit: `learner.total_env_steps` is passed to `env.step` *before* the
  matching `push` increments it — shaping anneal lags by one step. Harmless.

---

## 4. Simulation/market findings that affect the storage agent

* **The learning storage is included in the household (HN) optimiser** for bus 5:
  `build_agent_data_list` filters only on `flex ∈ {0,1,2,3}` (`model.py:427-486`), and
  the learning agent has `flex=1`. Every re-plan solves a Gurobi schedule for a battery
  whose results (`max_buy_price`, `optimal_power_*`, `updated=0`) are then written onto
  the learning agent (`model.py:585-590`) and ignored. Wasted solver time *and* the HN
  plan assumes the battery will behave optimally, distorting co-located bid prices.
  Exclude `method == "learning"` agents from the HN agent list.
* **Duplicate battery**: `model.py:207-213` creates optimisation storages for every
  grid row with `bus != 4` (all five rows qualify: buses 12, 9, 14, 6, 10), *then*
  creates the learning storage as a sixth battery reusing row 0's parameters at bus 5.
  The system has ~147 kWh more storage than the SimBench grid defines. If intentional
  (control vs treatment), document it; the extra optimisation twin of the same physical
  spec also competes for the same spreads.
* `model.py:210` — `self.grid.storage.loc[a]["id"] = id_count` is chained indexing:
  a **no-op** on a copy. Use `self.grid.storage.loc[a, "id"] = id_count`.
* `config.yaml:574` says `solver: "appsi_highs"` but `model.py:269` hardcodes
  `self.solver = "gurobi_direct"`. The config key is dead — either honour it or delete
  it (memory says gurobi_direct is deliberate; then remove the yaml key).
* `compute_price` (`model.py:34-41`) is dead code and would IndexError on an empty
  side; delete.
* RES agents sell at a constant price of 0 (`agents.py:138`, never updated). That is a
  modelling choice, but it means "local energy is free up to fees" — it is the only
  reason storage buys clear at all (§1.2). Flagging so it's a conscious choice.

---

## 5. Other bugs (outside the storage path)

* **heatpump.update_status** (`agents.py:326-339`): if `model.results` is non-empty but
  the lookup throws or `result` isn't a DataFrame, `energy` is never assigned →
  `NameError` at line 339. Also `result["Energy bought [kWh]"].values[0]` IndexErrors
  if the agent row is missing. Initialise `energy = 0` before the `try`.
* **EV.forecast_min** (`agents.py:1094-1151`): bare `except: print()` swallows errors;
  `n` can be unbound if the range is empty; `expected_soc` is a DataFrame used as a
  scratch array (slow, and `is np.nan` on line 1147 is never true for float NaN —
  use `np.isnan`). Debug prints (`print("")`, `print()`) left in.
* `ext_grid.step` (`agents.py:190-200`): missing price ⇒ `price=0, energy=0` silently;
  the whole market then clears against a 0-price grid for that step. Raise or log
  loudly — a data gap should not look like free energy.
* `csv_writer.write_entry` (`data/csv_writer.py:163-165`): the header line is written
  via `';'.join(map(str, entry))` (iterating a DataFrame yields column names — works,
  but by accident) and `data_entry_count` is only used as a header flag; the
  first-of-month rollover resets it but the header for the new file is only written if
  the next call happens to be the first (it is — fragile though). Also HN output is
  hardcoded to node 5 (`csv_writer.py:172`).
* `LEM.build_HEM_dict` (`model.py:387-425`): `target_yes_percentage = 1` makes the
  whole 60/40 switching machinery dead code — everything participates. Simplify or
  parameterise.

---

## 6. Efficiency

The Gurobi market solve (~2.25 s/step) dominates live runs, so these matter mostly for
multi-epoch live training and general hygiene — but several are large constant factors:

1. **Per-step pandas scans**: every load/RES agent does
   `df.loc[df["time"] == current_date]` each step (`agents.py:38`,
   `agents.py:191`, temperature lookups `agents.py:285,306,338`, `model.py:494`).
   That's a full-column comparison per agent per step. The storage agent already shows
   the right pattern — a `dict(zip(time, value))` cache
   (`agents.py:539-548`). Apply the same to `a_power_profile`, `energy_price`, and
   `temperature_df` (one shared dict on the model). Easily tens of ms/step across ~50+
   agents.
2. **`MarketOptimizer.compute_allocation`** (`market_optimizer.py:396-446`): finds each
   agent's index with `next(i for i in m.BIDS if m.id_bid[i] == …)` — O(n²) per step.
   Build `{agent_id: index}` dicts once in `__init__`.
3. **`process_pyomo_results`**: per step it deep-copies the results template
   (`market_optimizer.py:459`), deep-copies the entire pandapower grid and runs a power
   flow (`market_optimizer.py:683-691`), does `iterrows()` over demand/supply, and
   appends rows one-by-one to `agent_pw_sc/lec/ext` DataFrames
   (`market_optimizer.py:535-602`) **that are never read afterwards** — delete those
   three frames outright; build `net2` once and update its load/sgen values in place;
   consider making the power flow optional (`RUN_PF=0`) for training epochs where grid
   KPIs aren't needed.
4. `gc.collect()` every 96 steps (`main.py:93`) is a band-aid; with the deepcopies from
   item 3 removed it can likely go.
5. Surrogate training at UTD 1 is already fast; if you want more from the same wall
   clock, larger batch (512) beats UTD > 1 on CPU with `torch.set_num_threads(1)`.

---

## 7. Prioritised action list (for maximum arbitrage)

1. **Decide the fee question** (storage fee-exempt or not) and make
   reward = surrogate cost = live settlement (§1.1). Without this, the agent optimises
   a fictional market.
2. **Fix `action_to_bid` pricing** so intended fills actually clear (cross fees + ε, or
   exempt storage), and verify with a one-day live fill audit (§1.2).
3. **Add known future DA prices to the state** (§1.5) — biggest upside once economics
   are right.
4. **Add a hold deadband** in shared logic (§1.6) and the negative-price-robust feature
   set (§1.4).
5. **Best-checkpoint selection on a held-out validation window** in
   `train_surrogate.py` (§3).
6. Correct the inventory valuation (η, margins, γΦ) (§1.7); consider γ ≈ 0.996.
7. Exclude the learning agent from the HN optimiser; fix the small bugs in §4/§5;
   apply the cheap efficiency wins in §6 (items 1–3).
8. Re-benchmark against the perfect-foresight LP bound *with the same fee model* — the
   current baseline (`analysis/naive_baseline_q1_2023.py`) also ignores fees, so both
   the bound and the agent need the same correction before comparing.

---

## 8. Applied fixes (2026-07-04)

Decisions taken (Ansh): **storage is fee-exempt** (§118 EnWG precedent) and the
**learning battery replaces the row-0 SimBench storage at its real bus** (no phantom
sixth unit).

| # | Fix | Where |
|---|-----|-------|
| 1 | Storage exempted from grid fees & levies in settlement accounting | `market_optimizer.py` (fee columns zeroed for `Agent Type == "storage"`) |
| 2 | `action_to_bid` bids now cross margin + ext fee term + ε so fills actually clear; asks undercut by ε and respect the 0.01 floor (no sells at negative prices) | `agents.py` |
| 3 | Reward cash leg valued at the **settled slack price** (live) / price±margin (surrogate); `cumulative_profit` likewise | `agents.py`, `storage_logic.py`, `storage_env.py` |
| 4 | Inventory term = γ-corrected potential shaping at **liquidation value** (η·(p−margin_sell), floored at 0) — exactly policy-equivalent to realised cash | `storage_logic.mark_to_market_reward` |
| 5 | State 16-D → **20-D**: 4 forward features from *known* day-ahead prices (next-1h mean, next-6h mean/min/max, std-normalised) | `storage_logic.state_features`, `agents.build_state`, `storage_env` |
| 6 | Momentum/baseline/spread features switched from ratios to **std-normalised differences** (robust to the 2k negative / 5k near-zero prices) | `storage_logic.state_features` |
| 7 | **Hold deadband** `|a| < 0.05` — no order placed (shared `ACTION_DEADBAND`) | `storage_logic`, `agents.action_to_bid`, `storage_env.step` |
| 8 | γ 0.99 → **0.996** (half-life ≈ 1.8 days; cross-day arbitrage visible) | `agents._setup_learning`, env/eval scripts |
| 9 | Train/val/test split + **best-on-validation checkpointing** (train 2021-01→2022-09, val 2022-10→12, test = config Q1 2023); saves `surrogate_policy.pt` (best) and `_last.pt` | `train_surrogate.py` |
| 10 | Learning battery replaces row-0 storage at bus 12; chained-assignment id fix | `model.py` |
| 11 | Learning agent excluded from the HN Gurobi optimiser (agent list, prognosis, states) | `model.py` |
| 12 | Solver read from config (`config.yaml` now says `gurobi_direct` honestly) | `model.py`, `config.yaml` |
| 13 | heatpump `energy` NameError path fixed; ext_grid price-gap now warns and restores capacity afterwards; dead `compute_price`/`step_subnet` removed | `agents.py`, `model.py` |
| 14 | Efficiency: O(1) dict lookups for load/RES profiles + temperature; O(1) agent→index maps in market clearing; dead `agent_pw_*` frames removed | `agents.py`, `model.py`, `market_optimizer.py` |
| 15 | HN CSV logging follows the learning agent's actual bus instead of hardcoded node 5 | `csv_writer.py`, `main.py` |

Note: saved policies from before this change are incompatible (16-D state) — retrain
with `train_surrogate.py`.

### Verification results

* **Unit checks** (shared logic): 20-D features finite & clipped under negative/zero
  prices; reward-telescoping identity exact; deadband holds; negative-price sells
  blocked; buys billed at p+margin — all passed. SAC self-tests passed.
* **Live-market smoke** (forced actions, Jan 1 2023): night buys cleared **5/6 at full
  power** against the grid (0/∞ before the fix) — at *negative* spot prices, i.e. the
  battery got paid to charge; storage fee columns exactly 0; deadband held; sells at
  the negative-price night correctly blocked, and cleared **3/4** at the 18:00 positive
  price (the 4th hit the SOC floor — correct physics), earning +3.05 € on the swing.
* **Training** (60k surrogate steps, 7.7 min): validation profit (held-out Oct–Dec
  2022) peaked at **+1196.79 €** at 25k steps and plateaued after — the
  best-on-validation checkpoint (not the last policy) is what gets saved, which this
  run demonstrates mattered.
* **Held-out test** (Q1 2023 surrogate, real margins): **+412.24 €** for the quarter
  (~+4.6 €/day), vs the causal-threshold floor +358 € and the perfect-foresight
  ceiling +577 € — the policy beats the fixed-rule baseline and captures ~71 % of the
  theoretical ceiling, fully out-of-sample. The pre-fix agent lost −2.76 €/day.
* Remaining step: the full live Mesa-market eval over Q1 2023
  (`SAC_LOAD_POLICY=output/sac/surrogate_policy.pt SAC_EVAL=1 python main.py`,
  ~5 h Gurobi wall-clock) to confirm the number under real market fills.
