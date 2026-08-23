Arbitrage-Only LP: a Level-Field Baseline for SAC

Context

Ansh's question is "under the same parameters and setup, which process — optimisation or SAC —
performs better?" The whole-node HNOptimizer cannot answer that: it solves
(household-node cost minimisation) with different information (~21 h actual prices + device
forecasts). The fair contest is a battery-only LP for storage row 0 with st
parity: it plans over exactly the forward-price window SAC's state encodes (current price +
FORECAST_STEPS=24 known DA prices, 6 h), maximising trading profit + termin
the same objective SAC's reward expresses.

Scope decisions (Ansh): arbitrage-only LP now; the whole-node row-0 run stays available later
(toggle already committed, provenance guard makes late runs comparable). Ph
d72c493, e7d1c43; provenance guards) are done.

Design principle — maximum parity via reuse: the arbitrage branch converts its plan into
the same action fraction ∈ [−1, 1] SAC emits and calls the existing
storage.action_to_bid(action) (mesa_model/agents.py:597). Deadband, emergency SOC
overrides, fee-crossing bid price (p + margin_buy + fee_ext + 0.01), ask pr
(p − margin_sell − 0.01, ASK_PRICE_FLOOR) — all shared code, zero duplication. The two
methods differ only in how the action is chosen.

Step 1 — The LP (optimization/arbitrage_optimizer.py, new, ~100 lines)

plan_arbitrage(soc, prices, capacity, max_power_kw, dt_h, efficiency, disch) -> (buy_kwh, sell_kwh) for the first step. Pure LP via scipy.optimize.linprog
(HiGHS — no Gurobi license contention, ~ms per solve; scipy already in the venv — verify, else
pyomo+gurobi_direct).

- Variables: grid-side b_t, s_t ≥ 0 over T = len(prices) steps (25).
- Dynamics, mirroring storage_logic.soc_transition:
  soc_t = soc_{t−1}·discharge + (b_t·η − s_t/η)/C, with 0.05 ≤ soc_t ≤ 0.95
- Power caps, mirroring provide_power_kwh: b_t ≤ P·dt/η, s_t ≤ P·dt·η.
- Objective (maximise):
  Σ [s_t·(p_t − margin_sell) − b_t·(p_t + margin_buy)]/100 + κ·soc_T,
  κ = C·η·max(p_T − margin_sell, 0)/100 — liquidation_value as a constant c
  (terminal price known at solve time, so it stays linear).
- Margins as ex-ante prices match the reward's settled-price fallback. No b
  margins make simultaneous buy+sell strictly unprofitable.

Step 2 — Agent + model hooks

- mesa_model/agents.py — in storage.step(), new branch elif self.method == "arbitrage": next to the "learning" branch: build prices = [current] + get_future_prices(), solve, convert (b0, s0) → action fraction
  (b0_power/max_charge or −s0_power/max_discharge from provide_a_power()), call
  self.action_to_bid(action). In update_status, "arbitrage" follows the neu
  (non-learning) path — same SOC bookkeeping, no learner/replay (check existing method
  guards; the "optimisation" path is the template). Constructor: no SACLear
- mesa_model/model.py — accept "arbitrage" in the STORAGE0_METHOD validation
  (line ~200); exclude method == "arbitrage" from HNOptimizer everywhere
  method == "learning" is excluded (3 places — grep == "learning"), so the node LP
  doesn't double-plan the battery.

Step 3 — Harness support

- analysis/run_comparison_eval.py: --mode arbitrage → sets STORAGE0_METHOD=
  writes output/comparison/eval_arbitrage.jsonl. Meta already stamps
  git_sha/git_dirty/storage0_method.
- analysis/compare_sac_vs_optimisation.py: --baseline {optimisation,arbitrage}
  (default optimisation for back-compat) selecting eval_<baseline>.jsonl an
  All guards (SHA, dirty, window, per-agent margin sanity) apply unchanged — rows 1–4 still
  run the whole-node LP in every mode, so the cross-run check survives.

Step 4 — Verification (before any long run)

1. Unit check (scratchpad script): 50 random price series → simulate the LP
   with soc_transition, assert the SOC path matches the LP's internal SOC to ~1e-9 and all
   bounds hold; hand-check a 3-step case (cheap night / expensive morning)
2. Smoke: .venv/Scripts/python.exe analysis/run_comparison_eval.py --mode arbitrage --max-steps 20 — bids clear, provenance stamped, day records valid; re-smoke --mode sac
   to confirm no regression from the agents.py edit.
3. Commit as one commit: "feat: arbitrage-only LP baseline (level-field SAC comparison)".

Step 5 — Full runs (launched only on Ansh's go; machine must stay awake)

Sequential, on the new commit: --mode arbitrage (~market-clearing-bound, LP adds ~ms/step),
then --mode sac (same frozen policy output/sac/surrogate_policy.pt). Then
compare_sac_vs_optimisation.py --baseline arbitrage; per-agent sanity gate must pass.
Report update afterwards. (Whole-node --mode optimisation run stays an opti
baseline, unbuilt work: none.)

Stated residual asymmetries (for the writeup, not fixable)

- Bid prices are mechanical for the LP, learned for SAC — pricing is part of each process.
- Two separate runs ⇒ endogenous price impact; measured by the rows-1–4 che