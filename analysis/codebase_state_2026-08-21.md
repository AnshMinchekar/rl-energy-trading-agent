# Codebase State & Comparison Fairness Review — 2026-08-21

**Scope:** the uncommitted LP work on `algo/soft-actor-critic` (`optimization/HN_optimizer.py`,
`mesa_model/model.py`), the 500-step optimisation eval that landed 2026-08-18, and the standing
89-day SAC eval from 2026-08-09.
**Question 1:** is the SAC-vs-optimisation comparison fair now?
**Question 2:** what would it take for SAC to stay profitable if grid fees were imposed on storage?

---

## 1. Verdict on fairness: **not yet — one invalidating defect fixed, one new one introduced**

The August report (`analysis/sac_vs_optimisation_report.md`) named one defect that invalidated
the comparison outright: the LP's phantom 16.7 ct/kWh charging tax (§4.1). **That is fixed and
verified.** On the matched Jan 1–5 window, row 0 (agent 23, bus 12, 146.7 kWh):

| controller | profit | sold | margin |
|---|---:|---:|---:|
| SAC | €32.28 | 1,742 kWh | 1.853 ct/kWh |
| LP (all fixes) | €18.52 | 1,791 kWh | 1.034 ct/kWh |

Ratio **174%**, down from the report's 306%. Volume is now within 3%, so the old story —
"SAC only wins by cycling 3.1× harder at identical margin" — is dead. This is now a genuine
margin-vs-margin contest, and SAC wins it on skill, not throughput.

**But the comparison is currently *less* internally consistent than it was on 2026-08-09,**
because the two runs no longer share a codebase:

- `eval_sac.jsonl` was produced **2026-08-09**, a week before the five economic fixes and the
  solver rework. Rows 1–4 in that file run the *old, fee-handicapped* LP.
- `eval_optimisation.jsonl` (2026-08-17/18) runs the *fixed* LP everywhere.

The cross-run sanity check — same LP controller, same batteries, supposed to match — proves it
(Jan 1–5 margins, ct/kWh sold):

| agent | SAC-run margin | OPT-run margin |
|---:|---:|---:|
| 24 | 6.419 | 1.254 |
| 25 | 1.170 | 1.425 |
| 26 | 4.119 | 1.326 |
| 27 | 5.244 | 1.181 |

Those are two different optimizers, not price-impact noise. The aggregate totals happen to land
within 0.9% (€43.46 vs €43.06) by pure cancellation — a trap: the headline sanity check "passes"
while every per-agent number fails. **Until both modes are re-run on the same code, every
cross-run number — including the 174% — is uninterpretable.**

Remaining gaps, in descending order of severity:

1. **Stale SAC run** (above). Invalidating; fixable by re-running.
2. **Window mismatch.** 5.2 days of LP vs 89 days of SAC. Five January days are not a season.
   The LP's truncated Jan 6 alone swings −€35 if naively included.
3. **§4.2, structural asymmetry.** The LP battery is co-optimised with the node's load, EV and
   heat pump; SAC is a standalone arbitrageur. Scoring market cash only credits one of the LP's
   two objectives. Not fixable by re-running — state it as a limitation, or build an
   arbitrage-only LP for row 0.
4. **§4.3, price impact.** SAC's volume moves clearing prices against the other four batteries.
   Was ~24% of the old edge; unmeasurable now until (1) is fixed, because the sanity check that
   measures it is broken.
5. **No degradation cost.** The old break-even was ~2.2 ct/kWh of wear. Post-fix, *both*
   controllers' Jan margins (1.85 / 1.03 ct/kWh) are **below** that. If wear were priced at the
   old break-even, both lose money in January. The absolute case for the battery is weaker than
   the report stated, even though the relative comparison stands.

---

## 2. Current state of the codebase

### What the recent updates did (all uncommitted, working tree only)

**`optimization/HN_optimizer.py`** (+356/−103):

- **Fee-base decomposition** — new `fee_base` variable with a single inequality
  (`fee_base[t] >= power_net[t] − battery charging`); fees apply to `fee_base`, energy price to
  full `power_net`. No binary needed: `fee_base` enters only the buy branch with a positive
  coefficient under minimisation. Kills the phantom charging tax without exempting household load.
- **`buy_flag` Big-M disjunction → convex two-inequality form** — `cost_energy >= formula_buy`
  and `>= formula_sell` gives exactly `max(buy, sell)` because the buy line is strictly steeper.
  Removes 83 binaries per optimizer; 7 of 12 buses became pure LPs. Verified equivalent over 30
  randomised trials to ~1e-9.
- **`MIPGapAbs` = €0.15** — justified by receding horizon (only the first of 83 planned steps
  executes, so €0.15 of horizon-wide slack is under a cent on the executed action).
- **Terminal stored-energy credit** fixed (was ~100× overvalued — the hoarding incentive).
- **SOC band** [0.2, 0.8] → [0.05, 0.95], matching `agents.py`.
- Dead binaries (`A`, `zz`, `z_mode`) and `bigM`/`bigMM` deleted; idle-step fallback bid deleted;
  ask-price sign error on node-balanced steps fixed; per-solve JSONL diagnostics added
  (`SOLVE_LOG_DIR`, summarised by `analysis/summarize_solver_log.py`).

**`mesa_model/model.py`** (+8/−2): `STORAGE0_METHOD` env toggle routing row 0 through the LP for
the benchmark. Clean, validated, default unchanged.

### What the 500-step run proved (192 solves, `output/solver_diagnostics_500/`)

- Solver health: 108/192 pure LPs, median gap 0.000, max absolute slack **€0.73** vs a cycle
  worth €1.5–4.4. The one-step-tuned thresholds (€0.15 floor, 60 s limit) hold at scale.
- Bus 6 is the runtime tax: 9/18 solves hit the 60 s limit (one 1259 s wall outlier — machine
  sleep; Gurobi still reported `maxTimeLimit`).
- **The solver rework was economically a no-op**: margin 1.163 → 1.190 ct/kWh, +€0.37 over five
  days. Exactly as predicted — the incumbent was already near-optimal, only the *bound* was
  garbage. The work bought certification and a 1136 s → 2.0 s solve, not money.

### Housekeeping problems

- **All of the above is uncommitted** since 2026-08-16. Two weeks of load-bearing work,
  unversioned. Commit before anything else.
- **`HN_optimizer/HN_optimizer.py`** appeared untracked 2026-08-20: a pristine **pre-fix** copy
  (1,840 lines vs the fixed 1,531 — still has `buy_flag`, the old SOC band, the 100× credit).
  If it is an A/B reference it belongs somewhere labelled (`archive/`), not in a top-level
  directory shadowing the module name — anything importing `HN_optimizer` ambiguously could
  silently get the broken optimizer.
- `Comparision/` (sic) duplicates `analysis/` outputs under a typo'd name; consolidate.
- `output/comparison/` holds five generations of `eval_optimisation*.jsonl`; only timestamps
  distinguish them, and nothing records which code produced which.

---

## 3. What's missing

1. **A same-code, same-window pair of runs** — the only thing between here and a defensible number.
2. **A degradation model.** Even a constant ct/kWh-throughput cost. Both controllers now sit
   below the old 2.2 ct/kWh break-even; without wear in the LP objective *and* the SAC reward,
   both are optimising a fiction.
3. **An answer to the LP's remaining margin gap.** Post-fix the LP earns 1.03 vs SAC's 1.85
   ct/kWh at equal volume. Hypothesis to test next: the plan-driven fee-crossing bids/asks clear
   round trips whose spread doesn't cover the trip. The solver log + trade logs can decompose
   which planned trades were negative-spread ex post.
4. **§4.2 mitigation**, if a true upper bound is wanted: an arbitrage-only LP for row 0 (same
   horizon, same price knowledge, battery-only objective). Cheap — it's the existing model with
   one agent.
5. **Multi-seed / multi-quarter SAC evidence.** One frozen policy, one window, one seed.
6. **Experiment provenance**: eval JSONLs and solver logs aren't tied to the code SHA that
   produced them — the stale-SAC-run defect at the centre of this review is exactly the failure
   mode that habit invites.

---

## 4. How to make the comparison fair

**Tier 1 — required, mechanical (~1 overnight per mode):**

1. Commit the working tree (the fixes become the citable baseline SHA).
2. Re-run **both** modes on that SHA over the full config window (8,543 steps, Q1-2023):
   `run_comparison_eval.py --mode optimisation`, then `--mode sac`. Solver time extrapolates to
   ~4.2 h for the LP side; the 19 h wall of the 500-step run was mostly market clearing plus a
   sleeping machine.
3. Re-run `compare_sac_vs_optimisation.py`, but require the rows-1–4 sanity check to pass
   **per-agent** (e.g. <10% each), not in aggregate — the aggregate passed this week (+0.9%)
   while every per-agent margin was off by up to 5×.
4. Record the code SHA in each eval's `meta` line.

**Tier 2 — makes the claim strong rather than merely valid:**

5. Report **margin (ct/kWh)** alongside totals — post-fix it is the honest headline.
6. Add a degradation sensitivity table (0 / 1 / 2 / 3 ct/kWh-throughput) for both sides.
7. Quantify price impact from the now-valid sanity check and net it out of the headline edge.
8. State the irreducibles in the writeup: whole-node vs standalone objective (§4.2), and the
   LP's ~12 h actual-price foresight vs SAC's 6 h DA forwards (benchmark definition, but it
   must be stated).

**Tier 3 — optional upper bound:** the arbitrage-only LP for row 0. The claim then becomes
"SAC vs the best standalone planner", which no listed asymmetry can undercut.

---

## 5. Can SAC still be profitable if grid fees are imposed?

**Under current behaviour: unambiguously no.** The 89-day run bought 29,797 kWh at a
2.13 ct/kWh-bought margin. Fee rates from `data/config/config.yaml` (per kWh on buys):
LEC = 5.5 + 2.35 = **7.85 ct**, external = 11 + 4.7 = **15.7 ct**.

| fee scenario | 89-day profit | fee bill |
|---|---:|---:|
| none (today, §118(6) EnWG exemption) | **+€634** | — |
| LEC 7.85 ct on all buys | **−€1,705** | €2,339 |
| external 15.7 ct on all buys | **−€4,044** | €4,678 |

A 2.13 ct margin against a 7.85 ct hurdle: the churn strategy dies by ~3.7×. But that is the
*strategy*, not the method — SAC's value proposition is that it re-learns when the economics
change. Paths to fee-world profitability, in order of leverage:

1. **Align the reward first, or nothing else matters.** This exact failure already happened
   pre-July: a fee-blind reward against fee-charging settlement produced −€2.76/day of confident
   churn. Fees must appear in `mark_to_market_reward`, the surrogate env, *and* settlement
   simultaneously — an unpriced fee is an invisible tax the policy will walk straight into.
2. **Selectivity: trade the tails, not the mean.** Fees are per-kWh, so the optimal response is
   fewer, deeper trades on fat spreads — negative-price hours (charging is *paid*, partly
   offsetting the fee) and morning/evening scarcity peaks. Q1-2023 daily spreads run 5–8 ct;
   under LEC fees the profitable set shrinks to roughly the top decile of days and volume must
   fall ~5–10×. The existing hold-deadband plus a fee-aware reward should produce this without
   architecture changes; the risk is sparse-reward training, which the surrogate env offsets
   with cheap extra steps.
3. **Exploit the fee asymmetry: buy local.** LEC-internal energy carries 7.85 ct vs 15.7 ct
   external. Charging preferentially from local PV surplus (midday, when prices are low anyway)
   halves the hurdle. The state needs a local-generation feature (e.g. node PV forecast) for the
   policy to learn the distinction — today it cannot see it.
4. **Become a solar-shifter, not a grid arbitrageur.** The fee-world business case is
   buy-local-midday / sell-evening: a 5–8 ct spread against a 7.85 ct hurdle is marginal for
   pure arbitrage but turns positive once the local-fee discount and wear savings from lower
   cycling stack. This is also what real German storage does where not §118-exempt.
5. **Fee-crossing bids stay mandatory.** Bidding above the fee-inclusive price is what lets
   buys clear against the market's welfare fee term at all. Under the exemption those fills
   were free; under fees they are genuinely expensive — which is exactly what the reward in (1)
   teaches the policy to ration.
6. **Honesty clause:** at *external*-level fees (15.7 ct), pure price arbitrage on this price
   series is likely unprofitable for **any** controller — the LP included; its old
   phantom-fee behaviour (629 buys in 8,543 steps) was an accidental preview of the correct
   fee-world policy. Storage fee exemption exists in German law precisely because of this. The
   defensible claim then becomes "SAC loses least / degrades most gracefully", demonstrable
   with the same A/B run at fee levels 0 / 7.85 / 15.7.

---

## 6. Recommended order of work

1. Commit the LP fixes + toggle (baseline SHA).
2. Full-window re-run, both modes, same SHA → the first fair headline number.
3. Decompose the LP's residual 1.03-vs-1.85 margin gap from the trade logs.
4. Degradation sensitivity in the comparison script.
5. Optional: fee-scenario A/B (0 / 7.85 / 15.7 ct) with an aligned reward — answers §5
   empirically, and is a stronger result than the fee-free ratio alone.
