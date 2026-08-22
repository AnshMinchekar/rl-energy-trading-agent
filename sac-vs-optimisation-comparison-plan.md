# Plan: SAC vs `optimisation` (HN_optimizer) head-to-head comparison

**Status: code ready — only the two long simulation runs remain.**
See [How to run](#how-to-run) at the bottom.

## Background / why this is missing today

The SimBench grid (`1-LV-rural1--2-sw`) has **5 storage units**:

| Row | Agent ID | Bus | Capacity (kWh) | Method |
|---|---|---|---|---|
| 0 | 23 | 12 | 146.7 | `learning` (SAC) |
| 1 | 24 | 9 | 67.0 | `optimisation` |
| 2 | 25 | 14 | 61.1 | `optimisation` |
| 3 | 26 | 6 | 36.7 | `optimisation` |
| 4 | 27 | 10 | 100.5 | `optimisation` |

Rows 1-4 are scheduled by `optimization/HN_optimizer.py` (`HNOptimizer`) — a
rolling-horizon Pyomo/Gurobi LP that re-solves every `max_prognosis` steps
with up to ~12 h of actual future price knowledge, then feeds
`max_buy_price`/`min_sell_price` back into the `storage` agent's
`method == "optimisation"` bidding branch (`agents.py:858-885`).

Row 0 (SAC) was **structurally excluded** from `HNOptimizer`
(`model.py` skips `method == "learning"` agents in three places), so there
has never been a same-window comparison of SAC against the `optimisation`
method on the *same physical battery* — the README's "Known gaps" flags
exactly this.

## What was implemented

1. **`STORAGE0_METHOD` env toggle** (`mesa_model/model.py`, storage setup):
   `STORAGE0_METHOD=optimisation` routes row 0 through the same
   HNOptimizer/LP path as rows 1-4. Default (`learning`) is unchanged;
   invalid values raise. **Verified:** with the toggle on, bus 12's HN
   optimizer contains the battery (`flex=[0,1,3]`); without it, the battery
   is excluded (`flex=[0,3]`); all other buses identical between modes.

2. **`analysis/run_comparison_eval.py`** — one driver for both sides:
   - `--mode optimisation`: row 0 via the LP (sets the env toggle itself).
   - `--mode sac`: frozen-policy eval (`SAC_EVAL=1`, deterministic, no
     learning), loading `--policy` (default `output/sac/surrogate_policy.pt`).
   - Both write `output/comparison/eval_<mode>.jsonl`: a `meta` header, one
     `day` record per storage agent per calendar day, and a terminal
     `summary` per agent (incl. terminal SOC + liquidation value). Existing
     logs are archived with a timestamp, never overwritten.
   - Logs **all five** storage units, so rows 1-4 double as a cross-run
     sanity check.
   - `--max-steps N` for smoke tests. **Verified end-to-end in both modes
     with `--max-steps 5`** (real Gurobi steps).

3. **`analysis/compare_sac_vs_optimisation.py`** — reads the two logs and
   produces `output/comparison/sac_vs_optimisation.png` (per-day profit /
   SOC / trades for row 0) plus a stdout summary: raw and
   terminal-SOC-adjusted totals, the SAC/optimisation ratio vs the >75%
   success criterion, and the rows-1-4 cross-run sanity table (warns if
   they diverge >10%, i.e. the two runs' market outcomes drifted too far to
   compare row 0 fairly). **Verified against the smoke-test logs.**

4. **README update** — deferred until the real numbers exist (fill in
   "Current results", close the "Known gaps" bullet).

## Methodology decisions (baked into the scripts)

- **Profit is defined once, identically for both sides** and identically to
  the SAC agent's own accounting: market-settled energy from `m.results`
  valued at the settled slack price, falling back to spot ± margin when the
  slack price is unavailable. Fees never appear (storage is fee-exempt at
  settlement, §118 EnWG). Revenue/fee CSV columns are *not* used.
- **Terminal-SOC adjustment**: adjusted profit = cash profit + liquidation
  value of the terminal charge (`storage_logic.liquidation_value`, at the
  final spot price). Both runs start at SOC 0.40, so the start term cancels
  in the comparison. The final step's settled trade is applied to SOC before
  valuing it (the agent's own `soc` lags one step at run end).
- **Market-settled flows only**: bus 12 also hosts a load and a heat pump,
  so in optimisation mode the LP battery partly serves node self-supply,
  which bypasses market clearing. The comparison counts market-settled
  energy only; this is stated rather than corrected.
- **Two separate runs, endogenous prices**: slack prices differ between the
  runs. The rows-1-4 sanity check quantifies how much; in the 5-step smoke
  test they matched exactly.

## Known caveats — read before interpreting results

1. **The `optimisation` baseline is fee-handicapped as implemented.**
   `HNOptimizer`'s internal objective charges `gridfee + levies`
   (~15.7 ct/kWh) on battery purchases (`HN_optimizer.py:539-548`), and its
   battery bid prices embed the same fees (`HN_optimizer.py:1158, 1165`) —
   but market settlement *exempts* storage from those fees
   (`market_optimizer.py:630-631`). So the LP schedules the battery as a
   fee-paying asset it isn't: it only sees arbitrage where the spread beats
   the phantom fees, and will under-trade. It is a legitimate benchmark of
   "the optimisation method as currently implemented," **not** a true upper
   bound. In the smoke test, row 0 held while rows 1-4 charged at
   near-zero prices — consistent with this handicap.
   - **Optional phase 2** (only if a fair upper bound is wanted): remove the
     fee terms from the LP's *scheduling objective* for battery flows while
     keeping the *bid prices* fee-crossing — the fee-inflated bid doubles as
     the clearing device that lets external buys clear against the market
     LP's `gridfee_levies_ext` welfare term (same trick SAC's
     `action_to_bid` uses deliberately). Naively deleting the fees would
     break clearing. Re-run and report both baselines.
2. **Information asymmetry (by design)**: HNOptimizer re-plans with up to
   ~12 h of actual future prices; SAC sees 6 h of forward day-ahead
   features. This is the benchmark definition, but state it in any writeup.

## How to run

```bash
# 0. Optional smoke test (~1 min each):
python analysis/run_comparison_eval.py --mode optimisation --max-steps 20
python analysis/run_comparison_eval.py --mode sac --max-steps 20

# 1. The two full runs over config.yaml's window (Q1 2023, ~8.5k steps,
#    ~2.25 s/step Gurobi-bound -> several hours EACH; run sequentially):
python analysis/run_comparison_eval.py --mode optimisation
python analysis/run_comparison_eval.py --mode sac

# 2. Compare:
python analysis/compare_sac_vs_optimisation.py
#    -> output/comparison/sac_vs_optimisation.png + stdout summary
```

Then update `readme.md` ("Current results" + "Known gaps") with the numbers.


Everything is ready. Here's what's in place:

Code (all smoke-tested against the real Gurobi market):

1. mesa_model/model.py — new STORAGE0_METHOD env toggle. Default behavior is completely unchanged; STORAGE0_METHOD=optimisation routes the SAC battery (row 0, bus 12) through the same HNOptimizer LP as the other four units. I verified with a real step that bus 12's optimizer then contains the battery (flex=[0,1,3]) and correctly excludes it in SAC mode (flex=[0,3]).
2. analysis/run_comparison_eval.py — one driver for both sides of the comparison (--mode optimisation / --mode sac). Each run writes output/comparison/eval_<mode>.jsonl with per-day records for all five storage units plus terminal SOC/liquidation summaries, using the exact same profit formula as the SAC agent's own accounting. It archives old logs instead of overwriting, and has --max-steps for smoke tests. I ran both modes for 5 real Gurobi steps end-to-end (this caught and fixed two bugs: m.grid.storage is emptied after construction, and m.market_price doesn't exist before the first step).
3. analysis/compare_sac_vs_optimisation.py — produces the 3-panel plot plus a summary with terminal-SOC-adjusted totals, the ratio against the >75% criterion, and the rows-1–4 cross-run sanity check (they matched exactly in the smoke test). Tested against the smoke logs, then I deletart clean.
4. sac-vs-optimisation-comparison-plan.md — updated with implementation status, the methodology decisions, the fee-handicap caveat (with the optional phase-2 fix), and run commands.

When you're ready (each full run is several hours, Gurobi-bound; run them one after the other):

python analysis/run_comparison_eval.py --mode optimisation
python analysis/run_comparison_eval.py --mode sac
python analysis/compare_sac_vs_optimisation.py

One thing to watch: in the 5-step smoke test the optimisation-mode battery held whileed at near-zero prices — consistent with the fee-handicap caveat (its LP plans as ifcharging costs ~15.7 ct/kWh in fees it never actually pays). If the full run shows it barely trading, that's the handicap, not a bug, and the phase-2 fix in the plan is how to get a fair upper bound. README updates wait until the real numbers exist.