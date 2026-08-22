# SAC vs HNOptimizer — Head-to-Head Performance Report

**Unit under test:** storage row 0, agent 23, bus 12 — 146.7 kWh / 95% round-trip / SimBench `p_mw`
**Window:** 2023-01-01 00:00 → 2023-03-30 23:45 (8,543 steps × 15 min, 89 days)
**Runs:** `output/comparison/eval_sac.jsonl`, `output/comparison/eval_optimisation.jsonl`
**Produced by:** `analysis/run_comparison_eval.py` → `analysis/compare_sac_vs_optimisation.py`
**Date:** 2026-08-09

---

## 1. Headline

| | days | profit € | term. SOC | liq. € | **adjusted €** |
|---|---:|---:|---:|---:|---:|
| SAC (frozen policy) | 89 | 633.82 | 0.396 | 4.84 | **638.66** |
| optimisation (HNOptimizer LP) | 89 | 200.84 | 0.614 | 7.51 | **208.35** |

**SAC / optimisation = 306.5%**, against a >75% success criterion.

The criterion is met by a wide margin. **This number should not be reported as "SAC beats the
optimiser 3×" without the qualifications in §4** — the baseline is running with a modelling
defect that suppresses its trading, and the two agents are not solving the same problem.

The arithmetic is sound. The interpretation is not yet.

---

## 2. What actually happened

### 2.1 Volume and pricing

| | bought kWh | sold kWh | buy steps | sell steps | avg buy ct | avg sell ct | day-avg spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC | 29,797 | 26,883 | 3,939 (46%) | 3,201 (37%) | 11.11 | 13.46 | 2.35 ct |
| optimisation | 9,534 | 8,564 | **629 (7%)** | 2,741 (32%) | 10.05 | 13.37 | 3.32 ct |

The single most informative cell is the optimiser's **629 buy steps out of 8,543**. It is not
trading badly — it is barely charging at all. See §4.1.

### 2.2 Daily profit distribution

| | mean/day | median | std | losing days | worst | best |
|---|---:|---:|---:|---:|---:|---:|
| SAC | €7.12 | €5.91 | 8.33 | 16 (18%) | −19.70 | +32.85 |
| optimisation | €2.26 | €2.44 | 7.70 | 28 (31%) | −19.81 | +22.45 |

SAC's advantage is broad-based, not driven by a handful of outlier days: it wins on the median,
loses money on fewer days, and has a comparable downside tail. Monthly totals are stable for both.

| | Jan 2023 | Feb 2023 | Mar 2023 |
|---|---:|---:|---:|
| SAC | 210.05 | 200.62 | 223.15 |
| optimisation | 63.19 | 64.69 | 72.96 |

### 2.3 Cycling intensity

| | eq. cycles/day (mean) | peak day | total eq. cycles |
|---|---:|---:|---:|
| SAC | **2.17** | 3.70 | 193 |
| optimisation | 0.69 | 1.56 | 62 |

SAC cycles the battery **3.1× harder**. Both respect the `max_power` and SOC bounds enforced in
`agents.py:531-542` (`provide_a_power`), so the throughput is physically legal within the model.

---

## 3. The decomposition: SAC wins on volume, not on skill

Normalising profit by battery throughput — the quantity that actually consumes battery life:

| | throughput kWh | profit € | **profit per kWh throughput** |
|---|---:|---:|---:|
| SAC | 28,340 | 633.82 | **2.236 ct/kWh** |
| optimisation | 9,049 | 200.84 | **2.219 ct/kWh** |

**The two strategies are equally efficient per unit of battery wear — within 0.8% of each other.**

SAC's entire €433 advantage is explained by cycling 3.1× more, at the same margin per cycle. It is
not making better decisions per trade; it is making more of the same-quality trades.

The day-averaged spread in §2.1 (2.35 ct vs 3.32 ct) superficially suggests SAC trades *worse*, but
that statistic is an unweighted mean over days and is unreliable. Profit-per-throughput is the
volume-weighted truth, and it says the two are indistinguishable.

### 3.1 Degradation sensitivity

Because both earn the same margin per kWh cycled, a per-kWh wear cost scales both sides almost
identically and **the ratio is invariant**:

| wear cost | SAC € | optimisation € | ratio |
|---:|---:|---:|---:|
| 0.0 ct/kWh | +633.82 | +200.84 | 316% |
| 1.0 ct/kWh | +350.42 | +110.35 | 318% |
| 2.0 ct/kWh | +67.01 | +19.86 | 337% |
| 2.5 ct/kWh | −74.69 | −25.38 | n/a |

Break-even wear cost: **SAC 2.236 ct/kWh, optimisation 2.219 ct/kWh** — the same point.

Two conclusions, and they pull in opposite directions:

- Degradation does **not** differentially favour the optimiser. An earlier working hypothesis that
  pricing wear would collapse SAC's lead was wrong; it collapses both absolute profits equally.
- But no degradation cost is modelled anywhere, and **both strategies are only marginally profitable
  once wear exceeds ~2.2 ct/kWh** — well inside the plausible range for Li-ion. The absolute
  profitability of this entire setup is fragile even though the comparison between the two is not.

---

## 4. Threats to validity

### 4.1 The baseline pays a phantom 16.7 ct/kWh charging tax — **critical**

`optimization/HN_optimizer.py:540` and `:548` price grid purchases in the LP objective as:

```python
(m.prices[t] + self.margin_buy + self.levies + self.gridfee) / 100
```

= spot + 1.0 + 4.7 + 11.0 = **spot + 16.7 ct/kWh**.

Market settlement, however, explicitly zeroes fees for storage under §118(6) EnWG
(`optimization/market_optimizer.py:630-631`), and that fee-free settlement is what both runs are
scored on. The LP is therefore optimising against a cost ~16.7 ct/kWh above what it actually pays
to charge. Q1-2023 daily spreads are roughly 5–8 ct/kWh, so a 16.7 ct phantom cost makes nearly
every arbitrage trade look loss-making. Hence 629 buy steps.

SAC is unaffected because `agents.py:633` bids *above* the fee-inclusive price purely to guarantee a
fill, then settles fee-free. SAC is behaving correctly; the LP is not.

**This is the dominant driver of the 3× gap and it is a defect in the baseline, not a merit of SAC.**

Note the fix is **not** deleting the two fee terms. `HN_optimizer.py:500` defines the fee base as the
aggregate node exchange:

```python
m.power_net[t] == sum(m.power_buy[i,t] for i in m.AGENTS) - sum(m.power_sell[i,t] for i in m.AGENTS)
```

which bundles household load with battery charging. Removing the terms would wrongly exempt ordinary
household consumption from grid fees and perturb every household's dispatch. The correct change
decomposes the fee base — fees on `power_net` **minus** storage charging power, floored at zero,
while the energy price continues to apply to the full `power_net`. That requires a new variable and
a Big-M split, and care around the `buy_flag` sign conventions that also key off `power_net`.

### 4.2 The two agents are not solving the same problem — structural, not fixable by re-running

The LP battery is embedded in a whole-node optimisation that co-optimises household load, EV and
heat-pump dispatch. SAC row 0 is a standalone arbitrageur. The evaluation scores only market cash
flows, which is one of the two objectives the LP is actually pursuing and the only one SAC pursues.

Even with §4.1 fixed, this asymmetry remains and must be stated as a limitation. A fair contest
would require either an arbitrage-only LP for row 0, or crediting the LP for household cost savings.

### 4.3 ~24% of the edge is transferred, not created

Rows 1–4 run `optimisation` in **both** runs, so their totals should match. They do not:

| agent | bus | SAC-run € | opt-run € | diff |
|---:|---:|---:|---:|---:|
| 24 | 9 | 107.42 | 125.39 | −17.97 |
| 25 | 14 | 55.58 | 89.05 | −33.48 |
| 26 | 6 | 49.39 | 57.02 | −7.63 |
| 27 | 10 | 147.89 | 191.92 | −44.03 |
| **total** | | **360.28** | **463.38** | **−103.10** |

Up to 38% divergence on a single unit. SAC's 3× volume moves clearing prices against the other
batteries, costing them €103. Against SAC's €433 raw edge over the optimiser row 0, roughly **24% of
the measured advantage is value taken from other market participants rather than newly created**.

This also means the two runs are not the same market, which weakens any strict row-0 comparison.

---

## 5. Data integrity checks

Both runs pass the checks that would catch phantom energy or double-counted cash:

- **Energy balance (SAC).** Net purchase 2,913.8 kWh vs round-trip losses of
  `29,797.2 × 0.05 + 26,883.4 × (1/0.95 − 1)` = 2,904.8 kWh. Residual 9.0 kWh = 0.03% of throughput,
  absorbed by the SOC delta and self-discharge. No unexplained energy.
- **Energy balance (optimisation).** Net purchase 969.6 kWh vs losses 927.4 kWh, remainder consistent
  with the higher terminal SOC (0.614 vs 0.396).
- **Identical accounting.** Both runs value cash with the same formula and the same settled price,
  `(sold × p_sell − bought × p_buy)/100` — `run_comparison_eval.py:169`, matching the SAC agent's own
  `agents.py:740`.
- **Same starting state.** Both runs open at the same SOC, so the start-inventory term cancels; only
  terminal charge needs the liquidation adjustment, which is applied.
- **Grid identity.** `storage0_id = 23` in both runs; capacities, buses and efficiencies match across
  the two `meta` records.

The measurement is trustworthy. The experiment design is what is in question.

---

## 6. Verdict and recommendation

**As measured:** SAC returns 306.5% of the optimisation baseline over 89 days, meeting the >75%
criterion decisively, with a better median day and fewer losing days.

**As a defensible claim:** the result currently reads *"SAC beats a fee-handicapped LP baseline by
3×, by cycling the battery 3.1× harder at identical margin per cycle, with ~24% of the gap coming
from price impact on other units."* That is a materially weaker statement.

Priority order for making this publishable:

1. **Fix §4.1** — decompose the fee base in `HN_optimizer.py` so the LP's objective matches
   settlement, then re-run `--mode optimisation`. Expect the optimiser's buy-step count to rise
   sharply from 629 and the ratio to fall substantially toward 100%. This is the only defect that
   invalidates the comparison outright.
2. **Report profit-per-throughput alongside the ratio.** It is the metric that reveals the two
   policies are equally skilled, and it is the honest headline.
3. **State §4.2 and §4.3 as limitations** in the comparison plan. Neither is removable by re-running.
4. **Model a degradation cost**, even crudely. At ~2.2 ct/kWh both strategies break even, so the
   absolute case for this battery is unproven regardless of which controller wins.

Until item 1 is done, treat 306.5% as an upper bound on SAC's true advantage, not an estimate of it.
