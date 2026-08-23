# SAC vs the optimiser — the plain-English version

**What was tested:** one battery (storage row 0, agent 23, bus 12 — 146.7 kWh, 95% round-trip),
run twice over the same three months of 2023 (1 Jan → 30 Mar, 8,543 fifteen-minute steps).
Once controlled by the trained SAC policy, once by the existing HNOptimizer LP.

**Chart:** `Comparision/sac_vs_optimisation_chart.png`
**Numbers came from:** `output/comparison/eval_sac.jsonl`, `output/comparison/eval_optimisation.jsonl`
*(note: `eval_sac.jsonl` has since been overwritten by the 22 Aug 2026 rerun on commit
`f83c38c` — the original numbers below survive only in this document and the chart)*
**Rebuild both with:** `python Comparision/make_comparison_chart.py`

This is a simplified retelling of `analysis/sac_vs_optimisation_report.md`. Same data, same
conclusions, fewer equations. The long report is the one to cite.

> **Superseded headline (23 Aug 2026).** Everything below this box describes the *original*
> matchup against the fee-handicapped whole-node LP. That matchup has since been re-run as a
> fair fight — see the next section. The number to quote now is **≈98%**, not 306.5%.

---

## Update (23 Aug 2026): the level-field rematch — SAC reaches ≈98% of a perfect planner

The "what to fix" list at the bottom of this document asked for an arbitrage-only LP for row 0.
That baseline now exists (`optimization/arbitrage_optimizer.py`, commit `f83c38c`) and both
sides have been re-run over the same 89-day window on the same commit, clean tree, with the
comparison script's provenance guards passing.

**Why this matchup is fair where the old one wasn't.** The new LP controls the same battery
with exactly SAC's information (the current price plus the 24 known day-ahead prices — 6 h,
nothing more), maximises exactly the objective SAC's reward expresses (trading cash flow at
price ± margin, plus the terminal charge at liquidation value), and places its bids through
the *same* `action_to_bid()` pipeline — same deadband, same emergency overrides, same
fee-crossing bid price. No phantom fees, no household duties, no information edge in either
direction. The only difference left is how the action is chosen: a solver with a perfect
6-hour plan vs a learned policy.

**Chart:** `Comparision/sac_vs_arbitrage_chart.png` (canonical copy:
`output/comparison/sac_vs_arbitrage.png`)
**Numbers:** `output/comparison/eval_sac.jsonl`, `output/comparison/eval_arbitrage.jsonl`
**Rebuild:** `python analysis/compare_sac_vs_optimisation.py --baseline arbitrage`

| | profit over 89 days | energy sold | profit per kWh sold | terminal SOC | after valuing leftover charge |
|---|---:|---:|---:|---:|---:|
| SAC (frozen policy) | €598.37 | 26,876 kWh | 2.226 ct | 0.399 | **€603.25** |
| Arbitrage LP | €611.29 | 34,667 kWh | 1.763 ct | 0.074 | **€612.20** |

SAC / LP, terminal-SOC-adjusted: **98.5%** — the ">75% of the optimiser" criterion is **met**,
this time in a contest with nothing to apologise for.

**The interesting part is *how* they tie.** The solver churns: 29% more energy sold at a
thinner 1.76 ct/kWh margin, inventory run down to SOC 0.07 by window end. SAC is choosier:
fewer, fatter trades at 2.23 ct/kWh and more charge held back. Two different temperaments,
one almost identical total. Given the same 6-hour crystal ball, the learned policy extracts
essentially all the profit that is mathematically there to extract.

**One honest error bar.** The two runs are still separate markets: row 0 behaving differently
shifts clearing prices for everyone. The always-optimiser batteries 1–4 — which should earn
identically in both runs — diverged by up to 40% in relative terms on the smallest of them
(worst case ±€8.7 absolute, against row-0 profits of ~€600). That puts roughly a ±1.5% noise
band on the ratio. So: quote "**≈98%**", not "98.5%".

**Residual asymmetries, stated once:** bid prices are mechanical for the LP and learned for
SAC (pricing is part of each process, so this is a feature of the contest, not a flaw), and
the endogenous price impact above is measurable but not removable in separate runs. The
cycle-ageing caveat from the original analysis below applies to both sides *equally* here —
if anything more to the LP, which cycles harder.

---

## The one-paragraph answer

SAC made €634 over three months. The optimiser made €201. So SAC won by about 3×, and the
target was "SAC must reach at least 75% of the optimiser" — passed easily.

**But the win is mostly not real.** The optimiser was playing with a bug that makes it think
charging costs 16.7 ct/kWh more than it really does, so it barely charged at all. And when you
measure *quality* instead of *quantity*, the two are dead even. SAC didn't trade better — it
just traded far more often against an opponent that had mostly stopped trading.

---

## The three things the chart shows

### 1. SAC earns about 3× more money

| | profit over 89 days | after valuing leftover charge |
|---|---:|---:|
| SAC | €633.82 | **€638.66** |
| Optimiser | €200.84 | **€208.35** |

SAC / optimiser = **306.5%**. The lead is steady, not one lucky week — SAC wins on the median
day (€5.91 vs €2.44), loses money on fewer days (16 of 89 vs 28), and beats the optimiser in
each of January, February and March separately.

### 2. ...but each individual trade is equally good

Divide profit by the energy actually pushed through the battery:

| | energy cycled | profit | **profit per kWh cycled** |
|---|---:|---:|---:|
| SAC | 28,340 kWh | €633.82 | **2.236 ct/kWh** |
| Optimiser | 9,049 kWh | €200.84 | **2.219 ct/kWh** |

**0.8% apart.** That is a tie. Both controllers buy low and sell high by the same margin. SAC's
whole €433 lead comes from doing it 3.1× more often, not from doing it better. Same skill,
more repetitions.

(A quick look at the day-averaged buy/sell spread makes SAC look *worse* — 2.35 ct vs 3.32 ct.
Ignore that number. It averages small days and big days equally. Profit-per-kWh is the one
weighted by how much energy was actually moved, and it says "tie".)

### 3. The optimiser almost never charges — and that's a bug, not a strategy

Out of 8,543 possible moments to buy energy:

- SAC bought on **3,939** of them (46%)
- The optimiser bought on **629** of them (7%)

The optimiser isn't losing; it's abstaining. The reason is in `optimization/HN_optimizer.py:540`
and `:548`. When the LP decides whether charging is worth it, it prices grid purchases at
spot + 1.0 + 4.7 + 11.0 = **spot + 16.7 ct/kWh** in fees.

The market never charges that. Batteries are fee-exempt under §118(6) EnWG, and the settlement
code applies that exemption (`optimization/market_optimizer.py:630-631`) — which is what *both*
runs were paid on. Typical Q1-2023 daily price spreads are only 5–8 ct/kWh, so a phantom 16.7 ct
cost makes nearly every arbitrage trade look like a loser. The LP correctly refuses to make
trades that are, on its own books, unprofitable. Its books are wrong.

SAC dodges this because it bids above the fee-inclusive price just to guarantee a fill
(`agents.py:633`), then gets settled fee-free. SAC is behaving correctly here.

**This single defect is the main reason the scoreboard says 3×.**

---

## Not on the chart: neither one survives a realistic battery-wear cost

**First, what *is* modelled:** self-discharge (0.13%/day, `agents.py:414` and `:702`) and 95%
round-trip efficiency (`agents.py:700`). Both are real, both cost money — but both are *energy*
losses. You buy kWh and get fewer kWh back.

**What is not modelled is ageing:** `self.capacity` is set once in `agents.py:412` and never
changes, so the battery is 146.7 kWh on day 89 exactly as on day 1. Nothing shortens its life,
and neither the SAC reward (`storage_logic.py:113`) nor the LP objective contains any per-cycle
or per-kWh-throughput penalty. Cycling the pack is free in this simulation, so a controller
that cycles 3.1× harder pays nothing extra for it.

Add that cost back by hand — charge a price per kWh cycled — and both profits fall:

| wear cost | SAC | Optimiser |
|---:|---:|---:|
| 0.0 ct/kWh | +€634 | +€201 |
| 1.0 ct/kWh | +€350 | +€110 |
| 2.0 ct/kWh | +€67 | +€20 |
| 2.5 ct/kWh | −€75 | −€25 |

Both hit zero at about **2.2 ct/kWh**, which is inside the plausible range for real Li-ion.
Note this cuts against an earlier guess: charging for wear does **not** rescue the optimiser.
Because they earn the same margin per cycle, wear scales both down together and the ratio
barely moves (316% → 337%). It doesn't change who wins — it just makes both of them roughly
break-even in absolute terms.

---

## Two more reasons to be careful

**They aren't playing the same game.** The LP battery is part of a whole-house optimisation —
it's also juggling household load, the EV and the heat pump. SAC is a pure arbitrage bot with
one job. The scoring only counts market cash, which is everything SAC cares about and only half
of what the LP is trying to do. Fixing the fee bug won't fix this; it needs either an
arbitrage-only LP for row 0, or credit given to the LP for household savings.

**About a quarter of SAC's edge is taken, not created.** Batteries 1–4 run the same optimiser
code in *both* runs, so they should earn the same in both. They don't — they earned €103 less in
the SAC run, because SAC's extra volume moved clearing prices against them. Measured against
SAC's €433 raw edge, roughly **24% of the advantage is money moved off other participants**
rather than newly created. It also means the two runs weren't quite the same market, which
weakens any strict head-to-head.

---

## Is the measurement itself trustworthy?

Yes. The arithmetic checks out:

- **Energy adds up.** SAC's net purchase of 2,913.8 kWh matches its predicted round-trip losses
  of 2,904.8 kWh to within 9 kWh — 0.03% of throughput. No phantom energy. Same for the
  optimiser run.
- **Both sides counted identically.** Same cash formula, same settled price, for both runs.
- **Same starting battery, same grid.** Both open at the same SOC and the same agent IDs,
  capacities and buses.

The measurement is sound. It's the *matchup* that's unfair.

---

## What to actually say out loud

**Don't say:** "SAC beats the optimiser 3×." (That was the fee-handicapped matchup.)

**Do say:** "Against a perfect-information-parity LP — same forward prices, same objective, same
bidding pipeline — the trained SAC policy earns ≈98% of the optimum over 89 days, trading less
volume at a higher margin per kWh."

## What to fix, in order *(status as of 23 Aug 2026)*

1. ~~**An arbitrage-only LP for row 0** as the fair baseline~~ — **DONE.** Built
   (`optimization/arbitrage_optimizer.py`), verified (LP SOC path matches the live
   `soc_transition` to 1e-15 over randomised tests), run, and reported in the update section
   above. This resolves the "not playing the same game" objection outright, without needing
   the Big-M fee-base surgery in `HN_optimizer.py` (that fix — decomposing fees on net exchange
   minus storage charging — remains open, but now only matters for the *whole-node* baseline,
   which is optional context rather than the headline).
2. **Report profit-per-kWh next to the ratio** — **DONE** in the update table: SAC 2.226 ct/kWh
   vs LP 1.763 ct/kWh. In the fair matchup SAC's per-kWh skill is *higher*; the LP closes the
   gap on volume.
3. **Write down the limitations** — **DONE** ("residual asymmetries" in the update section:
   learned-vs-mechanical bid pricing, and the ±1.5% cross-run price-impact band).
4. **Put a cycle-ageing cost in the model** — still open, and still the most important honesty
   item in *absolute* terms: a per-kWh-throughput charge in both the reward and the LP objective.
   Both controllers break even at roughly 2.2 ct/kWh of wear, so whether this battery makes real
   money remains unproven no matter which controller wins. Note that in the fair matchup a wear
   cost would hit the *LP harder* (29% more throughput), so it cannot rescue an anti-SAC verdict —
   there is no anti-SAC verdict left to rescue.

The old caution — "treat 306.5% as a ceiling, not a measurement" — is now settled: the fair
measurement is **≈98%**, and the right way to read it is not "the LP wins by 1.5%" but
"a learned policy matches a perfect short-horizon planner to within the noise of the market
it trades in."
