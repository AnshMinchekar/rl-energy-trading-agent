"""Run one side of the SAC-vs-optimisation head-to-head over config.yaml's window.

Three modes, each a single pass over the live Mesa/Gurobi market:

    python analysis/run_comparison_eval.py --mode optimisation
        Routes storage row 0 (the SAC battery's bus/physics) through the same
        HNOptimizer/LP path as rows 1-4 (via STORAGE0_METHOD=optimisation).

    python analysis/run_comparison_eval.py --mode arbitrage
        Battery-only level-field LP baseline (STORAGE0_METHOD=arbitrage):
        plans row 0 over exactly SAC's information set (current price + the
        known DA forward window) maximising SAC's reward objective, and bids
        through the same action_to_bid pipeline. See
        optimization/arbitrage_optimizer.py.

    python analysis/run_comparison_eval.py --mode sac
        Frozen-policy SAC evaluation (SAC_EVAL=1, deterministic, no learning),
        loading SAC_LOAD_POLICY (default output/sac/surrogate_policy.pt).

Each run writes output/comparison/eval_<mode>.jsonl containing, for EVERY
storage unit (row 0 plus the four always-optimisation units, so rows 1-4
double as a cross-run sanity check):

  {"type": "meta", ...}          one header line: mode, window, agent ids
  {"type": "day",  ...}          one line per storage agent per calendar day
  {"type": "summary", ...}       one line per storage agent at the end,
                                 incl. terminal SOC + liquidation value

Profit is computed identically for both modes and identically to the SAC
agent's own accounting (agents.py update_status): market-settled energy from
m.results valued at the settled slack price, falling back to spot +/- margin
when the slack price is unavailable that step. Fees never appear: storage is
fee-exempt at settlement (paragraph 118 EnWG, market_optimizer.py).

Compare the two logs with analysis/compare_sac_vs_optimisation.py.

NOTE: ~2.25 s/step (Gurobi-bound) -> several hours for the default Q1-2023
window, same cost as any live run.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

OUT_DIR = "output/comparison"


def git_provenance():
    """Code identity of this run, for the meta record. The compare script
    refuses to compare two logs whose SHAs differ — the 2026-08 stale-run
    mismatch (eval_sac.jsonl a week older than the LP fixes) is exactly the
    failure this exists to prevent."""
    def _run(*cmd):
        return subprocess.run(("git",) + cmd, capture_output=True, text=True,
                              cwd=REPO_ROOT).stdout.strip()
    sha = _run("rev-parse", "HEAD") or None
    # -uno: untracked files (output/, old reports) don't change run behaviour;
    # modified tracked files do.
    dirty = bool(_run("status", "--porcelain", "-uno"))
    return sha, dirty


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["optimisation", "arbitrage", "sac"],
                   required=True)
    p.add_argument("--policy", default="output/sac/surrogate_policy.pt",
                   help="checkpoint for --mode sac (default: %(default)s)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after N steps (smoke test); default: full window")
    return p.parse_args()


def main():
    args = parse_args()

    # Env vars must be set BEFORE importing mesa_model.model (it builds the
    # LEM, and therefore the storage agents, at import time).
    if args.mode in ("optimisation", "arbitrage"):
        os.environ["STORAGE0_METHOD"] = args.mode
    else:
        if not os.path.exists(args.policy):
            sys.exit(f"Policy checkpoint not found: {args.policy} "
                     f"(train one with train_surrogate.py first)")
        os.environ["STORAGE0_METHOD"] = "learning"
        os.environ["SAC_LOAD_POLICY"] = args.policy
        os.environ["SAC_EVAL"] = "1"

    import numpy as np
    from mesa_model.model import model as m
    from mesa_model.storage_logic import liquidation_value, soc_transition

    storages = [a for a in m.agents if getattr(a, "typ", None) == "storage"]
    storages.sort(key=lambda a: a.unique_id)
    # m.grid.storage is emptied after agent construction, so identify row 0 as
    # the lowest-id storage agent (they are created in row order, model.py:198).
    storage0_id = int(storages[0].unique_id)
    margin_buy = m.market_price_margin_buy
    margin_sell = m.market_price_margin_sell
    # m.market_price is only assigned inside step(); read the same series from
    # the ext_grid agent (flex == 999), available at construction.
    price_df = [a for a in m.agents if a.flex == 999][0].energy_price
    spot = dict(zip(price_df["time"], price_df["price"]))  # ct/kWh

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"eval_{args.mode}.jsonl")
    if os.path.exists(out_path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.replace(out_path, out_path.replace(".jsonl", f"_{stamp}.jsonl"))
    out = open(out_path, "w", buffering=1)

    def emit(rec):
        out.write(json.dumps(rec) + "\n")

    git_sha, git_dirty = git_provenance()
    emit({"type": "meta", "mode": args.mode,
          "policy": args.policy if args.mode == "sac" else None,
          "git_sha": git_sha, "git_dirty": git_dirty,
          "storage0_method": os.environ["STORAGE0_METHOD"],
          "run_started": datetime.now().isoformat(timespec="seconds"),
          "total_steps": int(m.total_steps),
          "storage0_id": storage0_id,
          "margin_buy": margin_buy, "margin_sell": margin_sell,
          "storages": [{"agent_id": int(a.unique_id), "bus": int(a.bus),
                        "method": a.method, "capacity_kwh": a.capacity,
                        "efficiency": a.efficiency} for a in storages]})

    def fresh_day():
        return defaultdict(lambda: {"profit": 0.0, "bought": 0.0, "sold": 0.0,
                                    "n_buy": 0, "n_sell": 0,
                                    "buy_price_sum": 0.0, "sell_price_sum": 0.0,
                                    "soc": []})

    def flush_day(day, acc):
        for a in storages:
            d = acc[a.unique_id]
            socs = d["soc"] or [a.soc]
            emit({"type": "day", "date": str(day),
                  "agent_id": int(a.unique_id), "bus": int(a.bus),
                  "actual_profit_eur": round(d["profit"], 4),
                  "energy_bought_kwh": round(d["bought"], 4),
                  "energy_sold_kwh": round(d["sold"], 4),
                  "soc_avg": round(float(np.mean(socs)), 4),
                  "soc_min": round(float(np.min(socs)), 4),
                  "soc_max": round(float(np.max(socs)), 4),
                  "trade_count_buy": d["n_buy"], "trade_count_sell": d["n_sell"],
                  "avg_buy_price": round(d["buy_price_sum"] / d["n_buy"], 4) if d["n_buy"] else 0.0,
                  "avg_sell_price": round(d["sell_price_sum"] / d["n_sell"], 4) if d["n_sell"] else 0.0})

    totals = defaultdict(float)
    current_day, acc = None, fresh_day()
    last_trade = {}     # agent_id -> (bought, sold) of the final step, for terminal SOC
    last_spot = None

    n_steps = m.total_steps if args.max_steps is None else min(args.max_steps, m.total_steps)
    for n in range(n_steps):
        m.step()
        try:
            step_result = m.results[int(m.stepcount)]
            agents_df = step_result["agents"]
        except (KeyError, IndexError):
            continue

        p_now = float(spot.get(m.current_date, 30.0))
        last_spot = p_now
        try:
            p_settle = float(step_result["Input_Grid"]["Market Price [€/kWh]"].to_numpy()[0]) * 100.0
            if np.isnan(p_settle):
                p_settle = None
        except (KeyError, IndexError, TypeError, ValueError):
            p_settle = None
        # Same fallback the SAC agent uses when the slack price is unavailable.
        p_buy = p_settle if p_settle is not None else p_now + margin_buy
        p_sell = p_settle if p_settle is not None else p_now - margin_sell

        day = m.current_date.date()
        if current_day is None:
            current_day = day
        elif day != current_day:
            flush_day(current_day, acc)
            current_day, acc = day, fresh_day()

        for a in storages:
            row = agents_df[agents_df["Agent ID"] == a.unique_id]
            if len(row) == 0:
                continue
            bought = float(row["Energy bought [kWh]"].to_numpy()[0])
            sold = float(row["Energy sold [kWh]"].to_numpy()[0])
            profit = (sold * p_sell - bought * p_buy) / 100.0
            d = acc[a.unique_id]
            d["profit"] += profit
            d["bought"] += bought
            d["sold"] += sold
            d["soc"].append(a.soc)
            if bought > 0.01:
                d["n_buy"] += 1
                d["buy_price_sum"] += p_buy
            if sold > 0.01:
                d["n_sell"] += 1
                d["sell_price_sum"] += p_sell
            totals[a.unique_id] += profit
            last_trade[a.unique_id] = (bought, sold)

        if n % 96 == 0:
            print(f"[{args.mode}] day {n // 96 + 1}: storage0 cumulative profit "
                  f"{totals[storage0_id]:+.2f} EUR", flush=True)

    if current_day is not None:
        flush_day(current_day, acc)

    # Terminal state. agent.soc lags one step behind the last cleared trade
    # (update_status applies it at tau+1, which never runs after the final
    # step), so apply that trade here before valuing the remaining charge.
    for a in storages:
        bought, sold = last_trade.get(a.unique_id, (0.0, 0.0))
        terminal_soc = soc_transition(a.soc, bought, sold, a.capacity,
                                      a.efficiency, a.discharge)
        liq = liquidation_value(last_spot if last_spot is not None else 30.0,
                                terminal_soc, a.capacity, a.efficiency, margin_sell)
        emit({"type": "summary", "agent_id": int(a.unique_id), "bus": int(a.bus),
              "method": a.method,
              "total_profit_eur": round(totals[a.unique_id], 4),
              "terminal_soc": round(float(terminal_soc), 4),
              "terminal_liquidation_eur": round(float(liq), 4),
              "final_spot_ct": last_spot})

    out.close()
    print(f"\nDone. Wrote {out_path}")
    print(f"storage0 (id {storage0_id}) total profit: {totals[storage0_id]:+.2f} EUR")


if __name__ == "__main__":
    main()
