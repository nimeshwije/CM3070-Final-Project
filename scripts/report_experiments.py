#!/usr/bin/env python3
"""All quantitative experiments reported in the final report, in one run.

Writes reports/final_experiments.json.  Everything is reproducible from the
cached price snapshot in data_cache/ and the fixed seeds below.

Experiments
-----------
A  Champion (hardened engine) on the report snapshot: AAPL 2015-01-02 .. 2024-11-29, 70/30.
B  Ablation: prototype configuration (single-window Sharpe, no costs, no regularisation).
C  Multi-seed robustness: A and B repeated over 10 seeds.
D  Cost sensitivity of the champion (0 / 10 / 25 / 50 bp per side).
E  Rolling two-year windows over the full history to 2026.
F  Forward test: the frozen champion on data that did not exist when it was evolved
   (2024-11-30 .. 2026-09-11), and on the whole 2021-12-08 .. 2026-09-11 window.
G  Cross-ticker evaluation: the 14 rules trained through the admin area (70/30 on
   2015-01-01 .. 2026-09-11), re-evaluated here from their artifacts.
H  Rule transfer: the AAPL champion applied unchanged to other tickers, out-of-sample.
"""
from __future__ import annotations

import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from advisor.data import fetch_prices
from advisor.evaluation import chronological_split, evaluate_split, walk_forward_report
from advisor.evolution import GAConfig, evolve
from advisor.fitness import FitnessConfig
from advisor.genome import Genome
from advisor.backtest import run_backtest
from advisor.persistence import load_artifact, list_artifacts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "final_experiments.json")

SNAP_START, SNAP_END = "2015-01-01", "2024-11-29"
HARDENED = dict(n_folds=4, consistency_lambda=0.5, min_trades_per_year=2.0, sparsity_penalty=1.0, cost=0.001)
PROTOTYPE = dict(n_folds=1, consistency_lambda=0.0, min_trades_per_year=0.0, sparsity_penalty=0.0, cost=0.0)


def m2(m):  # compact metrics
    keys = ["total_return", "sharpe", "max_drawdown", "annual_volatility", "n_trades", "time_in_market",
            "benchmark_total_return", "benchmark_sharpe", "benchmark_max_drawdown", "benchmark_annual_volatility"]
    return {k: (None if m.get(k) is None else float(m[k])) for k in keys if k in m}


def run_ga(train, fit_kwargs, seed):
    ga = GAConfig(population_size=60, generations=40, seed=seed)
    fit = FitnessConfig(**fit_kwargs)
    t0 = time.time()
    res = evolve({"AAPL": train}, ga, fit)
    return res, time.time() - t0


def split_eval(prices, genome, cost):
    ev = evaluate_split(prices, genome, 0.7, cost=cost)
    return {"train": m2(ev.train_result.metrics), "test": m2(ev.test_result.metrics),
            "train_range": ev.train_range, "test_range": ev.test_range}


out = {}
aapl_snap = fetch_prices("AAPL", start=SNAP_START, end=SNAP_END).prices
aapl_full = fetch_prices("AAPL", start=SNAP_START).prices
train_snap, test_snap = chronological_split(aapl_snap, 0.7)
print(f"snapshot {len(aapl_snap)} days, full {len(aapl_full)} days")

# ---- A + B (seed 7) --------------------------------------------------------
res_h, t_h = run_ga(train_snap, HARDENED, 7)
res_p, t_p = run_ga(train_snap, PROTOTYPE, 7)
champ, proto = res_h.best_genome, res_p.best_genome
out["A_champion"] = {"genome": champ.to_dict(), "rule": champ.describe(), "seconds": t_h,
                     "history": res_h.history, **split_eval(aapl_snap, champ, 0.001)}
out["B_ablation"] = {"genome": proto.to_dict(), "rule": proto.describe(), "seconds": t_p,
                     **split_eval(aapl_snap, proto, 0.001),
                     "eval_no_cost": split_eval(aapl_snap, proto, 0.0)}
print("A", champ.to_dict(), out["A_champion"]["test"]["sharpe"])
print("B", proto.to_dict(), out["B_ablation"]["test"]["sharpe"])

# ---- C multi-seed ------------------------------------------------------------
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rows = []
for cfg_name, cfg in (("hardened", HARDENED), ("prototype", PROTOTYPE)):
    for s in seeds:
        res, secs = run_ga(train_snap, cfg, s)
        g = res.best_genome
        ev = split_eval(aapl_snap, g, 0.001)
        rows.append({"config": cfg_name, "seed": s, "genome": g.to_dict(),
                     "is_sharpe": ev["train"]["sharpe"], "oos_sharpe": ev["test"]["sharpe"],
                     "oos_return": ev["test"]["total_return"], "oos_mdd": ev["test"]["max_drawdown"],
                     "oos_vol": ev["test"]["annual_volatility"], "oos_trades": ev["test"]["n_trades"],
                     "bh_sharpe": ev["test"]["benchmark_sharpe"], "seconds": secs})
        print(f"  {cfg_name} seed {s}: IS {ev['train']['sharpe']:.2f} OOS {ev['test']['sharpe']:+.2f} trades {ev['test']['n_trades']}")
out["C_multiseed"] = rows

# ---- D cost sensitivity ------------------------------------------------------
out["D_costs"] = {}
for bp in (0, 10, 25, 50):
    ev = split_eval(aapl_snap, champ, bp / 10_000)
    out["D_costs"][str(bp)] = ev["test"]

# ---- E rolling windows (full history to 2026) --------------------------------
wf = walk_forward_report(aapl_full, champ, window_years=2.0, cost=0.001)
out["E_rolling"] = wf.to_dict(orient="records")

# ---- F forward test ----------------------------------------------------------
fwd = aapl_full[aapl_full.index > pd.Timestamp(SNAP_END)]
oos_all = aapl_full[aapl_full.index >= pd.Timestamp(out["A_champion"]["test_range"][0])]
# warm-up: indicators need history, so backtest on a window that includes the long
# window of prior prices, then score only the forward part.
def scored_window(full, start_ts, genome, cost=0.001):
    warm = genome.long_window + 5
    idx = full.index.get_indexer([start_ts], method="bfill")[0]
    chunk = full.iloc[max(0, idx - warm):]
    res = run_backtest(chunk, genome, cost)
    r = res.daily_returns[res.daily_returns.index >= start_ts]
    b = chunk.pct_change().fillna(0.0)[chunk.index >= start_ts]
    from advisor.backtest import sharpe_ratio, max_drawdown
    eq, beq = (1 + r).cumprod(), (1 + b).cumprod()
    pos = res.positions[res.positions.index >= start_ts] if hasattr(res, "positions") else None
    return {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "days": int(len(r)),
            "total_return": float(eq.iloc[-1] - 1), "sharpe": float(sharpe_ratio(r)),
            "max_drawdown": float(max_drawdown(eq)), "annual_volatility": float(r.std() * np.sqrt(252)),
            "benchmark_total_return": float(beq.iloc[-1] - 1), "benchmark_sharpe": float(sharpe_ratio(b)),
            "benchmark_max_drawdown": float(max_drawdown(beq)),
            "benchmark_annual_volatility": float(b.std() * np.sqrt(252))}
out["F_forward"] = {"post_snapshot": scored_window(aapl_full, fwd.index[0], champ),
                    "full_oos_to_2026": scored_window(aapl_full, pd.Timestamp(out["A_champion"]["test_range"][0]), champ)}
print("F", out["F_forward"])

# ---- G cross-ticker (admin-trained artifacts) --------------------------------
out["G_cross_ticker"] = []
for t in sorted(list_artifacts()):
    if t in ("AAPL", "AAPL_SNAPSHOT2024") or not t.isalpha():
        continue
    art = load_artifact(t)
    meta = art.get("meta", {})
    req = meta.get("request", {})
    ps = fetch_prices(t, start=req.get("start", "2015-01-01"), end=req.get("end"))
    ev = split_eval(ps.prices, art["genome_obj"], req.get("cost", 0.001))
    out["G_cross_ticker"].append({"ticker": t, "genome": art["genome"], "rule": art["rule_text"],
                                  "days": int(len(ps)), "source": ps.source, **ev})
    te = ev["test"]
    print(f"  {t:5s} OOS Sharpe {te['sharpe']:+.2f} vs {te['benchmark_sharpe']:+.2f}  MDD {te['max_drawdown']:.0%} vs {te['benchmark_max_drawdown']:.0%}")

# ---- H rule transfer (AAPL champion unchanged, other tickers, OOS = same test dates) --
out["H_transfer"] = []
for t in ["SPY", "AMD", "INTC", "MU", "STX", "GLW", "WBD", "WDC", "AMAT", "ADBE", "MRVL", "PODD"]:
    ps = fetch_prices(t, start=SNAP_START, end=SNAP_END)
    if len(ps) < 2000:
        continue
    ev = split_eval(ps.prices, champ, 0.001)
    out["H_transfer"].append({"ticker": t, **ev["test"], "test_range": ev["test_range"]})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
