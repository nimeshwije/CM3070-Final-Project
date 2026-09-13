#!/usr/bin/env python3
"""Train (evolve) a trading rule for one or more tickers from the terminal.

This is a thin command-line wrapper over `advisor.pipeline.train_rule` --
the same pipeline the web admin area runs -- so both entry points produce
identical artifacts for identical settings and seed.

Examples
--------
# Single-asset evolution on AAPL, 2015 onwards:
python scripts/train.py --ticker AAPL --start 2015-01-01

# Multi-asset evolution (rule must generalise across all three), saved per ticker:
python scripts/train.py --ticker AAPL MSFT SPY --start 2015-01-01

# Quick smoke test on synthetic data (no network needed):
python scripts/train.py --ticker DEMO --synthetic --generations 10 --population 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advisor.pipeline import TrainRequest, train_rule


def main() -> int:
    ap = argparse.ArgumentParser(description="Evolve a trading rule.")
    ap.add_argument("--ticker", nargs="+", required=True,
                    help="One ticker for single-asset evolution, several for multi-asset.")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="Chronological fraction used for training (rest is held out).")
    ap.add_argument("--population", type=int, default=60)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost", type=float, default=0.001, help="Transaction cost per side.")
    ap.add_argument("--folds", type=int, default=4, help="Walk-forward folds in fitness.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Force synthetic GBM data (offline smoke test).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    req = TrainRequest(
        tickers=[t.upper() for t in args.ticker],
        start=args.start, end=args.end, train_frac=args.train_frac,
        population=args.population, generations=args.generations, seed=args.seed,
        cost=args.cost, folds=args.folds, synthetic=args.synthetic,
    )
    outcome = train_rule(req, progress=lambda msg: print("  " + msg))

    print(f"\nBest genome: {outcome.genome}")
    print(f"Rule: {outcome.rule_text}\n")
    for row in outcome.per_ticker:
        tr, te = row["train"], row["test"]
        table = pd.DataFrame({
            "In-sample": [tr["total_return"], tr["sharpe"], tr["max_drawdown"], tr["n_trades"]],
            "Out-of-sample": [te["total_return"], te["sharpe"], te["max_drawdown"], te["n_trades"]],
            "Buy & hold (OOS)": [te["benchmark_total_return"], te["benchmark_sharpe"],
                                 te["benchmark_max_drawdown"], None],
        }, index=["Total return", "Sharpe ratio", "Max drawdown", "Number of trades"])
        print(f"=== {row['ticker']} ({row['data_source']}) ===")
        print(table.round(3).to_string())
        print(f"Train window: {row['train_range'][0]} .. {row['train_range'][1]}")
        print(f"Test window:  {row['test_range'][0]} .. {row['test_range'][1]}\n")
    for p in outcome.artifacts:
        print(f"Saved rule artifact -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
