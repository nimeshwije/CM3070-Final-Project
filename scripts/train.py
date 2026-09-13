#!/usr/bin/env python3
"""Train (evolve) a trading rule for one or more tickers.

This is the OFFLINE pipeline: data ingestion -> evolutionary engine ->
evaluation -> saved rule artifact.  The end user never runs this; the web
advisor only loads the resulting models/<ticker>.json.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advisor.data import fetch_prices, synthetic_gbm, PriceSeries
from advisor.evaluation import chronological_split, evaluate_split
from advisor.evolution import GAConfig, evolve
from advisor.fitness import FitnessConfig
from advisor.persistence import save_artifact


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

    # ---------------------------------------------------------- data
    series: dict[str, PriceSeries] = {}
    for t in args.ticker:
        if args.synthetic:
            series[t] = PriceSeries(t, synthetic_gbm(seed=abs(hash(t)) % 10_000), "synthetic")
        else:
            series[t] = fetch_prices(t, start=args.start, end=args.end)
        print(f"{t}: {len(series[t])} days ({series[t].source})")

    train_map, test_map = {}, {}
    for t, ps in series.items():
        tr, te = chronological_split(ps.prices, args.train_frac)
        train_map[t], test_map[t] = tr, te

    # ---------------------------------------------------------- evolve
    ga_cfg = GAConfig(population_size=args.population, generations=args.generations, seed=args.seed)
    fit_cfg = FitnessConfig(n_folds=args.folds, cost=args.cost)

    def progress(gen, best, mean):
        print(f"  gen {gen:3d}  best fitness {best:+.3f}  mean {mean:+.3f}")

    print(f"\nEvolving on {list(train_map)} "
          f"(pop {ga_cfg.population_size}, {ga_cfg.generations} generations)...")
    result = evolve(train_map, ga_cfg, fit_cfg, progress_callback=progress)
    genome = result.best_genome
    print(f"\nBest genome: {genome.to_dict()}")
    print(f"Rule: {genome.describe()}\n")

    # ---------------------------------------------------------- evaluate & save
    for t, ps in series.items():
        ev = evaluate_split(ps.prices, genome, args.train_frac, cost=args.cost)
        print(f"=== {t} ===")
        print(ev.summary_table().round(3).to_string())
        print(f"Train window: {ev.train_range[0]} .. {ev.train_range[1]}")
        print(f"Test window:  {ev.test_range[0]} .. {ev.test_range[1]}\n")
        path = save_artifact(
            t,
            genome,
            train_metrics=ev.train_result.metrics,
            test_metrics=ev.test_result.metrics,
            meta={
                "data_source": ps.source,
                "train_range": ev.train_range,
                "test_range": ev.test_range,
                "ga": vars(ga_cfg),
                "fitness": vars(fit_cfg),
                "history": result.history,
                "multi_asset_partners": [x for x in series if x != t],
            },
        )
        print(f"Saved rule artifact -> {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
