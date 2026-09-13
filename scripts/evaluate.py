#!/usr/bin/env python3
"""Full quantitative evaluation of a saved rule.

Produces, for a ticker with a trained artifact in models/:
  * the in-sample / out-of-sample / buy-and-hold summary table,
  * a rolling walk-forward table across the full history,
  * plots: equity curves (train+test), GA convergence, drawdown -- saved
    under reports/ for direct inclusion in the final report.

Example:
    python scripts/evaluate.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advisor.data import fetch_prices
from advisor.evaluation import evaluate_split, walk_forward_report
from advisor.persistence import load_artifact

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a saved evolved rule.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--cost", type=float, default=0.001)
    args = ap.parse_args()

    art = load_artifact(args.ticker)
    genome = art["genome_obj"]
    print(f"Loaded rule for {args.ticker}: {art['rule_text']}\n")

    ps = fetch_prices(args.ticker, start=args.start, end=args.end)
    print(f"Data: {len(ps)} days ({ps.source})\n")

    ev = evaluate_split(ps.prices, genome, args.train_frac, cost=args.cost)
    print("=== Summary (train / held-out test / benchmark) ===")
    print(ev.summary_table().round(3).to_string(), "\n")

    wf = walk_forward_report(ps.prices, genome, cost=args.cost)
    print("=== Rolling walk-forward windows ===")
    print(wf.round(3).to_string(index=False), "\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ---- equity curves ---------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for ax, res, title in (
        (axes[0], ev.train_result, f"{args.ticker} in-sample {ev.train_range}"),
        (axes[1], ev.test_result, f"{args.ticker} out-of-sample {ev.test_range}"),
    ):
        ax.plot(res.equity.index, res.equity.values, label="Evolved strategy")
        ax.plot(res.benchmark_equity.index, res.benchmark_equity.values,
                label="Buy & hold", alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("Equity (start = 1.0)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    eq_path = os.path.join(REPORTS_DIR, f"{args.ticker}_equity.png")
    fig.savefig(eq_path, dpi=150)
    print(f"Saved {eq_path}")

    # ---- GA convergence (if history stored) ------------------------------
    history = art.get("meta", {}).get("history")
    if history:
        h = pd.DataFrame(history)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(h["generation"], h["best_fitness"], label="Best fitness")
        ax2.plot(h["generation"], h["mean_fitness"], label="Mean fitness", alpha=0.8)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Walk-forward fitness")
        ax2.set_title(f"GA convergence -- {args.ticker}")
        ax2.legend()
        ax2.grid(alpha=0.3)
        conv_path = os.path.join(REPORTS_DIR, f"{args.ticker}_convergence.png")
        fig2.savefig(conv_path, dpi=150)
        print(f"Saved {conv_path}")

    # ---- drawdown --------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    eq = ev.test_result.equity
    dd = eq / eq.cummax() - 1.0
    beq = ev.test_result.benchmark_equity
    bdd = beq / beq.cummax() - 1.0
    ax3.fill_between(dd.index, dd.values, 0, alpha=0.5, label="Strategy drawdown")
    ax3.plot(bdd.index, bdd.values, color="tab:orange", alpha=0.8, label="Benchmark drawdown")
    ax3.set_title(f"{args.ticker} out-of-sample drawdown")
    ax3.legend()
    ax3.grid(alpha=0.3)
    dd_path = os.path.join(REPORTS_DIR, f"{args.ticker}_drawdown.png")
    fig3.savefig(dd_path, dpi=150)
    print(f"Saved {dd_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
