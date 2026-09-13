"""Out-of-sample evaluation against a buy-and-hold benchmark.

Implements the evaluation strategy from the report's design chapter: a
chronological train/test split (the GA never sees the test window), metrics
reported strategy-vs-benchmark, and an optional rolling walk-forward
evaluation across the full history for the final report.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import DEFAULT_COST, TRADING_DAYS, BacktestResult, run_backtest
from .genome import Genome


@dataclass
class SplitEvaluation:
    train_result: BacktestResult
    test_result: BacktestResult
    train_range: tuple[str, str]
    test_range: tuple[str, str]

    def summary_table(self) -> pd.DataFrame:
        """The report's headline table: in-sample vs out-of-sample vs benchmark."""
        tr, te = self.train_result.metrics, self.test_result.metrics
        rows = {
            "Total return": [tr["total_return"], te["total_return"], te["benchmark_total_return"]],
            "Sharpe ratio": [tr["sharpe"], te["sharpe"], te["benchmark_sharpe"]],
            "Max drawdown": [tr["max_drawdown"], te["max_drawdown"], te["benchmark_max_drawdown"]],
            "Annual volatility": [
                tr["annual_volatility"], te["annual_volatility"], te["benchmark_annual_volatility"],
            ],
            "Number of trades": [tr["n_trades"], te["n_trades"], None],
            "Time in market": [tr["time_in_market"], te["time_in_market"], 1.0],
        }
        df = pd.DataFrame(rows).T
        df.columns = ["In-sample", "Out-of-sample", "Buy & hold (OOS)"]
        return df


def chronological_split(prices: pd.Series, train_frac: float = 0.7) -> tuple[pd.Series, pd.Series]:
    """Split a price series chronologically into train (older) and test (recent)."""
    if not 0.1 <= train_frac <= 0.95:
        raise ValueError("train_frac should be in [0.1, 0.95]")
    cut = int(len(prices) * train_frac)
    return prices.iloc[:cut], prices.iloc[cut:]


def evaluate_split(
    prices: pd.Series,
    genome: Genome,
    train_frac: float = 0.7,
    cost: float | None = None,
) -> SplitEvaluation:
    """Backtest `genome` separately on the train and held-out test windows."""
    cost = DEFAULT_COST if cost is None else cost
    train, test = chronological_split(prices, train_frac)
    train_res = run_backtest(train, genome, cost)
    test_res = run_backtest(test, genome, cost)
    return SplitEvaluation(
        train_result=train_res,
        test_result=test_res,
        train_range=(str(train.index[0].date()), str(train.index[-1].date())),
        test_range=(str(test.index[0].date()), str(test.index[-1].date())),
    )


def walk_forward_report(
    prices: pd.Series,
    genome: Genome,
    window_years: float = 2.0,
    cost: float | None = None,
) -> pd.DataFrame:
    """Rolling evaluation: score the rule on consecutive windows of history.

    Used in the final report to show how performance varies by regime
    (cf. Potvin et al. 2004 on regime-dependence).
    """
    cost = DEFAULT_COST if cost is None else cost
    w = int(window_years * TRADING_DAYS)
    rows = []
    for start in range(0, len(prices) - w + 1, w):
        chunk = prices.iloc[start:start + w]
        res = run_backtest(chunk, genome, cost)
        m = res.metrics
        rows.append({
            "window_start": str(chunk.index[0].date()),
            "window_end": str(chunk.index[-1].date()),
            "strategy_return": m["total_return"],
            "benchmark_return": m["benchmark_total_return"],
            "strategy_sharpe": m["sharpe"],
            "benchmark_sharpe": m["benchmark_sharpe"],
            "max_drawdown": m["max_drawdown"],
            "n_trades": m["n_trades"],
        })
    return pd.DataFrame(rows)
