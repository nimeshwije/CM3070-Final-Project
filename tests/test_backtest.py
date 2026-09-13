import numpy as np
import pandas as pd
import pytest

from advisor.backtest import (
    compute_positions, max_drawdown, run_backtest, sharpe_ratio,
)
from advisor.data import synthetic_gbm
from advisor.genome import Genome

GENOME = Genome(short_window=10, long_window=40, rsi_period=14, rsi_buy=60, rsi_sell=95)


def test_positions_are_binary_and_flat_during_warmup():
    prices = synthetic_gbm(400, seed=3)
    pos = compute_positions(prices, GENOME)
    assert set(np.unique(pos)) <= {0.0, 1.0}
    # No position before the long window has data.
    assert (pos.iloc[: GENOME.long_window - 1] == 0).all()


def test_one_day_lag_no_lookahead():
    """The held position must equal the previous day's signal."""
    prices = synthetic_gbm(400, seed=4)
    res = run_backtest(prices, GENOME, cost=0.0)
    signal = compute_positions(prices, GENOME)
    assert (res.positions.iloc[1:].to_numpy() == signal.shift(1).iloc[1:].to_numpy()).all()
    assert res.positions.iloc[0] == 0.0


def test_costs_reduce_returns():
    prices = synthetic_gbm(800, seed=5)
    free = run_backtest(prices, GENOME, cost=0.0)
    costly = run_backtest(prices, GENOME, cost=0.005)
    if free.n_trades > 0:
        assert costly.metrics["total_return"] < free.metrics["total_return"]


def test_always_flat_rule_has_zero_return():
    g = Genome(short_window=10, long_window=40, rsi_period=14, rsi_buy=10, rsi_sell=95)
    prices = pd.Series(
        np.linspace(100, 200, 300),
        index=pd.bdate_range("2020-01-01", periods=300),
    )  # smooth rise -> RSI stays ~100, never below 10
    res = run_backtest(prices, g)
    assert res.n_trades == 0
    assert res.metrics["total_return"] == pytest.approx(0.0)


def test_sharpe_of_constant_returns_is_zero():
    r = pd.Series([0.0] * 100)
    assert sharpe_ratio(r) == 0.0


def test_max_drawdown_known_case():
    eq = pd.Series([1.0, 2.0, 1.0, 3.0])
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_benchmark_equity_matches_prices():
    prices = synthetic_gbm(300, seed=6)
    res = run_backtest(prices, GENOME)
    expected = prices.iloc[-1] / prices.iloc[0] - 1.0
    assert res.metrics["benchmark_total_return"] == pytest.approx(expected, rel=1e-9)
