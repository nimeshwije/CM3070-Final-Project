"""Transaction-cost-aware backtesting and performance metrics.

The backtester simulates the long/flat rule decoded from a genome over a
price history.  Two honesty measures from the report are built in:

  * signals are acted on ONE DAY LATER (no look-ahead bias) -- the return on
    day t is earned by the position decided on day t-1;
  * a proportional transaction cost is charged every time the position
    changes, so strategies that trade often are not flattered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .genome import Genome
from .indicators import rsi, sma

TRADING_DAYS = 252
DEFAULT_COST = 0.001  # 10 basis points per side -- a common retail assumption


def compute_positions(prices: pd.Series, genome: Genome) -> pd.Series:
    """Decode a genome into a daily position series (1 = long, 0 = flat).

    The rule is stateful: entry requires SMA_short > SMA_long AND RSI < buy
    threshold; exit occurs when SMA_short < SMA_long OR RSI > sell threshold.
    Days with insufficient indicator history stay flat.
    """
    s = sma(prices, genome.short_window)
    l = sma(prices, genome.long_window)
    r = rsi(prices, genome.rsi_period)

    above = (s > l).to_numpy()
    below = (s < l).to_numpy()
    r_np = r.to_numpy()
    valid = (~np.isnan(s.to_numpy())) & (~np.isnan(l.to_numpy())) & (~np.isnan(r_np))

    pos = np.zeros(len(prices), dtype=float)
    holding = False
    for i in range(len(prices)):
        if not valid[i]:
            pos[i] = 0.0
            continue
        if holding:
            if below[i] or r_np[i] > genome.rsi_sell:
                holding = False
        else:
            if above[i] and r_np[i] < genome.rsi_buy:
                holding = True
        pos[i] = 1.0 if holding else 0.0
    return pd.Series(pos, index=prices.index, name="position")


@dataclass
class BacktestResult:
    """Everything needed to evaluate (and plot) one backtest."""

    equity: pd.Series                 # strategy equity curve, starts at 1.0
    benchmark_equity: pd.Series       # buy-and-hold equity curve, starts at 1.0
    daily_returns: pd.Series          # strategy net daily returns
    positions: pd.Series              # 0/1 position actually held each day
    n_trades: int                     # number of position changes
    metrics: dict = field(default_factory=dict)


def sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio of a daily return series."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_annual / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, returned as a negative fraction."""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def run_backtest(
    prices: pd.Series,
    genome: Genome,
    cost: float = DEFAULT_COST,
) -> BacktestResult:
    """Simulate `genome`'s rule over `prices`, net of transaction costs."""
    prices = prices.dropna()
    asset_returns = prices.pct_change().fillna(0.0)

    signal_pos = compute_positions(prices, genome)
    # Act one day after the signal: today's return is earned by yesterday's decision.
    held_pos = signal_pos.shift(1).fillna(0.0)

    # Cost is charged on the day the position changes (both entries and exits).
    turnover = held_pos.diff().abs().fillna(held_pos.abs())
    strat_returns = held_pos * asset_returns - cost * turnover

    equity = (1.0 + strat_returns).cumprod()
    bench = (1.0 + asset_returns).cumprod()
    n_trades = int(turnover.sum())

    years = max(len(prices) / TRADING_DAYS, 1e-9)
    metrics = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "benchmark_total_return": float(bench.iloc[-1] - 1.0),
        "sharpe": sharpe_ratio(strat_returns),
        "benchmark_sharpe": sharpe_ratio(asset_returns),
        "max_drawdown": max_drawdown(equity),
        "benchmark_max_drawdown": max_drawdown(bench),
        "annual_volatility": float(strat_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "benchmark_annual_volatility": float(asset_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "n_trades": n_trades,
        "trades_per_year": n_trades / years,
        "time_in_market": float(held_pos.mean()),
    }
    return BacktestResult(
        equity=equity,
        benchmark_equity=bench,
        daily_returns=strat_returns,
        positions=held_pos,
        n_trades=n_trades,
        metrics=metrics,
    )
