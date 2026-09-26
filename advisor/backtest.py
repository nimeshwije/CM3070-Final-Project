"""The backtester: simulates a genome's rule over a price history, with costs.

Two honesty measures are built directly into this module rather than bolted
on later, because getting either wrong silently inflates every result:

  * signals are acted on ONE DAY LATER -- the return on day t is earned by
    the position decided on day t-1, so the rule can never trade on
    information it hasn't seen yet (no look-ahead bias);
  * a proportional transaction cost is charged every time the position
    changes, so a rule that trades constantly doesn't get flattered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .genome import Genome
from .indicators import rsi, sma

TRADING_DAYS = 252
DEFAULT_COST = 0.001  # 10 basis points per side -- a fairly standard retail assumption


def compute_positions(prices: pd.Series, genome: Genome) -> pd.Series:
    """Decode a genome into a daily position series (1 = long, 0 = flat).

    The rule is stateful, which is why this is a loop rather than a vectorised
    expression: whether we are in the market today depends on whether we were
    in it yesterday. Entry needs SMA_short > SMA_long AND RSI < buy threshold;
    exit happens when SMA_short < SMA_long OR RSI > sell threshold. Days where
    the indicators haven't warmed up yet stay flat.
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
    """Everything I need to evaluate (and later plot) a single backtest."""

    equity: pd.Series                 # strategy equity curve, starts at 1.0
    benchmark_equity: pd.Series       # buy-and-hold equity curve, starts at 1.0
    daily_returns: pd.Series          # strategy net daily returns
    positions: pd.Series              # 0/1 position actually held each day
    n_trades: int                     # number of position changes
    metrics: dict = field(default_factory=dict)


def sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio of a daily return series (0.0 for degenerate input)."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free_annual / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown, returned as a negative fraction."""
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
    """Run `genome`'s rule over `prices` and return equity, positions and metrics."""
    prices = prices.dropna()
    asset_returns = prices.pct_change().fillna(0.0)

    signal_pos = compute_positions(prices, genome)
    # The one-day lag: today's return goes to yesterday's decision. This single
    # shift(1) is what prevents look-ahead bias, and test_one_day_lag_no_lookahead
    # exists specifically to make sure nobody (including me) removes it.
    held_pos = signal_pos.shift(1).fillna(0.0)

    # Charge the cost on the day the position changes -- entries and exits both.
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
