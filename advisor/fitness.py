"""Walk-forward, regularised, multi-asset fitness.

This module addresses the overfitting finding of the feature prototype
(report Chapter 4): a single in-sample Sharpe rewards rules that memorise one
period.  Three counter-measures are applied *during* the search:

1. WALK-FORWARD FITNESS.  The training window is split into K contiguous
   folds.  For each fold k, the rule is scored on fold k using only data up
   to the end of that fold (indicators warm up on preceding data).  The
   fitness aggregates the per-fold Sharpe ratios as  mean - lambda * std,
   so a rule must perform *consistently across sub-periods*, not just once.

2. REGULARISATION AGAINST DEGENERATE RULES.  Rules that almost never trade
   (the prototype's "RSI < 20 and RSI > 90" pathology) get a penalty
   proportional to how far their trade count falls below a minimum
   trades-per-year target.

3. MULTI-ASSET EVOLUTION.  Fitness can be computed over several tickers and
   averaged (minus a dispersion penalty), rewarding rules that generalise
   across assets rather than fitting one stock's idiosyncrasies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import DEFAULT_COST, run_backtest, sharpe_ratio, compute_positions
from .genome import Genome


@dataclass
class FitnessConfig:
    n_folds: int = 4                  # walk-forward folds within the training window
    consistency_lambda: float = 0.5   # weight on the std of per-fold Sharpe
    min_trades_per_year: float = 2.0  # below this, apply the sparsity penalty
    sparsity_penalty: float = 1.0     # penalty per missing trade/year (scaled)
    dispersion_lambda: float = 0.25   # multi-asset: weight on cross-asset std
    cost: float = DEFAULT_COST        # transaction cost per side


def _fold_sharpes(prices: pd.Series, genome: Genome, cfg: FitnessConfig) -> list[float]:
    """Sharpe ratio of the rule on each contiguous walk-forward fold.

    Indicators are computed on the full history up to each fold's end, then
    returns are scored only inside the fold -- so later folds see realistic
    warmed-up indicators and no fold sees future data.
    """
    n = len(prices)
    if n < cfg.n_folds * 60:  # need a sane minimum per fold
        return [run_backtest(prices, genome, cfg.cost).metrics["sharpe"]]

    edges = np.linspace(0, n, cfg.n_folds + 1, dtype=int)
    full = run_backtest(prices, genome, cfg.cost)
    sharpes = []
    for k in range(cfg.n_folds):
        fold_ret = full.daily_returns.iloc[edges[k]:edges[k + 1]]
        sharpes.append(sharpe_ratio(fold_ret))
    return sharpes


def fitness_single_asset(prices: pd.Series, genome: Genome, cfg: FitnessConfig) -> float:
    """Walk-forward, sparsity-regularised fitness on one asset."""
    result = run_backtest(prices, genome, cfg.cost)
    sharpes = _fold_sharpes(prices, genome, cfg)

    mean_s = float(np.mean(sharpes))
    std_s = float(np.std(sharpes))
    score = mean_s - cfg.consistency_lambda * std_s

    # Sparsity penalty: rules that barely trade are un-testable and usually
    # overfit to a handful of historical episodes.
    tpy = result.metrics["trades_per_year"]
    if tpy < cfg.min_trades_per_year:
        score -= cfg.sparsity_penalty * (cfg.min_trades_per_year - tpy) / cfg.min_trades_per_year
    return score


def fitness(price_map: dict[str, pd.Series], genome: Genome, cfg: FitnessConfig | None = None) -> float:
    """Aggregate fitness across one or more assets.

    For a single asset this reduces to `fitness_single_asset`.  For several,
    the score is  mean(asset scores) - dispersion_lambda * std(asset scores),
    favouring rules whose quality is uniform across markets.
    """
    cfg = cfg or FitnessConfig()
    scores = [fitness_single_asset(p, genome, cfg) for p in price_map.values()]
    if len(scores) == 1:
        return scores[0]
    return float(np.mean(scores) - cfg.dispersion_lambda * np.std(scores))
