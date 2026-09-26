"""The fitness function: walk-forward, regularised, and optionally multi-asset.

This module is my answer to the biggest problem I found while building the
feature prototype (Chapter 4 of the report): if fitness is just one
in-sample Sharpe ratio, the GA happily evolves rules that memorise one lucky
period. I ended up applying three counter-measures *during* the search
itself, not just at evaluation time:

1. WALK-FORWARD FITNESS. The training window is split into K contiguous
   folds, and the rule is scored on each fold separately (indicators warm up
   on the data before the fold, so no fold ever sees the future). The
   fitness is then  mean - lambda * std  of the per-fold Sharpes, which
   means a rule has to work consistently across sub-periods, not just once.

2. A PENALTY FOR DEGENERATE RULES. My prototype kept evolving rules like
   "RSI < 20 and RSI > 90" that essentially never trade and therefore never
   lose. Rules whose trade count falls below a minimum trades-per-year
   target now get penalised in proportion to the shortfall.

3. MULTI-ASSET EVOLUTION. Fitness can be averaged over several tickers
   (minus a dispersion penalty), so the GA is rewarded for rules that
   generalise across assets instead of fitting one stock's quirks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import DEFAULT_COST, run_backtest, sharpe_ratio
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

    The trick here: I run one backtest over the whole training window (so
    indicators are properly warmed up everywhere), then score the returns
    fold by fold. Later folds therefore see realistic indicator state, and
    no fold can ever see data from after its own end.
    """
    n = len(prices)
    if n < cfg.n_folds * 60:  # too little data to fold sensibly -> one score
        return [run_backtest(prices, genome, cfg.cost).metrics["sharpe"]]

    edges = np.linspace(0, n, cfg.n_folds + 1, dtype=int)
    full = run_backtest(prices, genome, cfg.cost)
    sharpes = []
    for k in range(cfg.n_folds):
        fold_ret = full.daily_returns.iloc[edges[k]:edges[k + 1]]
        sharpes.append(sharpe_ratio(fold_ret))
    return sharpes


def fitness_single_asset(prices: pd.Series, genome: Genome, cfg: FitnessConfig) -> float:
    """The walk-forward, sparsity-penalised fitness score for one asset."""
    result = run_backtest(prices, genome, cfg.cost)
    sharpes = _fold_sharpes(prices, genome, cfg)

    mean_s = float(np.mean(sharpes))
    std_s = float(np.std(sharpes))
    score = mean_s - cfg.consistency_lambda * std_s

    # Sparsity penalty. A rule that barely trades can't really be tested and
    # in my experience was always overfit to a couple of historical episodes,
    # so trading too rarely costs fitness.
    tpy = result.metrics["trades_per_year"]
    if tpy < cfg.min_trades_per_year:
        score -= cfg.sparsity_penalty * (cfg.min_trades_per_year - tpy) / cfg.min_trades_per_year
    return score


def fitness(price_map: dict[str, pd.Series], genome: Genome, cfg: FitnessConfig | None = None) -> float:
    """Aggregate fitness across one or more assets.

    With one asset this is just `fitness_single_asset`. With several, the
    score is  mean(asset scores) - dispersion_lambda * std(asset scores),
    which prefers rules that are decent everywhere over rules that are
    brilliant on one ticker and terrible on the rest.
    """
    cfg = cfg or FitnessConfig()
    scores = [fitness_single_asset(p, genome, cfg) for p in price_map.values()]
    if len(scores) == 1:
        return scores[0]
    return float(np.mean(scores) - cfg.dispersion_lambda * np.std(scores))
