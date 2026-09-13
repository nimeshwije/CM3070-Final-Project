"""The end-to-end training pipeline.

    data ingestion -> chronological split -> evolutionary search
                   -> out-of-sample evaluation -> saved rule artifact

This is the single implementation used by BOTH entry points -- the command
line (`scripts/train.py`) and the web admin area -- so the report can state
that a rule trained from the browser is produced by exactly the same code,
with the same honesty measures, as one trained from the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

from .data import PriceSeries, fetch_prices, synthetic_gbm
from .evaluation import chronological_split, evaluate_split
from .evolution import GAConfig, evolve
from .fitness import FitnessConfig
from .persistence import is_valid_ticker, save_artifact

ProgressFn = Callable[[str], None]


@dataclass
class TrainRequest:
    """Everything needed to run one training job (mirrors the CLI flags)."""

    tickers: list[str]
    start: str = "2015-01-01"
    end: str | None = None
    train_frac: float = 0.7
    population: int = 60
    generations: int = 40
    seed: int = 7
    cost: float = 0.001
    folds: int = 4
    synthetic: bool = False          # force GBM data (offline demo / tests)

    def validate(self) -> None:
        if not self.tickers:
            raise ValueError("At least one ticker is required")
        for t in self.tickers:
            if not is_valid_ticker(t):
                raise ValueError(f"Invalid ticker symbol: {t!r}")
        if not 0.1 <= self.train_frac <= 0.95:
            raise ValueError("Training fraction must be between 0.1 and 0.95")
        if not 4 <= self.population <= 500:
            raise ValueError("Population must be between 4 and 500")
        if not 1 <= self.generations <= 500:
            raise ValueError("Generations must be between 1 and 500")
        if not 0.0 <= self.cost <= 0.05:
            raise ValueError("Transaction cost must be between 0 and 0.05")
        if not 1 <= self.folds <= 12:
            raise ValueError("Folds must be between 1 and 12")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainOutcome:
    """Result of a training run, JSON-serialisable for the web UI."""

    genome: dict
    rule_text: str
    best_fitness: float
    per_ticker: list[dict] = field(default_factory=list)   # summary rows per ticker
    history: list[dict] = field(default_factory=list)      # GA convergence
    artifacts: list[str] = field(default_factory=list)     # saved file paths

    def to_dict(self) -> dict:
        return asdict(self)


def train_rule(req: TrainRequest, progress: ProgressFn | None = None) -> TrainOutcome:
    """Run the full pipeline for `req`, reporting progress lines via `progress`."""
    req.validate()
    log = progress or (lambda msg: None)

    # ------------------------------------------------------------ data
    series: dict[str, PriceSeries] = {}
    for t in req.tickers:
        if req.synthetic:
            series[t] = PriceSeries(t, synthetic_gbm(seed=abs(hash(t)) % 10_000), "synthetic")
        else:
            series[t] = fetch_prices(t, start=req.start, end=req.end)
        log(f"{t}: {len(series[t])} days of data ({series[t].source})")

    train_map = {}
    for t, ps in series.items():
        tr, _ = chronological_split(ps.prices, req.train_frac)
        train_map[t] = tr

    # ------------------------------------------------------------ evolve
    ga_cfg = GAConfig(population_size=req.population, generations=req.generations, seed=req.seed)
    fit_cfg = FitnessConfig(n_folds=req.folds, cost=req.cost)
    log(f"Evolving on {list(train_map)} (population {ga_cfg.population_size}, "
        f"{ga_cfg.generations} generations, {fit_cfg.n_folds} walk-forward folds)")

    def on_gen(gen, best, mean):
        log(f"gen {gen:3d}  best fitness {best:+.3f}  mean {mean:+.3f}")

    result = evolve(train_map, ga_cfg, fit_cfg, progress_callback=on_gen)
    genome = result.best_genome
    log(f"Best genome: {genome.to_dict()}")
    log(f"Rule: {genome.describe()}")

    # ------------------------------------------------------------ evaluate & save
    outcome = TrainOutcome(
        genome=genome.to_dict(),
        rule_text=genome.describe(),
        best_fitness=result.best_fitness,
        history=result.history,
    )
    for t, ps in series.items():
        ev = evaluate_split(ps.prices, genome, req.train_frac, cost=req.cost)
        tr, te = ev.train_result.metrics, ev.test_result.metrics
        outcome.per_ticker.append({
            "ticker": t,
            "data_source": ps.source,
            "train_range": ev.train_range,
            "test_range": ev.test_range,
            "train": tr,
            "test": te,
        })
        log(f"{t}: OOS Sharpe {te['sharpe']:+.2f} vs buy&hold {te['benchmark_sharpe']:+.2f}; "
            f"OOS return {te['total_return']:+.1%} vs {te['benchmark_total_return']:+.1%}; "
            f"max drawdown {te['max_drawdown']:.1%}; trades {te['n_trades']}")
        path = save_artifact(
            t, genome,
            train_metrics=tr,
            test_metrics=te,
            meta={
                "data_source": ps.source,
                "train_range": ev.train_range,
                "test_range": ev.test_range,
                "ga": vars(ga_cfg),
                "fitness": vars(fit_cfg),
                "history": result.history,
                "multi_asset_partners": [x for x in series if x != t],
                "request": req.to_dict(),
            },
        )
        outcome.artifacts.append(path)
        log(f"Saved rule artifact -> {path}")
    return outcome
