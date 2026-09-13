"""The genetic algorithm.

Population-based search over the five-gene genome space, maximising the
walk-forward fitness.  Operators (as specified in the design chapter):

  * tournament selection (size 3),
  * uniform crossover,
  * per-gene Gaussian mutation with sigma proportional to the gene's range,
  * elitism (the best E genomes survive unchanged),
  * repair after every variation so all genomes stay valid.

The run is fully reproducible from a seed, and the returned history supports
the convergence plots used in the evaluation chapter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

from .fitness import FitnessConfig, fitness
from .genome import GENE_BOUNDS, Genome, random_genome


@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 40
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25       # per-gene probability
    mutation_sigma_frac: float = 0.15 # sigma as a fraction of the gene's range
    elitism: int = 2
    seed: int = 7


@dataclass
class GAResult:
    best_genome: Genome
    best_fitness: float
    history: list[dict] = field(default_factory=list)  # per-generation stats
    final_population: list[tuple[Genome, float]] = field(default_factory=list)


def _tournament(pop: list[tuple[Genome, float]], rng: random.Random, k: int) -> Genome:
    """Pick k random individuals, return (a copy of) the fittest."""
    contenders = rng.sample(pop, k)
    return max(contenders, key=lambda gf: gf[1])[0].copy()


def _crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Uniform crossover: each gene inherited from either parent with p=0.5."""
    genes = {}
    for name in GENE_BOUNDS:
        genes[name] = getattr(a if rng.random() < 0.5 else b, name)
    return Genome(**genes)


def _mutate(g: Genome, rng: random.Random, cfg: GAConfig) -> Genome:
    """Gaussian mutation: each gene perturbed with probability mutation_rate."""
    genes = g.to_dict()
    for name, (lo, hi) in GENE_BOUNDS.items():
        if rng.random() < cfg.mutation_rate:
            sigma = (hi - lo) * cfg.mutation_sigma_frac
            genes[name] = genes[name] + rng.gauss(0.0, sigma)
    return Genome(**{k: int(round(v)) for k, v in genes.items()})


def evolve(
    price_map: dict[str, pd.Series],
    ga_cfg: GAConfig | None = None,
    fit_cfg: FitnessConfig | None = None,
    progress_callback=None,
) -> GAResult:
    """Run the genetic algorithm over the training price data.

    `price_map` maps ticker -> training-window price series (one entry for
    single-asset evolution, several for multi-asset).  Returns the best
    genome, its fitness, and a per-generation history for convergence plots.
    """
    ga_cfg = ga_cfg or GAConfig()
    fit_cfg = fit_cfg or FitnessConfig()
    rng = random.Random(ga_cfg.seed)

    def evaluate(g: Genome) -> float:
        return fitness(price_map, g, fit_cfg)

    # Memoise: genome space is small and elites recur, so caching saves work.
    cache: dict[tuple, float] = {}

    def cached_eval(g: Genome) -> float:
        key = tuple(g.genes())
        if key not in cache:
            cache[key] = evaluate(g)
        return cache[key]

    population = [random_genome(rng) for _ in range(ga_cfg.population_size)]
    scored = [(g, cached_eval(g)) for g in population]
    history: list[dict] = []

    for gen in range(ga_cfg.generations):
        scored.sort(key=lambda gf: gf[1], reverse=True)
        best_g, best_f = scored[0]
        mean_f = sum(f for _, f in scored) / len(scored)
        history.append({
            "generation": gen,
            "best_fitness": best_f,
            "mean_fitness": mean_f,
            "best_genome": best_g.to_dict(),
        })
        if progress_callback:
            progress_callback(gen, best_f, mean_f)

        # Elites pass through unchanged.
        next_pop: list[Genome] = [g.copy() for g, _ in scored[: ga_cfg.elitism]]

        while len(next_pop) < ga_cfg.population_size:
            p1 = _tournament(scored, rng, ga_cfg.tournament_size)
            p2 = _tournament(scored, rng, ga_cfg.tournament_size)
            child = _crossover(p1, p2, rng) if rng.random() < ga_cfg.crossover_rate else p1
            child = _mutate(child, rng, ga_cfg).repair()
            next_pop.append(child)

        scored = [(g, cached_eval(g)) for g in next_pop]

    scored.sort(key=lambda gf: gf[1], reverse=True)
    best_g, best_f = scored[0]
    history.append({
        "generation": ga_cfg.generations,
        "best_fitness": best_f,
        "mean_fitness": sum(f for _, f in scored) / len(scored),
        "best_genome": best_g.to_dict(),
    })
    return GAResult(best_genome=best_g, best_fitness=best_f, history=history, final_population=scored)
