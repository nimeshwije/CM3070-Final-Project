from advisor.data import synthetic_gbm
from advisor.evolution import GAConfig, evolve
from advisor.fitness import FitnessConfig, fitness
from advisor.genome import GENE_BOUNDS, Genome


def small_run(seed=7):
    prices = synthetic_gbm(700, seed=11)
    ga = GAConfig(population_size=14, generations=6, seed=seed)
    fit = FitnessConfig(n_folds=3)
    return evolve({"SYN": prices}, ga, fit)


def test_ga_returns_valid_genome_and_history():
    res = small_run()
    g = res.best_genome
    for name, (lo, hi) in GENE_BOUNDS.items():
        assert lo <= getattr(g, name) <= hi
    assert g.short_window < g.long_window
    assert len(res.history) == 7  # 6 generations plus the final snapshot


def test_ga_best_fitness_never_decreases_with_elitism():
    res = small_run()
    best = [h["best_fitness"] for h in res.history]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:]))


def test_ga_reproducible_from_seed():
    r1, r2 = small_run(seed=3), small_run(seed=3)
    assert r1.best_genome == r2.best_genome
    assert r1.best_fitness == r2.best_fitness


def test_sparsity_penalty_hits_degenerate_rule():
    """A rule that never trades has to score worse than one that actually trades.

    This is the regression test for the degenerate-rule pathology I hit in
    the prototype (rules that never enter the market and so never lose).
    """
    prices = synthetic_gbm(700, seed=12)
    cfg = FitnessConfig(n_folds=3)
    never_trades = Genome(10, 40, 14, 10, 95)   # entry needs RSI < 10, which basically never happens
    plausible = Genome(10, 40, 14, 60, 95)
    assert fitness({"S": prices}, never_trades, cfg) < fitness({"S": prices}, plausible, cfg)


def test_multi_asset_fitness_runs():
    p1, p2 = synthetic_gbm(600, seed=1), synthetic_gbm(600, seed=2)
    g = Genome(10, 40, 14, 55, 90)
    score = fitness({"A": p1, "B": p2}, g, FitnessConfig(n_folds=3))
    assert isinstance(score, float)
