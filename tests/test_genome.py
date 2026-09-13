import random

from advisor.genome import GENE_BOUNDS, Genome, random_genome


def test_random_genomes_always_valid():
    rng = random.Random(1)
    for _ in range(500):
        g = random_genome(rng)
        for name, (lo, hi) in GENE_BOUNDS.items():
            assert lo <= getattr(g, name) <= hi
        assert g.short_window < g.long_window
        assert g.rsi_buy < g.rsi_sell


def test_repair_fixes_inverted_windows():
    g = Genome(short_window=60, long_window=20, rsi_period=14, rsi_buy=30, rsi_sell=70).repair()
    assert g.short_window < g.long_window


def test_repair_fixes_inverted_rsi_thresholds():
    g = Genome(short_window=10, long_window=50, rsi_period=14, rsi_buy=80, rsi_sell=60).repair()
    assert g.rsi_buy < g.rsi_sell


def test_repair_clamps_out_of_bounds():
    g = Genome(short_window=-5, long_window=999, rsi_period=1000, rsi_buy=-3, rsi_sell=999).repair()
    for name, (lo, hi) in GENE_BOUNDS.items():
        assert lo <= getattr(g, name) <= hi


def test_roundtrip_dict():
    g = Genome(10, 50, 14, 30, 70)
    assert Genome.from_dict(g.to_dict()) == g


def test_describe_mentions_all_genes():
    g = Genome(10, 50, 14, 30, 70)
    text = g.describe()
    for v in ("10", "50", "14", "30", "70"):
        assert v in text
