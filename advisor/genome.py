"""The five-gene strategy representation.

A candidate strategy is a genome of five integers:

    short_window : short simple-moving-average window (days)
    long_window  : long simple-moving-average window (days)
    rsi_period   : RSI lookback (days)
    rsi_buy      : RSI level below which entries are allowed
    rsi_sell     : RSI level above which the position is exited

decoded into a long/flat trading rule:

    ENTER (go long)  when  SMA_short > SMA_long  AND  RSI < rsi_buy
    EXIT  (go flat)  when  SMA_short < SMA_long  OR   RSI > rsi_sell

The representation is deliberately tiny so the evolved rule stays readable --
the interpretability claim in the report is bound to this compactness.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict


# Inclusive bounds for each gene.  short < long is enforced by repair().
GENE_BOUNDS = {
    "short_window": (5, 60),
    "long_window": (20, 250),
    "rsi_period": (5, 30),
    "rsi_buy": (10, 60),
    "rsi_sell": (55, 95),
}


@dataclass
class Genome:
    short_window: int
    long_window: int
    rsi_period: int
    rsi_buy: int
    rsi_sell: int

    # ------------------------------------------------------------------ API
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(**{k: int(d[k]) for k in GENE_BOUNDS})

    def genes(self) -> list[int]:
        return [self.short_window, self.long_window, self.rsi_period, self.rsi_buy, self.rsi_sell]

    def copy(self) -> "Genome":
        return Genome(*self.genes())

    def describe(self) -> str:
        """Human-readable statement of the rule -- used in reports and the UI."""
        return (
            f"Buy when the {self.short_window}-day average price rises above the "
            f"{self.long_window}-day average and the {self.rsi_period}-day RSI is "
            f"below {self.rsi_buy}; sell when the {self.short_window}-day average "
            f"falls below the {self.long_window}-day average or the RSI rises "
            f"above {self.rsi_sell}."
        )

    # ------------------------------------------------------------- validity
    def repair(self) -> "Genome":
        """Clamp genes into bounds and enforce structural constraints.

        Called after every crossover/mutation so variation operators can be
        simple and unconstrained.  Constraints: each gene within GENE_BOUNDS;
        short_window strictly less than long_window; rsi_buy < rsi_sell.
        """
        names = list(GENE_BOUNDS)
        vals = {}
        for name in names:
            lo, hi = GENE_BOUNDS[name]
            vals[name] = int(min(max(round(getattr(self, name)), lo), hi))
        # short strictly below long
        if vals["short_window"] >= vals["long_window"]:
            vals["long_window"] = min(vals["short_window"] + 10, GENE_BOUNDS["long_window"][1])
            if vals["short_window"] >= vals["long_window"]:
                vals["short_window"] = vals["long_window"] - 10
        # buy threshold strictly below sell threshold
        if vals["rsi_buy"] >= vals["rsi_sell"]:
            vals["rsi_buy"] = min(vals["rsi_buy"], vals["rsi_sell"] - 5)
            lo_buy, _ = GENE_BOUNDS["rsi_buy"]
            vals["rsi_buy"] = max(vals["rsi_buy"], lo_buy)
            if vals["rsi_buy"] >= vals["rsi_sell"]:
                vals["rsi_sell"] = vals["rsi_buy"] + 5
        return Genome(**vals)


def random_genome(rng: random.Random) -> Genome:
    """Sample a uniformly random (then repaired) genome."""
    g = Genome(**{name: rng.randint(lo, hi) for name, (lo, hi) in GENE_BOUNDS.items()})
    return g.repair()
