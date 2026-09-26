"""The five-gene strategy representation.

Every candidate strategy in my system is just five integers:

    short_window : short simple-moving-average window (days)
    long_window  : long simple-moving-average window (days)
    rsi_period   : RSI lookback (days)
    rsi_buy      : RSI level below which entries are allowed
    rsi_sell     : RSI level above which the position is exited

which decode into a long/flat trading rule:

    ENTER (go long)  when  SMA_short > SMA_long  AND  RSI < rsi_buy
    EXIT  (go flat)  when  SMA_short < SMA_long  OR   RSI > rsi_sell

I kept the representation this small on purpose: the whole point of the
project is that the evolved rule stays readable by a human, and the
interpretability argument I make in the report depends on that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict


# Inclusive bounds for each gene. The short < long constraint isn't encoded
# here -- repair() enforces it after every crossover/mutation.
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
        """Spell the rule out in plain English (shown in the UI and my report)."""
        return (
            f"Buy when the {self.short_window}-day average price rises above the "
            f"{self.long_window}-day average and the {self.rsi_period}-day RSI is "
            f"below {self.rsi_buy}; sell when the {self.short_window}-day average "
            f"falls below the {self.long_window}-day average or the RSI rises "
            f"above {self.rsi_sell}."
        )

    # ------------------------------------------------------------- validity
    def repair(self) -> "Genome":
        """Clamp genes back into bounds and fix any broken constraints.

        I call this after every crossover and mutation, which lets those
        operators stay simple -- they can produce anything and repair()
        cleans up. The constraints are: every gene inside GENE_BOUNDS,
        short_window strictly below long_window, and rsi_buy < rsi_sell.
        """
        names = list(GENE_BOUNDS)
        vals = {}
        for name in names:
            lo, hi = GENE_BOUNDS[name]
            vals[name] = int(min(max(round(getattr(self, name)), lo), hi))
        # If the windows ended up inverted, push the long window up (or the
        # short one down if we hit the upper bound).
        if vals["short_window"] >= vals["long_window"]:
            vals["long_window"] = min(vals["short_window"] + 10, GENE_BOUNDS["long_window"][1])
            if vals["short_window"] >= vals["long_window"]:
                vals["short_window"] = vals["long_window"] - 10
        # Same idea for the RSI thresholds: buy must sit below sell.
        if vals["rsi_buy"] >= vals["rsi_sell"]:
            vals["rsi_buy"] = min(vals["rsi_buy"], vals["rsi_sell"] - 5)
            lo_buy, _ = GENE_BOUNDS["rsi_buy"]
            vals["rsi_buy"] = max(vals["rsi_buy"], lo_buy)
            if vals["rsi_buy"] >= vals["rsi_sell"]:
                vals["rsi_sell"] = vals["rsi_buy"] + 5
        return Genome(**vals)


def random_genome(rng: random.Random) -> Genome:
    """Sample a uniformly random genome, then repair it so it's valid."""
    g = Genome(**{name: rng.randint(lo, hi) for name, (lo, hi) in GENE_BOUNDS.items()})
    return g.repair()
