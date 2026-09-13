"""The deterministic rule engine.

Given an evolved genome and the latest prices, this module produces the
advisor's decision -- BUY, SELL, or HOLD -- together with machine-readable
reasons.  This output is fully auditable and reproducible: the same rule and
the same prices always yield the same decision.  The language model in
`explain.py` NEVER makes or alters this decision; it only rewords it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd

from .backtest import compute_positions
from .genome import Genome
from .indicators import rsi, sma


@dataclass
class Decision:
    """A structured, machine-readable recommendation."""

    ticker: str
    action: str                  # "BUY" | "SELL" | "HOLD"
    stance: str                  # "in_market" | "in_cash" (rule's target state)
    as_of: str                   # date of the latest observation used
    price: float
    reasons: list[str] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)
    rule: dict = field(default_factory=dict)
    rule_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def decide(ticker: str, prices: pd.Series, genome: Genome) -> Decision:
    """Apply the evolved rule to the latest data and emit a Decision.

    The rule's position series is computed over the recent history; the
    decision is the transition implied for the next trading day:
      flat -> long  : BUY
      long -> flat  : SELL
      unchanged     : HOLD (with the current stance reported)
    """
    prices = prices.dropna()
    if len(prices) < genome.long_window + 5:
        raise ValueError(
            f"Need at least {genome.long_window + 5} days of prices, got {len(prices)}"
        )

    pos = compute_positions(prices, genome)
    today, yesterday = pos.iloc[-1], pos.iloc[-2]

    s_short = sma(prices, genome.short_window).iloc[-1]
    s_long = sma(prices, genome.long_window).iloc[-1]
    r = rsi(prices, genome.rsi_period).iloc[-1]

    trend_up = s_short > s_long
    reasons: list[str] = []
    reasons.append(
        f"{genome.short_window}-day average ({s_short:.2f}) is "
        f"{'above' if trend_up else 'below'} the {genome.long_window}-day average ({s_long:.2f})"
        f" -- the medium-term trend is {'up' if trend_up else 'down'}"
    )
    if r < genome.rsi_buy:
        rsi_state = f"below the buy threshold of {genome.rsi_buy}"
    elif r > genome.rsi_sell:
        rsi_state = f"above the sell threshold of {genome.rsi_sell}"
    else:
        rsi_state = (
            f"between the buy threshold ({genome.rsi_buy}) and the sell threshold ({genome.rsi_sell})"
        )
    reasons.append(f"{genome.rsi_period}-day RSI is {r:.0f}, {rsi_state}")

    if today > yesterday:
        action = "BUY"
        reasons.append("the rule's entry conditions have just been met: enter the market")
    elif today < yesterday:
        action = "SELL"
        reasons.append("the rule's exit conditions have just been met: move to cash")
    else:
        action = "HOLD"
        if today == 1.0:
            reasons.append("the rule remains invested: stay in the market")
        else:
            reasons.append("the rule's entry conditions are not met: stay in cash")

    return Decision(
        ticker=ticker,
        action=action,
        stance="in_market" if today == 1.0 else "in_cash",
        as_of=str(prices.index[-1].date()),
        price=float(prices.iloc[-1]),
        reasons=reasons,
        indicators={
            "sma_short": round(float(s_short), 4),
            "sma_long": round(float(s_long), 4),
            "rsi": round(float(r), 2),
        },
        rule=genome.to_dict(),
        rule_text=genome.describe(),
    )
