"""The deterministic rule engine.

Given an evolved genome and the latest prices, this module produces the
actual decision -- BUY, SELL or HOLD -- plus machine-readable reasons. The
output is fully auditable: the same rule, prices and user position always
give the same decision. The language model in `explain.py` NEVER makes or
changes this decision; its only job is to reword it.

Two ways of reading the rule (this took me a while to get right)
----------------------------------------------------------------
The evolved rule is really a *state machine*: on any given day it is either
"in the market" (long) or "in cash" (flat). I call that state the rule's
**stance**, and there are two sensible ways to turn it into advice:

* **Signal mode** (`holds_position=None`) reports the *transition* the rule
  makes today, exactly as the backtester trades it: BUY on the day the rule
  enters, SELL on the day it exits, HOLD otherwise. This is the right view
  for auditing the rule against the backtest -- but transitions only happen
  on a few percent of days, so a user checking in would see HOLD almost
  every single time, which I found genuinely confusing in early testing.

* **Position mode** (`holds_position=True/False`) answers the question a
  real user actually asks -- "given what *I* hold, what should I do?" -- by
  comparing the rule's stance with the user's own position:

      rule stance   user holds   ->  action
      in_market     yes          ->  HOLD   (keep the position)
      in_market     no           ->  BUY    (the rule would be invested)
      in_cash       yes          ->  SELL   (the rule would be in cash)
      in_cash       no           ->  HOLD   (stay in cash, wait)

  In both modes `signal_changed` flags the days the rule itself flipped, so
  the "fresh signal" information from signal mode is never lost.

One ethics point worth stating in the code: the user's position is a request
parameter and nothing more. It is never stored anywhere, which keeps the
system inside the project's ethics scope (no personal or financial data
retained).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd

from .backtest import compute_positions
from .genome import Genome
from .indicators import rsi, sma

IN_MARKET = "in_market"
IN_CASH = "in_cash"


@dataclass
class Decision:
    """The recommendation as a structured object, so everything is inspectable."""

    ticker: str
    action: str                  # "BUY" | "SELL" | "HOLD"
    stance: str                  # "in_market" | "in_cash" (rule's target state)
    as_of: str                   # date of the latest observation used
    price: float
    reasons: list[str] = field(default_factory=list)
    indicators: dict = field(default_factory=dict)
    rule: dict = field(default_factory=dict)
    rule_text: str = ""
    mode: str = "signal"         # "signal" | "position"  (see module docstring)
    holds_position: bool | None = None   # the user's stated position (position mode)
    signal_changed: bool = False # True on the day the rule enters or exits
    signal: str = "none"         # "enter" | "exit" | "none" -- today's rule transition

    def to_dict(self) -> dict:
        return asdict(self)


def position_action(stance: str, holds_position: bool) -> str:
    """The position-aware decision table. Pure function on purpose -- trivially testable."""
    if stance == IN_MARKET:
        return "HOLD" if holds_position else "BUY"
    return "SELL" if holds_position else "HOLD"


def decide(
    ticker: str,
    prices: pd.Series,
    genome: Genome,
    holds_position: bool | None = None,
) -> Decision:
    """Apply the evolved rule to the latest data and build a Decision.

    Passing `holds_position=None` gives signal mode (transition semantics);
    passing a bool gives position mode. The module docstring explains the
    difference and why both exist.
    """
    prices = prices.dropna()
    if len(prices) < genome.long_window + 5:
        raise ValueError(
            f"Need at least {genome.long_window + 5} days of prices, got {len(prices)}"
        )

    pos = compute_positions(prices, genome)
    today, yesterday = pos.iloc[-1], pos.iloc[-2]
    stance = IN_MARKET if today == 1.0 else IN_CASH
    if today > yesterday:
        signal = "enter"
    elif today < yesterday:
        signal = "exit"
    else:
        signal = "none"

    s_short = sma(prices, genome.short_window).iloc[-1]
    s_long = sma(prices, genome.long_window).iloc[-1]
    r = rsi(prices, genome.rsi_period).iloc[-1]

    # --- reasons from the indicators (these are the same in both modes) ----
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

    # --- then a reason describing what the rule itself is doing today ------
    if signal == "enter":
        reasons.append("the rule's entry conditions have just been met: it moves into the market today")
    elif signal == "exit":
        reasons.append("the rule's exit conditions have just been met: it moves to cash today")
    elif stance == IN_MARKET:
        reasons.append("the rule remains invested: its conditions still favour being in the market")
    else:
        reasons.append("the rule's entry conditions are not met: it stays in cash")

    # --- finally, turn the rule's stance into an action for this user -------
    if holds_position is None:
        mode = "signal"
        action = {"enter": "BUY", "exit": "SELL", "none": "HOLD"}[signal]
    else:
        mode = "position"
        action = position_action(stance, holds_position)
        you = "you currently hold this asset" if holds_position else "you do not currently hold this asset"
        rule_is = "in the market" if stance == IN_MARKET else "in cash"
        consequence = {
            "HOLD": ("keep your existing position" if holds_position
                     else "nothing to do -- stay in cash and wait for an entry signal"),
            "BUY": "buying would bring you in line with the rule",
            "SELL": "selling would bring you in line with the rule",
        }[action]
        reasons.append(f"{you}, while the rule is {rule_is}: {consequence}")

    return Decision(
        ticker=ticker,
        action=action,
        stance=stance,
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
        mode=mode,
        holds_position=holds_position,
        signal_changed=(signal != "none"),
        signal=signal,
    )
