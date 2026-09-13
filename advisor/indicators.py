"""Technical indicators used by the evolved trading rule.

Only two indicators are needed for the five-gene genome: the simple moving
average (SMA) and Wilder's Relative Strength Index (RSI).  Keeping the
indicator set small is a deliberate interpretability decision -- the evolved
rule must remain readable by a human (see the design chapter of the report).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` days (NaN until enough history)."""
    if window < 1:
        raise ValueError("SMA window must be >= 1")
    return prices.rolling(window=window, min_periods=window).mean()


def rsi(prices: pd.Series, period: int) -> pd.Series:
    """Wilder's Relative Strength Index in [0, 100].

    Uses Wilder's exponential smoothing (alpha = 1/period), the standard
    formulation.  Values are NaN until `period` observations are available.
    """
    if period < 1:
        raise ValueError("RSI period must be >= 1")
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 (all gains), RSI is 100 by convention.
    out = out.where(~((avg_loss == 0) & avg_gain.notna() & (avg_gain > 0)), 100.0)
    return out
