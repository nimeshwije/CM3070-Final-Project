"""The technical indicators the evolved rule is built from.

The five-gene genome only needs two indicators: the simple moving average
(SMA) and Wilder's Relative Strength Index (RSI). I deliberately did not add
more -- a bigger indicator library would give the GA more to search over,
but the evolved rule has to stay readable by a human, which is the point of
the project (I justify this in the design chapter of the report).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` days. NaN until there is enough history."""
    if window < 1:
        raise ValueError("SMA window must be >= 1")
    return prices.rolling(window=window, min_periods=window).mean()


def rsi(prices: pd.Series, period: int) -> pd.Series:
    """Wilder's Relative Strength Index, bounded in [0, 100].

    I use Wilder's exponential smoothing (alpha = 1/period), which is the
    textbook formulation. Values are NaN until `period` observations exist.
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
    # Edge case: if avg_loss == 0 (the series only went up) the division
    # above gives NaN, but by convention RSI should be 100 there.
    out = out.where(~((avg_loss == 0) & avg_gain.notna() & (avg_gain > 0)), 100.0)
    return out
