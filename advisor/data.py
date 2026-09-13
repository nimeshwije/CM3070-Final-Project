"""Price data ingestion.

Downloads daily adjusted closing prices via yfinance, caches them locally as
CSV so repeated runs (and offline demos) do not depend on the network, and
falls back to a synthetic geometric-Brownian-motion (GBM) series when no data
can be obtained at all.  The GBM fallback is a deliberate robustness decision:
the system must always run, e.g. during a live demonstration without internet.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Directory used to cache downloaded price series.
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")


@dataclass
class PriceSeries:
    """A daily close-price series plus provenance metadata."""

    ticker: str
    prices: pd.Series          # indexed by DatetimeIndex, values = adjusted close
    source: str                # "yfinance", "cache", or "synthetic"

    def __len__(self) -> int:  # convenience
        return len(self.prices)


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("/", "_").replace("^", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def _load_cache(ticker: str) -> pd.Series | None:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df["close"].astype(float)
    except Exception:  # corrupt cache -> ignore
        return None


def _save_cache(ticker: str, prices: pd.Series) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    prices.rename("close").to_frame().to_csv(_cache_path(ticker))


def synthetic_gbm(
    n_days: int = 2500,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.25,
    seed: int = 42,
) -> pd.Series:
    """Generate a synthetic daily price path with geometric Brownian motion.

    Parameters mirror a plausible equity: 8% annual drift, 25% annual vol.
    A fixed seed keeps the fallback reproducible across runs.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    steps = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n_days)
    log_path = np.concatenate([[np.log(s0)], np.log(s0) + np.cumsum(steps)])
    # Fixed start date keeps the fallback fully reproducible (and avoids the
    # pandas quirk where a non-business-day `end` shortens the range).
    dates = pd.bdate_range(start="2016-01-04", periods=len(log_path))
    return pd.Series(np.exp(log_path), index=dates, name="close")


def fetch_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    allow_synthetic: bool = True,
) -> PriceSeries:
    """Fetch daily close prices for `ticker`.

    Resolution order:
      1. yfinance download (auto-adjusted closes), which also refreshes the cache;
      2. the local CSV cache;
      3. a synthetic GBM series (if `allow_synthetic`), so the pipeline always runs.

    NOTE: when no `start` is given, the FULL available history is requested.
    yfinance's own default with no dates is period="1mo" (~22 trading days),
    which is far too little for the rule's long moving-average window -- this
    is exactly the "Need at least N days of prices, got 22" failure mode.
    """
    # --- 1. live download -------------------------------------------------
    try:
        import yfinance as yf

        if start is None and end is None:
            # No range requested -> full history, never yfinance's 1-month default.
            df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is not None and len(df) > 0:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance MultiIndex quirk
                close = close.iloc[:, 0]
            close = close.dropna().astype(float)
            if len(close) > 0:
                _save_cache(ticker, close)
                logger.info("Fetched %d days of %s from yfinance", len(close), ticker)
                return PriceSeries(ticker, close, "yfinance")
    except Exception as exc:  # network failure, bad ticker, etc.
        logger.warning("yfinance download failed for %s: %s", ticker, exc)

    # --- 2. cache ---------------------------------------------------------
    cached = _load_cache(ticker)
    if cached is not None and len(cached) > 0:
        if start:
            cached = cached[cached.index >= pd.Timestamp(start)]
        if end:
            cached = cached[cached.index <= pd.Timestamp(end)]
        if len(cached) > 0:
            logger.info("Loaded %d days of %s from cache", len(cached), ticker)
            return PriceSeries(ticker, cached, "cache")

    # --- 3. synthetic fallback -------------------------------------------
    if allow_synthetic:
        logger.warning("Falling back to synthetic GBM data for %s", ticker)
        return PriceSeries(ticker, synthetic_gbm(), "synthetic")

    raise RuntimeError(f"No price data available for {ticker}")
