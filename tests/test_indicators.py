import numpy as np
import pandas as pd
import pytest

from advisor.indicators import rsi, sma


def make_series(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_sma_matches_manual_mean():
    s = make_series([1, 2, 3, 4, 5, 6])
    out = sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[-1] == pytest.approx(5.0)


def test_sma_rejects_bad_window():
    with pytest.raises(ValueError):
        sma(make_series([1, 2, 3]), 0)


def test_rsi_bounds_and_direction():
    rng = np.random.default_rng(0)
    s = make_series(100 + np.cumsum(rng.normal(0, 1, 300)))
    out = rsi(s, 14).dropna()
    assert ((out >= 0) & (out <= 100)).all()


def test_rsi_all_gains_is_100():
    s = make_series(np.arange(1, 40, dtype=float))  # strictly rising
    out = rsi(s, 14).dropna()
    assert out.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_near_zero():
    s = make_series(np.arange(40, 1, -1, dtype=float))  # strictly falling
    out = rsi(s, 14).dropna()
    assert out.iloc[-1] < 1e-6
