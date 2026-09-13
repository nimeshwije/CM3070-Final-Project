# Evolutionary Financial Advisor Bot

A transparent, risk-aware investment advisor built around a trading rule
evolved by a genetic algorithm. CM3070 Final Project, Template 4.2
(Financial Advisor Bot), BSc Computer Science, University of London.

See **SETUP.md** for installation and usage instructions.

## Architecture

The system separates into an **offline training pipeline** (which the end
user never touches) and an **online advisor** (which they interact with):

```
OFFLINE  ─ scripts/train.py
  advisor/data.py        price ingestion (yfinance + cache + GBM fallback)
  advisor/genome.py      the 5-gene strategy representation
  advisor/indicators.py  SMA, RSI
  advisor/fitness.py     walk-forward, regularised, multi-asset fitness
  advisor/evolution.py   the genetic algorithm
  advisor/backtest.py    cost-aware backtester + Sharpe/drawdown metrics
  advisor/evaluation.py  train/test split + walk-forward evaluation
  advisor/persistence.py rule artifacts -> models/<ticker>.json

ONLINE   ─ webapp/app.py (Flask)
  advisor/rule_engine.py deterministic BUY/HOLD/SELL + machine-readable reasons
  advisor/explain.py     guardrailed local-LLM explanation (Ollama) + fallback
```

The evolved rule is only five integers, so the online advisor is
lightweight: it loads the rule, fetches recent prices, and emits a signal.
The language model **never makes or alters the decision** — it only rewords
the rule engine's output, and every LLM response is post-checked against the
decision before being shown (falling back to a deterministic template).

## Honesty measures baked into the evaluation

- Chronological train/test split — the GA never sees the held-out window.
- Signals are acted on one day later (no look-ahead bias).
- Transaction costs (10 bp per side by default) charged on every trade.
- Walk-forward fitness (mean − λ·std of per-fold Sharpe) penalises rules
  that only work in one sub-period.
- Sparsity regularisation penalises degenerate rules that almost never trade.
- Optional multi-asset evolution rewards rules that generalise across tickers.
- All results reported against a buy-and-hold benchmark.

## Disclaimer

This is an educational decision-support tool, not regulated financial
advice. No real money should be traded on its output.
