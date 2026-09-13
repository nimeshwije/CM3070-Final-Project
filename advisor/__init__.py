"""Evolutionary Financial Advisor Bot.

A transparent, risk-aware investment advisor built around a trading rule
evolved by a genetic algorithm (CM3070 Template 4.2, Financial Advisor Bot).

Package layout
--------------
data         -- price ingestion (yfinance) with a synthetic GBM fallback
indicators   -- technical indicators (SMA, RSI)
genome       -- the five-gene strategy representation and its bounds/repair
backtest     -- transaction-cost-aware backtester and performance metrics
fitness      -- walk-forward, regularised, multi-asset fitness function
evolution    -- the genetic algorithm (selection, crossover, mutation, elitism)
rule_engine  -- deterministic BUY/HOLD/SELL decisions with machine-readable reasons
explain      -- guardrailed local-LLM (Ollama) explanations with a safe fallback
persistence  -- saving/loading evolved rule artifacts as JSON
evaluation   -- out-of-sample evaluation against a buy-and-hold benchmark
"""

__version__ = "1.0.0"
