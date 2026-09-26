"""Evolutionary Financial Advisor Bot.

This package is the core of my CM3070 final project (Template 4.2, Financial
Advisor Bot): an investment advisor whose trading rule is evolved by a
genetic algorithm and kept deliberately simple enough to read.

A quick map of the modules, roughly in the order data flows through them:

data         -- gets price data (yfinance) with a synthetic GBM fallback
indicators   -- the two technical indicators I use (SMA, RSI)
genome       -- the five-gene strategy representation, its bounds and repair
backtest     -- backtester that charges transaction costs + Sharpe/drawdown
fitness      -- walk-forward, regularised, multi-asset fitness function
evolution    -- the genetic algorithm (selection, crossover, mutation, elitism)
pipeline     -- the end-to-end training pipeline shared by the CLI and web admin
jobs         -- background training jobs with live progress (for the web admin)
rule_engine  -- deterministic BUY/HOLD/SELL decisions with readable reasons
explain      -- local-LLM (Ollama) explanations behind guardrails, with fallback
persistence  -- saving/loading evolved rules as small JSON files
evaluation   -- out-of-sample evaluation against a buy-and-hold benchmark
"""

__version__ = "1.1.0"
