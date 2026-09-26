# Evolutionary Financial Advisor Bot

This is my CM3070 Final Project (Template 4.2 — Financial Advisor Bot, BSc
Computer Science, University of London). The idea is an investment advisor
that is transparent and risk-aware: instead of a black-box model, the
trading rule at the centre of the system is evolved by a genetic algorithm
and can be written down as five integers, so every recommendation can be
traced back to a rule a human can read.

Installation and usage are covered in **SETUP.md**.

## How the system is structured

I split the project into two parts — a **training system** (only the
administrator can use it) and an **online advisor** (open to any user).
Both are served by the same Flask process:

```
TRAINING SYSTEM  ─ webapp/admin.py (browser, behind a login)  or  scripts/train.py (CLI)
  advisor/pipeline.py    the single end-to-end training pipeline that both entry points share
  advisor/jobs.py        background job handling + live progress for the web admin
  advisor/data.py        gets price data (yfinance, with a cache and a GBM fallback)
  advisor/genome.py      the 5-gene representation of a strategy
  advisor/indicators.py  SMA and RSI implementations
  advisor/fitness.py     walk-forward, regularised, multi-asset fitness function
  advisor/evolution.py   the genetic algorithm itself
  advisor/backtest.py    backtester with transaction costs + Sharpe/drawdown metrics
  advisor/evaluation.py  train/test split and walk-forward evaluation
  advisor/persistence.py saves rule artifacts to models/<ticker>.json

ONLINE ADVISOR   ─ webapp/app.py (public)
  advisor/rule_engine.py deterministic BUY/HOLD/SELL decisions + machine-readable reasons
  advisor/explain.py     LLM explanation via local Ollama, with guardrails and a fallback
```

Because an evolved rule is just five integers, the online side stays very
lightweight: it loads the saved rule, pulls recent prices, and produces a
signal. One design decision I want to be clear about: the language model
**never makes or changes the decision**. Its only job is to reword what the
rule engine already decided, and every LLM output is checked against the
actual decision before it is shown to the user. If the check fails (or
Ollama isn't running), the system falls back to a deterministic template
explanation instead.

## The admin training area

The `/admin` page lets the administrator evolve new rules from the browser,
look at each saved rule's out-of-sample performance next to buy-and-hold,
and delete rules that are no longer wanted. A few notes on how I built it:

- There is a single admin account. The password comes from the
  `ADVISOR_ADMIN_PASSWORD` environment variable and is only ever stored as
  a salted PBKDF2 hash; the session cookie is HttpOnly and SameSite.
- I added a brute-force throttle (5 failed logins locks the page for 5
  minutes), CSRF tokens on every state-changing form, and strict validation
  of ticker strings before they are used in any file path.
- Training runs in a background thread and the page polls for progress.
  Only one job can run at a time — the GA is CPU-bound, so letting jobs run
  in parallel would just make them all slower.
- The browser and the CLI both call the same `train_rule()` function, so
  the same settings and seed always produce the same rule. This was
  important to me for reproducibility.

## Keeping the evaluation honest

A big worry with evolved trading rules is that it is very easy to fool
yourself with overfitting, so I built several safeguards directly into the
evaluation:

- The train/test split is chronological, and the GA never sees the
  held-out test window.
- Signals are only acted on the following day, to avoid look-ahead bias.
- Transaction costs (10 basis points per side by default) are charged on
  every simulated trade.
- The fitness function is walk-forward (mean minus λ times the standard
  deviation of per-fold Sharpe ratios), which punishes rules that only work
  in one particular sub-period.
- A sparsity regularisation term penalises degenerate rules that barely
  ever trade.
- There is an optional multi-asset mode, where a rule is rewarded for
  generalising across several tickers rather than fitting one.
- Every result is reported against a buy-and-hold benchmark, so the rule
  has something honest to be compared with.

## Disclaimer

This is an educational decision-support tool, not regulated financial
advice. No real money should be traded on its output.
