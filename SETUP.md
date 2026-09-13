# Setup Instructions — Evolutionary Financial Advisor Bot

## 1. Requirements

- Python 3.10 or newer (3.11 recommended)
- Internet access (for downloading price data via Yahoo Finance)
- Optional: [Ollama](https://ollama.com) for the local-LLM explanation layer.
  Without it, the system automatically falls back to rule-based explanations —
  nothing breaks.

## 2. Installation

```bash
# 1. Unzip the project and enter it
cd evo-advisor

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 3. (Optional) Set up the local LLM

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama serve                     # usually starts automatically after install
```

The advisor talks to Ollama at `http://localhost:11434`. To use a different
model, change `OLLAMA_MODEL` at the top of `advisor/explain.py`.

## 4. Verify the installation

```bash
python -m pytest tests/ -q
```

All 30 tests should pass in a few seconds (no network needed).

## 5. Train a rule (the offline pipeline)

```bash
# Single-asset evolution on Apple, 2015 onwards (takes a few minutes):
python scripts/train.py --ticker AAPL --start 2015-01-01

# Multi-asset evolution — the rule must generalise across all tickers:
python scripts/train.py --ticker AAPL MSFT SPY --start 2015-01-01

# Offline smoke test on synthetic data (no internet needed):
python scripts/train.py --ticker DEMO --synthetic --generations 10 --population 30
```

This prints per-generation fitness, the evolved rule in plain English, the
in-sample / out-of-sample / buy-and-hold summary table, and saves the rule
to `models/<TICKER>.json`.

Useful flags: `--generations`, `--population`, `--seed` (reproducibility),
`--cost` (transaction cost per side, default 0.001), `--folds` (walk-forward
folds in the fitness function), `--train-frac` (default 0.7).

## 6. Evaluate a trained rule

```bash
python scripts/evaluate.py --ticker AAPL
```

Prints the summary and rolling walk-forward tables, and saves report-ready
plots (equity curves, GA convergence, drawdown) into `reports/`.

## 7. Run the web advisor (the online system)

```bash
python webapp/app.py
```

Open **http://127.0.0.1:5000** in your browser. Pick an asset that has a
trained rule, and you'll see the BUY / HOLD / SELL recommendation, its
machine-readable reasons, the plain-language explanation (LLM or fallback —
the page labels which), the rule itself, and the recent backtest context.

There is also a JSON API for the auditable decision:
`GET http://127.0.0.1:5000/api/recommendation/AAPL`

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Falling back to synthetic GBM data" | Yahoo Finance unreachable or bad ticker. Check internet / ticker symbol. Previously downloaded data is cached in `data_cache/` and reused automatically. |
| Explanation labelled "rule-based fallback" | Ollama isn't running or the model isn't pulled. This is by design — advice is never blocked by the explanation layer. Run `ollama serve` and `ollama pull llama3.2`. |
| "No evolved rules found" in the web app | Train first (step 5) — the web app only loads artifacts from `models/`. |
| Slow training | Reduce `--generations` / `--population`, or use one ticker instead of several. |

## 9. Project layout

```
evo-advisor/
├── advisor/          the library (data, GA, backtest, rule engine, LLM layer)
├── scripts/          train.py (evolve + save), evaluate.py (tables + plots)
├── webapp/           Flask app + template
├── tests/            pytest suite (30 tests)
├── models/           saved rule artifacts (created by train.py)
├── data_cache/       cached price CSVs (created automatically)
├── reports/          evaluation plots (created by evaluate.py)
├── requirements.txt
└── README.md         architecture and design rationale
```

---
*This is an educational decision-support tool, not regulated financial
advice. No real money should be traded on its output.*
