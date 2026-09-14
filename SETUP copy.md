# Setup Instructions — Evolutionary Financial Advisor Bot (v1.1)

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

All 51 tests should pass in about 15 seconds (no network needed).

## 5. Set the admin password

The web app has an admin area for training rules. Set its password before
starting the server:

```bash
export ADVISOR_ADMIN_PASSWORD="choose-a-strong-password"   # macOS / Linux
set ADVISOR_ADMIN_PASSWORD=choose-a-strong-password         # Windows cmd
$env:ADVISOR_ADMIN_PASSWORD="choose-a-strong-password"      # Windows PowerShell
```

If you skip this, the app runs in **development mode** with the default
password `admin` and shows a warning banner on the login page.

Optional: `ADVISOR_SECRET_KEY` fixes the session-signing key so admin logins
survive a server restart (otherwise a random key is generated each run).

## 6. Run the web app

```bash
python webapp/app.py
```

| Page | URL | Who |
|---|---|---|
| Advisor | http://127.0.0.1:5000 | anyone — pick an asset, get BUY / HOLD / SELL with explanation |
| Admin login | http://127.0.0.1:5000/admin/login | administrator |
| Training system | http://127.0.0.1:5000/admin | administrator (after login) |
| JSON API | http://127.0.0.1:5000/api/recommendation/AAPL | anyone |

### Training a rule from the admin page

1. Log in at `/admin/login`.
2. Under **Evolve a new rule**, enter one ticker (single-asset) or several
   separated by spaces/commas (multi-asset — the rule must generalise across
   all of them). Adjust population, generations, seed, transaction cost,
   walk-forward folds and the date range if you wish.
3. Click **Start training**. The job runs in the background; the page shows
   live per-generation fitness, then the evolved rule and its out-of-sample
   results. A 60 × 40 run on ten years of daily data takes a few minutes.
4. When it finishes, the new ticker appears on the advisor page immediately.

Tick **Use synthetic data** to demo the whole flow without internet.

The **Saved rules** table lists every trained rule with its genome and its
out-of-sample Sharpe, return and drawdown next to the buy-and-hold benchmark
(green = beats the benchmark, red = doesn't). **Delete** removes a rule from
the advisor (e.g. to retire the DEMO rule before a demo video).

Only one training job runs at a time; the button is disabled while a job is
in progress.

## 7. Training from the command line (equivalent)

The admin page and the CLI call the same `advisor.pipeline.train_rule`, so
either can be used — identical settings and seed give identical rules.

```bash
# Single-asset evolution on Apple, 2015 onwards:
python scripts/train.py --ticker AAPL --start 2015-01-01

# Multi-asset evolution:
python scripts/train.py --ticker AAPL MSFT SPY --start 2015-01-01

# Offline smoke test on synthetic data:
python scripts/train.py --ticker DEMO --synthetic --generations 10 --population 30
```

Flags: `--generations`, `--population`, `--seed`, `--cost` (per side, default
0.001), `--folds` (walk-forward folds), `--train-frac` (default 0.7).

## 8. Evaluate a trained rule (plots for the report)

```bash
python scripts/evaluate.py --ticker AAPL
```

Prints the summary and rolling walk-forward tables and saves equity-curve,
GA-convergence and drawdown plots into `reports/`.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Falling back to synthetic GBM data" | Yahoo Finance unreachable or bad ticker. Check internet / ticker symbol. Downloaded data is cached in `data_cache/` and reused automatically. |
| "Need at least N days of prices, got 22" | Fixed in v1.1 (`advisor/data.py` now requests full history). If you still see it, delete `data_cache/<TICKER>.csv` and reload. |
| Explanation labelled "rule-based fallback" | Ollama isn't running or the model isn't pulled. By design — advice is never blocked by the explanation layer. Run `ollama serve` and `ollama pull llama3.2`. |
| Login page says "Development mode" | `ADVISOR_ADMIN_PASSWORD` is not set (step 5). Default password is `admin`. |
| "Too many failed attempts" | Five wrong passwords lock login for 5 minutes (brute-force throttle). |
| Admin session lost after restart | Set `ADVISOR_SECRET_KEY` to a fixed random string. |
| "A training job is already running" | Wait for the current job; one runs at a time. |
| Slow training | Reduce generations / population, or train one ticker at a time. |

## 10. Project layout

```
evo-advisor/
├── advisor/            the library
│   ├── data.py           price ingestion (yfinance + cache + GBM fallback)
│   ├── genome.py         5-gene strategy representation
│   ├── indicators.py     SMA, RSI
│   ├── backtest.py       cost-aware backtester + Sharpe/drawdown metrics
│   ├── fitness.py        walk-forward, regularised, multi-asset fitness
│   ├── evolution.py      genetic algorithm
│   ├── pipeline.py       end-to-end training pipeline (shared by CLI + web)
│   ├── jobs.py           background training jobs with live progress
│   ├── evaluation.py     train/test split + walk-forward evaluation
│   ├── persistence.py    rule artifacts -> models/<ticker>.json
│   ├── rule_engine.py    deterministic BUY/HOLD/SELL + reasons
│   └── explain.py        guardrailed Ollama explanation + fallback
├── scripts/            train.py, evaluate.py
├── webapp/             app.py (public advisor), admin.py (training system), templates/
├── tests/              pytest suite (51 tests)
├── models/             saved rule artifacts
├── data_cache/         cached price CSVs (auto-created)
├── reports/            evaluation plots (created by evaluate.py)
├── requirements.txt
└── README.md           architecture and design rationale
```

---
*This is an educational decision-support tool, not regulated financial
advice. No real money should be traded on its output.*
