# Setup Instructions — Evolutionary Financial Advisor Bot (v1.1)

## 1. What you need

- Python 3.10 or newer (I developed on 3.11)
- An internet connection, so the app can download price data from Yahoo
  Finance
- Optionally, [Ollama](https://ollama.com) if you want the local-LLM
  explanation layer. It is genuinely optional: if Ollama isn't installed,
  the system falls back to rule-based explanations and everything else
  still works.

## 2. Installing

```bash
# 1. Unzip the project and move into it
cd evo-advisor

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt
```

## 3. (Optional) Setting up the local LLM

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama serve                     # this usually starts by itself after installing
```

The advisor expects Ollama at `http://localhost:11434`. If you'd rather use
a different model, change `OLLAMA_MODEL` at the top of
`advisor/explain.py`.

## 4. Checking the installation works

```bash
python -m pytest tests/ -q
```

All 51 tests should pass in roughly 15 seconds, and none of them need the
network.

## 5. Setting the admin password

The web app has an admin area for training rules, and you should set its
password before starting the server:

```bash
export ADVISOR_ADMIN_PASSWORD="choose-a-strong-password"   # macOS / Linux
set ADVISOR_ADMIN_PASSWORD=choose-a-strong-password         # Windows cmd
$env:ADVISOR_ADMIN_PASSWORD="choose-a-strong-password"      # Windows PowerShell
```

If you skip this step, the app still runs, but in **development mode**: the
password defaults to `admin` and the login page displays a warning banner
so this can't happen silently.

You can also set `ADVISOR_SECRET_KEY` to a fixed random string if you want
admin logins to survive a server restart (otherwise a fresh signing key is
generated every run, which logs everyone out).

## 6. Running the web app

```bash
python webapp/app.py
```

| Page | URL | Who it's for |
|---|---|---|
| Advisor | http://127.0.0.1:5000 | anyone — choose an asset and get BUY / HOLD / SELL with an explanation |
| Admin login | http://127.0.0.1:5000/admin/login | the administrator |
| Training system | http://127.0.0.1:5000/admin | the administrator (once logged in) |
| JSON API | http://127.0.0.1:5000/api/recommendation/AAPL | anyone |

### Training a rule through the admin page

1. Log in at `/admin/login`.
2. In **Evolve a new rule**, type one ticker for single-asset evolution, or
   several tickers separated by spaces or commas for multi-asset evolution
   (in that mode the rule has to generalise across all of them). You can
   also change the population size, number of generations, random seed,
   transaction cost, number of walk-forward folds and the date range.
3. Press **Start training**. The job runs in the background and the page
   shows the fitness for each generation as it goes, followed by the
   evolved rule and its out-of-sample results. As a rough guide, a 60 × 40
   run over ten years of daily data takes a few minutes on my machine.
4. Once it finishes, the new ticker shows up on the public advisor page
   straight away.

If you tick **Use synthetic data**, the whole flow can be demonstrated
without any internet connection.

The **Saved rules** table shows every rule that has been trained, together
with its genome and its out-of-sample Sharpe ratio, return and drawdown
alongside the buy-and-hold benchmark (green means it beats the benchmark,
red means it doesn't). The **Delete** button removes a rule from the
advisor — I used this to retire the DEMO rule before recording my demo
video.

Only one training job can run at a time, and the button is disabled while
a job is in progress.

## 7. Training from the command line (does the same thing)

The admin page and the CLI both call the same
`advisor.pipeline.train_rule` function, so you can use whichever is more
convenient — identical settings and seed give identical rules.

```bash
# Single-asset evolution on Apple, from 2015 onwards:
python scripts/train.py --ticker AAPL --start 2015-01-01

# Multi-asset evolution:
python scripts/train.py --ticker AAPL MSFT SPY --start 2015-01-01

# Offline smoke test on synthetic data:
python scripts/train.py --ticker DEMO --synthetic --generations 10 --population 30
```

Available flags: `--generations`, `--population`, `--seed`, `--cost` (per
side, default 0.001), `--folds` (walk-forward folds) and `--train-frac`
(default 0.7).

## 8. Evaluating a trained rule (this makes the plots for the report)

```bash
python scripts/evaluate.py --ticker AAPL
```

This prints the summary and rolling walk-forward tables, and saves the
equity-curve, GA-convergence and drawdown plots into `reports/`.

## 9. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| "Falling back to synthetic GBM data" | Yahoo Finance couldn't be reached, or the ticker doesn't exist. Check your connection and the symbol. Downloaded data is cached in `data_cache/` and reused automatically. |
| "Need at least N days of prices, got 22" | This was a bug I fixed in v1.1 (`advisor/data.py` now requests the full history). If it still appears, delete `data_cache/<TICKER>.csv` and reload. |
| Explanation is labelled "rule-based fallback" | Ollama isn't running, or the model hasn't been pulled. This is deliberate — advice is never blocked just because the explanation layer is down. Run `ollama serve` and `ollama pull llama3.2`. |
| Login page says "Development mode" | You haven't set `ADVISOR_ADMIN_PASSWORD` (see step 5). The default password is `admin`. |
| "Too many failed attempts" | Five wrong passwords lock the login for 5 minutes — this is the brute-force throttle working as intended. |
| Admin session lost after a restart | Set `ADVISOR_SECRET_KEY` to a fixed random string. |
| "A training job is already running" | Wait for the current job to finish; only one runs at a time. |
| Training is slow | Lower the generations or population, or train one ticker at a time. |

## 10. Project layout

```
evo-advisor/
├── advisor/            the main library
│   ├── data.py           price ingestion (yfinance + cache + GBM fallback)
│   ├── genome.py         5-gene strategy representation
│   ├── indicators.py     SMA, RSI
│   ├── backtest.py       cost-aware backtester + Sharpe/drawdown metrics
│   ├── fitness.py        walk-forward, regularised, multi-asset fitness
│   ├── evolution.py      the genetic algorithm
│   ├── pipeline.py       end-to-end training pipeline (shared by CLI and web)
│   ├── jobs.py           background training jobs with live progress
│   ├── evaluation.py     train/test split + walk-forward evaluation
│   ├── persistence.py    rule artifacts -> models/<ticker>.json
│   ├── rule_engine.py    deterministic BUY/HOLD/SELL + reasons
│   └── explain.py        guardrailed Ollama explanation + fallback
├── scripts/            train.py and evaluate.py
├── webapp/             app.py (public advisor), admin.py (training system), templates/
├── tests/              pytest suite (51 tests)
├── models/             saved rule artifacts
├── data_cache/         cached price CSVs (created automatically)
├── reports/            evaluation plots (created by evaluate.py)
├── requirements.txt
└── README.md           architecture and design rationale
```

---
*This is an educational decision-support tool, not regulated financial
advice. No real money should be traded on its output.*
