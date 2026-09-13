#!/usr/bin/env python3
"""The user-facing web advisor (Flask).

This is the ONLINE half of the system: it loads a previously evolved rule
artifact, fetches recent prices, runs the deterministic rule engine, asks the
guardrailed local LLM for a plain-language explanation (falling back to a
template if unavailable), and renders everything with the backtest equity
curve for context.

Run with:  python webapp/app.py   (then open http://127.0.0.1:5000)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advisor.backtest import run_backtest
from advisor.data import fetch_prices
from advisor.explain import DISCLAIMER, explain
from advisor.persistence import list_artifacts, load_artifact
from advisor.rule_engine import decide

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

RECENT_DAYS = 600  # history fetched for the live signal + context chart


def _equity_png(prices, genome) -> str:
    """Backtest over the recent window and return the chart as base64 PNG."""
    res = run_backtest(prices, genome)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(res.equity.index, res.equity.values, label="Evolved strategy")
    ax.plot(res.benchmark_equity.index, res.benchmark_equity.values,
            label="Buy & hold", alpha=0.8)
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_title("Recent backtest context (net of costs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    tickers = list_artifacts()
    selected = request.args.get("ticker") or (tickers[0] if tickers else None)
    use_llm = request.args.get("llm", "on") != "off"

    if not tickers:
        return render_template("index.html", tickers=[], error=(
            "No evolved rules found. Train one first, e.g.: "
            "python scripts/train.py --ticker AAPL --start 2015-01-01"
        ), disclaimer=DISCLAIMER)

    try:
        art = load_artifact(selected)
        genome = art["genome_obj"]
        ps = fetch_prices(selected)
        recent = ps.prices.iloc[-RECENT_DAYS:]
        decision = decide(selected, recent, genome)
        explanation = explain(decision, use_llm=use_llm)
        chart = _equity_png(recent, genome)
        return render_template(
            "index.html",
            tickers=tickers,
            selected=selected,
            decision=decision,
            explanation=explanation,
            artifact=art,
            chart=chart,
            data_source=ps.source,
            use_llm=use_llm,
            disclaimer=DISCLAIMER,
            error=None,
        )
    except Exception as exc:
        logging.exception("Failed to build recommendation")
        return render_template("index.html", tickers=tickers, selected=selected,
                               error=str(exc), disclaimer=DISCLAIMER)


@app.route("/api/recommendation/<ticker>")
def api_recommendation(ticker: str):
    """Machine-readable endpoint: the auditable decision + explanation."""
    art = load_artifact(ticker)
    ps = fetch_prices(ticker)
    decision = decide(ticker, ps.prices.iloc[-RECENT_DAYS:], art["genome_obj"])
    explanation = explain(decision, use_llm=request.args.get("llm", "on") != "off")
    return jsonify({
        "decision": decision.to_dict(),
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
    })


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
