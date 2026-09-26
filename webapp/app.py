#!/usr/bin/env python3
"""The Flask app: the public advisor plus the admin training area.

I kept two clearly separated surfaces in one process:

  * PUBLIC  -- the online advisor. It loads a previously evolved rule,
    fetches recent prices, runs the deterministic rule engine, asks the
    guardrailed local LLM for a plain-language explanation (with the
    template fallback), and renders the lot with a backtest equity curve.
    No login needed.

  * ADMIN   -- the training system (implemented in webapp/admin.py). Behind
    a login; lets the administrator evolve new rules from the browser, see
    every saved rule's out-of-sample metrics, and delete rules. Training
    goes through the exact same pipeline as scripts/train.py.

Run with:  python webapp/app.py   (then open http://127.0.0.1:5000)
Admin at:  http://127.0.0.1:5000/admin   (password: ADVISOR_ADMIN_PASSWORD env var)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import secrets
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
from webapp.admin import admin_bp, init_admin

logging.basicConfig(level=logging.INFO)
RECENT_DAYS = 600  # how much history the live signal + context chart use


def _equity_png(prices, genome) -> str:
    """Backtest the recent window and return the equity chart as a base64 PNG."""
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


_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}


def _parse_holds(value: str | None) -> bool | None:
    """Parse ?holds= for the JSON API. Absent or unrecognised -> None, i.e. signal mode."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def create_app(config: dict | None = None) -> Flask:
    """Application factory -- mainly so the test suite can build isolated app instances."""
    app = Flask(__name__)
    # Session signing key: taken from the environment when deployed, random
    # per process otherwise -- the only consequence of the random one is that
    # admin sessions reset when the server restarts.
    app.config["SECRET_KEY"] = os.environ.get("ADVISOR_SECRET_KEY") or secrets.token_hex(32)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    if config:
        app.config.update(config)

    init_admin(app)
    app.register_blueprint(admin_bp, url_prefix="/admin")

    @app.context_processor
    def inject_globals():
        return {"disclaimer": DISCLAIMER}

    @app.route("/")
    def index():
        tickers = list_artifacts()
        selected = request.args.get("ticker") or (tickers[0] if tickers else None)
        # HTML gotcha that cost me a bug: an unticked checkbox is left out of
        # the query string entirely. So "absent" has to mean OFF once the form
        # has actually been submitted (i.e. ?ticker= is present), and can only
        # mean the default ON on a fresh visit with no query at all.
        if "ticker" in request.args:
            use_llm = request.args.get("llm") == "on"
        else:
            use_llm = request.args.get("llm", "on") != "off"
        # "I currently hold this asset" -- same unticked-checkbox rule as
        # above; a fresh visit assumes the user holds nothing. The web page
        # always runs in position mode, and the value is never stored.
        holds = request.args.get("holds") == "on"

        if not tickers:
            return render_template("index.html", tickers=[], error=(
                "No evolved rules found yet. An administrator can train one at /admin, "
                "or run: python scripts/train.py --ticker AAPL --start 2015-01-01"
            ))

        try:
            art = load_artifact(selected)
            genome = art["genome_obj"]
            ps = fetch_prices(selected)
            recent = ps.prices.iloc[-RECENT_DAYS:]
            decision = decide(selected, recent, genome, holds_position=holds)
            explanation = explain(decision, use_llm=use_llm)
            chart = _equity_png(recent, genome)
            return render_template(
                "index.html",
                tickers=tickers, selected=selected, decision=decision,
                explanation=explanation, artifact=art, chart=chart,
                data_source=ps.source, use_llm=use_llm, holds=holds, error=None,
            )
        except Exception as exc:
            logging.exception("Failed to build recommendation")
            return render_template("index.html", tickers=tickers, selected=selected,
                                   error=str(exc))

    @app.route("/api/recommendation/<ticker>")
    def api_recommendation(ticker: str):
        """The machine-readable endpoint: the full auditable decision + explanation.

        ?holds=1|0 selects position mode (the user does / does not hold the
        asset); leaving it out gives signal mode (the rule's own transition
        today).
        """
        art = load_artifact(ticker)
        ps = fetch_prices(ticker)
        decision = decide(ticker, ps.prices.iloc[-RECENT_DAYS:], art["genome_obj"],
                          holds_position=_parse_holds(request.args.get("holds")))
        explanation = explain(decision, use_llm=request.args.get("llm", "on") != "off")
        return jsonify({
            "decision": decision.to_dict(),
            "explanation": explanation,
            "disclaimer": DISCLAIMER,
        })

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
