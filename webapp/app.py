#!/usr/bin/env python3
"""The user-facing web advisor (Flask) plus the admin training area.

Two clearly separated surfaces share one process:

  * PUBLIC  -- the online advisor.  Loads a previously evolved rule, fetches
    recent prices, runs the deterministic rule engine, asks the guardrailed
    local LLM for a plain-language explanation (template fallback), and
    renders it with a backtest equity curve.  No login needed.

  * ADMIN   -- the training system (see webapp/admin.py).  Behind a login;
    lets an administrator evolve new rules from the browser, review every
    saved rule's out-of-sample metrics, and delete rules.  Training uses the
    exact same pipeline as scripts/train.py.

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


_TRUE = {"1", "true", "on", "yes"}
_FALSE = {"0", "false", "off", "no"}


def _parse_holds(value: str | None) -> bool | None:
    """?holds= for the JSON API: absent/unrecognised -> None (signal mode)."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None


def create_app(config: dict | None = None) -> Flask:
    """Application factory (lets the test-suite build isolated instances)."""
    app = Flask(__name__)
    # Session signing key: from the environment in deployment, random per
    # process otherwise (admin sessions then simply reset on restart).
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
        # An unticked checkbox is omitted from the query string entirely, so
        # "absent" must mean OFF once the form has been submitted (i.e. when
        # ?ticker= is present) and ON only on a fresh visit with no query.
        if "ticker" in request.args:
            use_llm = request.args.get("llm") == "on"
        else:
            use_llm = request.args.get("llm", "on") != "off"
        # "I currently hold this asset" -- same unticked-checkbox rule; a
        # fresh visit assumes the user holds nothing.  The web advisor always
        # runs in position mode; the value is never stored anywhere.
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
        """Machine-readable endpoint: the auditable decision + explanation.

        ?holds=1|0 selects position mode (the user does / does not hold the
        asset); omit it for signal mode (the rule's own transition today).
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
