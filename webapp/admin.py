"""Admin area: login, web-based training, model overview, deletion.

Security model (deliberately small, and documented for the report):

  * ONE admin account.  The password comes from the ADVISOR_ADMIN_PASSWORD
    environment variable; it is hashed at start-up (werkzeug, salted
    PBKDF2) and never held in plain text.  If the variable is unset a
    development default ("admin") is used and a warning is logged and
    shown in the UI.
  * Session-cookie login (signed with the app secret), HttpOnly + SameSite.
  * Brute-force throttle: after 5 failed attempts from one address the
    login is locked for 5 minutes.
  * CSRF token on every state-changing form (train, delete, logout).
  * Ticker inputs are validated by a strict pattern before touching disk.

Training runs in a background thread (advisor.jobs) and the page polls
/admin/job/status for live per-generation progress.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from functools import wraps

from flask import (
    Blueprint, abort, flash, jsonify, redirect, render_template, request,
    session, url_for, current_app,
)
from werkzeug.security import check_password_hash, generate_password_hash

from advisor.jobs import JobManager
from advisor.persistence import delete_artifact, is_valid_ticker, list_artifacts, load_artifact
from advisor.pipeline import TrainRequest

logger = logging.getLogger(__name__)
admin_bp = Blueprint("admin", __name__)

DEV_DEFAULT_PASSWORD = "admin"
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 300

_state: dict = {
    "password_hash": None,
    "using_default_password": False,
    "failed": {},            # ip -> [count, first_failure_ts]
    "jobs": None,            # JobManager
}


def init_admin(app) -> None:
    """Configure credentials + job manager for this app instance."""
    pw = app.config.get("ADMIN_PASSWORD") or os.environ.get("ADVISOR_ADMIN_PASSWORD")
    if not pw:
        pw = DEV_DEFAULT_PASSWORD
        _state["using_default_password"] = True
        logger.warning(
            "ADVISOR_ADMIN_PASSWORD is not set -- using the development default "
            "password. Set the environment variable before exposing this app."
        )
    else:
        _state["using_default_password"] = False
    _state["password_hash"] = generate_password_hash(pw)
    _state["failed"] = {}
    _state["jobs"] = app.config.get("JOB_MANAGER") or JobManager()


def jobs() -> JobManager:
    return _state["jobs"]


# ----------------------------------------------------------------- helpers
def _csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


def _check_csrf() -> None:
    token = request.form.get("csrf") or request.headers.get("X-CSRF-Token")
    if not token or token != session.get("csrf"):
        abort(400, "Invalid or missing CSRF token")


def _client_ip() -> str:
    return request.remote_addr or "unknown"


def _locked_out(ip: str) -> int:
    """Seconds of lockout remaining for `ip` (0 if not locked)."""
    rec = _state["failed"].get(ip)
    if not rec:
        return 0
    count, first_ts = rec
    if count < MAX_FAILED_LOGINS:
        return 0
    remaining = LOCKOUT_SECONDS - (time.time() - first_ts)
    if remaining <= 0:
        del _state["failed"][ip]
        return 0
    return int(remaining)


def _record_failure(ip: str) -> None:
    count, first_ts = _state["failed"].get(ip, (0, time.time()))
    _state["failed"][ip] = (count + 1, first_ts)


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("admin"):
            if request.path.startswith("/admin/job"):
                return jsonify({"error": "login required"}), 401
            return redirect(url_for("admin.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@admin_bp.context_processor
def inject_admin_globals():
    return {
        "csrf_token": _csrf_token,
        "using_default_password": _state["using_default_password"],
    }


# ------------------------------------------------------------------ auth
@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    ip = _client_ip()
    if request.method == "POST":
        wait = _locked_out(ip)
        if wait:
            flash(f"Too many failed attempts. Try again in {wait // 60 + 1} minute(s).", "error")
            return render_template("admin_login.html"), 429
        password = request.form.get("password", "")
        if check_password_hash(_state["password_hash"], password):
            session.clear()
            session["admin"] = True
            session.permanent = False
            _state["failed"].pop(ip, None)
            _csrf_token()
            nxt = request.args.get("next") or url_for("admin.dashboard")
            if not nxt.startswith("/"):        # open-redirect guard
                nxt = url_for("admin.dashboard")
            return redirect(nxt)
        _record_failure(ip)
        flash("Incorrect password.", "error")
        return render_template("admin_login.html"), 401
    if session.get("admin"):
        return redirect(url_for("admin.dashboard"))
    return render_template("admin_login.html")


@admin_bp.route("/logout", methods=["POST"])
def logout():
    _check_csrf()
    session.clear()
    return redirect(url_for("index"))


# ------------------------------------------------------------- dashboard
def _model_rows() -> list[dict]:
    rows = []
    for t in list_artifacts():
        try:
            art = load_artifact(t)
        except Exception as exc:  # corrupt file: still list it so it can be deleted
            rows.append({"ticker": t, "error": str(exc)})
            continue
        te = art.get("test_metrics", {})
        meta = art.get("meta", {})
        rows.append({
            "ticker": t,
            "genome": art["genome"],
            "rule_text": art.get("rule_text", ""),
            "oos_sharpe": te.get("sharpe"),
            "bench_sharpe": te.get("benchmark_sharpe"),
            "oos_return": te.get("total_return"),
            "bench_return": te.get("benchmark_total_return"),
            "max_dd": te.get("max_drawdown"),
            "bench_max_dd": te.get("benchmark_max_drawdown"),
            "n_trades": te.get("n_trades"),
            "data_source": meta.get("data_source", "?"),
            "test_range": meta.get("test_range"),
            "saved_at": (meta.get("saved_at") or "")[:16].replace("T", " "),
            "partners": meta.get("multi_asset_partners", []),
        })
    return rows


@admin_bp.route("/")
@admin_required
def dashboard():
    return render_template(
        "admin.html",
        models=_model_rows(),
        job=jobs().current(),
        history=jobs().history(),
        busy=jobs().is_busy(),
        defaults=TrainRequest(tickers=[]).to_dict(),
    )


# -------------------------------------------------------------- training
@admin_bp.route("/train", methods=["POST"])
@admin_required
def train():
    _check_csrf()
    f = request.form
    tickers = [t.strip().upper() for t in f.get("tickers", "").replace(",", " ").split() if t.strip()]
    try:
        req = TrainRequest(
            tickers=tickers,
            start=f.get("start") or "2015-01-01",
            end=f.get("end") or None,
            train_frac=float(f.get("train_frac") or 0.7),
            population=int(f.get("population") or 60),
            generations=int(f.get("generations") or 40),
            seed=int(f.get("seed") or 7),
            cost=float(f.get("cost") or 0.001),
            folds=int(f.get("folds") or 4),
            synthetic=f.get("synthetic") == "on",
        )
        jobs().start(req)
        flash(f"Training started for {', '.join(tickers)}.", "ok")
    except (ValueError, RuntimeError) as exc:
        flash(f"Could not start training: {exc}", "error")
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/job/status")
@admin_required
def job_status():
    """Polled by the dashboard for live progress."""
    return jsonify({"job": jobs().current(), "busy": jobs().is_busy()})


# ---------------------------------------------------------------- delete
@admin_bp.route("/models/<ticker>/delete", methods=["POST"])
@admin_required
def delete_model(ticker: str):
    _check_csrf()
    if not is_valid_ticker(ticker):
        abort(400, "Invalid ticker")
    if delete_artifact(ticker):
        flash(f"Deleted the saved rule for {ticker}.", "ok")
    else:
        flash(f"No saved rule found for {ticker}.", "error")
    return redirect(url_for("admin.dashboard"))
