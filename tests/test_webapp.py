"""Flask test-client tests: the public advisor, admin auth, web training and delete.

These test the app end to end through HTTP (well, the test client), which is
as close as the automated suite gets to what a user actually experiences.
"""
import re
import time

import pytest

import advisor.persistence as persistence
from advisor.persistence import list_artifacts
from advisor.pipeline import TrainRequest, train_rule
from webapp.app import create_app
from webapp import admin as admin_mod

PASSWORD = "correct horse battery staple"


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "MODELS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def app(models_dir):
    app = create_app({"TESTING": True, "ADMIN_PASSWORD": PASSWORD, "SECRET_KEY": "test"})
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, password=PASSWORD):
    return client.post("/admin/login", data={"password": password}, follow_redirects=False)


def csrf_of(client, path="/admin/"):
    html = client.get(path).get_data(as_text=True)
    m = re.search(r'name="csrf" value="([0-9a-f]+)"', html)
    assert m, "no CSRF token rendered"
    return m.group(1)


def wait_until_idle(client, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get("/admin/job/status").get_json()
        if not data["busy"]:
            return data["job"]
        time.sleep(0.05)
    raise AssertionError("training job did not finish")


# ---------------------------------------------------------------- public
def test_index_without_models_explains_how_to_train(client):
    html = client.get("/").get_data(as_text=True)
    assert "No evolved rules found" in html
    assert "/admin" in html


def test_index_serves_recommendation_for_synthetic_model(client, models_dir):
    train_rule(TrainRequest(tickers=["SYNX"], population=8, generations=2, folds=2, synthetic=True))
    # No network needed: fetch_prices falls back to synthetic data for SYNX.
    html = client.get("/?ticker=SYNX&llm=off").get_data(as_text=True)
    assert re.search(r'class="badge (BUY|SELL|HOLD)"', html)
    assert "rule-based fallback" in html


def test_llm_checkbox_can_be_unticked(client, models_dir):
    """Regression test for a real bug: an unticked checkbox is simply omitted
    from the submitted form, so the absence of ?llm= on a submitted form has
    to mean OFF -- my first version fell back to ON and the box could never
    be unticked."""
    train_rule(TrainRequest(tickers=["SYNZ"], population=8, generations=2, folds=2, synthetic=True))

    def checkbox_is_checked(html: str) -> bool:
        tag = html.split('name="llm"')[1].split(">")[0]   # attributes of the checkbox tag
        return "checked" in tag

    assert not checkbox_is_checked(client.get("/?ticker=SYNZ").get_data(as_text=True))       # unticked
    assert checkbox_is_checked(client.get("/?ticker=SYNZ&llm=on").get_data(as_text=True))    # ticked
    assert checkbox_is_checked(client.get("/").get_data(as_text=True))                       # fresh visit: ON


def test_api_recommendation_json(client, models_dir):
    train_rule(TrainRequest(tickers=["SYNY"], population=8, generations=2, folds=2, synthetic=True))
    data = client.get("/api/recommendation/SYNY?llm=off").get_json()
    assert data["decision"]["action"] in {"BUY", "SELL", "HOLD"}
    assert data["decision"]["mode"] == "signal"          # leaving out ?holds= means signal mode
    assert data["explanation"]["source"] == "template"


def test_api_holds_parameter_selects_position_mode(client, models_dir):
    train_rule(TrainRequest(tickers=["SYNP"], population=8, generations=2, folds=2, synthetic=True))
    yes = client.get("/api/recommendation/SYNP?llm=off&holds=1").get_json()["decision"]
    no = client.get("/api/recommendation/SYNP?llm=off&holds=0").get_json()["decision"]
    assert yes["mode"] == "position" and yes["holds_position"] is True
    assert no["mode"] == "position" and no["holds_position"] is False
    assert yes["stance"] == no["stance"]
    assert (yes["action"], no["action"]) == (
        ("HOLD", "BUY") if yes["stance"] == "in_market" else ("SELL", "HOLD"))
    # a junk value should degrade to signal mode, not blow up with a 500
    assert client.get("/api/recommendation/SYNP?llm=off&holds=maybe").get_json()["decision"]["mode"] == "signal"


def test_holds_checkbox_drives_position_aware_advice(client, models_dir):
    """The web page always runs in position mode: the checkbox states the
    user's position, and (same HTML quirk as the LLM box) absent-on-submit
    means OFF."""
    train_rule(TrainRequest(tickers=["SYNH"], population=8, generations=2, folds=2, synthetic=True))
    stance = client.get("/api/recommendation/SYNH?llm=off").get_json()["decision"]["stance"]

    def badge(html):
        return re.search(r'class="badge (BUY|SELL|HOLD)"', html).group(1)

    def holds_checked(html):
        return "checked" in html.split('name="holds"')[1].split(">")[0]

    fresh = client.get("/").get_data(as_text=True)                     # a first visit assumes you hold nothing
    assert not holds_checked(fresh) and "You: hold nothing" in fresh
    without = client.get("/?ticker=SYNH&llm=off").get_data(as_text=True)
    with_ = client.get("/?ticker=SYNH&llm=off&holds=on").get_data(as_text=True)
    assert not holds_checked(without) and holds_checked(with_)
    assert "You: hold this asset" in with_
    expected = ("BUY", "HOLD") if stance == "in_market" else ("HOLD", "SELL")
    assert (badge(without), badge(with_)) == expected
    assert ("Rule: in the market" if stance == "in_market" else "Rule: in cash") in without


def test_signal_changed_pill_only_on_transition_days(client, models_dir, monkeypatch):
    train_rule(TrainRequest(tickers=["SYNS"], population=8, generations=2, folds=2, synthetic=True))
    import webapp.app as app_mod
    from advisor import rule_engine

    real_decide = rule_engine.decide

    def forcing(signal):
        def forced(*a, **kw):
            d = real_decide(*a, **kw)
            d.signal, d.signal_changed = signal, signal != "none"
            return d
        return forced

    monkeypatch.setattr(app_mod, "decide", forcing("none"))
    assert "Signal changed today" not in client.get("/?ticker=SYNS&llm=off").get_data(as_text=True)
    monkeypatch.setattr(app_mod, "decide", forcing("enter"))
    html = client.get("/?ticker=SYNS&llm=off").get_data(as_text=True)
    assert "Signal changed today" in html and "entered the market" in html
    monkeypatch.setattr(app_mod, "decide", forcing("exit"))
    assert "moved to cash" in client.get("/?ticker=SYNS&llm=off").get_data(as_text=True)


# ------------------------------------------------------------------ auth
def test_admin_requires_login(client):
    r = client.get("/admin/")
    assert r.status_code == 302 and "/admin/login" in r.headers["Location"]
    assert client.get("/admin/job/status").status_code == 401
    assert client.post("/admin/train", data={}).status_code == 302


def test_wrong_password_rejected(client):
    r = login(client, "nope")
    assert r.status_code == 401
    assert client.get("/admin/").status_code == 302


def test_correct_password_grants_access(client):
    r = login(client)
    assert r.status_code == 302 and r.headers["Location"].endswith("/admin/")
    assert client.get("/admin/").status_code == 200


def test_lockout_after_repeated_failures(client, monkeypatch):
    monkeypatch.setattr(admin_mod, "MAX_FAILED_LOGINS", 3)
    for _ in range(3):
        login(client, "bad")
    assert login(client, "bad").status_code == 429
    assert login(client, PASSWORD).status_code == 429  # even the correct password has to wait out the lockout


def test_default_password_used_when_env_unset(models_dir, monkeypatch):
    monkeypatch.delenv("ADVISOR_ADMIN_PASSWORD", raising=False)
    app = create_app({"TESTING": True, "SECRET_KEY": "t"})
    c = app.test_client()
    assert "Development mode" in c.get("/admin/login").get_data(as_text=True)
    assert login(c, "admin").status_code == 302


def test_logout_requires_csrf_and_clears_session(client):
    login(client)
    assert client.post("/admin/logout", data={}).status_code == 400
    token = csrf_of(client)
    client.post("/admin/logout", data={"csrf": token})
    assert client.get("/admin/").status_code == 302


# -------------------------------------------------------------- training
def test_train_from_web_creates_model(client, models_dir):
    login(client)
    token = csrf_of(client)
    r = client.post("/admin/train", data={
        "csrf": token, "tickers": "webx, weby", "population": 8, "generations": 2,
        "folds": 2, "seed": 1, "cost": 0.001, "train_frac": 0.7, "synthetic": "on",
    }, follow_redirects=True)
    assert "Training started for WEBX, WEBY" in r.get_data(as_text=True)

    job = wait_until_idle(client)
    assert job["status"] == "done"
    assert any(l.startswith("gen ") for l in job["log"])
    assert sorted(list_artifacts()) == ["WEBX", "WEBY"]

    html = client.get("/admin/").get_data(as_text=True)
    assert "WEBX" in html and "multi-asset with WEBY" in html
    # ...and the public advisor page should offer them immediately
    assert "WEBX" in client.get("/?llm=off").get_data(as_text=True)


def test_train_rejects_invalid_ticker_and_missing_csrf(client, models_dir):
    login(client)
    assert client.post("/admin/train", data={"tickers": "AAPL"}).status_code == 400
    token = csrf_of(client)
    r = client.post("/admin/train", data={"csrf": token, "tickers": "../../x", "synthetic": "on"},
                    follow_redirects=True)
    assert "Could not start training" in r.get_data(as_text=True)
    assert list_artifacts() == []


def test_second_job_refused_while_running(client, models_dir):
    login(client)
    token = csrf_of(client)
    base = {"csrf": token, "population": 12, "generations": 6, "folds": 2, "synthetic": "on"}
    client.post("/admin/train", data={**base, "tickers": "BUSY1"})
    r = client.post("/admin/train", data={**base, "tickers": "BUSY2"}, follow_redirects=True)
    assert "already running" in r.get_data(as_text=True)
    wait_until_idle(client)


# ---------------------------------------------------------------- delete
def test_delete_model(client, models_dir):
    train_rule(TrainRequest(tickers=["GONE"], population=8, generations=2, folds=2, synthetic=True))
    login(client)
    token = csrf_of(client)
    r = client.post("/admin/models/GONE/delete", data={"csrf": token}, follow_redirects=True)
    assert "Deleted the saved rule for GONE" in r.get_data(as_text=True)
    assert list_artifacts() == []
    r = client.post("/admin/models/GONE/delete", data={"csrf": token}, follow_redirects=True)
    assert "No saved rule found" in r.get_data(as_text=True)
