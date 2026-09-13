"""Tests for the shared training pipeline and the background job manager."""
import time

import pytest

import advisor.persistence as persistence
from advisor.jobs import JobManager
from advisor.persistence import list_artifacts, load_artifact
from advisor.pipeline import TrainRequest, train_rule


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Redirect artifact storage to a temp folder so tests never touch models/."""
    monkeypatch.setattr(persistence, "MODELS_DIR", str(tmp_path))
    return tmp_path


def quick_request(*tickers, seed=7):
    return TrainRequest(tickers=list(tickers), population=10, generations=3, folds=2,
                        synthetic=True, seed=seed)


def test_request_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        TrainRequest(tickers=[]).validate()
    with pytest.raises(ValueError):
        TrainRequest(tickers=["../etc/passwd"]).validate()
    with pytest.raises(ValueError):
        TrainRequest(tickers=["AAPL"], population=2).validate()
    with pytest.raises(ValueError):
        TrainRequest(tickers=["AAPL"], train_frac=0.99).validate()
    TrainRequest(tickers=["BRK-B", "^GSPC", "BTC-USD"]).validate()  # legitimate symbols


def test_train_rule_saves_artifact_and_reports_progress(models_dir):
    lines = []
    outcome = train_rule(quick_request("SYN1"), progress=lines.append)
    assert "SYN1" in list_artifacts()
    art = load_artifact("SYN1")
    assert art["genome"] == outcome.genome
    assert any(l.startswith("gen ") for l in lines)
    assert outcome.per_ticker[0]["test"]["sharpe"] is not None
    assert len(outcome.history) == 4  # 3 generations + final snapshot


def test_multi_asset_training_saves_one_artifact_per_ticker(models_dir):
    outcome = train_rule(quick_request("A1", "B2"))
    assert sorted(list_artifacts()) == ["A1", "B2"]
    a, b = load_artifact("A1"), load_artifact("B2")
    assert a["genome"] == b["genome"]                 # one shared evolved rule
    assert a["meta"]["multi_asset_partners"] == ["B2"]
    assert len(outcome.artifacts) == 2


def test_cli_and_web_paths_are_identical(models_dir):
    """Same request + seed must give the same genome (single shared pipeline)."""
    g1 = train_rule(quick_request("SAME", seed=3)).genome
    g2 = train_rule(quick_request("SAME", seed=3)).genome
    assert g1 == g2


def wait_for(jm: JobManager, timeout=60):
    deadline = time.time() + timeout
    while jm.is_busy() and time.time() < deadline:
        time.sleep(0.05)
    assert not jm.is_busy(), "job did not finish in time"
    return jm.current()


def test_job_manager_runs_to_completion(models_dir):
    jm = JobManager()
    jm.start(quick_request("JOB1"))
    assert jm.is_busy()
    snap = wait_for(jm)
    assert snap["status"] == "done"
    assert snap["result"]["genome"]
    assert snap["progress_pct"] == 100
    assert "JOB1" in list_artifacts()


def test_job_manager_refuses_concurrent_jobs(models_dir):
    jm = JobManager()
    jm.start(quick_request("JOB2"))
    with pytest.raises(RuntimeError):
        jm.start(quick_request("JOB3"))
    wait_for(jm)


def test_job_manager_captures_failure(models_dir, monkeypatch):
    import advisor.jobs as jobs_mod

    def boom(req, progress=None):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(jobs_mod, "train_rule", boom)
    jm = JobManager()
    jm.start(quick_request("FAIL"))
    snap = wait_for(jm)
    assert snap["status"] == "failed"
    assert "simulated failure" in snap["error"]
    assert not jm.is_busy()          # a failed job frees the slot


def test_job_history_keeps_finished_jobs(models_dir):
    jm = JobManager()
    jm.start(quick_request("H1"))
    wait_for(jm)
    jm.start(quick_request("H2"))
    wait_for(jm)
    assert [h["request"]["tickers"] for h in jm.history()] == [["H1"]]
    assert jm.current()["request"]["tickers"] == ["H2"]
