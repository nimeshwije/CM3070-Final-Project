"""Saving, loading and deleting evolved rules ("artifacts").

An artifact is just a small JSON file holding the evolved genome, the
training configuration, and the metrics recorded at training time. Because
the rule itself is five integers, "deploying" a rule means writing one tiny
file -- the online advisor loads it, fetches recent prices, and emits a
signal, with no heavy model anywhere.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from .genome import Genome

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Ticker names come in from the web form, so they're validated with a strict
# pattern before they ever touch a file path -- otherwise something like
# "../etc/passwd" could escape models/. There's a test for exactly that.
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")


def is_valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_RE.match(ticker or ""))


def artifact_path(ticker: str) -> str:
    if not is_valid_ticker(ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    safe = ticker.replace("/", "_").replace("^", "_")
    return os.path.join(MODELS_DIR, f"{safe}.json")


def save_artifact(
    ticker: str,
    genome: Genome,
    train_metrics: dict,
    test_metrics: dict,
    meta: dict | None = None,
) -> str:
    """Write the evolved rule and its metrics to models/<ticker>.json."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    payload = {
        "ticker": ticker,
        "genome": genome.to_dict(),
        "rule_text": genome.describe(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "meta": {
            **(meta or {}),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    path = artifact_path(ticker)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_artifact(ticker: str) -> dict:
    """Load an artifact; a ready-made Genome object is added under 'genome_obj'."""
    with open(artifact_path(ticker)) as f:
        payload = json.load(f)
    payload["genome_obj"] = Genome.from_dict(payload["genome"])
    return payload


def delete_artifact(ticker: str) -> bool:
    """Remove a saved rule. Returns True if there actually was a file to delete."""
    path = artifact_path(ticker)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def list_artifacts() -> list[str]:
    """All tickers that currently have a saved evolved rule."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(MODELS_DIR) if f.endswith(".json"))
