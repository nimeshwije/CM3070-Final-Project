import json

import pytest

from advisor.data import synthetic_gbm
from advisor.explain import _passes_guardrails, _template_explanation, explain
from advisor.genome import Genome
from advisor.rule_engine import decide

GENOME = Genome(short_window=10, long_window=40, rsi_period=14, rsi_buy=60, rsi_sell=95)


def make_decision():
    prices = synthetic_gbm(400, seed=21)
    return decide("SYN", prices, GENOME)


def test_decision_structure():
    d = make_decision()
    assert d.action in {"BUY", "SELL", "HOLD"}
    assert d.stance in {"in_market", "in_cash"}
    assert len(d.reasons) >= 2
    assert d.rule == GENOME.to_dict()
    json.dumps(d.to_dict())  # must be JSON-serialisable


def test_decision_is_deterministic():
    prices = synthetic_gbm(400, seed=22)
    d1 = decide("SYN", prices, GENOME)
    d2 = decide("SYN", prices, GENOME)
    assert d1.to_dict() == d2.to_dict()


def test_decide_requires_enough_history():
    short = synthetic_gbm(20, seed=23)
    with pytest.raises(ValueError):
        decide("SYN", short, GENOME)


def test_template_fallback_states_action_and_disclaimer():
    d = make_decision()
    text = _template_explanation(d)
    assert d.action in text.upper()
    assert "educational" in text


def test_explain_without_llm_uses_template():
    d = make_decision()
    out = explain(d, use_llm=False)
    assert out["source"] == "template"
    assert d.action in out["text"].upper()


def test_explain_with_unreachable_llm_falls_back():
    d = make_decision()
    out = explain(d, url="http://localhost:1/api/generate")  # nothing listens here
    assert out["source"] == "template"


def test_guardrails_reject_contradiction():
    d = make_decision()
    good = f"The rule says {d.action}. The trend and RSI conditions support this stance today, so no change is needed."
    assert _passes_guardrails(good, d)
    other = ({"BUY", "SELL", "HOLD"} - {d.action}).pop()
    bad = f"Although the rule says {d.action}, I recommend {other} now because markets look strong."
    assert not _passes_guardrails(bad, d)
    assert not _passes_guardrails("too short", d)
