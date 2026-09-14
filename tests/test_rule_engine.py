import json

import pytest

from advisor.backtest import compute_positions
from advisor.data import synthetic_gbm
from advisor.explain import _passes_guardrails, _template_explanation, explain
from advisor.genome import Genome
from advisor.rule_engine import Decision, decide, position_action

GENOME = Genome(short_window=10, long_window=40, rsi_period=14, rsi_buy=60, rsi_sell=95)


def make_decision(holds_position=None):
    prices = synthetic_gbm(400, seed=21)
    return decide("SYN", prices, GENOME, holds_position=holds_position)


def test_decision_structure():
    d = make_decision()
    assert d.action in {"BUY", "SELL", "HOLD"}
    assert d.stance in {"in_market", "in_cash"}
    assert d.mode == "signal" and d.holds_position is None
    assert d.signal in {"enter", "exit", "none"}
    assert d.signal_changed == (d.signal != "none")
    assert len(d.reasons) >= 2
    assert d.rule == GENOME.to_dict()
    json.dumps(d.to_dict())  # must be JSON-serialisable


# ------------------------------------------------- position-aware advice
def test_position_action_table():
    assert position_action("in_market", True) == "HOLD"
    assert position_action("in_market", False) == "BUY"
    assert position_action("in_cash", True) == "SELL"
    assert position_action("in_cash", False) == "HOLD"


@pytest.mark.parametrize("seed", [21, 22, 23, 24, 25])
def test_position_mode_is_stance_vs_position(seed):
    prices = synthetic_gbm(400, seed=seed)
    sig = decide("SYN", prices, GENOME)
    for holds in (True, False):
        d = decide("SYN", prices, GENOME, holds_position=holds)
        assert d.mode == "position" and d.holds_position is holds
        assert d.action == position_action(d.stance, holds)
        # Everything the rule itself computed is identical across modes:
        assert (d.stance, d.signal, d.signal_changed, d.indicators) == \
               (sig.stance, sig.signal, sig.signal_changed, sig.indicators)
        assert d.reasons[:-1] == sig.reasons          # plus one position reason
        assert ("you currently hold" if holds else "you do not currently hold") in d.reasons[-1]


def test_signal_mode_matches_backtester_transition():
    """Signal mode must report exactly the transition the backtester trades."""
    prices = synthetic_gbm(400, seed=22)
    pos = compute_positions(prices, GENOME)
    d = decide("SYN", prices, GENOME)
    expected = {1.0: "BUY", -1.0: "SELL", 0.0: "HOLD"}[float(pos.iloc[-1] - pos.iloc[-2])]
    assert d.action == expected


def test_signal_changed_flag_on_a_transition_day():
    """Truncate the series at a day the rule flips: both modes must flag it."""
    prices = synthetic_gbm(600, seed=30)
    pos = compute_positions(prices, GENOME)
    flips = [i for i in range(GENOME.long_window + 10, len(pos)) if pos.iloc[i] != pos.iloc[i - 1]]
    assert flips, "synthetic series produced no trades; choose another seed"
    i = flips[-1]
    upto = prices.iloc[: i + 1]
    sig = decide("SYN", upto, GENOME)
    assert sig.signal_changed and sig.action in {"BUY", "SELL"}
    entered = pos.iloc[i] > pos.iloc[i - 1]
    assert sig.signal == ("enter" if entered else "exit")
    # Position mode on the same day: the flag survives, the action depends on the user.
    holder = decide("SYN", upto, GENOME, holds_position=True)
    nobody = decide("SYN", upto, GENOME, holds_position=False)
    assert holder.signal_changed and nobody.signal_changed
    if entered:
        assert (holder.action, nobody.action) == ("HOLD", "BUY")
    else:
        assert (holder.action, nobody.action) == ("SELL", "HOLD")


def _decision(action, stance, holds, signal="none"):
    return Decision(ticker="T", action=action, stance=stance, as_of="2026-01-01", price=1.0,
                    reasons=["r1"], mode="position", holds_position=holds,
                    signal=signal, signal_changed=(signal != "none"))


def test_template_explanation_is_position_aware():
    assert "keep the position" in _template_explanation(_decision("HOLD", "in_market", True))
    assert "stay in cash" in _template_explanation(_decision("HOLD", "in_cash", False))
    assert "BUYING" in _template_explanation(_decision("BUY", "in_market", False))
    assert "SELLING" in _template_explanation(_decision("SELL", "in_cash", True))
    fresh = _template_explanation(_decision("BUY", "in_market", False, signal="enter"))
    assert "signal changed today" in fresh and "entered the market" in fresh
    assert "signal changed" not in _template_explanation(_decision("BUY", "in_market", False))


def test_position_mode_explanations_pass_guardrails():
    for holds in (True, False):
        d = make_decision(holds_position=holds)
        out = explain(d, use_llm=False)
        assert out["source"] == "template"
        assert _passes_guardrails(out["text"], d)


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
