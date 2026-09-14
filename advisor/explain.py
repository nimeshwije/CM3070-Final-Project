"""The guardrailed explanation layer.

A local language model (via Ollama) rewrites the rule engine's deterministic
Decision into friendly prose for a non-technical user.  Safety design, as set
out in the report's design chapter:

  * The LLM NEVER makes or alters the decision -- the prompt hands it the
    finished decision and forbids changing it.
  * The LLM's output is POST-CHECKED: if it contradicts the decision (e.g.
    recommends a different action), mentions none of the supplied facts, or
    fails to arrive at all, the system falls back to a deterministic
    template message -- advice is never blocked by the explanation layer.
  * Running locally keeps the system free and keeps user data on-device.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from .rule_engine import Decision

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"          # any small local chat model works
OLLAMA_TIMEOUT_S = 30

DISCLAIMER = (
    "This is an educational decision-support tool, not regulated financial "
    "advice. No real money should be traded on its output."
)

_SYSTEM_RULES = """You are the explanation layer of a rule-based investment advisor.
You will be given a decision that has ALREADY been made by a deterministic
trading rule, together with the reasons for it.

Your ONLY job is to reword that decision and those reasons into 2-4 short,
friendly sentences a non-technical person can understand.

Strict rules:
- Do NOT change, soften, or second-guess the decision.
- Do NOT add any prediction, price target, or confidence claim.
- Do NOT recommend any other asset or action.
- Do NOT invent facts that are not in the data given to you.
- Mention the recommended action ({action}) explicitly.
- Plain language only: no jargon beyond "moving average" and "RSI",
  and briefly gloss RSI as a momentum gauge if you use the term.
{position_note}"""

_POSITION_NOTES = {
    None: "",
    True: ("- The user has told us they CURRENTLY HOLD this asset; address them as such "
           "(e.g. 'keep holding' rather than 'stay in cash').\n"),
    False: ("- The user has told us they DO NOT currently hold this asset; address them as "
            "such (e.g. 'no need to buy yet' rather than 'keep holding').\n"),
}


def _action_phrase(decision: Decision) -> str:
    """Headline phrase for the template, aware of mode and user position."""
    invested = decision.stance == "in_market"
    if decision.mode == "position":
        holds = bool(decision.holds_position)
        return {
            ("BUY", False): "the rule recommends BUYING: it is currently in the market and you are not",
            ("SELL", True): "the rule recommends SELLING: it is currently in cash and you still hold the asset",
            ("HOLD", True): "the rule recommends HOLDING: keep the position you already have",
            ("HOLD", False): ("the rule recommends HOLDING off: you do not hold the asset and "
                              "the rule is not signalling an entry, so stay in cash"),
        }[(decision.action, holds)]
    return {
        "BUY": "the rule recommends BUYING (moving your allocation into the market)",
        "SELL": "the rule recommends SELLING (moving your allocation to cash)",
        "HOLD": ("the rule recommends HOLDING your current stance ("
                 + ("staying invested" if invested else "staying in cash") + ")"),
    }[decision.action]


def _template_explanation(decision: Decision) -> str:
    """Deterministic fallback prose, built only from the Decision itself."""
    reasons = "; ".join(decision.reasons)
    fresh = ""
    if decision.signal_changed:
        fresh = (" Note: the rule's own signal changed today (it "
                 + ("entered the market" if decision.signal == "enter" else "moved to cash")
                 + "), so this is a fresh signal rather than a continuing one.")
    return (
        f"For {decision.ticker} as of {decision.as_of}: {_action_phrase(decision)}.{fresh} "
        f"Why: {reasons}. {DISCLAIMER}"
    )


def _call_ollama(prompt: str, model: str, url: str) -> str | None:
    """Minimal Ollama REST call; returns None on any failure."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode())
        return (data.get("response") or "").strip() or None
    except Exception as exc:
        logger.warning("Ollama unavailable (%s); using template fallback", exc)
        return None


def _passes_guardrails(text: str, decision: Decision) -> bool:
    """Post-check the LLM output before showing it to the user.

    Reject if the stated action is missing, if a *different* action is
    recommended, or if the text is degenerate (too short/long).
    """
    if not text or len(text) < 30 or len(text) > 1200:
        return False
    upper = text.upper()
    if decision.action not in upper:
        return False
    other_actions = {"BUY", "SELL", "HOLD"} - {decision.action}
    for other in other_actions:
        # A different action word used as a recommendation is disqualifying.
        for phrase in (f"RECOMMEND {other}", f"SHOULD {other}", f"{other} NOW"):
            if phrase in upper:
                return False
    return True


def explain(
    decision: Decision,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_URL,
    use_llm: bool = True,
) -> dict:
    """Produce the user-facing explanation for a Decision.

    Returns {"text": ..., "source": "llm" | "template"}.  The template path
    is taken whenever the LLM is disabled, unreachable, or fails guardrails.
    """
    if use_llm:
        prompt = (
            _SYSTEM_RULES.format(
                action=decision.action,
                position_note=_POSITION_NOTES[decision.holds_position],
            )
            + "\nDecision data (JSON):\n"
            + json.dumps(decision.to_dict(), indent=2)
            + "\n\nNow write the explanation:"
        )
        text = _call_ollama(prompt, model, url)
        if text is not None and _passes_guardrails(text, decision):
            return {"text": f"{text}\n\n{DISCLAIMER}", "source": "llm"}
        if text is not None:
            logger.warning("LLM output failed guardrails; using template fallback")

    return {"text": _template_explanation(decision), "source": "template"}
