#!/usr/bin/env python3
"""Generate the six explanations used in the comprehension study (report §5.3).

Three actions (BUY / SELL / HOLD) x two explanation paths (LLM / template),
all produced by the real system from the deployed AAPL rule on fixed dates,
so every participant sees exactly the same stimuli.

    python scripts/make_study_stimuli.py            # LLM stimuli need Ollama running
    python scripts/make_study_stimuli.py --no-llm   # template stimuli only

Writes reports/study_stimuli.json and reports/Participant_Sheet.md.
The participant sheet shuffles the six stimuli into a fixed order (S1..S6)
and never reveals which path produced each one.
"""
from __future__ import annotations

import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from advisor.data import fetch_prices
from advisor.persistence import load_artifact
from advisor.rule_engine import decide
from advisor.explain import explain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (as-of date, user holds?) chosen so the deployed rule yields each action.
CASES = [
    ("BUY",  "2026-07-17", False),   # rule in market, user not holding
    ("SELL", "2025-11-06", True),    # rule in cash, user still holding
    ("HOLD", "2026-07-17", True),    # rule in market, user holding -> keep
]
# Fixed presentation order (path hidden from participants).
ORDER = [("HOLD", "template"), ("BUY", "llm"), ("SELL", "template"),
         ("HOLD", "llm"), ("BUY", "template"), ("SELL", "llm")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()

    art = load_artifact(args.ticker)
    genome = art["genome_obj"]
    prices = fetch_prices(args.ticker, start="2015-01-01").prices

    stimuli = {}
    for action, as_of, holds in CASES:
        hist = prices[prices.index <= pd.Timestamp(as_of)]
        d = decide(args.ticker, hist, genome, holds_position=holds)
        assert d.action == action, f"expected {action} on {as_of}, got {d.action}"
        for path in ("template", "llm"):
            if path == "llm" and args.no_llm:
                continue
            e = explain(d, use_llm=(path == "llm"))
            if path == "llm" and e["source"] != "llm":
                print(f"WARNING: LLM unavailable/failed guardrails for {action}; got template instead")
            stimuli[f"{action}_{path}"] = {
                "action": action, "path": path, "source_actual": e["source"],
                "as_of": as_of, "holds_position": holds, "text": e["text"],
                "reasons": d.reasons, "signal_changed": d.signal_changed,
            }

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    json.dump(stimuli, open(os.path.join(ROOT, "reports", "study_stimuli.json"), "w"), indent=2)

    lines = ["# Advisor explanations - participant sheet", "",
             "You will read six short messages from an automated investment advisor. Each message is about the same company's shares, on a particular day. "
             "For each message, answer the four questions on your response sheet. There are no right or wrong opinions; we are testing the advisor's writing, not you.", ""]
    for i, (action, path) in enumerate(ORDER, 1):
        key = f"{action}_{path}"
        if key not in stimuli:
            lines += [f"## Message S{i}", "", "_[LLM message: generate with Ollama running]_", ""]
            continue
        s = stimuli[key]
        pos = "You told the advisor that you **currently hold** these shares." if s["holds_position"] else "You told the advisor that you **do not currently hold** these shares."
        lines += [f"## Message S{i}", "", pos, "", "> " + s["text"].replace("\n", "\n> "), ""]
    open(os.path.join(ROOT, "reports", "Participant_Sheet.md"), "w").write("\n".join(lines))
    print(json.dumps({k: v["text"][:110] + "..." for k, v in stimuli.items()}, indent=1))
    print("Answer key (S1..S6):", [f"S{i}={a} ({p})" for i, (a, p) in enumerate(ORDER, 1)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
