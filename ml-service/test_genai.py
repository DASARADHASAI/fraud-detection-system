"""
Tests for the Phase 4 GenAI layer.

The most important thing tested here is the FALLBACK path. That path is the
entire safety net of this phase, and it only executes when something has already
gone wrong -- so in normal use it is almost never exercised. A bug that lived
there survived several manual demos: LangGraph silently drops state keys that
are not declared on the state schema, so the template text arrived empty and
every fallback produced a BLANK explanation.

Run:
    ./venv/bin/python test_genai.py

Works with or without Ollama running -- it tests both modes deliberately.
"""

from __future__ import annotations

import sys

from explain import build_summary_sentence
from features import load_config
from genai_agent import (
    FraudGenAIAgent,
    find_contradictions,
    find_invented_amounts,
    find_speculation,
    ollama_is_available,
)


failures: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


# A realistic Phase 3 explanation to feed the agent.
SAMPLE_EXPLANATION = {
    "reasons": [
        {"feature": "orig_account_emptied", "value": 1.0, "impact": 0.09,
         "direction": "increases_risk",
         "reason": "this payment empties the sender's account completely"},
        {"feature": "amount", "value": 3244067.13, "impact": 0.03,
         "direction": "increases_risk",
         "reason": "the amount is Rs 3,244,067.13"},
    ],
    "protective_factors": [
        {"feature": "is_new_receiver", "value": 0.0, "impact": 0.01,
         "direction": "reduces_risk",
         "reason": "this receiver has been paid before"},
    ],
    "base_score": 0.5,
    "final_score": 0.96,
}


def main() -> int:
    cfg = load_config()

    # -------------------------------------------------------------------------
    print("\n1. Hallucinated amounts are detected")
    facts = "- Warning sign: the amount is Rs 3,244,067.13"
    check(find_invented_amounts("The customer moved Rs 3,244,067.13 today.", facts) == [],
          "an amount that IS in the facts is accepted")
    invented = find_invented_amounts("The customer moved Rs 50,000.00 today.", facts)
    check(invented != [], "an amount that is NOT in the facts is caught", str(invented))
    check(find_invented_amounts("Rs 3,244,067.13", "Rs 3244067.13") == [],
          "formatting differences do not count as invention")

    # -------------------------------------------------------------------------
    print("\n2. Invented criminal claims are detected")
    check(find_speculation("This suggests possible money laundering.") != [],
          "a money-laundering claim is caught")
    check(find_speculation("The account was emptied in one transfer.") == [],
          "an ordinary factual sentence is not flagged")

    # -------------------------------------------------------------------------
    print("\n3. Narration contradicting the decision is detected")
    check(find_contradictions("This payment is legitimate and low risk.", "BLOCK") != [],
          "calling a BLOCKED payment legitimate is caught")
    check(find_contradictions("This payment was cleared as normal.", "OTP_CONFIRM") != [],
          "calling a HELD payment cleared is caught")
    check(find_contradictions("This payment is highly suspicious.", "APPROVE") != [],
          "calling an APPROVED payment suspicious is caught")
    check(find_contradictions("The account was emptied completely.", "BLOCK") == [],
          "an accurate description of a blocked payment passes")

    # -------------------------------------------------------------------------
    print("\n4. The template fallback is never empty")
    # THE REGRESSION TEST. LangGraph only passes through state keys declared on
    # AgentState; facts_template was missing from the schema, so the fallback
    # text was dropped and every fallback returned "".
    agent = FraudGenAIAgent(cfg, verbose=False)
    for decision, score in [("BLOCK", 0.96), ("OTP_CONFIRM", 0.55), ("APPROVE", 0.05)]:
        result = agent.explain_transaction(SAMPLE_EXPLANATION, score, decision)
        check(bool(result["explanation"].strip()),
              f"{decision}: explanation is not empty",
              f"got {result['explanation']!r}")
        check(len(result["explanation"]) > 30,
              f"{decision}: explanation is a real sentence",
              f"got {result['explanation']!r}")
        check(bool(result["recommended_action"].strip()),
              f"{decision}: a recommended action is present")

    # -------------------------------------------------------------------------
    print("\n5. Forcing template mode still produces a usable explanation")
    # Simulate Ollama being down without stopping the real server.
    offline = FraudGenAIAgent(cfg, verbose=False)
    offline.available = False
    offline.llm = None
    result = offline.explain_transaction(SAMPLE_EXPLANATION, 0.96, "BLOCK")
    check(result["source"] == "template", "reports source as template when offline")
    check(len(result["explanation"]) > 30,
          "offline explanation is a real sentence", result["explanation"])
    expected = build_summary_sentence(SAMPLE_EXPLANATION, 0.96, "BLOCK")
    check(result["explanation"] == expected,
          "offline explanation matches the Phase 3 template exactly")

    # -------------------------------------------------------------------------
    print("\n6. The action always matches the decision, never the language model")
    for decision, must_contain in [("BLOCK", "block"), ("OTP_CONFIRM", "passcode"),
                                   ("APPROVE", "proceed")]:
        r = offline.explain_transaction(SAMPLE_EXPLANATION, 0.5, decision)
        check(must_contain in r["recommended_action"].lower(),
              f"{decision} produces the matching action",
              r["recommended_action"])

    # -------------------------------------------------------------------------
    print("\n7. The knowledge base loaded and retrieval works")
    kb = agent.knowledge_base
    check(kb.ready, f"knowledge base indexed ({len(kb.cases)} cases)")
    check(len(kb.cases) >= 8, f"at least 8 past cases present, found {len(kb.cases)}")
    outcomes = {c["outcome"] for c in kb.cases}
    check(outcomes == {"fraud", "legitimate"},
          "both fraud AND legitimate cases exist",
          "without legitimate cases every payment matches a fraud precedent")

    hits = kb.find_similar(["this payment empties the sender's account completely",
                            "the money is going to a receiver never used before"])
    check(len(hits) > 0, "a fraud-flavoured query retrieves cases")
    if hits:
        check(hits[0]["outcome"] == "fraud",
              "a fraud-flavoured query retrieves a FRAUD case first",
              f"got {hits[0]['case_id']}")
        check(all(h["similarity"] >= 0.15 for h in hits),
              "weak matches are filtered out")

    # -------------------------------------------------------------------------
    print("\n8. A degenerate explanation does not crash the agent")
    empty = {"reasons": [], "protective_factors": [], "base_score": 0.5,
             "final_score": 0.5}
    r = offline.explain_transaction(empty, 0.5, "OTP_CONFIRM")
    check(len(r["explanation"]) > 10, "an explanation with no factors still works",
          r["explanation"])

    # -------------------------------------------------------------------------
    print("\n9. Analyst chat answers in both modes")
    r = offline.chat("Why is a brand-new receiver risky?")
    check(len(r["answer"]) > 30, "offline chat returns the retrieved cases",
          r["answer"][:80])
    available, _ = ollama_is_available(cfg)
    if available:
        r = agent.chat("Does an OTP confirmation guarantee a payment is safe?")
        check(len(r["answer"]) > 30, "online chat returns an answer",
              r["answer"][:80])
    else:
        print("  SKIP  online chat (Ollama not running)")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
