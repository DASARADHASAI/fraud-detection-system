"""
=============================================================================
PHASE 8 -- Acceptance tests for the whole system.
=============================================================================

Every other test file checks one component. This one checks the PRODUCT: it
posts payments to the Java backend exactly as the dashboard does, lets them
travel through H2, the Python scorer, the rule layer and the step-up logic, and
asserts the decision a bank would actually see.

Run it with both services up:

    ./venv/bin/python app.py                     # terminal 1
    cd ../backend && mvn spring-boot:run         # terminal 2
    ./venv/bin/python acceptance_tests.py        # terminal 3

It writes docs/acceptance-tests.md from the real run - the document is
generated, never hand-written, so it cannot drift from what the code does.

>>> A NOTE ON WHAT "PASSING" MEANS HERE. <<<
These are behaviour tests, not accuracy metrics. They prove specific fraud
patterns produce specific automatic decisions. The accuracy numbers live in
docs/results.md, which train_model.py regenerates. Two documents because two
different scripts own them; conflating them would mean one overwrites the other.
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BACKEND = "http://localhost:8080/api"
ML = "http://localhost:5001"

CITIES = {
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Delhi": (28.7041, 77.1025),
}


# =============================================================================
# The test cases, as specified in the build prompt
# =============================================================================

def payment(amount, orig, dest, *, type="TRANSFER", old_org=None, old_dest=0.0,
            trusted=False, city=None):
    """Build one payment, keeping the balances arithmetically consistent."""
    old_org = amount if old_org is None else old_org
    body = {
        "type": type,
        "amount": amount,
        "nameOrig": orig,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": max(old_org - amount, 0.0),
        "nameDest": dest,
        "oldbalanceDest": old_dest,
        "newbalanceDest": old_dest + amount,
        "trustedDevice": trusted,
    }
    if city:
        lat, lon = CITIES[city]
        body.update({"location": city, "locationLat": lat, "locationLon": lon})
    return body


def build_cases(run: str) -> list[dict]:
    """
    One dict per test case.

    `expect` is a set of acceptable final decisions. Some cases have a single
    correct answer (R1 must block); others are specified as "flagged", which
    means anything except a silent approval.
    """
    return [
        {
            "id": "R0",
            "name": "Ordinary payment (control)",
            "why": "Proves the system is not simply blocking everything. Without "
                   "a negative control, every other pass here would be worthless.",
            "steps": [payment(2500, f"C-R0-{run}", f"M-SHOP-{run}",
                              type="PAYMENT", old_org=50000, old_dest=10000,
                              city="Mumbai")],
            "expect": {"APPROVE"},
            "expect_label": "APPROVE",
        },
        {
            "id": "R1",
            "name": "Huge amount, account emptied",
            "why": "The classic account-takeover shape: the entire balance moved "
                   "in one transfer to an account with no history.",
            "steps": [payment(3244067.13, f"C-R1-{run}", f"C-R1D-{run}",
                              old_org=3244067.13, city="Mumbai")],
            "expect": {"BLOCK"},
            "expect_label": "BLOCK",
        },
        {
            "id": "R2",
            "name": "Impossible travel",
            "why": "Two payments 1,650 km apart within the same hour. A physical "
                   "contradiction, so a hard rule decides it - not a probability.",
            "steps": [payment(2000, f"C-R2-{run}", f"M-R2A-{run}", type="PAYMENT",
                              old_org=60000, old_dest=8000, city="Mumbai"),
                      payment(2500, f"C-R2-{run}", f"M-R2B-{run}", type="PAYMENT",
                              old_org=58000, old_dest=9000, city="Kolkata")],
            "expect": {"BLOCK"},
            "expect_label": "BLOCK",
            "expect_rule": "impossible_travel",
        },
        {
            "id": "R3",
            "name": "Far above what this receiver normally collects",
            "why": "Tests the receiver-side behavioural features - the amount is "
                   "many times what the receiving account usually takes in.",
            "steps": [payment(900000, f"C-R3-{run}", f"C-R3D-{run}",
                              old_org=1200000, city="Mumbai")],
            "expect": {"BLOCK", "OTP_CONFIRM"},
            "expect_label": "flagged (not silently approved)",
        },
        {
            "id": "R5",
            "name": "Unfamiliar shape, unseen pattern",
            "why": "The case the unsupervised model exists for. See the caveat "
                   "below - a single live payment is weak evidence for this one.",
            "steps": [payment(1750000, f"C-R5-{run}", f"C-R5D-{run}",
                              type="CASH_OUT", old_org=1750000, city="Delhi")],
            "expect": {"BLOCK", "OTP_CONFIRM"},
            "expect_label": "flagged (not silently approved)",
        },
        {
            "id": "R6",
            "name": "Genuine but unusual, from a known device",
            "why": "A real customer making a large one-off payment. Must be "
                   "CONFIRMED, not blocked - blocking genuine payments is a "
                   "failure too, just a quieter one.",
            "steps": [payment(480000, f"C-R6-{run}", f"C-R6P-{run}",
                              old_org=1000000, trusted=True, city="Mumbai")],
            "expect": {"OTP_CONFIRM"},
            "expect_label": "OTP_CONFIRM (must NOT block)",
        },
        {
            "id": "R7",
            "name": "Rapid transfers from one account",
            "why": "An account being drained in a burst. The model cannot detect "
                   "this - PaySim contains no such pattern - so the rule layer "
                   "does. Five payments in quick succession.",
            # Balance deliberately stays high: each payment on its own is
            # unremarkable, so the model has no reason to stop it. Only the
            # BURST is suspicious. An earlier version let the balance run down
            # to zero, which made the last payment a full drain - the model
            # blocked it on its own and the test passed without proving the
            # rule did anything.
            "steps": [payment(40000, f"C-R7-{run}", f"C-R7D{i}-{run}",
                              old_org=500000, old_dest=5000,
                              city="Mumbai") for i in range(5)],
            "expect": {"BLOCK"},
            "expect_label": "BLOCK",
            "expect_rule": "rapid_sender_transfers",
            # The rule must be doing the work, not riding along behind the model.
            "expect_model_not": "BLOCK",
        },
    ]


# =============================================================================
# Running them
# =============================================================================

def services_up() -> tuple[bool, str]:
    try:
        health = requests.get(f"{BACKEND}/health", timeout=5).json()
    except requests.RequestException:
        return False, (f"No backend at {BACKEND}\n"
                       "  start it:  cd ../backend && mvn spring-boot:run")
    if health.get("mlService") != "ok":
        return False, ("The backend is up but cannot reach the ML service.\n"
                       "  start it:  ./venv/bin/python app.py")
    return True, "both services responding"


def run_case(case: dict) -> dict:
    """Fire every step; the LAST one carries the decision under test."""
    results = []
    for step in case["steps"]:
        response = requests.post(f"{BACKEND}/transactions", json=step, timeout=60)
        if response.status_code != 200:
            return {**case, "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "passed": False}
        results.append(response.json())
        time.sleep(0.25)          # let H2 settle between steps in a burst

    final = results[-1]
    decision = final["decision"]
    passed = decision in case["expect"]

    rule_ok = True
    if case.get("expect_rule"):
        rule_ok = final.get("ruleApplied") == case["expect_rule"]
        passed = passed and rule_ok

    # Some cases exist to prove a RULE catches what the model cannot. If the
    # model would have reached the same decision anyway, the case proves
    # nothing, so treat that as a failure rather than a pass.
    model_proves_rule = True
    if case.get("expect_model_not"):
        model_proves_rule = final.get("modelDecision") != case["expect_model_not"]
        passed = passed and model_proves_rule

    return {
        **case,
        "steps_run": len(results),
        "decision": decision,
        "model_decision": final.get("modelDecision"),
        "fraud_score": final.get("fraudScore"),
        "models_disagree": final.get("modelsDisagree"),
        "rule_applied": final.get("ruleApplied"),
        "quick_signal": final.get("quickSignalApplied"),
        "explanation": final.get("explanation", ""),
        "processing_ms": final.get("processingMillis"),
        "rule_ok": rule_ok,
        "model_proves_rule": model_proves_rule,
        "passed": passed,
        "transaction_id": final.get("id"),
        "error": None,
    }


# =============================================================================
# Writing the report
# =============================================================================

def write_report(results: list[dict], cfg_note: str) -> Path:
    path = Path(__file__).resolve().parent.parent / "docs" / "acceptance-tests.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for r in results if r["passed"])
    lines = [
        "# Phase 8 — Acceptance Tests",
        "",
        f"*Generated by `ml-service/acceptance_tests.py` on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}. Do not edit by hand — "
        f"re-run the script.*",
        "",
        f"**{passed} of {len(results)} cases passed.** Each payment was posted to "
        "the Java backend exactly as the dashboard posts it, and travelled the "
        "full path: H2 → Python scorer → rule layer → automatic routing.",
        "",
        cfg_note,
        "",
        "## Results",
        "",
        "| Case | Scenario | Expected | Actual | Score | Decided by |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        if r.get("error"):
            lines.append(f"| {r['id']} | {r['name']} | {r['expect_label']} | "
                         f"**ERROR** | — | {r['error'][:40]} |")
            continue
        mark = "✅" if r["passed"] else "❌"
        decided_by = (f"rule: `{r['rule_applied']}`" if r["rule_applied"]
                      else f"signal: `{r['quick_signal']}`" if r["quick_signal"]
                      else "the model")
        lines.append(
            f"| {r['id']} | {r['name']} | {r['expect_label']} | "
            f"{mark} {r['decision']} | {r['fraud_score']:.4f} | {decided_by} |")

    lines += ["", "## What each case proves", ""]
    for r in results:
        if r.get("error"):
            lines += [f"### {r['id']} — {r['name']}", "", f"**ERROR:** {r['error']}", ""]
            continue
        lines += [
            f"### {r['id']} — {r['name']}",
            "",
            r["why"],
            "",
            f"- **Payments sent:** {r['steps_run']}",
            f"- **Model score:** {r['fraud_score']:.4f} "
            f"(model alone would have said **{r['model_decision']}**)",
            f"- **Models disagreed:** {'yes' if r['models_disagree'] else 'no'}",
        ]
        if r["rule_applied"]:
            lines.append(f"- **Rule fired:** `{r['rule_applied']}` — this overrode the model")
        if r.get("expect_model_not"):
            lines.append(
                f"- **The rule was load-bearing:** the model alone said "
                f"`{r['model_decision']}`, so this decision came from the rule, "
                f"not from the model agreeing with it")
        if r["quick_signal"]:
            lines.append(f"- **Quick signal:** `{r['quick_signal']}`")
        lines += [
            f"- **Final decision:** **{r['decision']}** "
            f"({'as expected' if r['passed'] else 'NOT as expected'})",
            f"- **Decided in:** {r['processing_ms']} ms",
            "",
            f"> {r['explanation']}",
            "",
        ]

    lines += [
        "## Honest caveats",
        "",
        "**R4 is absent.** The build specification lists R1, R2, R3, R5, R6 and "
        "R7 — there is no R4. The numbering gap is inherited, not an omission "
        "here.",
        "",
        "**R5 is the weakest case in this table.** A single live payment cannot "
        "really demonstrate \"the unsupervised model catches an unseen pattern\", "
        "because nothing about one transaction is genuinely novel to a model "
        "already trained on millions like it. The real evidence for R5 is the "
        "holdout experiment in `docs/results.md`: every TRANSFER fraud was hidden "
        "from the Random Forest during training, and when shown those frauds it "
        "caught 96.0% while the Isolation Forest caught 47.7% — together 99.9%, "
        "with **32 frauds rescued** that the supervised model missed entirely. "
        "That is a measurement; this row is a demonstration.",
        "",
        "**R2 and R7 are decided by rules, not by the model, and that is the "
        "point.** PaySim has no location column at all, and measured across the "
        "full 6.3M rows it contains no mule-burst pattern — fraud receivers are "
        "*less* busy than legitimate ones (0.036 vs 0.189 prior payments per "
        "hour). A model cannot learn what is not in the data, so these typologies "
        "are encoded as explicit rules. The `Decided by` column above shows "
        "which mechanism actually made each call.",
        "",
        "**These are behaviour tests, not accuracy metrics.** They prove specific "
        "patterns produce specific decisions. For precision, recall, F1 and the "
        "confusion matrix see [results.md](results.md), which `train_model.py` "
        "regenerates.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "cd ml-service",
        "./venv/bin/python app.py                     # terminal 1",
        "cd ../backend && mvn spring-boot:run         # terminal 2",
        "cd ../ml-service && ./venv/bin/python acceptance_tests.py",
        "```",
        "",
        "Each run uses fresh account numbers, so results do not depend on what "
        "was in the database beforehand.",
        "",
    ]

    path.write_text("\n".join(lines))
    return path


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print("=" * 74)
    print("PHASE 8 -- ACCEPTANCE TESTS (full stack)")
    print("=" * 74)

    ok, message = services_up()
    print(f"\n{message}")
    if not ok:
        return 1

    try:
        requests.post(f"{ML}/reset", timeout=5)
    except requests.RequestException:
        pass

    health = requests.get(f"{BACKEND}/health", timeout=10).json()
    thresholds = health["thresholds"]
    cfg_note = (f"Thresholds in force: approve below **{thresholds['approveBelow']}**, "
                f"block above **{thresholds['blockAbove']}**, OTP timeout "
                f"**{thresholds['otpTimeoutSeconds']}s** — read from `config.yaml`, "
                f"which both services share.")
    print(cfg_note.replace("**", ""))

    run_id = datetime.now().strftime("%H%M%S")
    cases = build_cases(run_id)

    print(f"\nRunning {len(cases)} cases with account suffix -{run_id}\n")
    results = []
    for case in cases:
        result = run_case(case)
        results.append(result)

        if result.get("error"):
            print(f"  {result['id']}  ERROR  {result['error'][:60]}")
            continue

        mark = "PASS" if result["passed"] else "FAIL"
        decided_by = (result["rule_applied"] or result["quick_signal"] or "model")
        print(f"  {result['id']}  {mark}  expected {result['expect_label']:<34} "
              f"got {result['decision']:<12} score {result['fraud_score']:.4f}  "
              f"via {decided_by}")
        if not result["passed"]:
            print(f"        model alone said {result['model_decision']}; "
                  f"rule={result['rule_applied']}, signal={result['quick_signal']}")

    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 74)
    print(f"{passed} of {len(results)} cases passed")
    print("=" * 74)

    report = write_report(results, cfg_note)
    print(f"\nReport written to {report}")

    if passed < len(results):
        print("\nFailing cases are recorded in the report as failures. Do not")
        print("weaken the expectations to make them pass - either the system is")
        print("wrong or the expectation was.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
