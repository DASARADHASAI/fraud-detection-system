"""
Tests for the Phase 6 Spring Boot backend, driven over HTTP.

Written in Python purely so it can reuse this venv and the existing test style -
it exercises the JAVA service, not the Python one.

Run:
    ./venv/bin/python app.py                       # terminal 1
    cd ../backend && mvn spring-boot:run           # terminal 2
    ./venv/bin/python test_backend.py              # terminal 3

Some checks need a short OTP timeout to run in reasonable time. Set
`decision.otp_timeout_seconds: 8` in config.yaml and restart the backend before
running with --slow, otherwise the timeout check is skipped.
"""

from __future__ import annotations

import sys
import time

import requests

BACKEND = "http://localhost:8080/api"
ML = "http://localhost:5001"

failures: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


def submit(**overrides) -> dict:
    """Post a payment and return the decision."""
    txn = {
        "type": "PAYMENT", "amount": 2500.0, "nameOrig": "C-TEST",
        "oldbalanceOrg": 50000.0, "newbalanceOrig": 47500.0,
        "nameDest": "M-SHOP", "oldbalanceDest": 10000.0, "newbalanceDest": 12500.0,
    }
    txn.update(overrides)
    return requests.post(f"{BACKEND}/transactions", json=txn, timeout=30).json()


def backend_is_up() -> bool:
    try:
        return requests.get(f"{BACKEND}/health", timeout=5).ok
    except requests.RequestException:
        return False


def main() -> int:
    slow = "--slow" in sys.argv

    if not backend_is_up():
        print(f"No backend on {BACKEND}")
        print("Start it with:  cd ../backend && mvn spring-boot:run")
        return 1

    try:
        requests.post(f"{ML}/reset", timeout=5)
    except requests.RequestException:
        pass

    # -------------------------------------------------------------------------
    print("\n1. Both services are alive and share one config")
    health = requests.get(f"{BACKEND}/health", timeout=10).json()
    check(health["backend"] == "ok", "backend reports ok")
    check(health["mlService"] == "ok", "backend can reach the ML service",
          "start it with: ./venv/bin/python app.py")

    # The whole point of SharedConfig: if these drift, Java and Python quietly
    # disagree about what counts as fraud and nothing errors.
    ml_config = requests.get(f"{ML}/config", timeout=10).json()
    check(abs(health["thresholds"]["approveBelow"]
              - ml_config["decision_thresholds"]["approve_below"]) < 1e-9,
          "Java and Python agree on the APPROVE threshold",
          f"Java {health['thresholds']['approveBelow']} vs "
          f"Python {ml_config['decision_thresholds']['approve_below']}")
    check(abs(health["thresholds"]["blockAbove"]
              - ml_config["decision_thresholds"]["block_above"]) < 1e-9,
          "Java and Python agree on the BLOCK threshold")
    check("config.yaml" in health["configLoadedFrom"],
          "thresholds were read from the shared config.yaml, not hard-coded")

    # -------------------------------------------------------------------------
    print("\n2. Automatic routing, with no human step")
    ordinary = submit(nameOrig="C-R1")
    check(ordinary["decision"] == "APPROVE",
          "an ordinary payment is approved automatically", str(ordinary["decision"]))
    check("drain" in ordinary["explanation"] or "resembled" in ordinary["explanation"]
          or "does not" in ordinary["explanation"],
          "an approval is explained by what CLEARED it",
          f"got: {ordinary['explanation']}")
    check("signature of an automated account drain" not in ordinary["explanation"],
          "an approval is NOT explained by the worry that was outweighed",
          "verdict and evidence must point the same way")

    drained = submit(type="TRANSFER", amount=3244067.13, nameOrig="C-R2",
                     oldbalanceOrg=3244067.13, newbalanceOrig=0.0,
                     nameDest="C-R2DEST", oldbalanceDest=0.0, newbalanceDest=0.0)
    check(drained["decision"] == "BLOCK",
          "a fully drained account is blocked automatically")
    check(drained["fraudScore"] > 0.75, "and it scored above the block threshold")

    # -------------------------------------------------------------------------
    print("\n3. Quick signals settle what they can before asking the customer")
    blacklisted = submit(type="TRANSFER", amount=900000.0, nameOrig="C-R3",
                         oldbalanceOrg=1000000.0, newbalanceOrig=100000.0,
                         nameDest="C-KNOWN-FRAUD", oldbalanceDest=0.0,
                         newbalanceDest=900000.0)
    check(blacklisted["decision"] == "BLOCK",
          "a payment to a known-fraud account is blocked")
    check(blacklisted["quickSignalApplied"] == "known_fraud_receiver",
          "the known-fraud check is named in the response")
    check(blacklisted["modelDecision"] != "BLOCK",
          "and it escalated a decision the model had NOT blocked",
          f"model said {blacklisted['modelDecision']}")

    # -------------------------------------------------------------------------
    print("\n4. The OTP step-up replaces the human reviewer")
    uncertain = submit(type="TRANSFER", amount=480000.0, nameOrig="C-R4",
                       oldbalanceOrg=1000000.0, newbalanceOrig=520000.0,
                       nameDest="C-R4PAYEE", oldbalanceDest=0.0,
                       newbalanceDest=480000.0)
    check(uncertain["decision"] == "OTP_CONFIRM",
          "an uncertain payment triggers a customer challenge",
          f"got {uncertain['decision']} at score {uncertain['fraudScore']}")
    check("otp" in uncertain, "an OTP challenge is returned")

    if "otp" in uncertain:
        check(len(uncertain["otp"]["code"]) == 6, "the OTP code is 6 digits")
        check("Did you" in uncertain["otp"]["question"],
              "the customer is asked a plain yes/no question")

        confirmed = requests.post(
            f"{BACKEND}/transactions/{uncertain['id']}/otp",
            params={"confirmed": "true"}, timeout=30).json()
        check(confirmed["decision"] == "APPROVE",
              "answering YES approves the payment")

        # Answering twice must not be possible - a replayed confirmation would
        # let an attacker overturn a denial.
        again = requests.post(f"{BACKEND}/transactions/{uncertain['id']}/otp",
                              params={"confirmed": "false"}, timeout=30)
        check(again.status_code == 409,
              "answering the same challenge twice is refused",
              f"got HTTP {again.status_code}")

    denied = submit(type="TRANSFER", amount=480000.0, nameOrig="C-R5",
                    oldbalanceOrg=1000000.0, newbalanceOrig=520000.0,
                    nameDest="C-R5PAYEE", oldbalanceDest=0.0, newbalanceDest=480000.0)
    if denied["decision"] == "OTP_CONFIRM":
        answer = requests.post(f"{BACKEND}/transactions/{denied['id']}/otp",
                               params={"confirmed": "false"}, timeout=30).json()
        check(answer["decision"] == "BLOCK", "answering NO blocks the payment")

    # -------------------------------------------------------------------------
    print("\n5. The audit trail records every step")
    detail = requests.get(f"{BACKEND}/transactions/{uncertain['id']}", timeout=10).json()
    events = [e["event"] for e in detail["reviewLog"]]
    check("SCORED" in events, "the scoring is logged")
    check("OTP_SENT" in events, "the OTP challenge is logged")
    check("DECIDED" in events, "the final decision is logged")
    check("OTP_ANSWERED" in events, "the customer's answer is logged")
    check(any("no human" in e["detail"].lower() for e in detail["reviewLog"]),
          "the log states the decision was automatic")

    # -------------------------------------------------------------------------
    print("\n6. Validation rejects bad payments")
    for name, bad in [
        ("missing amount", {"type": "TRANSFER", "nameOrig": "C-X",
                            "oldbalanceOrg": 1.0, "newbalanceOrig": 1.0,
                            "nameDest": "C-Y", "oldbalanceDest": 1.0,
                            "newbalanceDest": 1.0}),
        ("negative amount", {"type": "TRANSFER", "amount": -5.0, "nameOrig": "C-X",
                             "oldbalanceOrg": 1.0, "newbalanceOrig": 1.0,
                             "nameDest": "C-Y", "oldbalanceDest": 1.0,
                             "newbalanceDest": 1.0}),
        ("blank sender", {"type": "TRANSFER", "amount": 5.0, "nameOrig": "",
                          "oldbalanceOrg": 1.0, "newbalanceOrig": 1.0,
                          "nameDest": "C-Y", "oldbalanceDest": 1.0,
                          "newbalanceDest": 1.0}),
    ]:
        resp = requests.post(f"{BACKEND}/transactions", json=bad, timeout=30)
        check(resp.status_code == 400, f"400 for {name}", f"got {resp.status_code}")

    # -------------------------------------------------------------------------
    print("\n7. Listing and statistics")
    check(len(requests.get(f"{BACKEND}/transactions", timeout=10).json()) > 0,
          "recent transactions can be listed")
    alerts = requests.get(f"{BACKEND}/alerts", timeout=10).json()
    check(all(a["finalDecision"] != "APPROVE" for a in alerts),
          "the alerts list contains no plain approvals")
    stats = requests.get(f"{BACKEND}/stats", timeout=10).json()
    check(stats["total"] > 0, "statistics are produced")
    check(stats["humanReviewers"] == 0,
          "the system reports zero human reviewers, as designed")

    # -------------------------------------------------------------------------
    print("\n8. Feedback is stored for future model improvement")
    fb = requests.post(f"{BACKEND}/transactions/{drained['id']}/feedback",
                       params={"wasFraud": "true", "note": "confirmed by customer"},
                       timeout=10).json()
    check(fb["confirmedFraud"] is True, "an outcome can be recorded")
    check("future scope" in fb["note"],
          "and it is honest that nothing retrains automatically")

    # -------------------------------------------------------------------------
    print("\n9. OTP timeout blocks the payment")
    if slow:
        pending = submit(type="TRANSFER", amount=480000.0, nameOrig="C-SILENT",
                         oldbalanceOrg=1000000.0, newbalanceOrig=520000.0,
                         nameDest="C-SILENTPAYEE", oldbalanceDest=0.0,
                         newbalanceDest=480000.0)
        if pending["decision"] == "OTP_CONFIRM":
            print("       waiting for the challenge to expire...")
            time.sleep(16)
            after = requests.get(f"{BACKEND}/transactions/{pending['id']}",
                                 timeout=10).json()
            check(after["transaction"]["finalDecision"] == "BLOCK",
                  "an unanswered challenge is blocked automatically",
                  "silence must never be treated as consent")
    else:
        print("  SKIP  needs --slow and otp_timeout_seconds set low in config.yaml")

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
