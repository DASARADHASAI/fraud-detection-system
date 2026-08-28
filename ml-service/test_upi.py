"""
Tests for the UPI adapter.

Two halves, same arrangement as test_api.py:

  A. MAPPING -- runs offline, no server needed. The important half. If the
     adapter maps a UPI payload onto the wrong internal fields, the score is
     silently computed from the wrong evidence and nothing reports an error.
  B. HTTP -- needs the server running on port 5001.

The single most important check here is EQUIVALENCE: the same payment,
expressed once in UPI's language and once in PaySim's, must produce the same
score. If it does not, the adapter is not an adapter -- it is a second model.

Run:
    ./venv/bin/python app.py          # in one terminal
    ./venv/bin/python test_upi.py     # in another
"""

from __future__ import annotations

import sys

import requests

from live_features import ValidationError, validate_transaction
from upi_adapter import upi_to_transaction


failures: list[str] = []
BASE_URL = "http://localhost:5001"


def check(condition: bool, description: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


def rejects(payload: dict, description: str, expect_in: str = "") -> None:
    """Assert the adapter refuses a payload, and refuses it for the right reason."""
    try:
        upi_to_transaction(payload)
    except ValidationError as e:
        joined = "; ".join(e.errors)
        if expect_in and expect_in.lower() not in joined.lower():
            check(False, description, f"rejected, but for: {joined}")
        else:
            check(True, description)
        return
    check(False, description, "payload was accepted when it should have been rejected")


# A person-to-person payment, in the shape a PSP switch emits.
P2P = {
    "txnId": "AXI4f9c2b1e7a3d",
    "txnType": "PAY",
    "amount": {"value": "48500.00", "currency": "INR"},
    "payer": {
        "vpa": "ravi.kumar@oksbi",
        "name": "Ravi Kumar",
        "accountBalance": "62000.00",
    },
    "payee": {
        "vpa": "quickcash9931@ybl",
        "name": "Quick Cash",
        "type": "PERSON",
    },
    "device": {"id": "dev-88", "trusted": True},
    "step": 400,
}


def a_basic_mapping() -> None:
    print("\n--- A1. A well-formed P2P payment maps onto the model's schema ---")
    txn, notes = upi_to_transaction(P2P)

    check(txn["type"] == "TRANSFER", "P2P becomes TRANSFER", txn["type"])
    check(txn["amount"] == 48500.0, "amount parsed from the string form", txn["amount"])
    check(txn["nameOrig"] == "ravi.kumar@oksbi", "payer VPA becomes nameOrig")
    check(txn["nameDest"] == "quickcash9931@ybl", "payee VPA becomes nameDest")
    check(txn["oldbalanceOrg"] == 62000.0, "payer balance becomes oldbalanceOrg")
    check(txn["newbalanceOrig"] == 13500.0,
          "newbalanceOrig computed as balance - amount", txn["newbalanceOrig"])
    check(txn["step"] == 400, "an explicit step is carried through")
    check(txn["trusted_device"] is True, "device.trusted reaches the rule layer")
    check(notes["upi_txn_id"] == "AXI4f9c2b1e7a3d", "the UPI reference is reported back")
    check(notes["payee_class"] == "person", "payee classified as a person")

    print("\n--- A2. The mapped transaction is one the scorer accepts ---")
    try:
        validate_transaction(txn)
        check(True, "mapped output passes the scorer's own validation")
    except ValidationError as e:
        check(False, "mapped output passes the scorer's own validation",
              "; ".join(e.errors))

    print("\n--- A3. VPAs are normalised, so history keys are stable ---")
    shouty = dict(P2P, payer=dict(P2P["payer"], vpa="  Ravi.Kumar@OKSBI  "))
    txn2, _ = upi_to_transaction(shouty)
    check(txn2["nameOrig"] == "ravi.kumar@oksbi",
          "VPA lowercased and trimmed", txn2["nameOrig"])


def a_merchant_classification() -> None:
    print("\n--- A4. P2M is a PAYMENT, not a TRANSFER ---")
    # This matters: is_risky_type is the model's 5th strongest feature, so
    # mapping every shop purchase to TRANSFER would inflate the score on
    # ordinary retail traffic.
    merchant = dict(P2P, payee={"vpa": "bigbazaar@paytm", "type": "MERCHANT"})
    txn, notes = upi_to_transaction(merchant)
    check(txn["type"] == "PAYMENT", "declared MERCHANT becomes PAYMENT", txn["type"])
    check(notes["payee_class"] == "merchant", "payee classified as a merchant")

    print("\n--- A5. A real MCC outranks whatever the payload declared ---")
    lying = dict(P2P, payee={"vpa": "store@axl", "type": "PERSON", "mcc": "5411"})
    txn, _ = upi_to_transaction(lying)
    check(txn["type"] == "PAYMENT",
          "a payee with a merchant category code is a merchant", txn["type"])

    placeholder = dict(P2P, payee={"vpa": "friend@ybl", "type": "PERSON", "mcc": "0000"})
    txn, _ = upi_to_transaction(placeholder)
    check(txn["type"] == "TRANSFER",
          "MCC 0000 is UPI's placeholder and does not mean merchant", txn["type"])


def a_invisible_beneficiary() -> None:
    print("\n--- A6. The beneficiary balance a real bank cannot see ---")
    txn, notes = upi_to_transaction(P2P)

    check(notes["beneficiary_balance_visible"] is False,
          "off-us payment is reported as having no beneficiary visibility")
    check(any("beneficiary balance" in a for a in notes["assumptions"]),
          "the imputation is disclosed in assumptions", str(notes["assumptions"]))

    # The point of imputing a CLEAN credit rather than 0/0: the destination
    # balance equation must reconcile exactly, so the imputation contributes
    # no fraud signal in either direction.
    residual = txn["oldbalanceDest"] + txn["amount"] - txn["newbalanceDest"]
    check(abs(residual) < 1e-9,
          "imputed destination balances reconcile exactly (no invented signal)",
          f"residual {residual}")
    check(txn["newbalanceDest"] == txn["amount"],
          "imputed as a clean credit of the full amount")

    print("\n--- A7. On-us payments, where the bank CAN see both sides ---")
    on_us = dict(P2P, payee=dict(P2P["payee"], accountBalance="15000.00"))
    txn, notes = upi_to_transaction(on_us)
    check(notes["beneficiary_balance_visible"] is True,
          "on-us payment reports the beneficiary as visible")
    check(txn["oldbalanceDest"] == 15000.0, "the real beneficiary balance is used")
    check(txn["newbalanceDest"] == 63500.0,
          "credit applied to the real balance", txn["newbalanceDest"])
    check(not any("not visible" in a for a in notes["assumptions"]),
          "no invisibility assumption is claimed when the balance was supplied")

    print("\n--- A8. A supplied balanceAfter is trusted over our arithmetic ---")
    explicit = dict(P2P, payer=dict(P2P["payer"], balanceAfter="9999.00"))
    txn, notes = upi_to_transaction(explicit)
    check(txn["newbalanceOrig"] == 9999.0,
          "the switch's own post-balance wins", txn["newbalanceOrig"])
    check(not any("balanceAfter was not supplied" in a and "payer" in a
                  for a in notes["assumptions"]),
          "no assumption is claimed for a field that was supplied")


def a_amount_forms() -> None:
    print("\n--- A9. The several shapes UPI sends money in ---")
    for label, amount in [
        ("wrapper object", {"value": "48500.00", "currency": "INR"}),
        ("bare string", "48500.00"),
        ("bare number", 48500.0),
        ("grouped string", "48,500.00"),
    ]:
        txn, _ = upi_to_transaction(dict(P2P, amount=amount))
        check(txn["amount"] == 48500.0, f"amount as a {label}", str(txn["amount"]))


def a_rejections() -> None:
    print("\n--- A10. What the adapter refuses, and why ---")
    rejects(dict(P2P, amount={"value": "100.00", "currency": "USD"}),
            "a non-rupee amount is refused, not silently scored as rupees",
            expect_in="INR")
    rejects(dict(P2P, payer={"vpa": "not-a-vpa", "accountBalance": "100"}),
            "a malformed payer VPA is refused", expect_in="valid UPI address")
    rejects(dict(P2P, payee={"vpa": "missing-at-sign"}),
            "a malformed payee VPA is refused", expect_in="valid UPI address")
    rejects({k: v for k, v in P2P.items() if k != "payer"},
            "an absent payer block is refused", expect_in="payer")
    rejects(dict(P2P, payer={"vpa": "ravi.kumar@oksbi"}),
            "an absent payer balance is refused, never defaulted to zero",
            expect_in="accountBalance")
    rejects(dict(P2P, amount="-500"),
            "a negative amount is refused", expect_in="negative")
    rejects(dict(P2P, amount="abc"),
            "a non-numeric amount is refused", expect_in="number")
    rejects([1, 2, 3], "a JSON array body is refused", expect_in="JSON object")

    print("\n--- A11. An unpinned clock is left unpinned ---")
    # The adapter must not invent an hour. hour_of_day and is_night are real
    # model features; a silently-invented value would be evidence we made up.
    no_step = {k: v for k, v in P2P.items() if k != "step"}
    txn, _ = upi_to_transaction(no_step)
    check("step" not in txn,
          "no step is fabricated; the scorer's own wall-clock fallback applies")


def a_equivalence() -> None:
    print("\n--- A12. EQUIVALENCE: the adapter is a translation, not a model ---")
    # The same payment written both ways must produce identical internal fields.
    # If this drifts, /predict and /predict/upi are scoring different things.
    upi_form = dict(P2P, payee=dict(P2P["payee"], accountBalance="15000.00"))
    mapped, _ = upi_to_transaction(upi_form)

    paysim_form = {
        "step": 400, "type": "TRANSFER", "amount": 48500.0,
        "nameOrig": "ravi.kumar@oksbi",
        "oldbalanceOrg": 62000.0, "newbalanceOrig": 13500.0,
        "nameDest": "quickcash9931@ybl",
        "oldbalanceDest": 15000.0, "newbalanceDest": 63500.0,
        "trusted_device": True,
    }
    scoring_fields = [
        "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "step",
    ]
    for field in scoring_fields:
        check(mapped[field] == paysim_form[field],
              f"{field} matches the hand-written equivalent",
              f"adapter={mapped[field]!r} expected={paysim_form[field]!r}")


# =============================================================================
# PART B -- over HTTP
# =============================================================================

def server_is_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def reset() -> None:
    """Clear the in-memory history so a receiver's past does not leak between cases."""
    requests.post(f"{BASE_URL}/reset", timeout=10)


def b_http() -> None:
    print("\n--- B1. /predict/upi answers with a decision ---")
    reset()
    r = requests.post(f"{BASE_URL}/predict/upi", json=P2P, timeout=30)
    check(r.status_code == 200, "a valid UPI payload scores", r.text[:200])
    body = r.json()
    check("fraud_score" in body and "decision" in body,
          "the response carries a score and a decision")
    check(body["upi"]["payee_vpa"] == "quickcash9931@ybl",
          "the upi block echoes the mapping")
    check(body["upi"]["beneficiary_balance_visible"] is False,
          "the upi block admits the beneficiary balance was imputed")
    check(len(body["shap_reasons"]) > 0, "SHAP reasons come back as usual")

    print("\n--- B2. Same payment, both languages, same score ---")
    # The strongest claim this project makes about the adapter. Each side gets
    # a clean history, so any difference is the adapter's fault and nothing else.
    on_us = dict(P2P, payee=dict(P2P["payee"], accountBalance="15000.00"))
    paysim_form = {
        "step": 400, "type": "TRANSFER", "amount": 48500.0,
        "nameOrig": "ravi.kumar@oksbi",
        "oldbalanceOrg": 62000.0, "newbalanceOrig": 13500.0,
        "nameDest": "quickcash9931@ybl",
        "oldbalanceDest": 15000.0, "newbalanceDest": 63500.0,
        "trusted_device": True,
    }
    reset()
    via_upi = requests.post(f"{BASE_URL}/predict/upi", json=on_us, timeout=30).json()
    reset()
    via_native = requests.post(f"{BASE_URL}/predict", json=paysim_form, timeout=30).json()

    check(via_upi["fraud_score"] == via_native["fraud_score"],
          "identical fraud score through both doors",
          f"upi={via_upi['fraud_score']} native={via_native['fraud_score']}")
    check(via_upi["decision"] == via_native["decision"],
          "identical decision through both doors",
          f"upi={via_upi['decision']} native={via_native['decision']}")
    check(via_upi["models_disagree"] == via_native["models_disagree"],
          "identical disagreement flag through both doors")

    print("\n--- B3. An account drain in UPI's language is still blocked ---")
    reset()
    drain = {
        "txnId": "AXIdrain001",
        "amount": {"value": "3244067.13", "currency": "INR"},
        "payer": {"vpa": "victim.account@okhdfcbank",
                  "accountBalance": "3244067.13", "balanceAfter": "0.00"},
        "payee": {"vpa": "mule8842@ybl", "type": "PERSON"},
        "device": {"trusted": False},
        "step": 400,
    }
    body = requests.post(f"{BASE_URL}/predict/upi", json=drain, timeout=30).json()
    check(body["decision"] == "BLOCK",
          "whole-balance transfer to an unknown payee is blocked",
          f"decision={body['decision']} score={body['fraud_score']}")

    print("\n--- B4. An ordinary shop payment is not blocked ---")
    reset()
    shop = {
        "txnId": "AXIshop001",
        "amount": {"value": "480.00", "currency": "INR"},
        "payer": {"vpa": "ravi.kumar@oksbi", "accountBalance": "62000.00"},
        "payee": {"vpa": "bigbazaar@paytm", "type": "MERCHANT", "mcc": "5411"},
        "device": {"trusted": True},
        "step": 400,
    }
    body = requests.post(f"{BASE_URL}/predict/upi", json=shop, timeout=30).json()
    check(body["decision"] == "APPROVE",
          "a small merchant payment from a trusted device is approved",
          f"decision={body['decision']} score={body['fraud_score']}")

    print("\n--- B5. Bad payloads come back as 400, not 500 ---")
    for label, payload in [
        ("a foreign currency", dict(P2P, amount={"value": "10", "currency": "USD"})),
        ("a malformed VPA", dict(P2P, payee={"vpa": "nope"})),
        ("a missing payer balance", dict(P2P, payer={"vpa": "a.b@oksbi"})),
    ]:
        r = requests.post(f"{BASE_URL}/predict/upi", json=payload, timeout=15)
        check(r.status_code == 400, f"{label} is a 400", f"got {r.status_code}")
        check("details" in r.json(), f"{label} explains itself in `details`")

    r = requests.post(f"{BASE_URL}/predict/upi",
                      data="not json", headers={"Content-Type": "application/json"},
                      timeout=15)
    check(r.status_code == 400, "an unparseable body is a 400", f"got {r.status_code}")


def main() -> int:
    print("=" * 70)
    print("PART A -- mapping (no server needed)")
    print("=" * 70)
    a_basic_mapping()
    a_merchant_classification()
    a_invisible_beneficiary()
    a_amount_forms()
    a_rejections()
    a_equivalence()

    print("\n" + "=" * 70)
    print("PART B -- HTTP behaviour")
    print("=" * 70)
    if server_is_up():
        b_http()
    else:
        print(f"\n  SKIPPED: no server on {BASE_URL}")
        print("  Start it with:  ./venv/bin/python app.py")

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
