"""
Tests for the Razorpay webhook receiver.

Two halves, same arrangement as the other suites:

  A. MAPPING AND SIGNATURES -- offline, no server. The important half. A webhook
     endpoint that trusts an unsigned body is an open door, and a paise/rupee
     mix-up is a silent 100x error in every amount the model ever sees.
  B. HTTP -- needs the server running with RAZORPAY_WEBHOOK_SECRET set.

Run:
    RAZORPAY_WEBHOOK_SECRET=test_secret ./venv/bin/python app.py   # terminal 1
    RAZORPAY_WEBHOOK_SECRET=test_secret ./venv/bin/python test_webhook.py
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import uuid

import requests

from live_features import ValidationError
from webhook_razorpay import (
    DECIDABLE_EVENT,
    PAISE_PER_RUPEE,
    razorpay_to_upi,
    verify_signature,
)


failures: list[str] = []
BASE_URL = "http://localhost:5001"
SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "test_secret")
MERCHANT = "unipaystore@razorpay"


def check(condition: bool, description: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


def rejects(event: dict, description: str, expect_in: str = "") -> None:
    try:
        razorpay_to_upi(event, MERCHANT)
    except ValidationError as e:
        joined = "; ".join(e.errors)
        if expect_in and expect_in.lower() not in joined.lower():
            check(False, description, f"rejected, but for: {joined}")
        else:
            check(True, description)
        return
    check(False, description, "accepted when it should have been rejected")


def event(**overrides) -> dict:
    """A payment.authorized event in Razorpay's real shape."""
    entity = {
        "id": "pay_" + uuid.uuid4().hex[:14],
        "entity": "payment",
        "amount": 4_850_000,          # paise -> Rs 48,500.00
        "currency": "INR",
        "status": "authorized",
        "order_id": "order_" + uuid.uuid4().hex[:14],
        "method": "upi",
        "vpa": "ravi.kumar@oksbi",
        "captured": False,
    }
    entity.update(overrides.pop("entity", {}))
    body = {
        "entity": "event",
        "account_id": "acc_TEST",
        "event": DECIDABLE_EVENT,
        "contains": ["payment"],
        "payload": {"payment": {"entity": entity}},
        "created_at": 1567692677,
    }
    body.update(overrides)
    return body


def sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# =============================================================================
# A1 -- Signature verification
# =============================================================================

def a_signatures() -> None:
    print("\n--- A1. Signatures ---")
    raw = json.dumps(event()).encode()
    good = sign(raw)

    check(verify_signature(raw, good, SECRET) is True,
          "a correctly signed body verifies")
    check(verify_signature(raw, good, "wrong_secret") is False,
          "the same body under a different secret does not")
    check(verify_signature(raw + b" ", good, SECRET) is False,
          "a body altered by one byte does not verify")
    check(verify_signature(raw, None, SECRET) is False,
          "a missing signature header does not verify")
    check(verify_signature(raw, good, None) is False,
          "no configured secret means nothing verifies -- fail closed")
    check(verify_signature(raw, good, "") is False,
          "an empty secret is treated as no secret")
    check(verify_signature(raw, "  " + good + "  ", SECRET) is True,
          "surrounding whitespace in the header is tolerated")
    check(verify_signature(raw, good.upper(), SECRET) is False,
          "a case-altered signature does not verify")


# =============================================================================
# A2 -- Money. Paise are not rupees.
# =============================================================================

def a_amounts() -> None:
    print("\n--- A2. Paise, not rupees ---")
    payload, notes = razorpay_to_upi(event(), MERCHANT)
    check(payload["amount"]["value"] == "48500.00",
          "4,850,000 paise becomes Rs 48,500.00", payload["amount"]["value"])
    check(notes["amount_paise"] == 4_850_000, "the original paise are reported")
    check(notes["amount_rupees"] == 48500.0, "and the rupee figure alongside it")
    check(PAISE_PER_RUPEE == 100, "the conversion constant is 100")

    payload, _ = razorpay_to_upi(event(entity={"amount": 1}), MERCHANT)
    check(payload["amount"]["value"] == "0.01",
          "one paise becomes Rs 0.01", payload["amount"]["value"])

    # Razorpay always sends an integer. Anything else means the payload was
    # reshaped in transit, and guessing the unit is how a 100x error happens.
    rejects(event(entity={"amount": 48500.0}),
            "a float amount is refused rather than guessed", expect_in="integer")
    rejects(event(entity={"amount": "4850000"}),
            "a string amount is refused rather than guessed", expect_in="integer")
    rejects(event(entity={"amount": -100}),
            "a negative amount is refused", expect_in="negative")
    rejects(event(entity={"amount": None}),
            "a missing amount is refused", expect_in="required")


# =============================================================================
# A3 -- Mapping
# =============================================================================

def a_mapping() -> None:
    print("\n--- A3. Razorpay's shape becomes UPI's shape ---")
    payload, notes = razorpay_to_upi(event(), MERCHANT)

    check(payload["payer"]["vpa"] == "ravi.kumar@oksbi",
          "the customer becomes the payer")
    check(payload["payee"]["vpa"] == MERCHANT,
          "the merchant becomes the payee")
    check(payload["payee"]["type"] == "MERCHANT",
          "a gateway payment is merchant-directed by definition")
    check(notes["decidable"] is True,
          "payment.authorized is flagged as a decidable moment")
    check(notes["razorpay_payment_id"].startswith("pay_"),
          "the Razorpay payment id is carried through")

    print("\n--- A4. The newer payload nests the VPA under `upi` ---")
    nested = event(entity={"vpa": None, "upi": {"vpa": "meera@ybl"}})
    payload, _ = razorpay_to_upi(nested, MERCHANT)
    check(payload["payer"]["vpa"] == "meera@ybl",
          "upi.vpa is read when the top-level vpa is absent")

    print("\n--- A5. Only UPI, only rupees ---")
    rejects(event(entity={"method": "card"}),
            "a card payment is refused by this route", expect_in="upi")
    rejects(event(entity={"method": "netbanking"}),
            "a netbanking payment is refused by this route", expect_in="upi")
    rejects(event(entity={"currency": "USD"}),
            "a non-rupee payment is refused", expect_in="INR")
    rejects(event(entity={"vpa": None}),
            "a payment with no payer address is refused", expect_in="UPI address")

    print("\n--- A6. Malformed envelopes ---")
    rejects({}, "an event with no name is refused", expect_in="'event' is required")
    rejects({"event": "payment.authorized"},
            "an event with no payload is refused", expect_in="payload")
    rejects({"event": "payment.authorized", "payload": {}},
            "an event with no payment is refused", expect_in="payment")
    rejects({"event": "payment.authorized", "payload": {"payment": {}}},
            "an event with no entity is refused", expect_in="entity")
    rejects([1, 2, 3], "a JSON array body is refused", expect_in="JSON object")


# =============================================================================
# A7 -- authorized decides; captured only reports
# =============================================================================

def a_decidability() -> None:
    print("\n--- A7. Which events are decisions and which are reports ---")
    _, notes = razorpay_to_upi(event(), MERCHANT)
    check(notes["decidable"] is True, "payment.authorized is a decision")
    check("capture" in notes["action"].lower(),
          "and says why -- capture or void is still open", notes["action"])

    for name in ("payment.captured", "payment.failed"):
        _, notes = razorpay_to_upi(event(event=name), MERCHANT)
        check(notes["decidable"] is False, f"{name} is NOT a decision")
        check("report" in notes["action"].lower(),
              f"{name} is described as a report", notes["action"])

    _, notes = razorpay_to_upi(event(event="payment.something_new"), MERCHANT)
    check(notes["decidable"] is False,
          "an unrecognised event is treated as an observation, not a decision")


# =============================================================================
# A8 -- The blindness, and the imputation chosen for it
# =============================================================================

def a_payer_blindness() -> None:
    print("\n--- A8. A merchant cannot see the payer's balance ---")
    payload, notes = razorpay_to_upi(event(), MERCHANT)

    check(notes["payer_balance_visible"] is False,
          "the response states the payer balance was not visible")
    check(any("not visible to a merchant" in a for a in notes["assumptions"]),
          "and the imputation is disclosed", str(notes["assumptions"]))

    # Both zero is PaySim's encoding for "balance not recorded". Measured on the
    # full file: that shape is 36.0% of legitimate merchant payments, whereas a
    # PERFECTLY RECONCILING balance is the shape of 99.5% of fraud. Imputing the
    # reconciling one would have pushed every gateway payment toward BLOCK.
    check(payload["payer"]["accountBalance"] == "0.00",
          "the payer balance is encoded as not-recorded, not invented")
    check(payload["payer"]["balanceAfter"] == "0.00",
          "and so is the post-balance")

    from upi_adapter import upi_to_transaction
    txn, _ = upi_to_transaction(payload)
    error_orig = txn["newbalanceOrig"] + txn["amount"] - txn["oldbalanceOrg"]
    check(abs(error_orig - txn["amount"]) < 0.01,
          "errorBalanceOrig lands on `amount`, NOT on the 99.5%-of-fraud value 0",
          f"got {error_orig}")
    emptied = (txn["newbalanceOrig"] == 0 and txn["oldbalanceOrg"] > 0
               and txn["amount"] > 0)
    check(not emptied,
          "orig_account_emptied stays false, so no drain signal is manufactured")


# =============================================================================
# PART B -- over HTTP
# =============================================================================

def server_is_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def post(body: dict, signature: str | None = "auto") -> requests.Response:
    raw = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if signature == "auto":
        headers["X-Razorpay-Signature"] = sign(raw)
    elif signature is not None:
        headers["X-Razorpay-Signature"] = signature
    return requests.post(f"{BASE_URL}/webhook/razorpay", data=raw,
                         headers=headers, timeout=30)


def b_http() -> None:
    print("\n--- B1. A genuine webhook is scored ---")
    r = post(event())
    if r.status_code == 503:
        print("  SKIPPED: server started without RAZORPAY_WEBHOOK_SECRET")
        print("  restart it with:  RAZORPAY_WEBHOOK_SECRET=test_secret ./venv/bin/python app.py")
        return

    check(r.status_code == 200, "a signed webhook is accepted", r.text[:200])
    body = r.json()
    check("fraud_score" in body and "decision" in body,
          "it comes back with a score and a decision")
    check(body["enforceable"] is True,
          "payment.authorized is marked enforceable")
    check(body["razorpay"]["payer_balance_visible"] is False,
          "the response admits the payer balance was imputed")

    print("\n--- B2. Forged and missing signatures are refused ---")
    r = post(event(), signature="0" * 64)
    check(r.status_code == 401, "a forged signature is a 401", f"got {r.status_code}")
    r = post(event(), signature=None)
    check(r.status_code == 401, "a missing signature is a 401", f"got {r.status_code}")

    # The body is signed, then altered. This is the attack the raw-bytes rule exists for.
    good = event()
    raw = json.dumps(good).encode()
    tampered = dict(good)
    tampered["payload"]["payment"]["entity"]["amount"] = 1
    r = requests.post(f"{BASE_URL}/webhook/razorpay",
                      data=json.dumps(tampered).encode(),
                      headers={"Content-Type": "application/json",
                               "X-Razorpay-Signature": sign(raw)}, timeout=15)
    check(r.status_code == 401,
          "a body altered after signing is a 401", f"got {r.status_code}")

    print("\n--- B3. Retries do not score twice ---")
    e = event()
    first = post(e).json()
    second = post(e).json()
    check(second.get("duplicate") is True,
          "a repeated payment id is reported as a duplicate")
    check(second["fraud_score"] == first["fraud_score"],
          "and answers with the original verdict, not a fresh score",
          f"{first['fraud_score']} vs {second['fraud_score']}")

    print("\n--- B4. captured is scored but marked unenforceable ---")
    body = post(event(event="payment.captured")).json()
    check(body["enforceable"] is False,
          "payment.captured comes back not enforceable")
    check("decision_note" in body,
          "and explains that the money has already moved")

    print("\n--- B5. Genuine but unmappable is a 400, not a 500 ---")
    for label, e in [
        ("a card payment", event(entity={"method": "card"})),
        ("a foreign currency", event(entity={"currency": "USD"})),
        ("a float amount", event(entity={"amount": 100.5})),
    ]:
        r = post(e)
        check(r.status_code == 400, f"{label} is a 400", f"got {r.status_code}")
        check("details" in r.json(), f"{label} explains itself in `details`")

    raw = b"not json at all"
    r = requests.post(f"{BASE_URL}/webhook/razorpay", data=raw,
                      headers={"X-Razorpay-Signature": sign(raw)}, timeout=15)
    check(r.status_code == 400, "an unparseable body is a 400", f"got {r.status_code}")


def b_diagnostics() -> None:
    print("\n--- B6. The receiver's own record, used to debug wiring ---")
    r = requests.get(f"{BASE_URL}/webhook/razorpay/recent", timeout=15)
    check(r.status_code == 200, "the diagnostic endpoint answers")
    body = r.json()
    check(body["configured"] is True,
          "it reports that a secret is configured")
    check("merchant_vpa" in body, "it names the merchant VPA being used")
    check(isinstance(body["events"], list), "it returns a list of events")

    # A scored event and a rejected one must be distinguishable -- that is the
    # whole point. "Nothing arrived" and "arrived but the secret is wrong" are
    # different faults with different fixes.
    before = len(requests.get(f"{BASE_URL}/webhook/razorpay/recent",
                              timeout=15).json()["events"])
    post(event())
    post(event(), signature="0" * 64)
    events = requests.get(f"{BASE_URL}/webhook/razorpay/recent",
                          timeout=15).json()["events"]
    check(len(events) >= before, "new events are recorded")

    outcomes = [e.get("outcome") for e in events]
    check("SCORED" in outcomes, "a scored webhook is recorded as SCORED")
    check("REJECTED" in outcomes, "a forged one is recorded as REJECTED")

    rejected = [e for e in events if e.get("outcome") == "REJECTED"]
    check(any(e.get("signature_present") is True for e in rejected),
          "a rejection records whether a signature was present at all",
          "this is how 'wrong secret' is told apart from 'never arrived'")

    scored = [e for e in events if e.get("outcome") == "SCORED"]
    check(all("decision" in e and "fraud_score" in e for e in scored),
          "scored entries carry the verdict")

    check(len(events) <= 50, "the log is bounded and cannot grow without limit",
          str(len(events)))


def main() -> int:
    print("=" * 70)
    print("PART A -- signatures and mapping (no server needed)")
    print("=" * 70)
    a_signatures()
    a_amounts()
    a_mapping()
    a_decidability()
    a_payer_blindness()

    print("\n" + "=" * 70)
    print("PART B -- HTTP behaviour")
    print("=" * 70)
    if server_is_up():
        b_http()
        b_diagnostics()
    else:
        print(f"\n  SKIPPED: no server on {BASE_URL}")

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
