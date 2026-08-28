"""
=============================================================================
RAZORPAY WEBHOOK -- a real payment gateway's events, scored by our engine.
=============================================================================

Phase 10 proved the engine speaks UPI. This proves something a panel can check:
that the payload comes from an actual payment company rather than from us.

    Razorpay  --POST-->  [ verify signature ]  -->  [ map ]  -->  scorer

>>> WHERE THIS WOULD ACTUALLY LIVE <<<
On the merchant's server, not inside the fraud engine. A merchant receives the
webhook, then asks its risk provider what to do. We mount it on the ML service
because that is where the mapping code is and running a third process for a
demonstration helps nobody -- but the boundary is real, and the endpoint calls
exactly the same scorer that /predict and /predict/upi call.

>>> THE MOMENT THAT MATTERS: authorized, NOT captured <<<
A gateway sends several events per payment. Only one of them is a decision:

    payment.authorized  the bank has held the funds, nobody has taken them yet.
                        With manual capture the merchant now chooses: capture,
                        or void. THIS is a point in the money path.
    payment.captured    the money has moved. A verdict here is a report, not a
                        decision, and can only trigger a refund afterwards.
    payment.failed      already dead.

So this receiver acts on `payment.authorized` and treats everything else as an
observation. Say this plainly if asked whether the webhook route can block: it
can, but only with manual capture enabled, and only at authorisation.

>>> THE HONEST PROBLEM, AND IT IS BIGGER THAN PHASE 10'S <<<
A merchant cannot see the payer's balance. Not "usually does not" -- cannot. It
is a customer of a different bank, and no gateway discloses it.

That is the mirror of the beneficiary problem in docs/INTEGRATION.md section 6,
and it is far more expensive, because the sender side is where this model's
evidence lives:

    errorBalanceOrig        0.2105
    absErrorBalanceOrig     0.1771
    amount_to_balance_ratio 0.1543
    orig_account_emptied    0.1411
    --------------------------------
                            0.6830   = 68.3% of total importance, all unavailable

We do NOT paper over that. The imputation is chosen from measurement, and the
degradation is measured rather than asserted -- see docs/INTEGRATION.md.

>>> WHY THE IMPUTATION IS "BOTH BALANCES ZERO" <<<
The tempting choice is a balance that reconciles exactly. Measured on the full
6.3M rows, that would have been a serious mistake:

    errorBalanceOrig == 0    in  99.5%  of fraud
                             in  45.7%  of legitimate merchant payments

A perfectly reconciling sender balance is close to the fraud signature itself.
Instead we use PaySim's own encoding for a sender balance that was not
recorded -- oldbalanceOrg = 0, newbalanceOrig = 0 -- which is the shape of 36.0%
of legitimate PAYMENT rows. Under it:

    errorBalanceOrig      = amount   (not the 99.5% fraud value)
    orig_account_emptied  = 0        (needs oldbalanceOrg > 0)
    amount_to_balance_ratio = 0      (division by zero -> 0)

We are not inventing a balance. We are using the dataset's existing convention
for "this balance is not known", which is precisely the merchant's situation.
=============================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import json

from live_features import ValidationError


# Razorpay sends money as an integer number of paise. 100 paise = 1 rupee.
# Getting this wrong is a silent 100x error in every amount, every threshold
# comparison and every score -- exactly the class of bug that passes testing
# because nothing raises.
PAISE_PER_RUPEE = 100

# The only event at which a decision is still possible. See the module docstring.
DECIDABLE_EVENT = "payment.authorized"

OBSERVED_EVENTS = {
    "payment.captured",   # money already moved; a verdict here is a report
    "payment.failed",
}

# We can only score what the UPI adapter understands.
SUPPORTED_METHOD = "upi"


# =============================================================================
# SECTION 1 -- Is this really from Razorpay?
# =============================================================================

def verify_signature(raw_body: bytes, signature: str | None, secret: str | None) -> bool:
    """
    Check the X-Razorpay-Signature header: HMAC-SHA256 of the RAW body.

    >>> THREE THINGS THAT ARE EASY TO GET WRONG <<<

    1. It must be the RAW bytes. Parsing the JSON and re-serialising it changes
       whitespace and key order, so the HMAC will not match and you will spend
       an afternoon believing the secret is wrong.

    2. The comparison must be constant-time. A plain `==` leaks how many leading
       bytes were correct through its timing, which is enough to forge a
       signature byte by byte.

    3. No secret means NO. An unverified webhook endpoint is an open door:
       anyone who learns the URL can post whatever payments they like and watch
       the fraud engine's verdicts. Failing closed is the only safe default.
    """
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())


# =============================================================================
# SECTION 2 -- Razorpay's shape -> UPI's shape
# =============================================================================

def _payment_entity(event: dict) -> dict:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValidationError(["'payload' is missing"])
    payment = payload.get("payment")
    if not isinstance(payment, dict):
        raise ValidationError(["'payload.payment' is missing"])
    entity = payment.get("entity")
    if not isinstance(entity, dict):
        raise ValidationError(["'payload.payment.entity' is missing"])
    return entity


def _payer_vpa(entity: dict) -> str | None:
    """Razorpay puts the payer's UPI address in `vpa`, or under `upi.vpa`."""
    upi = entity.get("upi") if isinstance(entity.get("upi"), dict) else {}
    return entity.get("vpa") or upi.get("vpa")


def razorpay_to_upi(event: dict, merchant_vpa: str) -> tuple[dict, dict]:
    """
    Map a Razorpay webhook event onto the UPI payload `upi_adapter` accepts.

    Returns (upi_payload, notes). `notes` records the event, whether it was a
    decidable moment, and every assumption the mapping had to make.

    Raises ValidationError -- becomes HTTP 400 -- when the event cannot be
    mapped. Same philosophy as the rest of the service: refuse loudly rather
    than score confidently from fields that were quietly defaulted.
    """
    if not isinstance(event, dict):
        raise ValidationError(["webhook body must be a JSON object"])

    name = str(event.get("event") or "").strip().lower()
    if not name:
        raise ValidationError(["'event' is required"])

    entity = _payment_entity(event)
    errors: list[str] = []
    assumptions: list[str] = []

    method = str(entity.get("method") or "").strip().lower()
    if method != SUPPORTED_METHOD:
        errors.append(
            f"only '{SUPPORTED_METHOD}' payments can be scored by this route, "
            f"got {method or 'nothing'!r}"
        )

    currency = str(entity.get("currency") or "INR").strip().upper()
    if currency != "INR":
        errors.append(f"only INR is supported, got {currency!r}")

    paise = entity.get("amount")
    if paise is None:
        errors.append("'amount' is required")
    elif not isinstance(paise, int) or isinstance(paise, bool):
        # Razorpay always sends an integer here. A float or a string means the
        # payload was reshaped somewhere, and a wrong unit is a 100x error.
        errors.append(f"'amount' must be an integer number of paise, got {paise!r}")
    elif paise < 0:
        errors.append(f"'amount' cannot be negative, got {paise}")

    payer = _payer_vpa(entity)
    if not payer:
        errors.append("the payer's UPI address is missing ('vpa' or 'upi.vpa')")

    if errors:
        raise ValidationError(errors)

    rupees = paise / PAISE_PER_RUPEE

    # >>> THE MERCHANT IS BLIND TO THE PAYER'S BALANCE <<<
    # See the module docstring for the measurement behind this choice.
    assumptions.append(
        "the payer's balance is not visible to a merchant; encoded as "
        "'balance not recorded' (both zero), the shape of 36.0% of legitimate "
        "merchant payments -- NOT a reconciling balance, which is the shape of "
        "99.5% of fraud"
    )

    upi_payload = {
        "txnId": entity.get("id"),
        "amount": {"value": f"{rupees:.2f}", "currency": "INR"},
        "payer": {
            "vpa": payer,
            "accountBalance": "0.00",
            "balanceAfter": "0.00",
        },
        "payee": {
            "vpa": merchant_vpa,
            "type": "MERCHANT",
            # A real acquirer knows its own merchant's category code. 5411 is
            # groceries; it is ours to state because we ARE the merchant.
            "mcc": "5411",
        },
    }

    notes = {
        "event": name,
        "decidable": name == DECIDABLE_EVENT,
        "razorpay_payment_id": entity.get("id"),
        "razorpay_order_id": entity.get("order_id"),
        "method": method,
        "amount_paise": paise,
        "amount_rupees": round(rupees, 2),
        "payer_vpa": payer,
        "merchant_vpa": merchant_vpa,
        "payer_balance_visible": False,
        "assumptions": assumptions,
    }

    if name == DECIDABLE_EVENT:
        notes["action"] = (
            "This is the authorisation moment. With manual capture the merchant "
            "can still capture or void, so a verdict here is a decision."
        )
    elif name in OBSERVED_EVENTS:
        notes["action"] = (
            "The money has already moved or already failed. A verdict here is a "
            "report, not a decision -- it can only prompt a refund afterwards."
        )
    else:
        notes["action"] = "Unrecognised event; scored as an observation only."

    return upi_payload, notes
