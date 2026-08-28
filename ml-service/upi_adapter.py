"""
=============================================================================
UPI ADAPTER -- speaking the payment industry's language, not ours.
=============================================================================

The scorer has always accepted PaySim's schema: nameOrig, oldbalanceOrg,
newbalanceDest and so on. That schema is a research dataset's, not a payment
network's. No real system will ever send it.

This module is the translation layer. It takes a UPI-shaped payload -- the
shape a PSP switch, a payment gateway webhook or a bank's core banking system
actually emits -- and maps it onto the schema the model was trained on.

    UPI payload  --->  [ this module ]  --->  scorer  --->  decision

>>> WHY A SEPARATE MODULE AND NOT A FEW LINES IN app.py <<<
Because this is the part a bank would rewrite and nothing else. The claim the
project makes is "any payment platform can adopt this engine by changing the
adapter". That claim is only credible if the adapter is genuinely the only
thing that would change -- so it lives in its own file, imports no model, and
holds no scoring logic whatsoever. It is a pure function from one dict to
another.

>>> THE HONEST PROBLEM: A BANK CANNOT SEE THE OTHER SIDE'S BALANCE <<<
PaySim gives every row four balance columns -- both sides, before and after.
Real UPI does not, and cannot. In a live payment the remitter bank sees its
own customer's account and nothing else; the beneficiary's balance sits in a
different bank's core, behind NPCI, and is never disclosed. It arrives only
for an *on-us* payment, where both accounts happen to be at the same bank.

We refuse to paper over that. Two rules:

  1. When the payee balance is genuinely absent we impute it as a *clean*
     credit -- oldbalanceDest = 0, newbalanceDest = amount -- so the balance
     equation reconciles exactly and the imputation contributes NO fraud
     signal. Imputing 0/0 instead would have been much worse: in PaySim an
     unchanged destination balance is part of the drain signature, so every
     off-us payment would have been handed a free push toward BLOCK.

  2. Whatever we assumed is reported back to the caller in `assumptions`. A
     score computed from partly imputed evidence must say so.

This costs less accuracy than it looks. Of the model's top ten features by
importance, the four largest are all sender-side --

    errorBalanceOrig        0.2105
    absErrorBalanceOrig     0.1771
    amount_to_balance_ratio 0.1543
    orig_account_emptied    0.1411
    --------------------------------
                            0.6830   = 68.3% of total importance

-- and every one of those is computable from what the remitter bank genuinely
holds: its own customer's balance before and after, and the amount. The
evidence a real bank is missing is not the evidence this model leans on.

>>> WHAT WE DO NOT DO <<<
This adapter does not score, does not decide, does not call the rule layer and
does not talk to the database. It maps fields and reports what it assumed.
=============================================================================
"""

from __future__ import annotations

import math
import re

from live_features import ValidationError


# =============================================================================
# SECTION 1 -- What a VPA looks like
# =============================================================================
#
# A UPI address is `handle@psp` -- ravi.kumar@oksbi, 9876543210@ybl. NPCI
# permits letters, digits, dot, hyphen and underscore in the handle.
#
# We validate the shape but never the PSP suffix against a list. New PSPs are
# onboarded regularly, and a hard-coded whitelist would start rejecting valid
# customers the first time NPCI adds one.
# =============================================================================

VPA_PATTERN = re.compile(r"^[A-Za-z0-9._-]{2,256}@[A-Za-z][A-Za-z0-9.]{1,64}$")

# UPI transaction types we accept, mapped to the model's vocabulary.
#
#   P2P -- person to person. Money leaves a customer account for another
#          customer account. In PaySim terms that is a TRANSFER.
#   P2M -- person to merchant. A purchase. PaySim calls that a PAYMENT, and
#          the model learned PAYMENT as the *least* risky type, which matches
#          reality: merchant VPAs are registered, KYC'd and traceable.
#
# The distinction is not cosmetic. `is_risky_type` is the 5th most important
# feature in the model (0.0640), so mapping every UPI payment to TRANSFER
# would quietly inflate the score on every legitimate shop purchase in India.
PAYEE_TYPE_TO_MODEL_TYPE = {
    "PERSON": "TRANSFER",
    "P2P": "TRANSFER",
    "MERCHANT": "PAYMENT",
    "P2M": "PAYMENT",
    "ENTITY": "PAYMENT",
}

# A payee carrying a real merchant category code is a merchant, whatever the
# payload called itself. "0000" is UPI's placeholder for "not a merchant".
NON_MERCHANT_MCC = {"", "0000", "none", "null"}


# =============================================================================
# SECTION 2 -- Small helpers
# =============================================================================

def _amount(value, field: str, errors: list[str]) -> float | None:
    """
    UPI sends money as a string -- {"value": "48500.00", "currency": "INR"}.
    Accept the string, the number, or the wrapper object.
    """
    if isinstance(value, dict):
        value = value.get("value")
    if value is None:
        errors.append(f"'{field}' is required")
        return None
    try:
        amount = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        errors.append(f"'{field}' must be a number, got {value!r}")
        return None
    if not math.isfinite(amount):
        errors.append(f"'{field}' must be a finite number, got {value!r}")
        return None
    if amount < 0:
        errors.append(f"'{field}' cannot be negative, got {amount}")
        return None
    return amount


def _optional_amount(value, field: str, errors: list[str]) -> float | None:
    """Same as _amount, but absence is allowed and is not an error."""
    if isinstance(value, dict):
        value = value.get("value")
    if value is None or value == "":
        return None
    return _amount(value, field, errors)


def _vpa(block: dict, side: str, errors: list[str]) -> str | None:
    """Pull and validate the VPA out of a payer/payee block."""
    if not isinstance(block, dict):
        errors.append(f"'{side}' must be an object")
        return None
    raw = block.get("vpa") or block.get("va") or block.get("address")
    if not raw:
        errors.append(f"'{side}.vpa' is required")
        return None
    vpa = str(raw).strip().lower()
    if not VPA_PATTERN.match(vpa):
        errors.append(f"'{side}.vpa' is not a valid UPI address, got {raw!r}")
        return None
    return vpa


def _currency(payload: dict, errors: list[str]) -> None:
    """
    Reject anything that is not rupees.

    The model was trained on rupee amounts and every threshold in config.yaml
    is a rupee figure. Scoring a dollar amount against them would not fail --
    it would silently score a $500 payment as though it were Rs 500. That is
    exactly the class of bug that never shows up in testing.
    """
    amount = payload.get("amount")
    currency = None
    if isinstance(amount, dict):
        currency = amount.get("currency")
    currency = currency or payload.get("currency")
    if currency and str(currency).strip().upper() != "INR":
        errors.append(f"only INR is supported, got {currency!r}")


def _model_type(payee: dict, payload: dict) -> str:
    """Decide whether this is a TRANSFER (P2P) or a PAYMENT (P2M)."""
    mcc = str(payee.get("mcc") or payee.get("merchantCategoryCode") or "").strip()
    if mcc.lower() not in NON_MERCHANT_MCC:
        return "PAYMENT"

    declared = (
        payee.get("type")
        or payload.get("txnType")
        or payload.get("transactionType")
        or ""
    )
    return PAYEE_TYPE_TO_MODEL_TYPE.get(str(declared).strip().upper(), "TRANSFER")


def _step(payload: dict) -> int | None:
    """
    PaySim's clock: one step = one hour. `hour_of_day` and `is_night` are real
    model features, so the hour a payment happened genuinely moves the score.

    A caller may pin it explicitly (the demo does). Otherwise we leave it
    unset and let the scorer fall back to wall-clock hours, rather than
    inventing a value here and hiding where it came from.
    """
    for field in ("step", "hour"):
        if payload.get(field) is not None:
            try:
                return int(payload[field])
            except (TypeError, ValueError):
                return None
    return None


# =============================================================================
# SECTION 3 -- The mapping itself
# =============================================================================

def upi_to_transaction(payload: dict) -> tuple[dict, dict]:
    """
    Map a UPI payload onto the scorer's schema.

    Returns (transaction, notes).

      transaction -- ready to hand straight to the scorer, exactly as though
                     it had arrived on /predict.
      notes       -- what the adapter did: the UPI reference, the payee class
                     it inferred, and every assumption it had to make.

    Raises ValidationError -- becomes HTTP 400 -- when the payload cannot be
    mapped. Same philosophy as the rest of the service: refuse loudly rather
    than score confidently from fields that were quietly defaulted to zero.
    """
    if not isinstance(payload, dict):
        raise ValidationError(["request body must be a JSON object"])

    errors: list[str] = []
    assumptions: list[str] = []

    payer = payload.get("payer") if isinstance(payload.get("payer"), dict) else {}
    payee = payload.get("payee") if isinstance(payload.get("payee"), dict) else {}

    _currency(payload, errors)

    payer_vpa = _vpa(payload.get("payer"), "payer", errors)
    payee_vpa = _vpa(payload.get("payee"), "payee", errors)
    amount = _amount(payload.get("amount"), "amount", errors)

    # --- The sender's side. This the remitter bank always has. ---
    payer_before = _optional_amount(
        payer.get("accountBalance", payer.get("balance")),
        "payer.accountBalance", errors,
    )
    payer_after = _optional_amount(
        payer.get("balanceAfter"), "payer.balanceAfter", errors,
    )

    if payer_before is None:
        errors.append(
            "'payer.accountBalance' is required -- the remitter bank always "
            "holds its own customer's balance, and four of the model's five "
            "strongest features are derived from it"
        )

    if amount is not None and payer_before is not None and payer_after is None:
        payer_after = max(0.0, payer_before - amount)
        assumptions.append(
            "payer.balanceAfter was not supplied; computed as balance - amount"
        )

    # --- The receiver's side. Usually invisible. See the module docstring. ---
    payee_before = _optional_amount(
        payee.get("accountBalance", payee.get("balance")),
        "payee.accountBalance", errors,
    )
    payee_after = _optional_amount(
        payee.get("balanceAfter"), "payee.balanceAfter", errors,
    )

    beneficiary_visible = payee_before is not None
    if not beneficiary_visible and amount is not None:
        # Impute a clean credit: the balance equation reconciles exactly, so
        # the imputation contributes no fraud signal in either direction.
        payee_before, payee_after = 0.0, amount
        assumptions.append(
            "beneficiary balance not visible to the remitter bank (off-us "
            "payment); imputed as a clean credit so it adds no fraud signal"
        )
    elif beneficiary_visible and payee_after is None and amount is not None:
        payee_after = payee_before + amount
        assumptions.append(
            "payee.balanceAfter was not supplied; computed as balance + amount"
        )

    if errors:
        raise ValidationError(errors)

    model_type = _model_type(payee, payload)

    transaction = {
        "type": model_type,
        "amount": amount,
        # VPAs are the account identifiers. They are stable across payments,
        # which is what makes receiver history work -- and unlike PaySim's
        # minted-per-row customer ids, they are stable on the sender side too.
        "nameOrig": payer_vpa,
        "oldbalanceOrg": payer_before,
        "newbalanceOrig": payer_after,
        "nameDest": payee_vpa,
        "oldbalanceDest": payee_before,
        "newbalanceDest": payee_after,
    }

    step = _step(payload)
    if step is not None:
        transaction["step"] = step

    # --- Optional signals the rule layer uses, none of them model features ---
    device = payload.get("device") if isinstance(payload.get("device"), dict) else {}
    if device.get("trusted") is not None:
        transaction["trusted_device"] = bool(device["trusted"])

    location = device.get("location") if isinstance(device.get("location"), dict) else {}
    location = location or (
        payload.get("location") if isinstance(payload.get("location"), dict) else {}
    )
    if location.get("city"):
        transaction["location"] = str(location["city"]).strip()
    for source, target in (("lat", "location_lat"), ("lon", "location_lon")):
        if location.get(source) is not None:
            try:
                transaction[target] = float(location[source])
            except (TypeError, ValueError):
                pass

    notes = {
        "upi_txn_id": str(
            payload.get("txnId") or payload.get("txnid") or payload.get("id") or ""
        ) or None,
        "payer_vpa": payer_vpa,
        "payee_vpa": payee_vpa,
        "payee_class": "merchant" if model_type == "PAYMENT" else "person",
        "mapped_type": model_type,
        "beneficiary_balance_visible": beneficiary_visible,
        "assumptions": assumptions,
    }
    return transaction, notes
