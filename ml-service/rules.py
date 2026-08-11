"""
=============================================================================
THE RULE LAYER -- hard constraints that outrank the model.
=============================================================================

A machine learning model can only detect patterns that exist in its training
data. Two things follow from that, and this file handles both.

  1. SOME PATTERNS ARE PHYSICALLY IMPOSSIBLE, NOT MERELY UNLIKELY.
     A customer cannot pay in Mumbai and then in Kolkata fifteen minutes later.
     That is not a probability to be weighed against other evidence; it is a
     contradiction. Handing it to a model that might score it 0.13 and approve
     it would be a category error.

  2. SOME PATTERNS ARE ABSENT FROM THE DATA ENTIRELY.
     We measured all 6,362,620 PaySim rows for the classic "mule account
     collects a burst of transfers" typology:

         prior payments to this receiver     legitimate    fraud
           within 1 hour                         0.189     0.036
           within 4 hours                        0.453     0.085
           within 24 hours                       1.132     0.338
         receiver reused at any point            57.2%     37.0%

     Every window points the same way, and only 0.58% of fraud rows had a
     receiver collecting 2+ payments within an hour. PaySim fraud targets
     fresh, QUIET receivers. The model therefore learned -- correctly, for this
     dataset -- that a busy receiver is LESS suspicious.

     That pattern is real in actual banking. It is simply not in this
     simulator. No feature engineering recovers a signal that is absent and
     points backwards, so we encode the typology as an explicit rule and say so
     in the report, rather than pretending the model detects it.

>>> RULES MAY ONLY ESCALATE. <<<
APPROVE -> OTP_CONFIRM -> BLOCK, never the reverse. A rule that could soften a
decision would let a crude heuristic overrule a measured model. The same
principle governs models_disagree in Phase 2.

Java mirrors this logic in Phase 6. Both sides read the thresholds from
config.yaml so the two implementations cannot drift apart.
=============================================================================
"""

from __future__ import annotations

# Ordered from most permissive to most restrictive. Used to decide which of
# several triggered rules wins, and to guarantee a rule never softens anything.
SEVERITY = {"APPROVE": 0, "OTP_CONFIRM": 1, "BLOCK": 2}


def _fmt(n: int) -> str:
    return f"{n} payment" + ("" if n == 1 else "s")


def check_impossible_travel(travel: dict, cfg: dict) -> dict | None:
    """
    Could this customer physically have got here from their last payment?

    `travel` is computed in live_features.py, which owns the location history.
    This function only decides what to DO about it.
    """
    rule_cfg = cfg.get("rules", {}).get("impossible_travel", {})
    if not rule_cfg.get("enabled", True):
        return None
    if not travel.get("checked") or not travel.get("impossible"):
        return None

    return {
        "rule": "impossible_travel",
        "action": rule_cfg.get("action", "BLOCK"),
        "detail": travel.get("detail", "physically impossible journey"),
        "evidence": {
            "from": travel.get("from"),
            "to": travel.get("to"),
            "distance_km": travel.get("distance_km"),
            "hours_elapsed": travel.get("hours_elapsed"),
            "implied_speed_kmph": travel.get("implied_speed_kmph"),
            "limit_kmph": travel.get("limit_kmph"),
        },
        # Wording for the customer. Never mention "the model" -- this decision
        # did not come from the model.
        "customer_message": (
            f"This payment was made in {travel.get('to')}, but the previous "
            f"payment on this account was in {travel.get('from')} only "
            f"{travel.get('hours_elapsed')} hours earlier. That journey is not "
            f"physically possible, so the payment was stopped."
        ),
    }


def check_rapid_sender_transfers(history, txn: dict, cfg: dict) -> dict | None:
    """
    An account being emptied in a burst -- Phase 8's test case R7.

    Fires on 0.000% of historical PaySim rows (the busiest sender in the whole
    file had exactly one prior payment within an hour), so this is dormant on
    the dataset and active in a live demo. That is the intended behaviour: it
    adds no false positives to the reported metrics, and still catches a drain
    in progress.
    """
    rule_cfg = cfg.get("rules", {}).get("rapid_sender_transfers", {})
    if not rule_cfg.get("enabled", True):
        return None

    window = int(rule_cfg.get("window_steps", 1))
    limit = int(rule_cfg.get("max_allowed", 3))
    account = txn.get("nameOrig")
    if not account:
        return None

    recent = history.count_within(account, int(txn["step"]), window)
    if recent <= limit:
        return None

    return {
        "rule": "rapid_sender_transfers",
        "action": rule_cfg.get("action", "BLOCK"),
        "detail": (f"this account already made {_fmt(recent)} in the previous "
                   f"{window} hour(s), above the limit of {limit}"),
        "evidence": {"recent_transactions": recent, "window_hours": window,
                     "limit": limit},
        "customer_message": (
            f"This account has made {_fmt(recent)} in the last {window} hour(s), "
            f"which is far more than usual. Payments have been stopped as a "
            f"precaution while the account is checked."
        ),
    }


def check_rapid_receiver_collection(history, txn: dict, cfg: dict) -> dict | None:
    """
    One account collecting payments from many senders -- the mule pattern.

    Escalates to OTP rather than blocking, because the suspicious behaviour
    belongs to the RECEIVER while the payment being held belongs to a sender who
    has done nothing wrong. Blocking them would punish the wrong person.
    """
    rule_cfg = cfg.get("rules", {}).get("rapid_receiver_collection", {})
    if not rule_cfg.get("enabled", True):
        return None

    window = int(rule_cfg.get("window_steps", 1))
    limit = int(rule_cfg.get("max_allowed", 4))
    account = txn.get("nameDest")
    if not account:
        return None

    recent = history.count_within(account, int(txn["step"]), window)
    if recent <= limit:
        return None

    return {
        "rule": "rapid_receiver_collection",
        "action": rule_cfg.get("action", "OTP_CONFIRM"),
        "detail": (f"the receiving account has taken in {_fmt(recent)} in the "
                   f"previous {window} hour(s), above the limit of {limit}"),
        "evidence": {"recent_transactions": recent, "window_hours": window,
                     "limit": limit},
        "customer_message": (
            f"The account you are paying has received {_fmt(recent)} from "
            f"different people in the last {window} hour(s). Please confirm you "
            f"meant to send this."
        ),
    }


def evaluate(txn: dict, signals: dict, history, cfg: dict,
             model_decision: str) -> dict:
    """
    Run every rule and work out the final decision.

    Returns the decision after rules, which rule (if any) changed it, and the
    full list of everything that fired -- the dashboard and the audit log both
    want to show all of it, not just the winner.
    """
    triggered = []
    for result in (
        check_impossible_travel(signals.get("impossible_travel", {}), cfg),
        check_rapid_sender_transfers(history, txn, cfg),
        check_rapid_receiver_collection(history, txn, cfg),
    ):
        if result:
            triggered.append(result)

    final = model_decision
    applied = None
    for rule in triggered:
        # Escalate only. A rule must never soften what the model decided.
        if SEVERITY.get(rule["action"], 0) > SEVERITY.get(final, 0):
            final = rule["action"]
            applied = rule["rule"]

    return {
        "decision": final,
        "model_decision": model_decision,
        "rule_applied": applied,
        "rules_triggered": triggered,
        "overridden": applied is not None,
    }
