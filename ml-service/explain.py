"""
=============================================================================
PHASE 3 -- Explainable AI. Why did the model decide that?
=============================================================================

A fraud score of 0.94 is useless on its own. Nobody can act on it, nobody can
challenge it, and a regulator will not accept "the computer said so". This file
turns the score into reasons a human can read.

HOW SHAP WORKS (the one-paragraph version):
Imagine the model starts at a neutral opinion -- "this transaction is average,
50/50". Then each feature nudges that opinion up or down. "The account was
emptied" might push it +0.30 toward fraud. "The receiver is well established"
might pull it -0.05 back. SHAP calculates the exact size of every nudge, and
those nudges always sum to the final score. So we can rank them and report the
biggest ones.

The value of SHAP over a plain feature-importance list: importance tells you
what matters ON AVERAGE, across all transactions. SHAP tells you what mattered
FOR THIS ONE. Two transactions with the same score can have completely
different explanations.

RUN IT:
    ./venv/bin/python explain.py

Requires the models from Phase 2 (run train_model.py first).

SPEED: about 4 ms per transaction, so this is safe to run inside the real-time
/predict endpoint in Phase 5. The slow GenAI narration in Phase 4 is what gets
called separately.
=============================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from features import load_config, resolve_path, build_feature_matrix


# =============================================================================
# SECTION 1 -- Turning a feature name + value into an English phrase
# =============================================================================
#
# The model thinks in column names like "errorBalanceOrig". A customer needs
# "the sender's balance doesn't add up". Each entry below is a small function
# that takes the feature's REAL (unscaled) value and returns a phrase.
#
# We pass the real value, not the scaled one, so the phrase can quote actual
# figures: "moves 100% of the balance" rather than "scaled value 3.2".
# =============================================================================

def _money(v: float) -> str:
    """Format an amount the way a bank statement would."""
    return f"Rs {v:,.2f}"


PHRASE_BUILDERS = {
    # --- Balance arithmetic: the strongest signals in PaySim ---
    #
    # Read these carefully -- the direction is counter-intuitive, and it is a
    # property of the DATA, not a mistake. In PaySim a fraudulent drain leaves
    # the books reconciling to the exact rupee, while ordinary payments are
    # frequently inconsistent. So "the arithmetic is perfect" is a fraud signal
    # here. Say that explicitly rather than leaving a reader baffled.
    "errorBalanceOrig": lambda v: (
        f"the sender's balance does not reconcile (off by {_money(abs(v))})"
        if abs(v) > 1 else
        "the sender's balance reconciles to the exact rupee, the signature of "
        "an automated account drain"
    ),
    "absErrorBalanceOrig": lambda v: (
        f"a {_money(v)} accounting mismatch on the sender's side"
        if v > 1 else "the sender's side reconciles exactly, with no rounding at all"
    ),
    "errorBalanceDest": lambda v: (
        f"the receiver's balance does not reconcile (off by {_money(abs(v))})"
        if abs(v) > 1 else "the receiver's balance reconciles to the exact rupee"
    ),
    "absErrorBalanceDest": lambda v: (
        f"a {_money(v)} accounting mismatch on the receiver's side"
        if v > 1 else "the receiver's side reconciles exactly"
    ),

    # --- Account-draining fingerprints ---
    # NOTE the negative phrasing below. `orig_account_emptied` is 0 both when
    # money remains AND when the account was already empty to begin with, so
    # "the sender still has money left" would be an outright false statement in
    # the second case. State only what the flag actually guarantees.
    "orig_account_emptied": lambda v: (
        "this payment empties the sender's account completely"
        if v >= 0.5 else "this payment does not drain the sender's account"
    ),
    "dest_was_empty": lambda v: (
        "the receiving account was empty beforehand, which is typical of a "
        "freshly-created mule account"
        if v >= 0.5 else "the receiving account already held a balance"
    ),
    "amount_to_balance_ratio": lambda v: (
        f"this moves {min(v, 1.0) * 100:.0f}% of the sender's available balance"
        if v > 0 else "the sender's account was already empty before this payment"
    ),

    # --- Amount ---
    "amount": lambda v: f"the amount is {_money(v)}",
    "oldbalanceOrg": lambda v: f"the sender held {_money(v)} before this payment",
    "newbalanceOrig": lambda v: f"the sender is left with {_money(v)}",
    "oldbalanceDest": lambda v: f"the receiver held {_money(v)} before this payment",
    "newbalanceDest": lambda v: f"the receiver now holds {_money(v)}",

    # --- Transaction type ---
    "is_risky_type": lambda v: (
        "this is a TRANSFER or CASH_OUT, the only two types where fraud occurs "
        "in this data"
        if v >= 0.5 else "this transaction type has never been associated with fraud"
    ),
    "type_TRANSFER": lambda v: "this is a TRANSFER" if v >= 0.5 else "this is not a transfer",
    "type_CASH_OUT": lambda v: "this is a CASH_OUT" if v >= 0.5 else "this is not a cash-out",
    "type_PAYMENT": lambda v: "this is an ordinary PAYMENT" if v >= 0.5 else "this is not an ordinary payment",
    "type_CASH_IN": lambda v: "this is a CASH_IN" if v >= 0.5 else "this is not a cash-in",
    "type_DEBIT": lambda v: "this is a DEBIT" if v >= 0.5 else "this is not a debit",

    # --- Timing ---
    "hour_of_day": lambda v: f"the payment was made at {int(v):02d}:00",
    "is_night": lambda v: (
        "the payment was made in the middle of the night"
        if v >= 0.5 else "the payment was made during normal hours"
    ),

    # --- Receiver behaviour ---
    "is_new_receiver": lambda v: (
        "the money is going to a receiver never used before"
        if v >= 0.5 else "this receiver has been paid before"
    ),

    # --- Group D: the SENDER's own behaviour (Phase 14) ---------------------
    #
    # Off on PaySim, where 99.85% of senders appear exactly once. Live, and on
    # any dataset with repeat customers, these are among the most intuitive
    # reasons a person can be given -- they compare someone against THEMSELVES
    # rather than against a population, which is the difference between "this
    # is unusual for you" and "this is unusual for somebody".
    "user_txn_count": lambda v: (
        f"this customer has {int(v)} earlier transactions on record"
        if v >= 1 else "this customer has no earlier transactions on record"
    ),
    "user_mean_amount": lambda v: (
        f"this customer's payments usually run around {_money(v)}"
        if v > 0 else "this customer has no established typical amount yet"
    ),
    "user_std_amount": lambda v: (
        f"this customer's payment amounts typically vary by about {_money(v)}"
        if v >= 1 else
        "this customer's payment amounts are usually very consistent"
    ),
    "user_has_thin_history": lambda v: (
        "this customer has almost no history yet, so their normal is not "
        "established"
        if v >= 0.5 else
        "this customer has enough history to judge what is normal for them"
    ),
    "user_amount_deviation": lambda v: (
        f"the amount is {v:.1f} standard deviations above this customer's own "
        f"normal"
        if v > 0 else "the amount is within this customer's own normal range"
    ),
    "user_amount_ratio": lambda v: (
        f"the amount is {v:.1f} times this customer's usual payment"
        if v > 1 else
        "the amount is no larger than this customer's usual payment"
    ),
    "user_velocity_count": lambda v: (
        f"this customer made {int(v)} other transactions in the recent window"
        if v >= 1 else "this customer has made no other recent transactions"
    ),
    "dest_txn_count": lambda v: (
        f"this receiving account has been used {int(v)} times before"
        if v > 0 else "this receiving account has never been seen before"
    ),
    "dest_has_thin_history": lambda v: (
        "the receiving account is almost brand new, so there is little history "
        "to judge it against"
        if v >= 0.5 else "the receiving account has an established history"
    ),
    "dest_amount_deviation": lambda v: (
        f"the amount is {v:.1f} standard deviations above what this receiver "
        f"normally collects"
        if v > 0.5 else "the amount is in line with what this receiver normally collects"
    ),
    "dest_amount_ratio": lambda v: (
        f"the amount is {v:.1f}x what this receiver normally collects"
        if v > 1.5 else "the amount is normal for this receiver"
    ),
    "dest_velocity_count": lambda v: (
        f"this receiver has taken in {int(v)} payments in the last 24 hours"
        if v > 0 else "this receiver has had no other recent activity"
    ),
    "dest_mean_amount": lambda v: f"this receiver usually collects around {_money(v)}",
    "dest_std_amount": lambda v: f"amounts to this receiver typically vary by {_money(v)}",
}


# Several features describe the SAME underlying idea. Without grouping, a top-5
# list can read "balance mismatch / balance mismatch / balance mismatch", which
# looks broken to a human even though it is mathematically correct. We keep only
# the strongest feature from each group.
CONCEPT_GROUPS = {
    "errorBalanceOrig": "sender_balance_mismatch",
    "absErrorBalanceOrig": "sender_balance_mismatch",
    "errorBalanceDest": "receiver_balance_mismatch",
    "absErrorBalanceDest": "receiver_balance_mismatch",
    # "the account was emptied", "100% of the balance moved" and "the sender is
    # left with Rs 0.00" are three descriptions of one event -- group them.
    "orig_account_emptied": "account_drained",
    "amount_to_balance_ratio": "account_drained",
    "newbalanceOrig": "account_drained",
    "amount": "amount_size",
    "oldbalanceOrg": "sender_balances",
    # oldbalanceDest == 0 is literally the definition of dest_was_empty, so
    # letting both through produced the same sentence twice.
    "oldbalanceDest": "receiver_was_empty",
    "newbalanceDest": "receiver_balances",
    "is_risky_type": "transaction_type",
    "type_TRANSFER": "transaction_type",
    "type_CASH_OUT": "transaction_type",
    "type_PAYMENT": "transaction_type",
    "type_CASH_IN": "transaction_type",
    "type_DEBIT": "transaction_type",
    "hour_of_day": "timing",
    "is_night": "timing",
    "is_new_receiver": "receiver_is_new",
    "dest_txn_count": "receiver_is_new",
    "dest_has_thin_history": "receiver_is_new",
    "dest_was_empty": "receiver_was_empty",
    "dest_amount_deviation": "receiver_amount_unusual",
    "dest_amount_ratio": "receiver_amount_unusual",
    "dest_mean_amount": "receiver_amount_unusual",
    "dest_std_amount": "receiver_amount_unusual",
    "dest_velocity_count": "receiver_velocity",
}


def build_phrase(feature: str, value: float) -> str:
    """
    Turn one feature and its real value into a readable phrase.

    Deliberately catches EVERY exception. This function only produces display
    text -- if it fails, the right outcome is a plainer sentence, never a failed
    payment decision. The live API accepts JSON from the Java service, so values
    like infinity or NaN can arrive here even though Phase 1 scrubs them from
    the training data (int(inf) raises OverflowError, which is easy to miss).
    """
    value = float(value) if value is not None else 0.0
    if not np.isfinite(value):
        return f"{feature.replace('_', ' ')} could not be read for this transaction"

    builder = PHRASE_BUILDERS.get(feature)
    if builder is not None:
        try:
            return builder(value)
        except Exception:
            pass
    # Fallback for any feature we forgot to write a phrase for.
    return f"{feature.replace('_', ' ')} = {value:,.2f}"


# =============================================================================
# SECTION 2 -- The explainer
# =============================================================================

class FraudExplainer:
    """
    Wraps SHAP so the rest of the system can ask one simple question:
    "why did this transaction get that score?"

    Build it ONCE at application startup (Phase 5 does this) -- constructing it
    is cheap, but doing it per request would still be waste.
    """

    # A factor contributing less than this to the score is noise, not a reason.
    # Without this floor the list padded itself out to five entries with things
    # like "amounts to this receiver typically vary by Rs 178,656 (impact 0.000)",
    # which is worse than saying nothing.
    MIN_IMPACT = 0.002

    def __init__(self, rf, scaler, feature_columns: list[str], top_n: int = 5):
        self.rf = rf
        self.scaler = scaler
        self.feature_columns = list(feature_columns)
        self.top_n = int(top_n)
        # TreeExplainer computes SHAP values exactly for tree models, and does
        # it fast -- no sampling, no background dataset needed.
        self.explainer = shap.TreeExplainer(rf)

    # -------------------------------------------------------------------------
    def _shap_for_fraud_class(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Pull out the SHAP values for the FRAUD class (class 1).

        SHAP's output shape has changed across versions, so we handle both:
          * older: a list [array_for_class_0, array_for_class_1]
          * newer: one array shaped (rows, features, classes)
        """
        values = self.explainer.shap_values(X_scaled)

        if isinstance(values, list):
            return np.asarray(values[1])
        values = np.asarray(values)
        if values.ndim == 3:
            return values[:, :, 1]
        return values

    # -------------------------------------------------------------------------
    def explain(self, X_row: pd.DataFrame) -> dict:
        """
        Explain ONE transaction.

        `X_row` is a single-row DataFrame of UNSCALED features, in the same
        column order the model was trained on.

        Returns a dict with:
          reasons            -- top factors that pushed this TOWARD fraud
          protective_factors -- top factors that pushed it toward legitimate
          base_score / final_score -- where the model started and ended up

        We return protective factors as well because the system must explain
        every decision, including the approvals. "Approved because the receiver
        is well established and the balances reconcile" is a real explanation;
        silence is not.
        """
        if len(X_row) != 1:
            raise ValueError(f"explain() takes exactly one row, got {len(X_row)}")

        X_scaled = self.scaler.transform(X_row[self.feature_columns])
        shap_values = self._shap_for_fraud_class(X_scaled)[0]
        raw_values = X_row[self.feature_columns].iloc[0].to_numpy(dtype=float)

        contributions = [
            {
                "feature": feat,
                "value": float(raw),
                "shap": float(sv),
                "group": CONCEPT_GROUPS.get(feat, feat),
            }
            for feat, raw, sv in zip(self.feature_columns, raw_values, shap_values)
        ]

        pushed_to_fraud = [c for c in contributions if c["shap"] > 0]
        pushed_to_legit = [c for c in contributions if c["shap"] < 0]

        base = self.explainer.expected_value
        base_score = float(base[1] if np.ndim(base) > 0 else base)

        return {
            "reasons": self._top_by_group(pushed_to_fraud, descending=True),
            "protective_factors": self._top_by_group(pushed_to_legit, descending=False),
            "base_score": base_score,
            "final_score": base_score + float(np.sum(shap_values)),
        }

    # -------------------------------------------------------------------------
    def _top_by_group(self, contributions: list[dict], descending: bool) -> list[dict]:
        """
        Rank contributions, keep only the strongest one from each concept group,
        and attach a human phrase.

        The grouping is what stops the output reading like
        "balance mismatch, balance mismatch, balance mismatch".
        """
        ordered = sorted(contributions, key=lambda c: abs(c["shap"]), reverse=True)

        seen_groups: set[str] = set()
        results: list[dict] = []
        for c in ordered:
            if abs(c["shap"]) < self.MIN_IMPACT:
                break            # everything after this is even smaller
            if c["group"] in seen_groups:
                continue
            seen_groups.add(c["group"])
            results.append({
                "feature": c["feature"],
                "value": c["value"],
                "impact": round(abs(c["shap"]), 4),
                "direction": "increases_risk" if descending else "reduces_risk",
                "reason": build_phrase(c["feature"], c["value"]),
            })
            if len(results) >= self.top_n:
                break
        return results

    # -------------------------------------------------------------------------
    def explain_batch(self, X: pd.DataFrame) -> list[dict]:
        """Explain many transactions at once (used for testing in Phase 8)."""
        return [self.explain(X.iloc[[i]]) for i in range(len(X))]


# =============================================================================
# SECTION 3 -- Plain-sentence summary (also the Phase 4 fallback)
# =============================================================================

def build_summary_sentence(explanation: dict, fraud_score: float,
                           decision: str | None = None) -> str:
    """
    Stitch the SHAP reasons into one plain sentence, with no AI involved.

    This is deliberately dumb and completely reliable. Phase 4 will hand the
    same reasons to a local language model for a nicer narration, but if Ollama
    is not running -- or is slow, or produces nonsense -- the system falls back
    to THIS. The product must never be unable to explain itself just because an
    optional component is down.
    """
    reasons = explanation.get("reasons", [])
    protective = explanation.get("protective_factors", [])

    def join(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    # >>> THE VERDICT AND THE EVIDENCE MUST POINT THE SAME WAY. <<<
    # An earlier version always quoted the fraud-pushing factors, which produced
    # sentences like "This transaction looks legitimate because the sender is
    # left with Rs 0.00" -- a legitimate verdict justified with suspicious
    # evidence. When the model cleared a payment, explain what CLEARED it.
    if fraud_score >= 0.75:
        opening = f"This transaction was rated high risk ({fraud_score:.0%} fraud score)"
        body = f" because {join([r['reason'] for r in reasons[:3]])}" if reasons else ""
    elif fraud_score >= 0.35:
        opening = f"This transaction was rated uncertain ({fraud_score:.0%} fraud score)"
        body = f" because {join([r['reason'] for r in reasons[:3]])}" if reasons else ""
    else:
        opening = f"This transaction looks legitimate ({fraud_score:.0%} fraud score)"
        if protective:
            body = f" because {join([p['reason'] for p in protective[:2]])}"
        else:
            body = " and nothing about it resembles known fraud"

    # For a cleared payment, still mention anything that gave pause -- an
    # explanation that hides the counter-evidence is not an explanation.
    caveat = ""
    if fraud_score < 0.35 and reasons:
        caveat = f" The only mild concern was that {reasons[0]['reason']}."

    tail = ""
    if decision == "BLOCK":
        tail = " The payment was blocked automatically."
    elif decision == "OTP_CONFIRM":
        tail = " We have asked the customer to confirm the payment by OTP."
    elif decision == "APPROVE":
        tail = " The payment was approved automatically."

    return f"{opening}{body}.{caveat}{tail}"


# =============================================================================
# SECTION 4 -- Loading everything the explainer needs
# =============================================================================

def load_explainer(cfg: dict | None = None):
    """
    Load the trained model and wrap it in an explainer.
    Phase 5's Flask app calls this ONCE at startup.

    Returns (explainer, models_dict) so the caller also has the raw model
    objects for scoring.
    """
    cfg = cfg or load_config()
    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])

    required = ["random_forest.joblib", "isolation_forest.joblib",
                "scaler.joblib", "model_metadata.json"]
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"\n\nMissing model files in {models_dir}: {missing}\n"
            "Run Phase 2 first:  ./venv/bin/python train_model.py\n"
        )

    rf = joblib.load(models_dir / "random_forest.joblib")
    iso = joblib.load(models_dir / "isolation_forest.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    with open(models_dir / "model_metadata.json") as f:
        metadata = json.load(f)

    explainer = FraudExplainer(
        rf=rf,
        scaler=scaler,
        feature_columns=metadata["feature_columns"],
        top_n=int(cfg["explain"]["top_n_reasons"]),
    )
    models = {
        "rf": rf,
        "iso": iso,
        "scaler": scaler,
        "metadata": metadata,
        "calibration": metadata["anomaly_calibration"],
        "feature_columns": metadata["feature_columns"],
    }
    return explainer, models


# =============================================================================
# SECTION 5 -- Demo
# =============================================================================

def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _show(explainer, models, cfg, X_row, actual_label, title):
    """Score one transaction and print its full explanation."""
    from train_model import score_transactions, route_decisions

    scores = score_transactions(
        models["rf"], models["iso"], models["scaler"], X_row, cfg,
        models["calibration"],
    )
    decision = route_decisions(scores, cfg)[0]
    score = float(scores["fraud_score"][0])
    explanation = explainer.explain(X_row)

    print(f"\n{title}")
    print("-" * 78)
    print(f"  Actual label : {'FRAUD' if actual_label == 1 else 'legitimate'}")
    print(f"  Fraud score  : {score:.4f}   "
          f"(RF {scores['rf_prob'][0]:.3f} / anomaly {scores['iso_norm'][0]:.3f})")
    print(f"  Decision     : {decision}")
    if scores["models_disagree"][0]:
        print(f"  NOTE         : the two models disagreed on this one")

    print(f"\n  Why the model leaned toward FRAUD:")
    if explanation["reasons"]:
        for i, r in enumerate(explanation["reasons"], 1):
            print(f"    {i}. {r['reason']}")
            print(f"       (impact {r['impact']:.3f}, from {r['feature']})")
    else:
        print("    nothing pushed this toward fraud")

    print(f"\n  What reassured the model:")
    if explanation["protective_factors"]:
        for i, r in enumerate(explanation["protective_factors"][:3], 1):
            print(f"    {i}. {r['reason']}")
            print(f"       (impact {r['impact']:.3f}, from {r['feature']})")
    else:
        print("    nothing pushed this toward legitimate")

    print(f"\n  Plain-sentence summary (the no-AI fallback used in Phase 4):")
    print(f"    \"{build_summary_sentence(explanation, score, decision)}\"")


def main() -> int:
    import time

    cfg = load_config()
    _print_header("PHASE 3 -- EXPLAINING INDIVIDUAL DECISIONS WITH SHAP")

    try:
        explainer, models = load_explainer(cfg)
    except FileNotFoundError as e:
        print(e)
        return 1
    print(f"Loaded model with {len(models['feature_columns'])} features.")
    print(f"Showing the top {cfg['explain']['top_n_reasons']} reasons per decision.")

    df = pd.read_csv(resolve_path(cfg, cfg["paths"]["processed_data"]))
    X, y, _ = build_feature_matrix(df, cfg)

    # -------------------------------------------------------------------------
    # Sanity check: SHAP values must add up to the model's actual output.
    # If this fails, the explanation is not describing the real decision.
    # -------------------------------------------------------------------------
    _print_header("CORRECTNESS CHECK -- DO THE EXPLANATIONS MATCH THE MODEL?")
    sample = X.iloc[:200]
    scaled = models["scaler"].transform(sample)
    model_probs = models["rf"].predict_proba(scaled)[:, 1]
    shap_totals = explainer._shap_for_fraud_class(scaled).sum(axis=1)
    base = explainer.explainer.expected_value
    base_val = float(base[1] if np.ndim(base) > 0 else base)
    reconstructed = base_val + shap_totals
    max_error = float(np.max(np.abs(reconstructed - model_probs)))

    print(f"Checked 200 transactions.")
    print(f"Largest gap between (base + SHAP contributions) and the model's own")
    print(f"predicted probability: {max_error:.2e}")
    if max_error < 1e-4:
        print("PASS -- the explanations describe the real decision, not an approximation.")
    else:
        print("*** FAIL -- explanations do not reconstruct the model output. ***")
        return 1

    # -------------------------------------------------------------------------
    _print_header("SPEED CHECK -- CAN THIS RUN INSIDE A REAL-TIME API?")
    t0 = time.perf_counter()
    for i in range(50):
        explainer.explain(X.iloc[[i]])
    per_txn_ms = 1000 * (time.perf_counter() - t0) / 50
    print(f"Average time to explain one transaction: {per_txn_ms:.1f} ms")
    print(f"Budget for the Phase 5 /predict endpoint: ~1000 ms")
    print("PASS -- fast enough to explain every transaction in real time."
          if per_txn_ms < 100 else "*** TOO SLOW -- move explanation off /predict. ***")

    # -------------------------------------------------------------------------
    _print_header("EXAMPLE EXPLANATIONS")

    from train_model import score_transactions
    all_scores = score_transactions(
        models["rf"], models["iso"], models["scaler"], X, cfg, models["calibration"]
    )
    fraud_score = all_scores["fraud_score"]
    y_arr = y.to_numpy()

    # Pick genuinely interesting cases rather than the first few rows.
    picks = []
    caught = np.where((y_arr == 1) & (fraud_score > 0.75))[0]
    if len(caught):
        picks.append((caught[0], "CASE 1 -- fraud, correctly blocked"))

    missed = np.where((y_arr == 1) & (fraud_score < 0.35))[0]
    if len(missed):
        picks.append((missed[0], "CASE 2 -- fraud the model MISSED (worth studying)"))

    false_alarm = np.where((y_arr == 0) & (fraud_score > 0.75))[0]
    if len(false_alarm):
        picks.append((false_alarm[0], "CASE 3 -- legitimate payment wrongly flagged"))

    uncertain = np.where((fraud_score >= 0.35) & (fraud_score <= 0.75))[0]
    if len(uncertain):
        picks.append((uncertain[0], "CASE 4 -- uncertain, routed to OTP confirmation"))

    clean = np.where((y_arr == 0) & (fraud_score < 0.05))[0]
    if len(clean):
        picks.append((clean[0], "CASE 5 -- ordinary payment, approved"))

    for idx, title in picks:
        _show(explainer, models, cfg, X.iloc[[idx]], int(y_arr[idx]), title)

    _print_header("PHASE 3 COMPLETE")
    print("explain.py provides:")
    print("  FraudExplainer.explain(row)   -> ranked reasons for one transaction")
    print("  build_summary_sentence(...)   -> plain sentence, no AI needed")
    print("  load_explainer()              -> loads everything once at startup")
    print("\nNext: Phase 4 -- hand these reasons to Ollama for a nicer narration,")
    print("with build_summary_sentence() as the fallback when Ollama is absent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
