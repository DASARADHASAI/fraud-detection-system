"""
Tests for the Phase 3 explanation layer.

Plain Python with assertions -- no test framework to learn. Run it directly:

    ./venv/bin/python test_explain.py

Exits 0 if everything passes, 1 if anything fails, so it can be wired into a
build later. Requires the Phase 2 models.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from explain import (
    CONCEPT_GROUPS,
    PHRASE_BUILDERS,
    FraudExplainer,
    build_phrase,
    build_summary_sentence,
    load_explainer,
)
from features import MODEL_FEATURES, build_feature_matrix, load_config, resolve_path


failures: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> None:
    """Record a pass or a failure without stopping the run."""
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


def main() -> int:
    cfg = load_config()
    print("Loading model and explainer...")
    explainer, models = load_explainer(cfg)
    feature_cols = models["feature_columns"]

    df = pd.read_csv(resolve_path(cfg, cfg["paths"]["processed_data"])).head(300)
    X, y, _ = build_feature_matrix(df, cfg)

    # -------------------------------------------------------------------------
    print("\n1. Every feature the model uses has a human phrase")
    # A missing phrase falls back to "dest_amount_ratio = 8.20", which is not
    # something we want appearing on a customer's screen.
    missing = [f for f in MODEL_FEATURES if f not in PHRASE_BUILDERS]
    check(not missing, "all model features have a written phrase", f"missing: {missing}")

    # -------------------------------------------------------------------------
    print("\n2. Phrase building survives hostile values")
    # The live API takes JSON from the Java service, so anything can arrive.
    # A display helper must never be the reason a payment decision fails.
    edge_values = [0.0, 1.0, -1.0, 1e9, -1e9, 0.5, 23.0, np.nan, np.inf, -np.inf, None]
    crashes = []
    for feat in MODEL_FEATURES:
        for v in edge_values:
            try:
                phrase = build_phrase(feat, v)
                if not isinstance(phrase, str) or not phrase:
                    crashes.append(f"{feat}={v}: empty phrase")
            except Exception as e:
                crashes.append(f"{feat}={v}: {type(e).__name__}: {e}")
    check(not crashes, "no crashes on extreme, infinite, NaN or missing values",
          "; ".join(crashes[:5]))

    # -------------------------------------------------------------------------
    print("\n3. Explanations reconstruct the model's actual output")
    # If SHAP values do not sum back to the model's probability, the explanation
    # is describing something other than the decision that was made.
    scaled = models["scaler"].transform(X[feature_cols])
    model_probs = models["rf"].predict_proba(scaled)[:, 1]
    shap_totals = explainer._shap_for_fraud_class(scaled).sum(axis=1)
    base = explainer.explainer.expected_value
    base_val = float(base[1] if np.ndim(base) > 0 else base)
    max_error = float(np.max(np.abs(base_val + shap_totals - model_probs)))
    check(max_error < 1e-6,
          f"SHAP values sum to the model output (max error {max_error:.2e})")

    # -------------------------------------------------------------------------
    print("\n4. No reason list repeats the same concept twice")
    # "balance mismatch, balance mismatch, balance mismatch" is mathematically
    # correct and completely useless to a reader.
    dupes = 0
    for i in range(len(X)):
        e = explainer.explain(X.iloc[[i]])
        for key in ("reasons", "protective_factors"):
            groups = [CONCEPT_GROUPS.get(r["feature"], r["feature"]) for r in e[key]]
            if len(groups) != len(set(groups)):
                dupes += 1
    check(dupes == 0, f"no duplicate concept groups across {len(X)} transactions",
          f"{dupes} lists contained a repeat")

    # -------------------------------------------------------------------------
    print("\n5. Trivial contributions are filtered out")
    below = 0
    for i in range(len(X)):
        e = explainer.explain(X.iloc[[i]])
        for key in ("reasons", "protective_factors"):
            below += sum(1 for r in e[key] if r["impact"] < FraudExplainer.MIN_IMPACT)
    check(below == 0, "no reason falls below the impact threshold",
          f"{below} near-zero reasons leaked through")

    # -------------------------------------------------------------------------
    print("\n6. Reasons are ordered strongest first")
    unordered = 0
    for i in range(len(X)):
        e = explainer.explain(X.iloc[[i]])
        for key in ("reasons", "protective_factors"):
            impacts = [r["impact"] for r in e[key]]
            if impacts != sorted(impacts, reverse=True):
                unordered += 1
    check(unordered == 0, "impacts are in descending order",
          f"{unordered} lists were out of order")

    # -------------------------------------------------------------------------
    print("\n7. Risk and protective factors point in opposite directions")
    wrong = 0
    for i in range(len(X)):
        e = explainer.explain(X.iloc[[i]])
        if any(r["direction"] != "increases_risk" for r in e["reasons"]):
            wrong += 1
        if any(r["direction"] != "reduces_risk" for r in e["protective_factors"]):
            wrong += 1
    check(wrong == 0, "every factor is labelled with the correct direction")

    # -------------------------------------------------------------------------
    print("\n8. The summary sentence agrees with the verdict")
    # The bug this guards against: explaining a cleared payment using the
    # factors that argued for fraud, e.g. "looks legitimate because the sender's
    # account was emptied". Verdict and evidence must point the same way.
    sample = X.iloc[[0]]
    e = explainer.explain(sample)
    low = build_summary_sentence(e, 0.04, "APPROVE")
    high = build_summary_sentence(e, 0.95, "BLOCK")
    check("looks legitimate" in low and "approved automatically" in low,
          "a low score produces a 'legitimate' sentence", low)
    check("high risk" in high and "blocked automatically" in high,
          "a high score produces a 'high risk' sentence", high)

    if e["protective_factors"]:
        top_protective = e["protective_factors"][0]["reason"]
        check(top_protective in low,
              "a cleared payment is explained by what reassured the model")
    if e["reasons"]:
        top_risk = e["reasons"][0]["reason"]
        check(top_risk in high,
              "a blocked payment is explained by what alarmed the model")

    # -------------------------------------------------------------------------
    print("\n9. A blank transaction still explains itself")
    blank = pd.DataFrame([{c: 0.0 for c in feature_cols}])
    try:
        e = explainer.explain(blank)
        sentence = build_summary_sentence(e, 0.02, "APPROVE")
        check(isinstance(sentence, str) and len(sentence) > 20,
              "an all-zero transaction produces a usable sentence")
    except Exception as exc:
        check(False, "an all-zero transaction produces a usable sentence",
              f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------------------------------
    print("\n10. explain() rejects the wrong number of rows")
    # Silently explaining only the first of several rows would be worse than
    # refusing outright.
    try:
        explainer.explain(X.iloc[:3])
        check(False, "passing several rows raises an error")
    except ValueError:
        check(True, "passing several rows raises an error")
    except Exception as exc:
        check(False, "passing several rows raises ValueError",
              f"raised {type(exc).__name__} instead")

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
