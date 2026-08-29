"""
Tests for the data boundary (Phase 14).

schema.py is the only place that knows what the user's columns are called.
If it maps a column wrongly, every downstream number is computed from the
wrong evidence and nothing raises -- exactly the class of bug that passes
testing. So these checks are deliberately picky about failure messages.

Run:
    ./venv/bin/python test_schema.py
"""

from __future__ import annotations

import sys

import pandas as pd

import schema

failures: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {description}")
    else:
        print(f"  FAIL  {description}")
        if detail:
            print(f"        {detail}")
        failures.append(description)


def rejects(cfg: dict, description: str, expect_in: str) -> None:
    """Assert load_mapping refuses a config, and refuses it for the right reason."""
    try:
        schema.load_mapping(cfg)
    except schema.SchemaError as e:
        joined = "; ".join(e.errors)
        check(expect_in.lower() in joined.lower(), description,
              f"expected {expect_in!r} in: {joined}")
    else:
        check(False, description, "no SchemaError was raised")


def a_identity_default() -> None:
    print("\n1. An empty schema block means identity mapping")
    mapping = schema.load_mapping({})
    for field in schema.CANONICAL_FIELDS:
        check(mapping[field] == field, f"{field} defaults to itself")


def a_explicit_mapping() -> None:
    print("\n2. An explicit mapping is honoured")
    cfg = {"schema": {"amount": "amt", "nameOrig": "cc_num", "nameDest": "merchant"}}
    mapping = schema.load_mapping(cfg)
    check(mapping["amount"] == "amt", "amount -> amt")
    check(mapping["nameOrig"] == "cc_num", "nameOrig -> cc_num")
    check(mapping["step"] == "step", "unlisted fields still default to identity")


def a_rejections() -> None:
    print("\n3. Bad mappings are refused, with a usable reason")
    rejects({"schema": {"nonsense": "x"}},
            "an unknown canonical name is refused", "nonsense")
    rejects({"schema": {"amount": "amt", "step": "amt"}},
            "two canonical fields sharing one source column is refused", "amt")
    rejects({"schema": {"amount": None}},
            "a required field mapped to null is refused", "amount")


def _sparkov_like() -> pd.DataFrame:
    """A frame shaped like a real non-PaySim dataset: no balance columns."""
    return pd.DataFrame({
        "hour": [1, 2, 3],
        "category": ["PAYMENT", "TRANSFER", "PAYMENT"],
        "amt": ["100.50", "250.00", "75.25"],
        "cc_num": ["C1", "C2", "C1"],
        "merchant": ["M1", "M1", "M2"],
        "is_fraud": [0, 1, 0],
    })


SPARKOV_MAP = {
    "schema": {
        "step": "hour", "type": "category", "amount": "amt",
        "nameOrig": "cc_num", "nameDest": "merchant", "isFraud": "is_fraud",
        "oldbalanceOrg": None, "newbalanceOrig": None,
        "oldbalanceDest": None, "newbalanceDest": None,
    }
}


def b_normalise_renames() -> None:
    print("\n4. normalize() renames to canonical names")
    mapping = schema.load_mapping(SPARKOV_MAP)
    out, report = schema.normalize(_sparkov_like(), mapping)
    for field in ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]:
        check(field in out.columns, f"'{field}' exists after normalisation")
    check("amt" not in out.columns, "the original 'amt' name is gone")
    check(out["amount"].dtype.kind == "f",
          "amount was coerced from string to float",
          f"dtype is {out['amount'].dtype}")
    check(sorted(report.absent) == sorted(schema.OPTIONAL_CANONICAL),
          "all four balance columns are reported absent", str(report.absent))
    check("amount" in report.mapped, "the report lists what WAS supplied")


def b_normalise_requires() -> None:
    print("\n5. A missing REQUIRED column is refused")
    df = _sparkov_like().drop(columns=["amt"])
    mapping = schema.load_mapping(SPARKOV_MAP)
    try:
        schema.normalize(df, mapping)
    except schema.SchemaError as e:
        joined = "; ".join(e.errors)
        check("amt" in joined, "the error names the missing SOURCE column", joined)
        check("amount" in joined, "the error names the canonical field too", joined)
    else:
        check(False, "a missing required column must raise SchemaError")


def c_absent_balances() -> None:
    print("\n6. Unsupplied balances use the 'not recorded' encoding")
    mapping = schema.load_mapping(SPARKOV_MAP)
    out, report = schema.normalize(_sparkov_like(), mapping)

    for field in schema.OPTIONAL_CANONICAL:
        check(field in out.columns, f"'{field}' was created")
        check((out[field] == 0.0).all(),
              f"'{field}' is zero -- PaySim's 'not recorded' convention",
              str(out[field].tolist()))

    check(sorted(report.absent) == sorted(schema.OPTIONAL_CANONICAL),
          "the report lists the four absent balance fields", str(report.absent))
    check("amount" in report.mapped, "the report lists what WAS supplied")

    text = report.render()
    check("not recorded" in text.lower(),
          "the rendered report states the encoding used")
    check("oldbalanceOrg" in text,
          "the rendered report names each absent field")


def d_derived_step() -> None:
    print("\n7. step can be derived from a timestamp column")
    df = pd.DataFrame({
        "ts": ["2026-01-01 00:30:00", "2026-01-01 02:30:00", "2026-01-02 00:30:00"],
        "category": ["PAYMENT", "TRANSFER", "PAYMENT"],
        "amt": [10.0, 20.0, 30.0],
        "cc_num": ["C1", "C2", "C1"],
        "merchant": ["M1", "M1", "M2"],
    })
    cfg = {"schema": {
        "step": "ts", "type": "category", "amount": "amt",
        "nameOrig": "cc_num", "nameDest": "merchant", "isFraud": None,
        "oldbalanceOrg": None, "newbalanceOrig": None,
        "oldbalanceDest": None, "newbalanceDest": None,
    }}
    out, _ = schema.normalize(df, schema.load_mapping(cfg))

    check(out["step"].dtype.kind in "iu", "step is an integer",
          f"dtype is {out['step'].dtype}")
    steps = out["step"].tolist()
    check(steps[0] == 1, "the earliest row is step 1", str(steps))
    check(steps[1] == 3, "two hours later is step 3", str(steps))
    check(steps[2] == 25, "the next day is step 25", str(steps))


def d_integer_step_untouched() -> None:
    print("\n7b. An integer step is left exactly as it is")
    mapping = schema.load_mapping(SPARKOV_MAP)
    out, _ = schema.normalize(_sparkov_like(), mapping)
    check(out["step"].tolist() == [1, 2, 3],
          "an already-integer hour index is not rebased", str(out["step"].tolist()))


def d_unparseable_step() -> None:
    print("\n7c. A step that is neither integer nor timestamp is refused")
    df = _sparkov_like()
    df["hour"] = ["not a time", "nonsense", "junk"]
    mapping = schema.load_mapping(SPARKOV_MAP)
    try:
        schema.normalize(df, mapping)
    except schema.SchemaError as e:
        joined = "; ".join(e.errors)
        check("step" in joined, "the error names step", joined)
    else:
        check(False, "an unparseable step must raise SchemaError")


def e_identity_is_paysim() -> None:
    print("\n8. The default config is an identity mapping (PaySim unchanged)")
    import features
    cfg = features.load_config()
    mapping = schema.load_mapping(cfg)
    for field in schema.CANONICAL_FIELDS:
        check(mapping[field] == field,
              f"default config maps {field} to itself", mapping.get(field))
    check(hasattr(features, "LAST_CAPABILITY_REPORT"),
          "features exposes LAST_CAPABILITY_REPORT for the training report")


def e_paysim_frame_survives() -> None:
    print("\n8b. A PaySim-shaped frame passes through unchanged")
    paysim = pd.DataFrame({
        "step": [1, 2],
        "type": ["TRANSFER", "CASH_OUT"],
        "amount": [1000.0, 2000.0],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrg": [5000.0, 3000.0],
        "newbalanceOrig": [4000.0, 1000.0],
        "nameDest": ["D1", "D2"],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [1000.0, 2100.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
    })
    out, report = schema.normalize(paysim, schema.load_mapping({}))

    check(report.absent == [], "nothing is reported absent", str(report.absent))
    check("isFlaggedFraud" in out.columns,
          "the leaky column survives normalisation so drop_columns can remove it")
    check(out["step"].tolist() == [1, 2], "step is untouched")
    check(out["oldbalanceOrg"].tolist() == [5000.0, 3000.0],
          "real balances are not overwritten by the zero encoding")
    check(len(out) == 2 and len(out.columns) == 11,
          "shape is unchanged", f"{out.shape}")


def f_feature_groups() -> None:
    print("\n9. Feature groups resolve from what the data supports")
    import json
    import features

    counts = {name: len(cols) for name, cols in features.FEATURE_GROUPS.items()}
    check(counts["A core transaction"] == 9, "group A has 9 features", str(counts))
    check(counts["B balance"] == 11, "group B has 11 features", str(counts))
    check(counts["C receiver behaviour"] == 8, "group C has 8 features", str(counts))
    check(counts["D sender behaviour"] == 7, "group D has 7 features", str(counts))

    # PaySim: everything supplied, but sender history is inert (99.85% of
    # senders appear once), so group D stays off and the list is the old 28.
    full = schema.CapabilityReport(mapped=list(schema.CANONICAL_FIELDS), absent=[])
    resolved = features.resolve_features(full, enable_sender_group=False)
    check(len(resolved) == 28, "PaySim resolves to the original 28 features",
          str(len(resolved)))
    check(resolved == list(features.MODEL_FEATURES),
          "MODEL_FEATURES is exactly the resolver's PaySim output")

    # THE ORDER MATTERS. build_feature_matrix does df[feature_cols], so a
    # reordered list silently feeds the model its columns in the wrong order.
    trained = json.load(open("models/model_metadata.json"))["feature_columns"]
    check(resolved == trained,
          "the resolved order is identical to the trained model's column order",
          f"first difference at {next((i for i,(a,b) in enumerate(zip(resolved,trained)) if a!=b), None)}")

    # No balances: group B drops out entirely.
    no_bal = schema.CapabilityReport(
        mapped=[f for f in schema.CANONICAL_FIELDS
                if f not in schema.OPTIONAL_CANONICAL],
        absent=list(schema.OPTIONAL_CANONICAL))
    resolved = features.resolve_features(no_bal, enable_sender_group=False)
    check(len(resolved) == 17, "without balances, 17 features remain",
          str(len(resolved)))
    check(not any(f.startswith(("errorBalance", "absErrorBalance")) for f in resolved),
          "no balance-derived feature survives")
    check("amount_to_balance_ratio" not in resolved,
          "the balance ratio is gone too")

    # No balances but repeat senders: group D compensates.
    resolved = features.resolve_features(no_bal, enable_sender_group=True)
    check(len(resolved) == 24, "group D brings it back to 24", str(len(resolved)))
    check("user_velocity_count" in resolved, "sender velocity is included")
    check(no_bal.groups["B balance"] is False, "the report records B as off")
    check(no_bal.active_feature_count == 24, "the report records the count")

    # A partial balance set is NOT good enough -- errorBalanceOrig needs both
    # sender balances, so half the group would be computed from a zero.
    partial = schema.CapabilityReport(
        mapped=[f for f in schema.CANONICAL_FIELDS if f != "newbalanceDest"],
        absent=["newbalanceDest"])
    resolved = features.resolve_features(partial, enable_sender_group=False)
    check(len(resolved) == 17,
          "one missing balance column switches the whole group off",
          str(len(resolved)))


def g_group_d_phrases() -> None:
    print("\n10. Every group D feature can be explained in words")
    import features
    from explain import PHRASE_BUILDERS

    group_d = features.FEATURE_GROUPS["D sender behaviour"]

    for feat in group_d:
        check(feat in PHRASE_BUILDERS, f"'{feat}' has a phrase builder")

    for feat in group_d:
        if feat not in PHRASE_BUILDERS:
            continue
        # Exercise both branches: a large value and a zero.
        for value in (3.0, 0.0):
            text = PHRASE_BUILDERS[feat](value)
            check(isinstance(text, str) and len(text) > 10,
                  f"'{feat}' reads as a sentence at value {value}", repr(text))
            check(not text.endswith("."),
                  f"'{feat}' has no trailing full stop at value {value}, "
                  f"matching the other 28", repr(text))

    # The real guard: turning group D on must not produce a feature the
    # explanation layer cannot describe. That is what test_explain.py:57
    # enforces for MODEL_FEATURES, and it must hold for the 35-feature set too.
    every = schema.CapabilityReport(mapped=list(schema.CANONICAL_FIELDS), absent=[])
    full = features.resolve_features(every, enable_sender_group=True)
    missing = [f for f in full if f not in PHRASE_BUILDERS]
    check(not missing,
          "every feature in the FULL 35-feature set can be explained", str(missing))
    check(len(full) == 35, "the full set is 35 features", str(len(full)))


def h_health_reports() -> None:
    print("\n11. /health reports what the data supports")
    import requests
    try:
        r = requests.get("http://localhost:5001/health", timeout=3)
    except Exception:
        print("  SKIPPED: no server on port 5001")
        return

    body = r.json()
    check("data_capability" in body, "/health has a data_capability block",
          str(list(body.keys())))
    cap = body.get("data_capability", {})
    check("groups" in cap, "it lists the feature groups")
    check("active_feature_count" in cap, "it states the active feature count")
    check(cap.get("active_feature_count") == body.get("feature_count"),
          "the count agrees with feature_count",
          f"{cap.get('active_feature_count')} vs {body.get('feature_count')}")

    # The honesty guard: groups must be derived from the columns the model was
    # ACTUALLY trained on. The serving process never calls load_raw_data, so a
    # default of all-True would advertise a capability the model may not have.
    groups = cap.get("groups", {})
    check(groups.get("D sender behaviour") is False,
          "group D reads FALSE, because this model was trained without it",
          str(groups))
    check(groups.get("B balance") is True,
          "group B reads TRUE, because PaySim supplied the balances",
          str(groups))


def main() -> int:
    print("=" * 70)
    print("SCHEMA -- the data boundary")
    print("=" * 70)
    a_identity_default()
    a_explicit_mapping()
    a_rejections()
    b_normalise_renames()
    b_normalise_requires()
    c_absent_balances()
    d_derived_step()
    d_integer_step_untouched()
    d_unparseable_step()
    e_identity_is_paysim()
    e_paysim_frame_survives()
    f_feature_groups()
    g_group_d_phrases()
    h_health_reports()

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
