"""
Tests for the Phase 5 Flask API.

Two halves:
  A. LIVE FEATURE CONSISTENCY -- runs offline, no server needed. This is the
     most important test in the project. If the features built from a live JSON
     request do not mean exactly what they meant during training, every score is
     silently wrong and nothing reports an error.
  B. HTTP BEHAVIOUR -- needs the server running on port 5001.

Run:
    ./venv/bin/python app.py          # in one terminal
    ./venv/bin/python test_api.py     # in another

If the server is not up, part B is skipped and part A still runs.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import requests

from features import build_feature_matrix, load_config, resolve_path
from live_features import (
    LiveFeatureBuilder,
    TransactionHistory,
    ValidationError,
    validate_transaction,
)


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


GOOD_TXN = {
    "step": 400, "type": "TRANSFER", "amount": 50000.0,
    "nameOrig": "C1", "oldbalanceOrg": 80000.0, "newbalanceOrig": 30000.0,
    "nameDest": "C2", "oldbalanceDest": 10000.0, "newbalanceDest": 60000.0,
}


# =============================================================================
# PART A -- offline
# =============================================================================

def test_feature_consistency(cfg) -> None:
    """
    THE CRITICAL TEST: live features must equal training features.

    We take real rows from the processed dataset, rebuild each one as if it had
    just arrived as JSON, hand the builder the same receiver history the batch
    pipeline had, and require the numbers to match. Any drift here corrupts
    every score the service will ever produce, with no error message.
    """
    print("\n1. Live features match the training pipeline exactly")
    meta_path = resolve_path(cfg, cfg["paths"]["models_dir"]) / "model_metadata.json"
    columns = json.loads(meta_path.read_text())["feature_columns"]

    df = pd.read_csv(resolve_path(cfg, cfg["paths"]["processed_data"]))
    X, _, _ = build_feature_matrix(df, cfg)
    builder = LiveFeatureBuilder(cfg, columns, TransactionHistory())

    worst: dict[str, float] = {}
    rows = 300
    for i in range(rows):
        r = df.iloc[i]
        txn = validate_transaction({
            "step": int(r["step"]), "type": r["type"], "amount": float(r["amount"]),
            "nameOrig": r["nameOrig"], "oldbalanceOrg": float(r["oldbalanceOrg"]),
            "newbalanceOrig": float(r["newbalanceOrig"]), "nameDest": r["nameDest"],
            "oldbalanceDest": float(r["oldbalanceDest"]),
            "newbalanceDest": float(r["newbalanceDest"]),
        })
        # Supply the history the batch pipeline computed over the whole file --
        # a live service gets this from the bank's database.
        txn["dest_history"] = {
            "txn_count": int(r["dest_txn_count"]),
            "mean_amount": float(r["dest_mean_amount"]),
            "std_amount": float(r["dest_std_amount"]),
            "velocity_count": int(r["dest_velocity_count"]),
        }
        live, _ = builder.build(txn)
        for c in columns:
            diff = abs(float(live[c].iloc[0]) - float(X[c].iloc[i]))
            worst[c] = max(worst.get(c, 0.0), diff)

    mismatched = {c: d for c, d in worst.items() if d > 1e-6}
    check(not mismatched,
          f"all {len(columns)} features match over {rows} real rows",
          f"drifted: {mismatched}")

    check(list(live.columns) == columns,
          "feature columns are produced in the model's exact order")


def test_validation(cfg) -> None:
    print("\n2. Bad input is rejected, not silently scored")
    # Silently defaulting a missing field to zero would produce a confident
    # score computed from nothing, and no one downstream would ever know.
    cases = [
        ("missing amount", {k: v for k, v in GOOD_TXN.items() if k != "amount"}),
        ("missing nameDest", {k: v for k, v in GOOD_TXN.items() if k != "nameDest"}),
        ("negative amount", {**GOOD_TXN, "amount": -1}),
        ("unknown type", {**GOOD_TXN, "type": "BITCOIN"}),
        ("infinite amount", {**GOOD_TXN, "amount": float("inf")}),
        ("NaN amount", {**GOOD_TXN, "amount": float("nan")}),
        ("text amount", {**GOOD_TXN, "amount": "lots"}),
        ("blank account", {**GOOD_TXN, "nameOrig": "   "}),
        ("not an object", ["not", "a", "dict"]),
    ]
    for name, bad in cases:
        try:
            validate_transaction(bad)
            check(False, f"rejects: {name}", "it was accepted")
        except ValidationError:
            check(True, f"rejects: {name}")

    ok = validate_transaction(GOOD_TXN)
    check(ok["amount"] == 50000.0, "a valid transaction passes through")
    check("step" in validate_transaction({k: v for k, v in GOOD_TXN.items()
                                          if k != "step"}),
          "a missing step defaults to wall-clock hours rather than failing")
    check(validate_transaction({**GOOD_TXN, "type": "transfer"})["type"] == "TRANSFER",
          "transaction type is case-insensitive")


def test_history_ordering(cfg) -> None:
    print("\n3. History is recorded AFTER scoring, never before")
    # Training computed every history feature from prior rows only. Recording
    # before scoring would make a first-ever payment score with
    # dest_txn_count = 1 where training would have said 0 -- a mismatch on
    # every single request.
    meta_path = resolve_path(cfg, cfg["paths"]["models_dir"]) / "model_metadata.json"
    columns = json.loads(meta_path.read_text())["feature_columns"]
    history = TransactionHistory()
    builder = LiveFeatureBuilder(cfg, columns, history)

    txn = validate_transaction(GOOD_TXN)
    row, _ = builder.build(txn)
    check(float(row["dest_txn_count"].iloc[0]) == 0,
          "a first-ever receiver scores with dest_txn_count = 0")
    check(float(row["is_new_receiver"].iloc[0]) == 1,
          "a first-ever receiver is flagged as new")

    history.record(txn)
    row2, _ = builder.build(txn)
    check(float(row2["dest_txn_count"].iloc[0]) == 1,
          "the second payment to that receiver sees a count of 1")
    check(float(row2["is_new_receiver"].iloc[0]) == 0,
          "the receiver is no longer new on the second payment")


def test_thin_history_fallback(cfg) -> None:
    print("\n4. Thin history falls back to the training global averages")
    meta_path = resolve_path(cfg, cfg["paths"]["models_dir"]) / "model_metadata.json"
    columns = json.loads(meta_path.read_text())["feature_columns"]
    builder = LiveFeatureBuilder(cfg, columns, TransactionHistory())
    row, _ = builder.build(validate_transaction(GOOD_TXN))

    check(float(row["dest_has_thin_history"].iloc[0]) == 1,
          "an unknown receiver is marked as having thin history")
    check(abs(float(row["dest_mean_amount"].iloc[0]) - builder.global_mean) < 1e-6,
          "it uses the SAME global average that training used",
          "a different fallback value would shift every score for new receivers")


# =============================================================================
# PART B -- needs the server
# =============================================================================

def server_is_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def test_http(cfg) -> None:
    print("\n5. HTTP endpoints")
    requests.post(f"{BASE_URL}/reset", timeout=10)

    health = requests.get(f"{BASE_URL}/health", timeout=10).json()
    check(health["status"] == "ok", "/health reports ok")
    check(health["models_loaded"], "/health confirms models are loaded")

    conf = requests.get(f"{BASE_URL}/config", timeout=10).json()
    check("decision_thresholds" in conf,
          "/config publishes thresholds so Java need not duplicate them")

    r = requests.post(f"{BASE_URL}/predict", json=GOOD_TXN, timeout=30)
    check(r.status_code == 200, "/predict accepts a valid transaction")
    body = r.json()
    for field in ("fraud_score", "models_disagree", "decision", "shap_reasons"):
        check(field in body, f"/predict returns '{field}'")
    check(body["decision"] in ("APPROVE", "OTP_CONFIRM", "BLOCK"),
          "/predict returns a valid decision", body["decision"])
    check(0.0 <= body["fraud_score"] <= 1.0, "fraud_score is within 0-1")

    print("\n6. Bad requests get 400, not 500")
    for name, payload in [
        ("missing field", {k: v for k, v in GOOD_TXN.items() if k != "amount"}),
        ("negative amount", {**GOOD_TXN, "amount": -5}),
        ("unknown type", {**GOOD_TXN, "type": "CRYPTO"}),
    ]:
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        check(resp.status_code == 400, f"400 for {name}", f"got {resp.status_code}")
        check("details" in resp.json() or "error" in resp.json(),
              f"an explanatory message for {name}")

    print("\n7. /predict stays inside the real-time budget")
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        requests.post(f"{BASE_URL}/predict", json=GOOD_TXN, timeout=30)
        timings.append(1000 * (time.perf_counter() - started))
    p95 = float(np.percentile(timings, 95))
    check(p95 < 1000, f"p95 latency {p95:.0f} ms is under the 1000 ms budget")

    print("\n8. /explain works by id and by full transaction")
    predicted = requests.post(f"{BASE_URL}/predict", json=GOOD_TXN, timeout=30).json()
    by_id = requests.post(f"{BASE_URL}/explain",
                          json={"transaction_id": predicted["transaction_id"]},
                          timeout=120).json()
    check(len(by_id.get("explanation", "")) > 20,
          "/explain by transaction_id returns a real explanation")
    check(by_id["source"] in ("ollama", "template"),
          "/explain says whether the text came from the model or the template",
          "the dashboard must not present a fallback sentence as model prose")
    check(by_id["decision"] == predicted["decision"],
          "/explain describes the SAME decision /predict returned")

    missing = requests.post(f"{BASE_URL}/explain",
                            json={"transaction_id": "nope"}, timeout=30)
    check(missing.status_code == 404, "an unknown transaction_id returns 404")

    fresh = requests.post(f"{BASE_URL}/explain",
                          json={"transaction": GOOD_TXN}, timeout=120).json()
    check(len(fresh.get("explanation", "")) > 20,
          "/explain also accepts a full transaction")

    print("\n9. /chat")
    chat = requests.post(f"{BASE_URL}/chat",
                         json={"question": "Why is a drained account suspicious?"},
                         timeout=120).json()
    check(len(chat.get("answer", "")) > 20, "/chat answers a question")
    blank = requests.post(f"{BASE_URL}/chat", json={"question": "  "}, timeout=30)
    check(blank.status_code == 400, "/chat rejects an empty question")

    print("\n10a. R7: rapid transfers from one account are blocked by rule")
    # The model CANNOT do this. Measured across all 6.3M PaySim rows, fraud
    # receivers are LESS busy than legitimate ones, so the model learned that
    # burst activity is reassuring. The typology is real in banking and absent
    # from the simulator, so it lives in rules.py instead.
    requests.post(f"{BASE_URL}/reset", timeout=10)
    drain = {"step": 300, "type": "TRANSFER", "amount": 40000,
             "nameOrig": "C-VICTIM", "oldbalanceOrg": 200000,
             "newbalanceOrig": 160000, "oldbalanceDest": 5000,
             "newbalanceDest": 45000}
    outcomes = []
    for i in range(5):
        outcomes.append(requests.post(
            f"{BASE_URL}/predict", json={**drain, "nameDest": f"C-D{i}"},
            timeout=30).json())

    check(outcomes[-1]["decision"] == "BLOCK",
          "the 5th rapid transfer in an hour is blocked")
    check(outcomes[-1]["rule_applied"] == "rapid_sender_transfers",
          "the rapid-transfer rule is named in the response")
    check(outcomes[-1]["model_decision"] == "APPROVE",
          "and the MODEL had approved it",
          "this is the whole point: the rule catches what the data cannot teach")
    check(all(o["decision"] == "APPROVE" for o in outcomes[:4]),
          "the first four transfers are not blocked",
          "the rule must need a genuine burst, not fire immediately")

    print("\n10b. A receiver collecting from many senders is escalated to OTP")
    requests.post(f"{BASE_URL}/reset", timeout=10)
    collected = []
    for i in range(6):
        collected.append(requests.post(
            f"{BASE_URL}/predict",
            json={**drain, "nameOrig": f"C-S{i}", "nameDest": "C-MULE"},
            timeout=30).json())
    check(collected[-1]["decision"] == "OTP_CONFIRM",
          "the 6th payment into one account triggers a step-up")
    check(collected[-1]["rule_applied"] == "rapid_receiver_collection",
          "the mule rule is named in the response")
    check(collected[-1]["decision"] != "BLOCK",
          "it asks rather than blocks",
          "the suspicious account is the RECEIVER; the sender did nothing wrong")

    print("\n10c. Rules stay silent on ordinary traffic")
    requests.post(f"{BASE_URL}/reset", timeout=10)
    fired = 0
    for i in range(40):
        r = requests.post(f"{BASE_URL}/predict", json={
            "step": 300 + i, "type": "PAYMENT", "amount": 1500 + i * 10,
            "nameOrig": f"C-CUST{i}", "oldbalanceOrg": 60000,
            "newbalanceOrig": 58500, "nameDest": f"M-SHOP{i % 5}",
            "oldbalanceDest": 90000, "newbalanceDest": 91500}, timeout=30).json()
        if r["rule_applied"]:
            fired += 1
    check(fired == 0, "40 ordinary payments trigger no rules",
          f"{fired} false positives -- thresholds are too tight")

    print("\n10d. Rules escalate only, never soften")
    # A crude heuristic must not be able to release what a measured model
    # blocked. Same principle as models_disagree in Phase 2.
    requests.post(f"{BASE_URL}/reset", timeout=10)
    blocked = requests.post(f"{BASE_URL}/predict", json={
        "step": 300, "type": "TRANSFER", "amount": 3244067.13, "nameOrig": "C-F",
        "oldbalanceOrg": 3244067.13, "newbalanceOrig": 0, "nameDest": "C-X",
        "oldbalanceDest": 0, "newbalanceDest": 0}, timeout=30).json()
    check(blocked["decision"] == "BLOCK",
          "a model BLOCK survives the rule layer untouched")

    print("\n11. The impossible-travel rule overrides the model")
    requests.post(f"{BASE_URL}/reset", timeout=10)
    base = {"type": "PAYMENT", "amount": 2000, "nameOrig": "C77",
            "oldbalanceOrg": 50000, "newbalanceOrig": 48000, "nameDest": "M1",
            "oldbalanceDest": 9000, "newbalanceDest": 11000}
    requests.post(f"{BASE_URL}/predict",
                  json={**base, "step": 100, "location": "Mumbai",
                        "location_lat": 19.0760, "location_lon": 72.8777}, timeout=30)
    jump = requests.post(f"{BASE_URL}/predict",
                         json={**base, "step": 100, "location": "Kolkata",
                               "location_lat": 22.5726, "location_lon": 88.3639},
                         timeout=30).json()
    check(jump["decision"] == "BLOCK",
          "a physically impossible journey is blocked")
    check(jump["rule_applied"] == "impossible_travel",
          "the response says WHICH rule fired")
    check(jump["fraud_score"] < 0.75,
          "and it blocked despite a LOW model score",
          "this proves the rule overrides the model rather than agreeing with it")


def main() -> int:
    cfg = load_config()
    print("=" * 70)
    print("PART A -- feature consistency and validation (no server needed)")
    print("=" * 70)
    test_feature_consistency(cfg)
    test_validation(cfg)
    test_history_ordering(cfg)
    test_thin_history_fallback(cfg)

    print("\n" + "=" * 70)
    print("PART B -- HTTP behaviour")
    print("=" * 70)
    if server_is_up():
        test_http(cfg)
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
