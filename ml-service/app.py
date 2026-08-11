"""
=============================================================================
PHASE 5 -- The Flask API.
=============================================================================

This is the seam between the two halves of the system. Java handles the money;
Python handles the maths. They talk over these endpoints.

  GET  /health   is the service up, are the models loaded, is Ollama there
  GET  /config   the decision thresholds, so Java does not duplicate them
  POST /predict  score a transaction  -- FAST (~28 ms), every payment
  POST /explain  plain-language write-up -- SLOW (~3.7 s), only when opened
  POST /chat     analyst question -> answer
  POST /reset    clear the demo history (testing only)

>>> WHY /predict AND /explain ARE SEPARATE ENDPOINTS <<<
The obvious design is one endpoint returning score and explanation together.
It would be wrong. Measured on this machine:

    POST /predict  (scoring + SHAP, end to end over HTTP)   ~28 ms
    POST /explain  (the same, plus llama3.2 narration)    ~3700 ms

That is a 135x difference.

A payment cannot wait three seconds for prose nobody may ever read. So
/predict returns the score and the SHAP reasons immediately -- those are
already human-readable -- and /explain fetches the language-model version
later, only for the transactions someone actually opens.

>>> PORT 5001, NOT 5000 <<<
macOS runs AirPlay Receiver on port 5000. Flask will appear to start and then
every request 403s, which wastes an afternoon. Configured in config.yaml.

RUN IT:
    ./venv/bin/python app.py
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from threading import Lock

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from explain import load_explainer
from features import load_config
from genai_agent import FraudGenAIAgent
from live_features import (
    LiveFeatureBuilder,
    TransactionHistory,
    ValidationError,
    validate_transaction,
)
from train_model import route_decisions, score_transactions


# =============================================================================
# SECTION 1 -- Start-up. Everything heavy is loaded exactly once.
# =============================================================================
#
# Loading a 7 MB Random Forest per request would add hundreds of milliseconds
# to every payment for no reason. These are module-level singletons, built
# while the process starts, before the first request arrives.
# =============================================================================

print("=" * 70)
print("FRAUD DETECTION ML SERVICE -- starting up")
print("=" * 70)

CONFIG = load_config()
_startup = time.perf_counter()

print("\nLoading models...")
EXPLAINER, MODELS = load_explainer(CONFIG)
FEATURE_COLUMNS = MODELS["feature_columns"]
print(f"  Random Forest + Isolation Forest + scaler loaded "
      f"({len(FEATURE_COLUMNS)} features)")

HISTORY = TransactionHistory()
FEATURE_BUILDER = LiveFeatureBuilder(CONFIG, FEATURE_COLUMNS, HISTORY)
print(f"  Live feature builder ready "
      f"(global fallback average: Rs {FEATURE_BUILDER.global_mean:,.2f})")

print("\nLoading the optional GenAI layer...")
AGENT = FraudGenAIAgent(CONFIG, verbose=True)

print(f"\nStartup finished in {time.perf_counter() - _startup:.1f}s")

app = Flask(__name__)
# The dashboard is served by Spring Boot on a different port, so the browser
# treats these calls as cross-origin and blocks them without this.
CORS(app)


# =============================================================================
# SECTION 2 -- A small cache so /explain does not rescore from scratch
# =============================================================================

class ScoreCache:
    """
    Remembers recent /predict results by transaction_id.

    Lets the Java service call /explain with just an id instead of resending the
    whole transaction, and guarantees the explanation describes exactly the
    score that was returned -- not a fresh re-score that could differ because
    the history moved on in between.
    """

    def __init__(self, max_entries: int = 500):
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._max = max_entries
        self._lock = Lock()

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def get(self, key: str) -> dict | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value


CACHE = ScoreCache()


# =============================================================================
# SECTION 3 -- Scoring, shared by /predict and /explain
# =============================================================================

def score_one(txn: dict) -> dict:
    """
    Validate, build features, score, and explain one transaction.

    Note the ORDER at the end: we score first and record the transaction into
    the history afterwards. Training built every history feature from prior
    rows only, so recording first would make a first-ever payment score with
    dest_txn_count = 1 where training would have said 0.
    """
    clean = validate_transaction(txn)
    row, signals = FEATURE_BUILDER.build(clean)

    scores = score_transactions(MODELS["rf"], MODELS["iso"], MODELS["scaler"],
                                row, CONFIG, MODELS["calibration"])
    decision = str(route_decisions(scores, CONFIG)[0])
    explanation = EXPLAINER.explain(row)

    # The impossible-travel RULE overrides the model. It is a physical
    # contradiction, not a probability -- no score should be able to wave
    # through a payment that could not have happened.
    travel = signals["impossible_travel"]
    rule_applied = None
    if travel.get("impossible"):
        decision = "BLOCK"
        rule_applied = "impossible_travel"

    HISTORY.record(clean)          # only AFTER scoring

    return {
        "transaction": clean,
        "fraud_score": round(float(scores["fraud_score"][0]), 4),
        "models_disagree": bool(scores["models_disagree"][0]),
        "decision": decision,
        "rule_applied": rule_applied,
        "shap_reasons": explanation["reasons"],
        "protective_factors": explanation["protective_factors"],
        "explanation_obj": explanation,
        "model_detail": {
            "random_forest_probability": round(float(scores["rf_prob"][0]), 4),
            "anomaly_score": round(float(scores["iso_norm"][0]), 4),
            "isolation_forest_flags_anomaly": bool(scores["iso_flags_anomaly"][0]),
            "novel_pattern": bool(scores["novel_pattern_flag"][0]),
        },
        "signals": {
            "impossible_travel": travel,
            "receiver_profile": signals["receiver_profile"],
            "sender_profile": signals["sender_profile"],
            "trusted_device": clean.get("trusted_device", False),
        },
    }


def public_view(result: dict, transaction_id: str, latency_ms: float) -> dict:
    """Trim the internal scoring result down to what callers should see."""
    return {
        "transaction_id": transaction_id,
        "fraud_score": result["fraud_score"],
        "models_disagree": result["models_disagree"],
        "decision": result["decision"],
        "rule_applied": result["rule_applied"],
        "shap_reasons": result["shap_reasons"],
        "protective_factors": result["protective_factors"],
        "model_detail": result["model_detail"],
        "signals": result["signals"],
        "latency_ms": round(latency_ms, 1),
    }


# =============================================================================
# SECTION 4 -- Endpoints
# =============================================================================

@app.get("/health")
def health():
    """
    Cheap liveness check. The Java backend polls this to decide whether to use
    its own fallback path, so it must never do real work.
    """
    return jsonify({
        "status": "ok",
        "models_loaded": True,
        "feature_count": len(FEATURE_COLUMNS),
        "model_trained_at": MODELS["metadata"].get("trained_at"),
        "genai": {
            "available": AGENT.available,
            "status": AGENT.status,
            "knowledge_base_cases": len(AGENT.knowledge_base.cases),
        },
        "accounts_in_memory": HISTORY.size(),
    })


@app.get("/config")
def get_config():
    """
    Publish the decision thresholds.

    Java needs these to render the dashboard, and hard-coding them in two
    languages guarantees they drift apart. One source of truth: config.yaml.
    """
    return jsonify({
        "decision_thresholds": CONFIG["decision"],
        "hybrid_weights": CONFIG["model"]["hybrid"],
        "feature_columns": FEATURE_COLUMNS,
    })


@app.post("/predict")
def predict():
    """
    Score one transaction. This is the real-time path -- keep it fast.

    Returns fraud_score, models_disagree, decision and shap_reasons. No language
    model is invoked here, by design.
    """
    started = time.perf_counter()
    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({"error": "request body must be valid JSON"}), 400

        result = score_one(payload)
        transaction_id = str(uuid.uuid4())
        CACHE.put(transaction_id, result)

        latency = 1000 * (time.perf_counter() - started)
        return jsonify(public_view(result, transaction_id, latency))

    except ValidationError as e:
        return jsonify({"error": "invalid transaction", "details": e.errors}), 400
    except Exception as e:
        # Never leak a stack trace to the caller, but do print it for us.
        traceback.print_exc()
        return jsonify({"error": "scoring failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500


@app.post("/explain")
def explain():
    """
    The plain-language write-up. SLOW (seconds) when the language model runs,
    so this is called lazily -- when someone opens a transaction, not on every
    payment.

    Accepts either {"transaction_id": "..."} from a previous /predict, or a
    full transaction to score afresh.
    """
    started = time.perf_counter()
    try:
        payload = request.get_json(force=True, silent=True) or {}

        transaction_id = payload.get("transaction_id")
        if transaction_id:
            result = CACHE.get(transaction_id)
            if result is None:
                return jsonify({
                    "error": "unknown transaction_id",
                    "detail": "it may have aged out of the cache; send the "
                              "full transaction instead",
                }), 404
        else:
            source = payload.get("transaction", payload)
            result = score_one(source)
            transaction_id = str(uuid.uuid4())
            CACHE.put(transaction_id, result)

        narration = AGENT.explain_transaction(
            explanation=result["explanation_obj"],
            fraud_score=result["fraud_score"],
            decision=result["decision"],
            models_disagree=result["models_disagree"],
        )

        return jsonify({
            "transaction_id": transaction_id,
            "fraud_score": result["fraud_score"],
            "decision": result["decision"],
            "explanation": narration["explanation"],
            "recommended_action": narration["recommended_action"],
            "similar_cases": narration["similar_cases"],
            # "ollama" or "template" -- the dashboard shows which, so nobody
            # mistakes a fallback sentence for model-written prose.
            "source": narration["source"],
            "notes": narration["notes"],
            "shap_reasons": result["shap_reasons"],
            "latency_ms": round(1000 * (time.perf_counter() - started), 1),
        })

    except ValidationError as e:
        return jsonify({"error": "invalid transaction", "details": e.errors}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "explanation failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500


@app.post("/chat")
def chat():
    """
    Analyst follow-up questions, grounded in knowledge_base/.

    This changes no decisions. It exists so a human reading the audit log
    afterwards can interrogate the system in plain English.
    """
    started = time.perf_counter()
    try:
        payload = request.get_json(force=True, silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "'question' is required"}), 400
        if len(question) > 2000:
            return jsonify({"error": "'question' is too long (max 2000 chars)"}), 400

        # Optionally ground the answer in a specific transaction.
        context = None
        transaction_id = payload.get("transaction_id")
        if transaction_id:
            cached = CACHE.get(transaction_id)
            if cached:
                context = {
                    "fraud_score": cached["fraud_score"],
                    "decision": cached["decision"],
                    "reasons": [r["reason"] for r in cached["shap_reasons"]],
                    "protective": [p["reason"] for p in cached["protective_factors"]],
                    "models_disagree": cached["models_disagree"],
                }

        answer = AGENT.chat(question, context=context)
        answer["latency_ms"] = round(1000 * (time.perf_counter() - started), 1)
        return jsonify(answer)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "chat failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500


@app.post("/reset")
def reset():
    """Clear the in-memory demo history. For testing and repeatable demos."""
    HISTORY.clear()
    return jsonify({"status": "history cleared", "accounts_in_memory": HISTORY.size()})


@app.errorhandler(404)
def not_found(_):
    return jsonify({
        "error": "no such endpoint",
        "available": ["GET /health", "GET /config", "POST /predict",
                      "POST /explain", "POST /chat", "POST /reset"],
    }), 404


@app.errorhandler(405)
def wrong_method(_):
    return jsonify({"error": "wrong HTTP method for this endpoint"}), 405


# =============================================================================
# SECTION 5 -- Run
# =============================================================================

if __name__ == "__main__":
    service = CONFIG["services"]["ml_service"]
    port = int(service["port"])

    print("\n" + "=" * 70)
    print(f"Listening on http://localhost:{port}")
    print("=" * 70)
    print("  GET  /health    service and model status")
    print("  GET  /config    decision thresholds")
    print("  POST /predict   score a transaction  (fast, ~28 ms)")
    print("  POST /explain   plain-language write-up  (slow, ~3.7 s with Ollama)")
    print("  POST /chat      analyst question")
    print("  POST /reset     clear demo history")
    print("\nTry it:")
    print(f"  curl http://localhost:{port}/health")
    print("\nPress Ctrl+C to stop.\n")

    app.run(
        host=service["host"],
        port=port,
        debug=bool(service.get("debug", False)),
        # threaded=True so a slow /explain cannot block a fast /predict.
        # Measured: with three /explain calls in flight, /predict's median goes
        # from 28 ms to 111 ms -- degraded but nowhere near the 3.7 s it would
        # be if requests were serialised.
        threaded=True,
    )
