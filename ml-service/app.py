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
import os
import sys
import time
import traceback
import uuid
from collections import OrderedDict, deque
from threading import Lock

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from explain import load_explainer
from features import load_config
from genai_agent import FraudGenAIAgent
import rules
from live_features import (
    LiveFeatureBuilder,
    TransactionHistory,
    ValidationError,
    validate_transaction,
)
from train_model import route_decisions, score_transactions
from upi_adapter import upi_to_transaction
from webhook_razorpay import (
    DECIDABLE_EVENT,
    razorpay_to_upi,
    verify_signature,
)


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

# --- Razorpay webhook configuration -----------------------------------------
#
# The secret is read from the environment, never from a file in the repository.
# If it is unset the webhook refuses every request: an unverified webhook
# endpoint is an open door, and anyone who learned the URL could post payments
# and read the fraud engine's verdicts. Failing closed is the only safe default.
RAZORPAY_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
MERCHANT_VPA = os.environ.get("RAZORPAY_MERCHANT_VPA", "unipaystore@razorpay")

# Gateways retry on any non-2xx, and a retried payment must not be scored twice
# -- the second pass would see the first in its own history and score it
# differently. Keyed on Razorpay's payment id, which is stable across retries.
WEBHOOK_SEEN = ScoreCache(max_entries=1000)

# What actually arrived, newest last. Purely diagnostic: when you are wiring a
# gateway up, the question you cannot otherwise answer is "did anything reach me
# at all, and was it signed correctly?" -- stdout scrolls away, this does not.
#
# Bounded, in memory, and never persisted. It holds payer UPI addresses, so it
# is a localhost diagnostic and nothing more.
WEBHOOK_LOG: deque = deque(maxlen=50)
WEBHOOK_LOG_LOCK = Lock()


def record_webhook(entry: dict) -> None:
    with WEBHOOK_LOG_LOCK:
        WEBHOOK_LOG.append(dict(entry, at=time.strftime("%H:%M:%S")))


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
    model_decision = str(route_decisions(scores, CONFIG)[0])
    explanation = EXPLAINER.explain(row)

    # The RULE LAYER runs after the model and may only escalate its decision.
    # It covers what the model cannot: physical contradictions, and typologies
    # that are absent from PaySim entirely (see rules.py for the measurements).
    verdict = rules.evaluate(clean, signals, HISTORY, CONFIG, model_decision)

    HISTORY.record(clean)          # only AFTER scoring, never before

    return {
        "transaction": clean,
        "fraud_score": round(float(scores["fraud_score"][0]), 4),
        "models_disagree": bool(scores["models_disagree"][0]),
        "decision": verdict["decision"],
        # Kept separate so the dashboard and the audit log can show that the
        # model said one thing and a rule overrode it.
        "model_decision": verdict["model_decision"],
        "rule_applied": verdict["rule_applied"],
        "rules_triggered": verdict["rules_triggered"],
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
            "impossible_travel": signals["impossible_travel"],
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
        "model_decision": result["model_decision"],
        "rule_applied": result["rule_applied"],
        "rules_triggered": result["rules_triggered"],
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


@app.post("/predict/upi")
def predict_upi():
    """
    Score a payment that arrived in UPI's language rather than PaySim's.

    Identical engine, identical thresholds, identical rule layer -- the only
    thing that differs is the shape of the request. That is the whole point:
    a payment platform adopts this system by changing an adapter, not a model.

    The response is byte-for-byte the /predict response plus one extra `upi`
    block saying what the adapter mapped and, importantly, what it had to
    assume. See upi_adapter.py for why a remitter bank usually cannot see the
    beneficiary's balance.
    """
    started = time.perf_counter()
    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({"error": "request body must be valid JSON"}), 400

        txn, notes = upi_to_transaction(payload)
        result = score_one(txn)
        transaction_id = str(uuid.uuid4())
        CACHE.put(transaction_id, result)

        latency = 1000 * (time.perf_counter() - started)
        response = public_view(result, transaction_id, latency)
        response["upi"] = notes
        return jsonify(response)

    except ValidationError as e:
        return jsonify({"error": "invalid UPI payload", "details": e.errors}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "scoring failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500


@app.post("/webhook/razorpay")
def webhook_razorpay():
    """
    Receive a Razorpay webhook, prove it is genuine, and score it.

    >>> THE ORDER HERE IS THE SECURITY <<<
    Verify the signature FIRST, against the raw bytes, before parsing anything.
    Everything after the signature check treats the body as trusted; everything
    before it must treat the body as hostile.

    Status codes are chosen for how a gateway reacts to them:
      401  bad or missing signature  -- never acknowledged, never retried by us
      400  genuine but unmappable    -- do not ask Razorpay to retry a payload
                                        that will fail identically forever
      200  scored, or a duplicate    -- acknowledged so retries stop
    """
    started = time.perf_counter()

    raw = request.get_data()
    signature = request.headers.get("X-Razorpay-Signature")

    if not RAZORPAY_WEBHOOK_SECRET:
        # Refusing to run beats running without authentication.
        return jsonify({
            "error": "webhook not configured",
            "detail": "RAZORPAY_WEBHOOK_SECRET is unset, so signatures cannot "
                      "be verified and every request is refused.",
        }), 503

    if not verify_signature(raw, signature, RAZORPAY_WEBHOOK_SECRET):
        record_webhook({
            "outcome": "REJECTED",
            "reason": "signature verification failed",
            "signature_present": bool(signature),
            "bytes": len(raw),
        })
        return jsonify({"error": "signature verification failed"}), 401

    try:
        event = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return jsonify({"error": "body must be valid JSON"}), 400

    try:
        upi_payload, notes = razorpay_to_upi(event, MERCHANT_VPA)

        # Replay: answer with the verdict we already reached, and do NOT let the
        # payment enter the account history a second time.
        payment_id = notes.get("razorpay_payment_id")
        if payment_id:
            previous = WEBHOOK_SEEN.get(payment_id)
            if previous is not None:
                return jsonify(dict(previous, duplicate=True)), 200

        txn, upi_notes = upi_to_transaction(upi_payload)
        result = score_one(txn)
        transaction_id = str(uuid.uuid4())
        CACHE.put(transaction_id, result)

        latency = 1000 * (time.perf_counter() - started)
        response = public_view(result, transaction_id, latency)
        response["upi"] = upi_notes
        response["razorpay"] = notes

        # An observation is not a decision. Say so in the response rather than
        # letting a caller mistake a post-hoc report for an authorisation.
        response["enforceable"] = notes["decidable"]
        if not notes["decidable"]:
            response["decision_note"] = (
                f"'{notes['event']}' is not the authorisation moment, so this "
                f"verdict is a report. Only '{DECIDABLE_EVENT}' can be acted on."
            )

        record_webhook({
            "outcome": "SCORED",
            "event": notes["event"],
            "payment_id": payment_id,
            "amount_rupees": notes["amount_rupees"],
            "payer_vpa": notes["payer_vpa"],
            "decision": result["decision"],
            "fraud_score": result["fraud_score"],
            "enforceable": notes["decidable"],
        })

        if payment_id:
            WEBHOOK_SEEN.put(payment_id, response)
        return jsonify(response), 200

    except ValidationError as e:
        record_webhook({"outcome": "UNMAPPABLE", "details": e.errors})
        return jsonify({"error": "unmappable webhook", "details": e.errors}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "scoring failed",
                        "detail": f"{type(e).__name__}: {e}"}), 500


@app.get("/webhook/razorpay/recent")
def webhook_recent():
    """
    What has actually arrived on the webhook, newest last.

    The first question when wiring a gateway is never "what score did it get" --
    it is "did anything reach me, and was it signed correctly?" A 401 here with
    `signature_present: true` means the secret is wrong; nothing at all means
    the tunnel or the dashboard URL is wrong. Those are different problems and
    this is how you tell them apart.
    """
    with WEBHOOK_LOG_LOCK:
        events = list(WEBHOOK_LOG)
    return jsonify({
        "configured": bool(RAZORPAY_WEBHOOK_SECRET),
        "merchant_vpa": MERCHANT_VPA,
        "received": len(events),
        "events": events,
    })


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
