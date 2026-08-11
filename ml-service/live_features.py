"""
=============================================================================
PHASE 5 (part 1) -- Turning ONE live transaction into the model's 28 features.
=============================================================================

THE PROBLEM THIS FILE SOLVES:
Phase 1 engineered features over 6.3 million rows at once. It could compute
"how many payments has this receiver taken before?" by looking at the whole
file. A live request has none of that -- it is a single JSON object arriving
from the Java backend, with no history attached.

If the numbers we build here do not mean EXACTLY what they meant during
training, the model reads them wrong and the score is silently garbage. That
failure has no error message: the API returns 200 and a plausible-looking
number. So this file is written to mirror features.py line for line, and
test_live_features.py checks the two against each other.

WHERE HISTORY COMES FROM AT SERVING TIME, in order of preference:

  1. The caller supplies it. The Java backend has an H2 database of past
     transactions, so it can pass dest_txn_count and friends directly. This is
     the correct source in production.

  2. This service's own in-memory record of what it has seen. This is what
     makes the demo work: submit three payments to the same receiver and the
     velocity count genuinely rises.

  3. Global averages. Exactly what training did for a receiver with too little
     history -- and on PaySim that was 56% of rows, so this is a well-trodden
     path, not an edge case.

The in-memory history is deliberately NOT persisted. It is demo scaffolding;
production reads from the bank's database. Persisting it would create a second
source of truth that silently drifts from H2.
=============================================================================
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from features import TRANSACTION_TYPES, resolve_path


# =============================================================================
# SECTION 1 -- Validating what the Java service sent us
# =============================================================================

REQUIRED_FIELDS = [
    "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
]
NUMERIC_FIELDS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]


class ValidationError(ValueError):
    """Raised when the incoming transaction cannot be scored. Becomes HTTP 400."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_transaction(txn: dict) -> dict:
    """
    Check and clean one incoming transaction.

    Rejecting loudly here is the whole point. A missing field silently treated
    as zero would produce a confident fraud score computed from nothing, and
    nobody downstream would ever know. Better a 400 the Java side can log.
    """
    if not isinstance(txn, dict):
        raise ValidationError(["request body must be a JSON object"])

    errors: list[str] = []
    clean: dict = {}

    for field in REQUIRED_FIELDS:
        if field not in txn or txn[field] is None:
            errors.append(f"'{field}' is required")

    # --- Numbers must be finite. NaN and infinity break scaling, scoring and
    #     the explanation phrases in three different ways. ---
    for field in NUMERIC_FIELDS:
        if field not in txn or txn[field] is None:
            continue
        try:
            value = float(txn[field])
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a number, got {txn[field]!r}")
            continue
        if not math.isfinite(value):
            errors.append(f"'{field}' must be a finite number, got {txn[field]!r}")
            continue
        if value < 0:
            errors.append(f"'{field}' cannot be negative, got {value}")
            continue
        clean[field] = value

    # --- Transaction type must be one the model was trained on ---
    txn_type = str(txn.get("type", "")).strip().upper()
    if txn_type and txn_type not in TRANSACTION_TYPES:
        errors.append(f"'type' must be one of {TRANSACTION_TYPES}, got {txn_type!r}")
    clean["type"] = txn_type

    # --- Account identifiers ---
    for field in ("nameOrig", "nameDest"):
        value = txn.get(field)
        if value is not None:
            text = str(value).strip()
            if not text:
                errors.append(f"'{field}' cannot be blank")
            clean[field] = text

    # --- step is optional. It is PaySim's clock, one unit = one hour. When the
    #     caller does not supply it we use real wall-clock hours, which keeps
    #     hour_of_day meaningful and lets the velocity window work. ---
    if "step" in txn and txn["step"] is not None:
        try:
            clean["step"] = int(txn["step"])
        except (TypeError, ValueError):
            errors.append(f"'step' must be a whole number, got {txn['step']!r}")
    else:
        clean["step"] = int(time.time() // 3600)

    # --- Optional extras used by the rule layer, not by the model ---
    if txn.get("location"):
        clean["location"] = str(txn["location"]).strip()
    for optional in ("location_lat", "location_lon"):
        if txn.get(optional) is not None:
            try:
                clean[optional] = float(txn[optional])
            except (TypeError, ValueError):
                errors.append(f"'{optional}' must be a number")
    clean["trusted_device"] = bool(txn.get("trusted_device", False))

    if errors:
        raise ValidationError(errors)
    return clean


# =============================================================================
# SECTION 2 -- Remembering what we have seen
# =============================================================================

class TransactionHistory:
    """
    A small in-memory record of recent activity per account.

    Thread-safe: Flask handles requests on multiple threads, and two payments
    arriving at once must not corrupt the counters.

    Bounded on purpose. An unbounded dict keyed by account id is a memory leak
    with extra steps -- this keeps only the most recent transactions per account
    and only the most recently active accounts.
    """

    MAX_PER_ACCOUNT = 50
    MAX_ACCOUNTS = 20_000

    def __init__(self):
        self._lock = threading.Lock()
        # account -> deque of (step, amount)
        self._history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.MAX_PER_ACCOUNT))
        # account -> (step, lat, lon, city) for the impossible-travel rule
        self._last_location: dict[str, tuple] = {}

    # -------------------------------------------------------------------------
    def snapshot(self, account: str) -> list[tuple]:
        """Everything we currently know about this account, as (step, amount)."""
        with self._lock:
            return list(self._history.get(account, ()))

    # -------------------------------------------------------------------------
    def last_location(self, account: str):
        with self._lock:
            return self._last_location.get(account)

    # -------------------------------------------------------------------------
    def record(self, txn: dict) -> None:
        """
        Store a transaction AFTER it has been scored.

        >>> ORDER MATTERS AND IT IS EASY TO GET BACKWARDS. <<<
        Training computed every history feature from PRIOR rows only, excluding
        the current one. If we recorded before scoring, a first-ever payment
        would score with dest_txn_count = 1 instead of 0 -- so the live number
        would mean something different from the trained one, on every single
        request. Callers must score first, then record.
        """
        with self._lock:
            for account in (txn.get("nameOrig"), txn.get("nameDest")):
                if account:
                    self._history[account].append(
                        (int(txn["step"]), float(txn["amount"])))

            if txn.get("nameOrig") and txn.get("location_lat") is not None:
                self._last_location[txn["nameOrig"]] = (
                    int(txn["step"]), float(txn["location_lat"]),
                    float(txn["location_lon"]), txn.get("location", ""),
                )

            # Evict the least recently touched accounts if we have grown too big.
            if len(self._history) > self.MAX_ACCOUNTS:
                for account in list(self._history)[:len(self._history) - self.MAX_ACCOUNTS]:
                    self._history.pop(account, None)

    # -------------------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._last_location.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._history)


# =============================================================================
# SECTION 3 -- Building the 28 features
# =============================================================================

class LiveFeatureBuilder:
    """
    Produces the exact feature row the trained model expects.

    Column ORDER is taken from model_metadata.json, not hard-coded, so a
    retrained model with different features cannot silently mismatch.
    """

    def __init__(self, cfg: dict, feature_columns: list[str],
                 history: TransactionHistory | None = None):
        self.cfg = cfg
        self.feature_columns = list(feature_columns)
        self.history = history or TransactionHistory()

        stats_path = resolve_path(cfg, cfg["paths"]["models_dir"]) / "global_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} is missing. Run Phase 1 (features.py) first -- the "
                "live API must use the SAME global averages that training used."
            )
        self.stats = json.loads(stats_path.read_text())
        self.global_mean = float(self.stats["global_mean_amount"])
        self.global_std = float(self.stats["global_std_amount"])
        self.min_history = int(self.stats["min_user_history"])
        self.velocity_window = int(self.stats["velocity_window_steps"])

    # -------------------------------------------------------------------------
    def _account_profile(self, account: str, step: int, amount: float,
                         supplied: dict | None) -> dict:
        """
        Work out the history features for one account.

        `supplied` lets the Java backend pass figures straight from H2, which
        beats anything this service can know. Whatever it does not supply, we
        compute from memory, and whatever we cannot compute falls back to the
        global averages -- the same fallback training used.
        """
        supplied = supplied or {}

        past = self.history.snapshot(account)
        count = int(supplied.get("txn_count", len(past)))

        if "mean_amount" in supplied:
            mean_amount = float(supplied["mean_amount"])
            std_amount = float(supplied.get("std_amount", self.global_std))
            thin = count < self.min_history
        elif count >= self.min_history and past:
            amounts = np.array([a for _, a in past], dtype=float)
            mean_amount = float(amounts.mean())
            std_amount = float(amounts.std())
            thin = False
        else:
            # Too little history to trust -- exactly the training fallback.
            mean_amount = self.global_mean
            std_amount = self.global_std
            thin = True

        if thin:
            mean_amount = self.global_mean
            std_amount = self.global_std

        # Mirror of features.py: guard a near-zero spread before dividing.
        safe_std = std_amount if std_amount > 1.0 else self.global_std
        deviation = float(np.clip((amount - mean_amount) / safe_std, -20, 20))
        ratio = float(min(amount / max(mean_amount, 1.0), 100.0))

        if "velocity_count" in supplied:
            velocity = int(supplied["velocity_count"])
        else:
            velocity = sum(1 for s, _ in past if s > step - self.velocity_window)

        return {
            "txn_count": count,
            "mean_amount": mean_amount,
            "std_amount": std_amount,
            "has_thin_history": int(thin),
            "amount_deviation": deviation,
            "amount_ratio": ratio,
            "velocity_count": velocity,
        }

    # -------------------------------------------------------------------------
    def build(self, txn: dict) -> tuple[pd.DataFrame, dict]:
        """
        Turn one validated transaction into (feature_row, extra_signals).

        `extra_signals` carries things the RULE layer uses but the model does
        not see: the sender profile (inert on PaySim, but real for a live
        customer) and the impossible-travel check.
        """
        amount = float(txn["amount"])
        step = int(txn["step"])
        old_orig = float(txn["oldbalanceOrg"])
        new_orig = float(txn["newbalanceOrig"])
        old_dest = float(txn["oldbalanceDest"])
        new_dest = float(txn["newbalanceDest"])

        f: dict[str, float] = {}

        # --- Raw amounts ---
        f["amount"] = amount
        f["oldbalanceOrg"] = old_orig
        f["newbalanceOrig"] = new_orig
        f["oldbalanceDest"] = old_dest
        f["newbalanceDest"] = new_dest

        # --- Balance errors (identical arithmetic to features.py) ---
        f["errorBalanceOrig"] = new_orig + amount - old_orig
        f["errorBalanceDest"] = old_dest + amount - new_dest
        f["absErrorBalanceOrig"] = abs(f["errorBalanceOrig"])
        f["absErrorBalanceDest"] = abs(f["errorBalanceDest"])

        # --- Draining fingerprints ---
        f["orig_account_emptied"] = int(new_orig == 0 and old_orig > 0 and amount > 0)
        f["dest_was_empty"] = int(old_dest == 0)
        f["amount_to_balance_ratio"] = (
            float(min(amount / old_orig, 10.0)) if old_orig > 0 else 0.0
        )

        # --- Transaction type ---
        for t in TRANSACTION_TYPES:
            f[f"type_{t}"] = int(txn["type"] == t)
        f["is_risky_type"] = int(txn["type"] in self.cfg["data"]["fraud_capable_types"])

        # --- Time ---
        f["hour_of_day"] = step % 24
        f["is_night"] = int(f["hour_of_day"] in (0, 1, 2, 3, 4, 5))

        # --- Receiver profile: what the model actually learned from ---
        dest = self._account_profile(txn["nameDest"], step, amount,
                                     txn.get("dest_history"))
        f["dest_txn_count"] = dest["txn_count"]
        f["dest_mean_amount"] = dest["mean_amount"]
        f["dest_std_amount"] = dest["std_amount"]
        f["dest_has_thin_history"] = dest["has_thin_history"]
        f["dest_amount_deviation"] = dest["amount_deviation"]
        f["dest_amount_ratio"] = dest["amount_ratio"]
        f["dest_velocity_count"] = dest["velocity_count"]
        f["is_new_receiver"] = int(dest["txn_count"] == 0)

        # --- Sender profile: computed for the rule layer and the explanation
        #     text, deliberately NOT a model input. See MODEL_FEATURES. ---
        sender = self._account_profile(txn["nameOrig"], step, amount,
                                       txn.get("user_history"))

        missing = [c for c in self.feature_columns if c not in f]
        if missing:
            raise RuntimeError(
                f"Live feature builder did not produce {missing}. It has drifted "
                "from the trained model -- retrain or fix live_features.py."
            )

        row = pd.DataFrame([[f[c] for c in self.feature_columns]],
                           columns=self.feature_columns)

        return row, {
            "sender_profile": sender,
            "receiver_profile": dest,
            "impossible_travel": self._check_impossible_travel(txn),
        }

    # -------------------------------------------------------------------------
    def _check_impossible_travel(self, txn: dict) -> dict:
        """
        The physical-contradiction rule: could this customer really have got
        from their last payment's location to this one in the time available?

        This is a RULE, not a model feature -- there is no probability involved.
        Either the journey is possible or it is not. Phase 7's dashboard shows
        it; Phase 6 lets it override the score.
        """
        result = {"checked": False, "impossible": False, "detail": None}
        if txn.get("location_lat") is None or txn.get("location_lon") is None:
            return result

        previous = self.history.last_location(txn.get("nameOrig", ""))
        if previous is None:
            result["detail"] = "no previous location on record for this customer"
            return result

        prev_step, prev_lat, prev_lon, prev_city = previous
        from features import _haversine_km
        distance = float(_haversine_km(prev_lat, prev_lon,
                                       txn["location_lat"], txn["location_lon"]))
        hours = max(int(txn["step"]) - int(prev_step), 0)
        hours = max(hours, 0.25)          # floor at 15 min; never divide by zero
        speed = distance / hours

        limit = float(self.cfg["simulated_location"]["max_plausible_speed_kmph"])
        result.update({
            "checked": True,
            "impossible": bool(speed > limit),
            "from": prev_city,
            "to": txn.get("location", ""),
            "distance_km": round(distance, 1),
            "hours_elapsed": round(hours, 2),
            "implied_speed_kmph": round(speed, 1),
            "limit_kmph": limit,
        })
        if result["impossible"]:
            result["detail"] = (
                f"{prev_city} to {txn.get('location', 'here')} is "
                f"{distance:,.0f} km in {hours:.2f} h, implying "
                f"{speed:,.0f} km/h - physically impossible"
            )
        else:
            result["detail"] = (
                f"{distance:,.0f} km in {hours:.2f} h ({speed:,.0f} km/h) "
                f"is physically possible"
            )
        return result
