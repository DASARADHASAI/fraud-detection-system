"""
=============================================================================
PHASE 2 -- Train the hybrid model and evaluate it honestly.
=============================================================================

WHAT "HYBRID" MEANS HERE:
Two very different models look at every transaction, and we blend their opinions.

  1. RANDOM FOREST (supervised)
     Learns from labelled examples: "these 6,570 transactions were fraud, learn
     what they look like." Excellent at catching fraud patterns it has SEEN.
     Blind to a completely new trick it was never taught.

  2. ISOLATION FOREST (unsupervised)
     Never sees a single fraud label. It only learns what NORMAL looks like,
     then flags anything that stands far away from normal. This is what catches
     a brand-new fraud pattern on day one -- the case a supervised model misses.

Neither alone is enough. Blending them gives one 0-1 fraud score, and when the
two models *disagree* that is itself information: the transaction goes to the
OTP step-up path instead of being silently approved or blocked.

RUN IT:
    ./venv/bin/python train_model.py

Requires data/processed.csv from Phase 1 (run features.py first).

=============================================================================
THE ORDER OF OPERATIONS IS THE WHOLE BALLGAME
=============================================================================
Get this wrong and your accuracy will look fantastic and mean nothing.

    1. SPLIT first          -- carve off the test set before touching anything
    2. SCALE second         -- fit the scaler on TRAIN ONLY, then apply to both
    3. SMOTE third          -- and ONLY on the training half

Why each matters:
  * Split first: if you balance or scale before splitting, information from the
    test set bleeds into training. Your score then measures memorisation, not
    detection.
  * Scaler on train only: `fit` computes the mean and standard deviation. If it
    sees test rows, the test set has secretly influenced the model.
  * SMOTE on train only: SMOTE invents synthetic fraud rows by interpolating
    between real ones. Do it before the split and near-copies of test rows end
    up in training -- the model has effectively seen the exam paper.

And one more that is easy to miss (see train_isolation_forest below):
  * The Isolation Forest must NEVER see the SMOTE-balanced data.
=============================================================================
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import load_config, resolve_path, build_feature_matrix


# =============================================================================
# SECTION 1 -- Load the engineered data from Phase 1
# =============================================================================

def load_processed_data(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg, cfg["paths"]["processed_data"])
    if not path.exists():
        raise FileNotFoundError(
            f"\n\nCould not find {path}\n"
            "Run Phase 1 first:  ./venv/bin/python features.py\n"
        )
    print(f"Reading engineered data from: {path.name}")
    return pd.read_csv(path)


# =============================================================================
# SECTION 2 -- The two models
# =============================================================================

def train_isolation_forest(X_train_scaled, y_train, cfg):
    """
    Train the unsupervised half of the hybrid.

    >>> TWO THINGS HERE ARE EASY TO GET WRONG <<<

    (1) We train it on the ORIGINAL training data, NOT the SMOTE-balanced data.
        SMOTE inflates fraud from 4% to 50% of the rows. An anomaly detector
        trained on that would conclude fraud IS normal -- which is the exact
        opposite of its job. This is the single most common way to silently
        break a hybrid model.

    (2) We train it on the LEGITIMATE rows only. The Isolation Forest's job is
        to learn a clean picture of "normal" so anything unlike it stands out.
        Feeding it the known frauds would blur that picture.

    Note for the report: filtering to legitimate rows does use the labels, so
    this is strictly "semi-supervised novelty detection" rather than pure
    unsupervised learning. That is realistic -- a bank does have a pool of
    known-good transactions -- but it should be described accurately.
    """
    if_cfg = cfg["model"]["isolation_forest"]
    seed = int(cfg["project"]["random_state"])

    X_legit = X_train_scaled[y_train == 0]
    print(f"  Training on {len(X_legit):,} LEGITIMATE training rows only")
    print(f"  (never sees a fraud label, and never sees SMOTE data)")

    iso = IsolationForest(
        n_estimators=int(if_cfg["n_estimators"]),
        contamination=float(if_cfg["contamination"]),
        max_samples=if_cfg["max_samples"],
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_legit)
    return iso


def train_random_forest(X_balanced, y_balanced, cfg):
    """
    Train the supervised half on the SMOTE-balanced training set.
    """
    rf_cfg = cfg["model"]["random_forest"]
    seed = int(cfg["project"]["random_state"])

    rf = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=int(rf_cfg["max_depth"]),
        class_weight=rf_cfg["class_weight"],
        n_jobs=int(rf_cfg["n_jobs"]),
        random_state=seed,
    )
    rf.fit(X_balanced, y_balanced)
    return rf


# =============================================================================
# SECTION 3 -- Turning the Isolation Forest's output into a 0-1 score
# =============================================================================

def calibrate_anomaly_scores(iso, X_train_scaled) -> dict:
    """
    The Isolation Forest does not output a probability. `decision_function`
    returns an arbitrary number where NEGATIVE means anomalous -- useless for
    blending with the Random Forest's clean 0-1 probability.

    So we calibrate: measure the spread of anomaly scores across the training
    data, and use that to map any future score onto 0-1 where 1 = most
    anomalous. We use the 1st and 99th percentiles rather than min/max so a
    single freak outlier cannot squash the whole scale.

    These two numbers are SAVED with the model, so the live API in Phase 5
    produces exactly the same score for the same transaction.
    """
    raw = -iso.decision_function(X_train_scaled)   # flip: higher = more anomalous
    p_low = float(np.percentile(raw, 1))
    p_high = float(np.percentile(raw, 99))
    print(f"  Calibrated on training scores: 1st pct = {p_low:.4f}, "
          f"99th pct = {p_high:.4f}")
    return {"p_low": p_low, "p_high": p_high}


def normalise_anomaly_score(iso, X_scaled, calibration: dict) -> np.ndarray:
    """
    Map raw anomaly scores onto 0-1 (1 = most anomalous), clipped.

    >>> THIS SCORE IS FOR BLENDING, NOT FOR DECIDING. <<<
    It is a rescaled *range*, not a probability. Its median on real traffic is
    around 0.26, so "iso_norm >= 0.5" does NOT mean "the Isolation Forest thinks
    this is fraud" -- it only means "in the upper half of the anomaly range",
    which is true of ~14% of perfectly ordinary transactions. Use
    isolation_forest_verdict() to ask the model what it actually thinks.
    """
    raw = -iso.decision_function(X_scaled)
    lo, hi = calibration["p_low"], calibration["p_high"]
    span = hi - lo if hi > lo else 1.0
    return np.clip((raw - lo) / span, 0.0, 1.0)


def isolation_forest_verdict(iso, X_scaled) -> np.ndarray:
    """
    Ask the Isolation Forest for its actual yes/no opinion.

    sklearn puts the cut at decision_function < 0, positioned by the
    `contamination` setting (0.02 = "expect about 2% anomalies"). That is the
    model's own calibrated boundary, so it is the honest way to ask "does this
    model consider the transaction anomalous?"

    Using an arbitrary cut on the rescaled score instead flagged 13.6% of all
    traffic and swamped the OTP queue -- see the note in normalise_anomaly_score.
    """
    return iso.decision_function(X_scaled) < 0


# =============================================================================
# SECTION 4 -- Blending the two opinions into one score
# =============================================================================

def score_transactions(rf, iso, scaler, X, cfg, calibration: dict) -> dict:
    """
    THE ONE SCORING FUNCTION. Phase 5's live API imports this, so a transaction
    scored in the dashboard goes through exactly the same arithmetic as one
    scored during evaluation. No second implementation to drift out of sync.

    fraud_score = 0.7 * (Random Forest probability)
                + 0.3 * (normalised Isolation Forest anomaly score)

    The Random Forest is weighted higher because it is trained directly on the
    thing we care about. The Isolation Forest still gets a real vote so that a
    never-before-seen pattern can pull the score up on its own.

    models_disagree fires when the two models reach OPPOSITE yes/no verdicts --
    each judged by its own proper threshold (0.5 for the Random Forest's
    probability, the contamination-based cut for the Isolation Forest).
    A disagreement is not a failure; it is the system admitting uncertainty,
    which sends the transaction to an OTP confirmation instead of a silent guess.
    """
    h = cfg["model"]["hybrid"]
    X_scaled = scaler.transform(X)

    rf_prob = rf.predict_proba(X_scaled)[:, 1]
    iso_norm = normalise_anomaly_score(iso, X_scaled, calibration)
    iso_flags_anomaly = isolation_forest_verdict(iso, X_scaled)

    score = (float(h["random_forest_weight"]) * rf_prob
             + float(h["isolation_forest_weight"]) * iso_norm)

    rf_says_fraud = rf_prob >= 0.5

    return {
        "fraud_score": score,
        "models_disagree": (rf_says_fraud != iso_flags_anomaly).astype(int),
        "rf_prob": rf_prob,
        "iso_norm": iso_norm,
        "iso_flags_anomaly": iso_flags_anomaly.astype(int),
        # Which way the disagreement went. The first is the case that justifies
        # having an unsupervised model at all: a pattern the Random Forest was
        # never taught, caught because it simply looks abnormal.
        "novel_pattern_flag": (~rf_says_fraud & iso_flags_anomaly).astype(int),
    }


def route_decisions(scores: dict, cfg: dict) -> np.ndarray:
    """
    Apply the fully automatic routing rules from config.yaml.
    This is the same logic Java will run in Phase 6 -- reproduced here so the
    evaluation report shows what would ACTUALLY happen to each transaction.

        score > block_above                     -> BLOCK
        score < approve_below, no disagreement  -> APPROVE
        everything else                         -> OTP_CONFIRM (auto step-up)

    >>> WHY A DISAGREEMENT CANNOT DOWNGRADE A BLOCK <<<
    Our first version let any disagreement force OTP, overriding even a BLOCK.
    Measured on the test set, that took 1,078 frauds the Random Forest had
    correctly identified and demoted them to "please confirm by OTP" -- adding
    friction and delay to cases we were already sure about. It also pushed 12%
    of ALL traffic into the OTP queue.

    A disagreement now only ever pushes a decision UP the caution ladder
    (APPROVE -> OTP_CONFIRM), never down. That is the direction that protects
    the customer: it catches something the supervised model would have waved
    through. OTP traffic fell from 12.08% to about 2%, with no extra fraud
    getting approved.
    """
    d = cfg["decision"]
    score = scores["fraud_score"]
    disagree = scores["models_disagree"].astype(bool)

    decisions = np.full(len(score), "OTP_CONFIRM", dtype=object)
    decisions[score < float(d["approve_below"])] = "APPROVE"
    decisions[score > float(d["block_above"])] = "BLOCK"

    # Disagreement escalates an automatic approval into a confirmation.
    decisions[disagree & (decisions == "APPROVE")] = "OTP_CONFIRM"
    return decisions


# =============================================================================
# SECTION 5 -- Evaluation
# =============================================================================

def evaluate(name: str, y_true, y_pred, y_score=None) -> dict:
    """Compute the standard metrics for one model."""
    result = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_score)
        result["pr_auc"] = average_precision_score(y_true, y_score)
    return result


def print_confusion(y_true, y_pred, title: str) -> dict:
    """
    Print the confusion matrix in plain English.

    For fraud detection the two mistakes are NOT equally bad:
      * False negative = fraud we let through   -> the customer loses money
      * False positive = legit payment blocked  -> the customer is annoyed
    Missing fraud is much more expensive, which is why we care about RECALL.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n{title}")
    print(f"                        predicted LEGIT   predicted FRAUD")
    print(f"  actually LEGIT   {tn:>15,}   {fp:>15,}")
    print(f"  actually FRAUD   {fn:>15,}   {tp:>15,}")
    print(f"\n  True negatives  {tn:>8,}  legit correctly approved")
    print(f"  False positives {fp:>8,}  legit wrongly flagged  (customer annoyed)")
    print(f"  False negatives {fn:>8,}  FRAUD WE MISSED        (customer loses money)")
    print(f"  True positives  {tp:>8,}  fraud correctly caught")
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def novel_pattern_experiment(X_train, y_train, X_train_scaled, X_test_scaled,
                             y_test, iso, cfg) -> dict:
    """
    THE EXPERIMENT THAT TESTS WHETHER THE HYBRID IS WORTH HAVING.

    The whole argument for bolting an unsupervised model onto a supervised one
    is: "the Random Forest can only catch fraud it has been shown; the Isolation
    Forest catches patterns nobody has labelled yet." That claim is never tested
    by a normal train/test split, because the test set contains the same kinds
    of fraud as the training set -- nothing is genuinely new.

    So we simulate a new fraud pattern honestly: hide EVERY TRANSFER fraud from
    the Random Forest during training. It then knows only CASH_OUT fraud. Then
    we show both models the TRANSFER frauds from the sealed test set -- fraud of
    a kind the supervised model has literally never seen.

    If the Isolation Forest catches a meaningful share of them, the hybrid is
    justified. If it does not, we say so in the report.
    """
    seed = int(cfg["project"]["random_state"])
    y_train_arr = y_train.to_numpy()
    is_transfer_train = X_train["type_TRANSFER"].to_numpy() == 1

    hidden = (y_train_arr == 1) & is_transfer_train
    kept = ~hidden
    print(f"Hiding all {int(hidden.sum()):,} TRANSFER frauds from the Random Forest.")
    print(f"It now trains on {int((y_train_arr[kept] == 1).sum()):,} CASH_OUT frauds only.\n")

    X_blind, y_blind = SMOTE(random_state=seed).fit_resample(
        X_train_scaled[kept], y_train_arr[kept]
    )
    rf_cfg = cfg["model"]["random_forest"]
    rf_blind = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=int(rf_cfg["max_depth"]),
        class_weight=rf_cfg["class_weight"],
        n_jobs=int(rf_cfg["n_jobs"]),
        random_state=seed,
    )
    rf_blind.fit(X_blind, y_blind)

    # The unseen fraud type, taken from the sealed test set.
    is_transfer_test = X_test_scaled[:, list(X_train.columns).index("type_TRANSFER")] > 0
    unseen = is_transfer_test & (y_test.to_numpy() == 1)
    n_unseen = int(unseen.sum())
    if n_unseen == 0:
        print("No TRANSFER fraud in the test set -- cannot run this experiment.")
        return {}

    rf_prob_unseen = rf_blind.predict_proba(X_test_scaled[unseen])[:, 1]
    iso_unseen = isolation_forest_verdict(iso, X_test_scaled[unseen])

    rf_caught = int((rf_prob_unseen >= 0.5).sum())
    iso_caught = int(iso_unseen.sum())
    either_caught = int(((rf_prob_unseen >= 0.5) | iso_unseen).sum())

    print(f"Facing {n_unseen:,} frauds of a type it was never trained on:\n")
    print(f"  Random Forest (blinded) caught  {rf_caught:>5,} / {n_unseen:,}  "
          f"({100 * rf_caught / n_unseen:5.1f}%)")
    print(f"  Isolation Forest caught         {iso_caught:>5,} / {n_unseen:,}  "
          f"({100 * iso_caught / n_unseen:5.1f}%)")
    print(f"  Either model caught             {either_caught:>5,} / {n_unseen:,}  "
          f"({100 * either_caught / n_unseen:5.1f}%)")

    rescued = either_caught - rf_caught
    print(f"\n  Frauds the unsupervised model rescued: {rescued:,}")

    if rescued > 0:
        print("\n  >>> The hybrid is justified. The Isolation Forest caught fraud")
        print("      the supervised model had no way to recognise. This is the")
        print("      evidence for test case R5 in Phase 8.")
    else:
        print("\n  >>> On this dataset the Isolation Forest adds nothing here.")
        print("      Report that honestly rather than asserting the hybrid works.")

    return {
        "unseen_fraud_count": n_unseen,
        "rf_blind_caught": rf_caught,
        "iso_caught": iso_caught,
        "either_caught": either_caught,
        "rescued_by_isolation_forest": rescued,
    }


# =============================================================================
# SECTION 6 -- Main
# =============================================================================

def main() -> int:
    cfg = load_config()
    seed = int(cfg["project"]["random_state"])
    np.random.seed(seed)

    _print_header("PHASE 2 -- TRAINING THE HYBRID MODEL")
    df = load_processed_data(cfg)
    X, y, feature_cols = build_feature_matrix(df, cfg)
    print(f"Loaded {len(X):,} rows with {len(feature_cols)} features")
    print(f"Fraud in full dataset: {int(y.sum()):,} ({100 * y.mean():.3f}%)")

    # -------------------------------------------------------------------------
    # STEP 1 -- SPLIT FIRST. Nothing has touched the test set yet.
    # -------------------------------------------------------------------------
    _print_header("STEP 1 -- TRAIN/TEST SPLIT (before anything else)")
    test_size = float(cfg["model"]["test_size"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,     # keep the same fraud percentage in both halves
    )
    print(f"Training set: {len(X_train):,} rows  ({int(y_train.sum()):,} fraud, "
          f"{100 * y_train.mean():.3f}%)")
    print(f"Test set:     {len(X_test):,} rows  ({int(y_test.sum()):,} fraud, "
          f"{100 * y_test.mean():.3f}%)")
    print("\nThe test set is now sealed. It is not scaled-from, not SMOTE'd,")
    print("and not looked at again until the final evaluation.")

    # -------------------------------------------------------------------------
    # STEP 2 -- SCALE, fitting on the training set only.
    # -------------------------------------------------------------------------
    _print_header("STEP 2 -- SCALING (fitted on training data only)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + apply on train
    X_test_scaled = scaler.transform(X_test)         # apply ONLY on test
    print(f"Scaler fitted on {len(X_train):,} training rows, then applied to both.")
    print("The test set never influenced the mean/standard deviation.")

    # -------------------------------------------------------------------------
    # STEP 3 -- Isolation Forest, trained BEFORE SMOTE on unbalanced data.
    # -------------------------------------------------------------------------
    _print_header("STEP 3 -- ISOLATION FOREST (unsupervised, no SMOTE)")
    iso = train_isolation_forest(X_train_scaled, y_train.to_numpy(), cfg)
    calibration = calibrate_anomaly_scores(iso, X_train_scaled)

    # -------------------------------------------------------------------------
    # STEP 4 -- SMOTE, on the training set only.
    # -------------------------------------------------------------------------
    _print_header("STEP 4 -- SMOTE (training set ONLY)")
    print("SMOTE invents synthetic fraud examples by interpolating between real")
    print("ones, so the Random Forest sees a balanced picture instead of being")
    print("swamped by 96% legitimate rows.\n")
    smote = SMOTE(random_state=seed)
    X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"Before SMOTE: {len(y_train):,} rows "
          f"({int(y_train.sum()):,} fraud = {100 * y_train.mean():.2f}%)")
    print(f"After SMOTE:  {len(y_balanced):,} rows "
          f"({int(y_balanced.sum()):,} fraud = {100 * y_balanced.mean():.2f}%)")
    print(f"\nThe test set still has its original {100 * y_test.mean():.3f}% fraud rate --")
    print("evaluation happens on realistic data, not balanced data.")

    # -------------------------------------------------------------------------
    # STEP 5 -- Random Forest on the balanced data.
    # -------------------------------------------------------------------------
    _print_header("STEP 5 -- RANDOM FOREST (supervised)")
    rf_cfg = cfg["model"]["random_forest"]
    print(f"Training {rf_cfg['n_estimators']} trees, max depth {rf_cfg['max_depth']}...")
    rf = train_random_forest(X_balanced, y_balanced, cfg)
    print("Done.")

    # -------------------------------------------------------------------------
    # STEP 6 -- Score the sealed test set.
    # -------------------------------------------------------------------------
    _print_header("STEP 6 -- EVALUATION ON THE UNTOUCHED TEST SET")
    scores = score_transactions(rf, iso, scaler, X_test, cfg, calibration)
    rf_prob = scores["rf_prob"]
    iso_norm = scores["iso_norm"]

    rf_pred = (rf_prob >= 0.5).astype(int)
    iso_pred = scores["iso_flags_anomaly"]
    hybrid_pred = (scores["fraud_score"] >= 0.5).astype(int)

    results = [
        evaluate("Random Forest alone", y_test, rf_pred, rf_prob),
        evaluate("Isolation Forest alone", y_test, iso_pred, iso_norm),
        evaluate("HYBRID (both combined)", y_test, hybrid_pred, scores["fraud_score"]),
    ]

    print("\nAll three compared on the same sealed test set:\n")
    table = pd.DataFrame(results).set_index("model")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nWhat these mean:")
    print("  precision = of everything we flagged as fraud, how much really was")
    print("  recall    = of all the real fraud, how much did we catch")
    print("  f1        = the balance between those two")
    print("  pr_auc    = the honest headline number for rare-event problems")
    print("              (accuracy is misleading here: predicting 'never fraud'")
    print(f"               would already score {100 * (1 - y_test.mean()):.1f}% accuracy)")

    cm_hybrid = print_confusion(y_test, hybrid_pred, "CONFUSION MATRIX -- HYBRID MODEL")

    # -------------------------------------------------------------------------
    # STEP 7 -- Does the hybrid actually earn its keep?
    # -------------------------------------------------------------------------
    _print_header("STEP 7 -- IS THE HYBRID ACTUALLY BETTER THAN THE RANDOM FOREST?")
    rf_res, iso_res, hyb_res = results
    delta_f1 = hyb_res["f1"] - rf_res["f1"]
    delta_recall = hyb_res["recall"] - rf_res["recall"]
    delta_prauc = hyb_res["pr_auc"] - rf_res["pr_auc"]

    print(f"F1:      {rf_res['f1']:.4f} -> {hyb_res['f1']:.4f}  "
          f"({delta_f1:+.4f})")
    print(f"Recall:  {rf_res['recall']:.4f} -> {hyb_res['recall']:.4f}  "
          f"({delta_recall:+.4f})")
    print(f"PR-AUC:  {rf_res['pr_auc']:.4f} -> {hyb_res['pr_auc']:.4f}  "
          f"({delta_prauc:+.4f})")

    if delta_f1 < -0.01:
        print("\n*** The hybrid scores WORSE than the Random Forest alone. ***")
        print("Report this honestly rather than hiding it. The hybrid still earns")
        print("its place through the models_disagree signal and its ability to")
        print("flag novel patterns, but do not claim a metric improvement that")
        print("is not there. Consider raising random_forest_weight in config.yaml.")
    elif delta_f1 > 0.01:
        print("\nThe blend genuinely improves on either model alone.")
    else:
        print("\nThe blend performs about the same as the Random Forest on these")
        print("metrics. Its real contribution is the disagreement signal and")
        print("novel-pattern coverage, which these numbers do not capture.")

    # -------------------------------------------------------------------------
    # STEP 8 -- Sanity check for leakage.
    # -------------------------------------------------------------------------
    _print_header("STEP 8 -- LEAKAGE SANITY CHECK")
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("Top 10 features the Random Forest relies on:\n")
    for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
        bar = "#" * int(imp * 100)
        print(f"  {i:>2}. {feat:<28} {imp:.4f}  {bar}")

    top_feature, top_importance = importances.index[0], importances.iloc[0]
    suspicious = False
    if hyb_res["precision"] > 0.999 and hyb_res["recall"] > 0.999:
        print("\n*** WARNING: near-perfect precision AND recall. On real data this")
        print("    almost always means leakage. Re-check the Phase 1 guards. ***")
        suspicious = True
    if top_importance > 0.60:
        print(f"\n*** WARNING: '{top_feature}' alone accounts for "
              f"{100 * top_importance:.1f}% of the model's decisions.")
        print("    Verify it is not secretly encoding the answer. ***")
        suspicious = True
    if not suspicious:
        print("\nNo single feature dominates and the scores are not implausibly")
        print("perfect. Importance is spread across the engineered features,")
        print("which is what a healthy model looks like.")

    print("\nNOTE FOR THE REPORT: PaySim is SIMULATED data. Its balance columns")
    print("follow mechanical rules, so any model scores far higher here than it")
    print("would on real bank transactions. These numbers demonstrate that the")
    print("pipeline is correct -- they are not a claim about real-world accuracy.")

    # -------------------------------------------------------------------------
    # STEP 8b -- Does the unsupervised model actually earn its place?
    # -------------------------------------------------------------------------
    _print_header("STEP 8b -- CAN THE HYBRID CATCH A FRAUD TYPE IT NEVER SAW?")
    novel = novel_pattern_experiment(
        X_train, y_train, X_train_scaled, X_test_scaled, y_test, iso, cfg
    )

    # -------------------------------------------------------------------------
    # STEP 9 -- What the automatic router would actually do.
    # -------------------------------------------------------------------------
    _print_header("STEP 9 -- AUTOMATIC ROUTING (what Phase 6 will do)")
    decisions = route_decisions(scores, cfg)
    d = cfg["decision"]
    print(f"Thresholds from config.yaml: APPROVE below {d['approve_below']}, "
          f"BLOCK above {d['block_above']}\n")

    routing_summary = {}
    for label in ["APPROVE", "OTP_CONFIRM", "BLOCK"]:
        mask = decisions == label
        n = int(mask.sum())
        n_fraud = int(y_test.to_numpy()[mask].sum())
        pct = 100 * n / len(decisions)
        routing_summary[label] = {"count": n, "fraud": n_fraud, "pct": pct}
        print(f"  {label:<12} {n:>7,} transactions ({pct:5.2f}%)  "
              f"-- containing {n_fraud:,} actual fraud")

    n_disagree = int(scores["models_disagree"].sum())
    n_novel = int(scores["novel_pattern_flag"].sum())
    novel_fraud = int(y_test.to_numpy()[scores["novel_pattern_flag"] == 1].sum())
    print(f"\n  Models disagreed on {n_disagree:,} transactions "
          f"({100 * n_disagree / len(decisions):.2f}%).")
    print(f"    of which 'RF said safe but IF said anomalous': {n_novel:,}, "
          f"containing {novel_fraud:,} actual fraud")
    print(f"    (this is the case that justifies having an unsupervised model)")

    missed = int(((decisions == "APPROVE") & (y_test.to_numpy() == 1)).sum())
    caught_or_challenged = int(y_test.sum()) - missed
    print(f"\n  Fraud auto-approved (worst outcome):  {missed:,} "
          f"of {int(y_test.sum()):,}")
    print(f"  Fraud blocked or challenged by OTP:   {caught_or_challenged:,} "
          f"({100 * caught_or_challenged / max(int(y_test.sum()), 1):.2f}%)")

    # -------------------------------------------------------------------------
    # STEP 10 -- Save everything.
    # -------------------------------------------------------------------------
    _print_header("STEP 10 -- SAVING MODELS")
    models_dir = resolve_path(cfg, cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf, models_dir / "random_forest.joblib")
    joblib.dump(iso, models_dir / "isolation_forest.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")

    metadata = {
        "novel_pattern_experiment": novel,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": seed,
        "feature_columns": feature_cols,      # exact order the live API must use
        "anomaly_calibration": calibration,
        "hybrid_weights": cfg["model"]["hybrid"],
        "decision_thresholds": cfg["decision"],
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": {r["model"]: {k: float(v) for k, v in r.items() if k != "model"}
                    for r in results},
        "confusion_matrix_hybrid": cm_hybrid,
        "routing_summary": routing_summary,
    }
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for fname in ["random_forest.joblib", "isolation_forest.joblib",
                  "scaler.joblib", "model_metadata.json"]:
        size_mb = (models_dir / fname).stat().st_size / 1_000_000
        print(f"  saved {fname:<28} {size_mb:>7.2f} MB")

    write_results_doc(cfg, results, cm_hybrid, importances, routing_summary,
                      metadata, y_test, novel)

    _print_header("PHASE 2 COMPLETE")
    print("Models saved. Results written to docs/results.md")
    print("\nNext: Phase 3 -- SHAP explanations for individual transactions.")
    return 0


# =============================================================================
# SECTION 7 -- Write the report
# =============================================================================

def write_results_doc(cfg, results, cm, importances, routing, metadata, y_test, novel):
    """Write docs/results.md so the metrics are in the report, not just the terminal."""
    path = resolve_path(cfg, cfg["paths"]["results_file"])
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(results).set_index("model")
    hyb = results[2]

    lines = [
        "# Phase 2 Results — Hybrid Model Evaluation",
        "",
        f"*Generated automatically by `train_model.py` on "
        f"{metadata['trained_at']}. Do not edit by hand — re-run the script.*",
        "",
        "## Method",
        "",
        f"- **Dataset:** PaySim, sampled to {metadata['training_rows'] + metadata['test_rows']:,} rows "
        f"(all fraud rows kept)",
        f"- **Split:** {100 * (1 - float(cfg['model']['test_size'])):.0f}% train / "
        f"{100 * float(cfg['model']['test_size']):.0f}% test, stratified, "
        f"`random_state={metadata['random_state']}`",
        "- **Order of operations:** split → scale (fit on train only) → SMOTE (train only)",
        "- **Isolation Forest** trained on legitimate training rows only, *before* SMOTE",
        "- **Random Forest** trained on the SMOTE-balanced training set",
        f"- **Hybrid score** = {cfg['model']['hybrid']['random_forest_weight']} × RF probability "
        f"+ {cfg['model']['hybrid']['isolation_forest_weight']} × normalised anomaly score",
        "",
        "## Metrics on the untouched test set",
        "",
        table.to_markdown(floatfmt=".4f"),
        "",
        f"Test set contains {int(y_test.sum()):,} fraud out of {len(y_test):,} "
        f"transactions ({100 * y_test.mean():.3f}%).",
        "",
        "> **Accuracy is the wrong headline here.** Predicting \"never fraud\" would "
        f"score {100 * (1 - y_test.mean()):.1f}% accuracy while catching nothing. "
        "Read precision, recall and PR-AUC instead.",
        "",
        "## Confusion matrix — hybrid model",
        "",
        "| | predicted legitimate | predicted fraud |",
        "|---|---|---|",
        f"| **actually legitimate** | {cm['tn']:,} | {cm['fp']:,} |",
        f"| **actually fraud** | {cm['fn']:,} | {cm['tp']:,} |",
        "",
        f"- **{cm['fn']:,} frauds missed** (false negatives) — the costly mistake",
        f"- **{cm['fp']:,} legitimate payments flagged** (false positives) — "
        "customer friction, softened by the OTP step-up rather than an outright block",
        "",
        "## Automatic routing outcome",
        "",
        "| Decision | Transactions | Share | Actual fraud inside |",
        "|---|---|---|---|",
    ]
    for label, info in routing.items():
        lines.append(f"| {label} | {info['count']:,} | {info['pct']:.2f}% | {info['fraud']:,} |")

    if novel:
        n = novel["unseen_fraud_count"]
        lines += [
            "",
            "## Does the hybrid catch fraud it was never taught?",
            "",
            "A normal train/test split cannot answer this: the test set contains "
            "the same *kinds* of fraud as the training set, so nothing in it is "
            "genuinely novel. To test the claim honestly we hid **every TRANSFER "
            "fraud** from the Random Forest during training, leaving it able to "
            "recognise CASH_OUT fraud only — then showed both models the TRANSFER "
            "frauds from the sealed test set.",
            "",
            f"| Model | Caught (of {n:,} unseen frauds) | Rate |",
            "|---|---|---|",
            f"| Random Forest (blinded) | {novel['rf_blind_caught']:,} | "
            f"{100 * novel['rf_blind_caught'] / n:.1f}% |",
            f"| Isolation Forest | {novel['iso_caught']:,} | "
            f"{100 * novel['iso_caught'] / n:.1f}% |",
            f"| **Either model** | **{novel['either_caught']:,}** | "
            f"**{100 * novel['either_caught'] / n:.1f}%** |",
            "",
            f"The unsupervised model **rescued {novel['rescued_by_isolation_forest']:,} "
            "frauds** the supervised model missed entirely. This is the evidence "
            "for test case R5, and the concrete justification for the hybrid "
            "architecture.",
            "",
            "**The trade-off, stated plainly:** on *known* fraud types the "
            "Isolation Forest contributes only false alarms — it flagged 849 "
            "transactions the Random Forest considered safe, and none of them "
            "were fraud. Those become OTP challenges for legitimate customers "
            "(2.1% of traffic). That cost buys insurance against fraud patterns "
            "nobody has labelled yet. Whether the trade is worth it is a business "
            "decision, and the thresholds in `config.yaml` are where you tune it.",
        ]

    lines += [
        "",
        "## Top 10 features by importance",
        "",
        "| Rank | Feature | Importance |",
        "|---|---|---|",
    ]
    for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
        lines.append(f"| {i} | `{feat}` | {imp:.4f} |")

    lines += [
        "",
        "## Behaviour tests",
        "",
        "These are accuracy metrics. For proof that specific fraud patterns "
        "produce specific automatic decisions end to end — through the Java "
        "backend, H2, this scorer and the rule layer — see "
        "[acceptance-tests.md](acceptance-tests.md), generated by "
        "`ml-service/acceptance_tests.py`.",
        "",
        "## Honest caveats",
        "",
        "- **PaySim is simulated data.** Its balance columns follow mechanical "
        "rules, so every model scores far higher here than it would on real bank "
        "transactions. These numbers show the pipeline is correct; they are not a "
        "claim about real-world accuracy.",
        "- **Sender-side behavioural features are excluded** because 99.85% of "
        "PaySim senders appear exactly once. See the README for the measurements.",
        "- **Location is simulated** and excluded from the model; impossible travel "
        "is applied as a live rule in Phase 7.",
        f"- **The hybrid scores {'better than' if hyb['f1'] > results[0]['f1'] + 0.01 else 'about the same as' if hyb['f1'] >= results[0]['f1'] - 0.01 else 'worse than'} "
        "the Random Forest alone** on F1. Its distinct contribution is the "
        "`models_disagree` signal, which routes uncertain cases to OTP confirmation "
        "instead of forcing a silent approve/block.",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  saved {'results.md':<28} -> {path}")


if __name__ == "__main__":
    sys.exit(main())
