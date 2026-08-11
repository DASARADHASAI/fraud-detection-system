"""
=============================================================================
PHASE 1 -- Data loading, cleaning, and feature engineering.
=============================================================================

WHAT THIS FILE DOES (in plain English):
A raw bank transaction only tells you "user X sent 5000 rupees to user Y".
That alone is not enough to spot fraud. This file turns each raw row into a
richer set of *clues* (called "features") that a machine learning model can
actually learn from -- things like "this amount is 12x bigger than this user
normally sends" or "the sender's balance maths does not add up".

RUN IT LIKE THIS (from inside the ml-service/ folder):
    ./venv/bin/python features.py

It will print the data shape, the class balance (how much fraud there is),
and the first few rows, then save the engineered data to data/processed.csv.

KEY RULES FOLLOWED HERE:
  * random_state = 42 everywhere, so every run gives identical results.
  * NO SMOTE in this file. Oversampling belongs in Phase 2, AFTER the
    train/test split, otherwise we leak test data into training.
  * `isFlaggedFraud` is DROPPED. It is PaySim's own built-in naive rule --
    using it would be cheating (the model would just copy the answer).
  * Per-user history is built "causally": when we compute a user's normal
    spending, we only ever look at that user's EARLIER transactions, never
    later ones. A real bank cannot see the future, so neither do we.
=============================================================================
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# SECTION 0 -- Config loading
# =============================================================================

def load_config(config_path: str | None = None) -> dict:
    """
    Read config.yaml so that no setting is hard-coded in the source files.

    We walk UP the folder tree looking for config.yaml, so this works whether
    you run the script from the project root or from inside ml-service/.
    """
    if config_path is not None:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        candidate = folder / "config.yaml"
        if candidate.exists():
            with open(candidate, "r") as f:
                cfg = yaml.safe_load(f)
            # Remember where the project root is; other modules need it.
            cfg["_project_root"] = str(folder)
            return cfg

    raise FileNotFoundError(
        "Could not find config.yaml. It should sit in the project root folder."
    )


def resolve_path(cfg: dict, relative_path: str) -> Path:
    """Turn a path from config.yaml into a real absolute path."""
    return (Path(__file__).resolve().parent / relative_path).resolve()


# =============================================================================
# SECTION 1 -- Loading the raw PaySim CSV
# =============================================================================

# The exact columns PaySim ships with. We check for these so that a wrong or
# corrupted download fails immediately with a clear message, instead of
# producing a broken model 20 minutes later.
PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

# PaySim's five transaction types. We fix the order so that the one-hot
# columns are always identical between training and live prediction.
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_raw_data(cfg: dict) -> pd.DataFrame:
    """
    Load the PaySim CSV, sanity-check it, and take a manageable random sample.

    PaySim has ~6.3 million rows. Training on all of them on a laptop is slow,
    so we sample. BUT we keep EVERY fraud row -- fraud is only ~0.13% of the
    data, so throwing any away would cripple the model.
    """
    csv_path = resolve_path(cfg, cfg["paths"]["raw_data"])

    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n\nDataset not found at:\n    {csv_path}\n\n"
            "Download PaySim from https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            "then place the CSV there and rename it to 'transactions.csv'.\n"
            "See docs/DATASET.md for exact step-by-step instructions.\n"
        )

    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- Sanity check: are the columns the ones we expect? ---
    missing = [c for c in PAYSIM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The CSV is missing expected PaySim columns: {missing}\n"
            f"Columns actually found: {list(df.columns)}\n"
            "You may have downloaded a different fraud dataset."
        )

    print(f"Full file loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Drop the leaky column (PaySim's own naive rule -- not a real clue) ---
    for col in cfg["data"]["drop_columns"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped leaky column: '{col}' (this is the dataset's own "
                  f"built-in rule, using it would be cheating)")

    # --- Sample down, but keep every fraud row ---
    sample_size = int(cfg["data"]["sample_size"])
    if sample_size and len(df) > sample_size:
        seed = int(cfg["project"]["random_state"])
        fraud_rows = df[df["isFraud"] == 1]
        legit_rows = df[df["isFraud"] == 0]

        n_legit_needed = max(sample_size - len(fraud_rows), 0)
        n_legit_needed = min(n_legit_needed, len(legit_rows))

        legit_sample = legit_rows.sample(n=n_legit_needed, random_state=seed)
        df = pd.concat([fraud_rows, legit_sample], ignore_index=True)

        # Shuffle so fraud is not all bunched at the top.
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        print(f"Sampled down to {len(df):,} rows "
              f"(kept ALL {len(fraud_rows):,} fraud rows + "
              f"{len(legit_sample):,} random legitimate rows)")

    return df


# =============================================================================
# SECTION 2 -- Simulated location (clearly flagged as NOT real data)
# =============================================================================
#
#   >>> HONESTY NOTE FOR THE REPORT <<<
#   PaySim contains NO location, city, latitude, or longitude column.
#   Real "impossible travel" detection needs one. So we GENERATE a plausible
#   city for each transaction purely so the feature can be demonstrated.
#   This is simulated data, not evidence. It is labelled as such everywhere,
#   and by default it is NOT fed to the machine learning model (see
#   build_feature_matrix below) -- it is used as a live RULE in the dashboard
#   layer (Phase 7) instead, which is where impossible-travel checks belong.
#
# =============================================================================

def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two points on Earth, in kilometres.
    (Standard haversine formula -- works on whole numpy arrays at once.)
    """
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def add_simulated_location(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Give every transaction a simulated city, then work out how fast the user
    would have had to travel between consecutive transactions.

    How the simulation works:
      1. Each user gets a "home city" (chosen deterministically from a seed).
      2. Most of their transactions happen in that home city.
      3. A small fraction of users get a deliberately IMPOSSIBLE jump injected
         (e.g. Mumbai -> Kolkata within the same hour) so that the
         impossible-travel demo has something real to catch.
    """
    loc_cfg = cfg["simulated_location"]
    if not loc_cfg.get("enabled", True):
        return df

    seed = int(cfg["project"]["random_state"])
    rng = np.random.default_rng(seed)

    cities = loc_cfg["cities"]
    city_names = np.array([c["name"] for c in cities])
    city_lat = np.array([c["lat"] for c in cities], dtype=float)
    city_lon = np.array([c["lon"] for c in cities], dtype=float)
    n_cities = len(cities)

    # --- Step 1: assign each user a home city ---
    users = df["nameOrig"].unique()
    home_city_idx = rng.integers(0, n_cities, size=len(users))
    user_to_home = dict(zip(users, home_city_idx))

    home_idx = df["nameOrig"].map(user_to_home).to_numpy()

    # --- Step 2: 90% of transactions happen at home, 10% elsewhere ---
    travels = rng.random(len(df)) < 0.10
    random_idx = rng.integers(0, n_cities, size=len(df))
    city_idx = np.where(travels, random_idx, home_idx)

    # --- Step 3: inject impossible jumps for a small fraction of users ---
    fraction = float(loc_cfg.get("impossible_jump_user_fraction", 0.02))
    n_injected = int(len(users) * fraction)
    injected_users = set(rng.choice(users, size=n_injected, replace=False)) if n_injected else set()

    df = df.assign(_city_idx=city_idx)
    is_injected_user = df["nameOrig"].isin(injected_users)

    # For those users, send their LAST transaction to the city farthest from home.
    if injected_users:
        # Pre-compute, for each city, the index of the farthest other city.
        dist_matrix = _haversine_km(
            city_lat[:, None], city_lon[:, None], city_lat[None, :], city_lon[None, :]
        )
        farthest_from = dist_matrix.argmax(axis=1)

        target_rows = (
            df[is_injected_user]
            .sort_values(["nameOrig", "step"], kind="mergesort")
            .groupby("nameOrig", sort=False)
            .tail(1)
            .index
        )
        df.loc[target_rows, "_city_idx"] = farthest_from[home_idx[target_rows]]

    df["location"] = city_names[df["_city_idx"].to_numpy()]
    df["location_lat"] = city_lat[df["_city_idx"].to_numpy()]
    df["location_lon"] = city_lon[df["_city_idx"].to_numpy()]
    df["location_is_simulated"] = True   # Never let anyone forget this.
    df = df.drop(columns=["_city_idx"])

    # --- Step 4: how fast would the user have had to travel? ---
    # Sort by user then time so "previous transaction" means the right thing.
    ordered = df.sort_values(["nameOrig", "step"], kind="mergesort")
    grp = ordered.groupby("nameOrig", sort=False)

    prev_lat = grp["location_lat"].shift(1)
    prev_lon = grp["location_lon"].shift(1)
    prev_step = grp["step"].shift(1)

    distance_km = _haversine_km(
        prev_lat, prev_lon, ordered["location_lat"], ordered["location_lon"]
    )

    # PaySim's "step" column is one hour of simulated time.
    # Floor the gap at 0.25h (15 min) so we never divide by zero.
    hours_elapsed = (ordered["step"] - prev_step).clip(lower=0.25)
    implied_speed = distance_km / hours_elapsed

    # Assigning Series back to df -- pandas lines them up by index automatically,
    # so the original row order is preserved.
    df["km_from_prev_txn"] = distance_km.fillna(0.0)
    df["hours_since_prev_txn"] = hours_elapsed.fillna(0.0)
    df["implied_speed_kmph"] = implied_speed.fillna(0.0)

    max_speed = float(loc_cfg.get("max_plausible_speed_kmph", 900.0))
    df["impossible_travel"] = (df["implied_speed_kmph"] > max_speed).astype(int)

    n_impossible = int(df["impossible_travel"].sum())
    print(f"Simulated location added. Impossible-travel cases created: "
          f"{n_impossible:,} (SIMULATED -- PaySim has no real geo data)")

    return df


# =============================================================================
# SECTION 3 -- The actual feature engineering
# =============================================================================

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Build all the fraud clues. Each block below is one family of clues.
    """
    print("\nEngineering features...")
    df = df.copy()

    # -------------------------------------------------------------------------
    # 3a. BALANCE ERRORS -- the single strongest signal in PaySim.
    #
    # In an honest transaction the maths balances:
    #     new balance = old balance - amount        (for the sender)
    # So this should come out to zero:
    #     newbalanceOrig + amount - oldbalanceOrg
    # When a fraudster drains an account the simulator leaves the books
    # inconsistent, so this number becomes large and non-zero.
    # -------------------------------------------------------------------------
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # The *size* of the mismatch matters more than its direction.
    df["absErrorBalanceOrig"] = df["errorBalanceOrig"].abs()
    df["absErrorBalanceDest"] = df["errorBalanceDest"].abs()

    # -------------------------------------------------------------------------
    # 3b. BALANCE PATTERN FLAGS -- classic account-draining fingerprints.
    # -------------------------------------------------------------------------
    # Sender's account left at exactly zero after a non-zero transfer.
    df["orig_account_emptied"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0) & (df["amount"] > 0)
    ).astype(int)

    # Receiver's account was empty before the money arrived -- typical of a
    # freshly created "mule" account used to receive stolen funds.
    df["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype(int)

    # What fraction of the sender's balance is being moved?
    # 1.0 means "everything you have". Fraud very often moves 100%.
    df["amount_to_balance_ratio"] = df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)
    df["amount_to_balance_ratio"] = df["amount_to_balance_ratio"].fillna(0.0).clip(upper=10.0)

    # -------------------------------------------------------------------------
    # 3c. TRANSACTION TYPE
    #
    # In PaySim, fraud ONLY ever occurs in TRANSFER and CASH_OUT. We keep the
    # type as a feature (one-hot encoded) and add an explicit "risky type" flag.
    # -------------------------------------------------------------------------
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    risky = cfg["data"]["fraud_capable_types"]
    df["is_risky_type"] = df["type"].isin(risky).astype(int)

    # -------------------------------------------------------------------------
    # 3d. TIME OF DAY
    #
    # PaySim's "step" is hours since the simulation started, so step % 24 gives
    # the hour of day. Money moving at 3am is more suspicious than at 3pm.
    # -------------------------------------------------------------------------
    df["hour_of_day"] = df["step"] % 24
    df["is_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # -------------------------------------------------------------------------
    # 3e. PER-USER PROFILE -- "is this normal FOR THIS PERSON?"
    #
    # 5,000 rupees is nothing for a business account and enormous for a student.
    # So we compare each amount against that user's own past behaviour.
    #
    # CRITICAL: we use only the user's EARLIER transactions (an "expanding"
    # window that excludes the current row). A real bank cannot see a customer's
    # future payments, so neither can we. Doing otherwise is data leakage.
    #
    # PaySim's users mostly appear only a handful of times, so most rows will
    # have little or no history -- those fall back to GLOBAL averages, exactly
    # as a real system does for a brand-new customer.
    # -------------------------------------------------------------------------
    min_history = int(cfg["data"].get("min_user_history", 3))

    ordered = df.sort_values(["nameOrig", "step"], kind="mergesort")
    amt = ordered["amount"]
    grp = ordered.groupby("nameOrig", sort=False)

    prior_count = grp.cumcount()                       # transactions BEFORE this one
    prior_sum = grp["amount"].cumsum() - amt           # their total
    prior_sq_sum = grp["amount"].transform(lambda s: (s ** 2).cumsum()) - amt ** 2

    safe_count = prior_count.replace(0, np.nan)
    prior_mean = prior_sum / safe_count
    prior_var = (prior_sq_sum / safe_count) - prior_mean ** 2
    prior_std = np.sqrt(prior_var.clip(lower=0.0))

    df["user_txn_count"] = prior_count
    df["user_mean_amount"] = prior_mean
    df["user_std_amount"] = prior_std

    # Global fallback values, saved to disk so the live API uses the SAME numbers.
    global_mean = float(df["amount"].mean())
    global_std = float(df["amount"].std())

    thin_history = df["user_txn_count"] < min_history
    df["has_thin_history"] = thin_history.astype(int)
    df["user_mean_amount"] = df["user_mean_amount"].fillna(global_mean)
    df["user_std_amount"] = df["user_std_amount"].fillna(global_std)
    df.loc[thin_history, "user_mean_amount"] = global_mean
    df.loc[thin_history, "user_std_amount"] = global_std

    # Guard against a zero standard deviation (a user who always sends exactly
    # the same amount) -- that would make the division below blow up.
    safe_std = df["user_std_amount"].replace(0, np.nan).fillna(global_std)
    safe_std = safe_std.where(safe_std > 1.0, global_std)

    # "How many standard deviations above this user's normal is this payment?"
    df["amount_deviation"] = (df["amount"] - df["user_mean_amount"]) / safe_std
    df["amount_deviation"] = df["amount_deviation"].clip(-20, 20)

    # "How many times bigger than this user's usual payment is this one?"
    df["amount_vs_user_mean_ratio"] = (
        df["amount"] / df["user_mean_amount"].clip(lower=1.0)
    ).clip(upper=100.0)

    print(f"  Per-user profile built (users with fewer than {min_history} past "
          f"transactions fall back to global averages: "
          f"{int(thin_history.sum()):,} rows, "
          f"{100 * thin_history.mean():.1f}%)")

    # -------------------------------------------------------------------------
    # 3f. VELOCITY -- "how busy has this account suddenly become?"
    #
    # Fraudsters drain an account in a burst of rapid transfers. We count how
    # many transactions this user made in the previous N steps (hours).
    # -------------------------------------------------------------------------
    window = int(cfg["data"]["velocity_window_steps"])
    df["velocity_count"] = _count_recent_transactions(ordered, window)

    # -------------------------------------------------------------------------
    # 3g. RECEIVER HISTORY -- "have we ever seen this receiver before?"
    # -------------------------------------------------------------------------
    dest_ordered = df.sort_values(["nameDest", "step"], kind="mergesort")
    df["dest_seen_count"] = dest_ordered.groupby("nameDest", sort=False).cumcount()
    df["is_new_receiver"] = (df["dest_seen_count"] == 0).astype(int)

    # Tidy up any stray infinities/NaNs before they reach the model.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Save the global fallback stats so Phase 5's live API scores a brand-new
    # user exactly the way training did.
    stats_path = resolve_path(cfg, cfg["paths"]["models_dir"]) / "global_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(
            {
                "global_mean_amount": global_mean,
                "global_std_amount": global_std,
                "min_user_history": min_history,
                "velocity_window_steps": window,
            },
            f,
            indent=2,
        )
    print(f"  Global fallback stats saved to {stats_path.name}")

    return df


def _count_recent_transactions(ordered: pd.DataFrame, window: int) -> pd.Series:
    """
    For each row: how many EARLIER transactions did this same user make within
    the last `window` steps (hours)?

    `ordered` must already be sorted by (nameOrig, step).
    Because each user's steps are sorted, we can use a binary search
    (np.searchsorted) instead of comparing every row to every other row.
    """
    counts = np.zeros(len(ordered), dtype=np.int32)
    position = 0

    for _, group in ordered.groupby("nameOrig", sort=False):
        steps = group["step"].to_numpy()
        n = len(steps)
        # Index of the first earlier transaction still inside the time window.
        first_in_window = np.searchsorted(steps, steps - window, side="left")
        # Everything from there up to (but not including) this row.
        counts[position:position + n] = np.arange(n) - first_in_window
        position += n

    return pd.Series(counts, index=ordered.index)


# =============================================================================
# SECTION 4 -- Selecting the columns the model actually sees
# =============================================================================

# These are the clues handed to the machine learning model, in a FIXED order.
# Phase 2 (training) and Phase 5 (live prediction) both import this list, which
# guarantees the live service builds its input in exactly the same shape.
MODEL_FEATURES = [
    # Raw transaction facts
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    # Balance errors -- the strongest PaySim signal
    "errorBalanceOrig",
    "errorBalanceDest",
    "absErrorBalanceOrig",
    "absErrorBalanceDest",
    # Balance pattern flags
    "orig_account_emptied",
    "dest_was_empty",
    "amount_to_balance_ratio",
    # Transaction type
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "is_risky_type",
    # Time
    "hour_of_day",
    "is_night",
    # Per-user profile
    "user_txn_count",
    "user_mean_amount",
    "user_std_amount",
    "has_thin_history",
    "amount_deviation",
    "amount_vs_user_mean_ratio",
    # Velocity
    "velocity_count",
    # Receiver history
    "dest_seen_count",
    "is_new_receiver",
]

# Plain-English names, used in Phase 3 to turn SHAP output into sentences a
# bank customer can actually understand.
FEATURE_DESCRIPTIONS = {
    "amount": "the transaction amount",
    "oldbalanceOrg": "the sender's balance before the transaction",
    "newbalanceOrig": "the sender's balance after the transaction",
    "oldbalanceDest": "the receiver's balance before the transaction",
    "newbalanceDest": "the receiver's balance after the transaction",
    "errorBalanceOrig": "the sender's balance does not add up after this transaction",
    "errorBalanceDest": "the receiver's balance does not add up after this transaction",
    "absErrorBalanceOrig": "a large accounting mismatch on the sender's side",
    "absErrorBalanceDest": "a large accounting mismatch on the receiver's side",
    "orig_account_emptied": "this transaction empties the sender's account completely",
    "dest_was_empty": "the receiving account was empty beforehand (possible mule account)",
    "amount_to_balance_ratio": "the share of the sender's balance being moved",
    "type_CASH_IN": "the transaction type is CASH_IN",
    "type_CASH_OUT": "the transaction type is CASH_OUT",
    "type_DEBIT": "the transaction type is DEBIT",
    "type_PAYMENT": "the transaction type is PAYMENT",
    "type_TRANSFER": "the transaction type is TRANSFER",
    "is_risky_type": "the transaction is a TRANSFER or CASH_OUT (the only types where fraud occurs)",
    "hour_of_day": "the hour of day the transaction happened",
    "is_night": "the transaction happened during the night",
    "user_txn_count": "how many past transactions this user has",
    "user_mean_amount": "this user's typical transaction amount",
    "user_std_amount": "how much this user's amounts normally vary",
    "has_thin_history": "this user has almost no transaction history yet",
    "amount_deviation": "the amount is far above what is normal for this user",
    "amount_vs_user_mean_ratio": "the amount is many times this user's usual payment",
    "velocity_count": "an unusual number of transactions in a short time",
    "dest_seen_count": "how often this receiver has been seen before",
    "is_new_receiver": "the money is going to a receiver never used before",
}

# Simulated-location columns. Carried in the dataset for the Phase 7 dashboard
# demo, but deliberately NOT in MODEL_FEATURES -- see the note below.
LOCATION_FEATURES = [
    "km_from_prev_txn",
    "hours_since_prev_txn",
    "implied_speed_kmph",
    "impossible_travel",
]


def build_feature_matrix(df: pd.DataFrame, cfg: dict | None = None):
    """
    Split the engineered table into X (the clues) and y (the answer).

    >>> WHY LOCATION IS EXCLUDED FROM X BY DEFAULT <<<
    The location columns are randomly generated by us, so they carry no genuine
    relationship to fraud -- feeding them to the model would just add noise and
    would make the reported accuracy dishonest. Instead, impossible travel is
    applied as a hard RULE in the live layer (Phase 7): two transactions too far
    apart in too little time trigger an automatic block, regardless of the
    model's score. Set `data.include_simulated_location_features: true` in
    config.yaml if you want to experiment with including them anyway.
    """
    feature_cols = list(MODEL_FEATURES)

    if cfg and cfg["data"].get("include_simulated_location_features", False):
        feature_cols += [c for c in LOCATION_FEATURES if c in df.columns]
        print("NOTE: simulated location features ARE included in the model input.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Engineered data is missing these columns: {missing}")

    X = df[feature_cols].copy()
    y = df["isFraud"].copy() if "isFraud" in df.columns else None
    return X, y, feature_cols


# =============================================================================
# SECTION 5 -- Run everything and print the confirmation report
# =============================================================================

def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    cfg = load_config()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    _print_header("PHASE 1 -- LOADING RAW DATA")
    try:
        df_raw = load_raw_data(cfg)
    except FileNotFoundError as e:
        print(e)
        return 1

    # ---- 1. Shape -----------------------------------------------------------
    _print_header("1. DATA SHAPE")
    print(f"Rows:    {df_raw.shape[0]:,}")
    print(f"Columns: {df_raw.shape[1]}")
    print(f"\nColumn names and types:")
    print(df_raw.dtypes.to_string())

    # ---- 2. Class balance ---------------------------------------------------
    _print_header("2. CLASS BALANCE (how much fraud is in the data?)")
    counts = df_raw["isFraud"].value_counts().sort_index()
    total = len(df_raw)
    legit = int(counts.get(0, 0))
    fraud = int(counts.get(1, 0))
    print(f"Legitimate (isFraud=0): {legit:>9,}  ({100 * legit / total:6.3f}%)")
    print(f"Fraudulent (isFraud=1): {fraud:>9,}  ({100 * fraud / total:6.3f}%)")
    if fraud:
        print(f"\nImbalance ratio: 1 fraud for every {legit / fraud:.0f} legitimate "
              f"transactions.")
    print("This is why Phase 2 uses SMOTE -- but ONLY on the training half,")
    print("never before the train/test split.")

    print("\nFraud broken down by transaction type:")
    breakdown = (
        df_raw.groupby("type")["isFraud"]
        .agg(total_txns="size", fraud_txns="sum")
        .assign(fraud_rate_pct=lambda d: 100 * d["fraud_txns"] / d["total_txns"])
        .sort_values("fraud_txns", ascending=False)
    )
    print(breakdown.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n(Confirms the known PaySim quirk: fraud appears ONLY in TRANSFER "
          "and CASH_OUT.)")

    # ---- 3. First rows ------------------------------------------------------
    _print_header("3. FIRST 5 ROWS OF RAW DATA")
    print(df_raw.head().to_string())

    # ---- 4. Feature engineering --------------------------------------------
    _print_header("4. FEATURE ENGINEERING")
    df_loc = add_simulated_location(df_raw, cfg)
    df_feat = engineer_features(df_loc, cfg)

    X, y, feature_cols = build_feature_matrix(df_feat, cfg)
    print(f"\nEngineered feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Features used by the model ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {col}")

    _print_header("5. SAMPLE OF THE ENGINEERED FEATURES")
    preview_cols = [
        "amount", "type", "errorBalanceOrig", "errorBalanceDest",
        "orig_account_emptied", "amount_deviation", "velocity_count",
        "is_new_receiver", "location", "impossible_travel", "isFraud",
    ]
    preview_cols = [c for c in preview_cols if c in df_feat.columns]
    print(df_feat[preview_cols].head(10).to_string(float_format=lambda v: f"{v:,.2f}"))

    _print_header("6. DO THE FEATURES ACTUALLY SEPARATE FRAUD FROM LEGIT?")
    print("(Average value of each clue, for legitimate vs fraudulent rows.\n"
          " Big gaps mean the clue is genuinely useful.)\n")
    check_cols = [
        "amount", "errorBalanceOrig", "errorBalanceDest", "orig_account_emptied",
        "dest_was_empty", "amount_to_balance_ratio", "is_risky_type",
        "amount_deviation", "velocity_count", "is_new_receiver",
    ]
    comparison = df_feat.groupby("isFraud")[check_cols].mean().T
    comparison.columns = ["legitimate (0)", "fraud (1)"]
    print(comparison.to_string(float_format=lambda v: f"{v:,.3f}"))

    # ---- 5. Save ------------------------------------------------------------
    out_path = resolve_path(cfg, cfg["paths"]["processed_data"])
    df_feat.to_csv(out_path, index=False)

    _print_header("PHASE 1 COMPLETE")
    print(f"Engineered data saved to: {out_path}")
    print(f"Rows: {len(df_feat):,}   Model features: {len(feature_cols)}")
    print("\nNo SMOTE was applied here -- that happens in Phase 2, after the")
    print("train/test split, and only on the training half.")
    print("\nNext: Phase 2 -- train the hybrid Isolation Forest + Random Forest model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
