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

import schema
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
# Set by load_raw_data so train_model.py can print and persist the report
# without threading a return value through four call sites. schema.py now owns
# the canonical column vocabulary -- two sources of truth for a column list is
# exactly how they drift apart.
LAST_CAPABILITY_REPORT = None

# PaySim's five transaction types. We fix the order so that the one-hot
# columns are always identical between training and live prediction.
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_raw_data(cfg: dict) -> pd.DataFrame:
    """
    Load the FULL PaySim CSV and sanity-check it.

    Note there is no sampling here. Sampling happens later, in sample_data(),
    AFTER the account-history features have been built -- see the long comment
    on sample_data() for why that order matters.
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

    # --- The data boundary: rename THEIR columns to OUR canonical names. ---
    # Everything downstream speaks canonical names only and never learns that
    # the file called them something else. See schema.py.
    global LAST_CAPABILITY_REPORT
    mapping = schema.load_mapping(cfg)
    df, LAST_CAPABILITY_REPORT = schema.normalize(df, mapping)
    print(LAST_CAPABILITY_REPORT.render())

    print(f"Full file loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Drop the leaky column (PaySim's own naive rule -- not a real clue) ---
    for col in cfg["data"]["drop_columns"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped leaky column: '{col}' (this is the dataset's own "
                  f"built-in rule, using it would be cheating)")

    return df


def sample_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Cut the data down to a size a laptop can train on quickly, keeping EVERY
    fraud row (fraud is only 0.129% of PaySim -- throwing any away would
    cripple the model).

    >>> WHY THIS RUNS *AFTER* FEATURE ENGINEERING <<<
    An account's history is a relationship BETWEEN rows. If you sample first,
    you keep ~3% of the rows, so the earlier transactions that gave an account
    its history are almost always among the 97% you threw away -- and every
    account then looks brand new. We measured exactly that: sampling first made
    99.5% of receivers look like first-timers, when the true figure in the full
    file is 83%. So: engineer history on all 6.3M rows, THEN sample.
    """
    sample_size = int(cfg["data"]["sample_size"])
    if not sample_size or len(df) <= sample_size:
        return df.reset_index(drop=True)

    seed = int(cfg["project"]["random_state"])
    fraud_rows = df[df["isFraud"] == 1]
    legit_rows = df[df["isFraud"] == 0]

    n_legit_needed = max(sample_size - len(fraud_rows), 0)
    n_legit_needed = min(n_legit_needed, len(legit_rows))

    legit_sample = legit_rows.sample(n=n_legit_needed, random_state=seed)
    sampled = pd.concat([fraud_rows, legit_sample], ignore_index=True)

    # Shuffle so fraud is not all bunched at the top.
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"Sampled down to {len(sampled):,} rows "
          f"(kept ALL {len(fraud_rows):,} fraud rows + "
          f"{len(legit_sample):,} random legitimate rows)")
    return sampled


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

    # --- Step 3: inject impossible jumps ---
    #
    # An "impossible jump" only makes sense for a customer who has at least TWO
    # transactions -- you need a previous location to travel FROM. As measured,
    # 99.85% of PaySim senders appear exactly once, so only a handful of
    # accounts in any sample are even eligible. We inject into all of them and
    # report the true count rather than silently producing zero.
    #
    # The real demonstration of this feature is the Phase 7 dashboard, where a
    # demo customer submits several transactions in a row and the
    # impossible-travel rule fires live.
    txns_per_user = df["nameOrig"].value_counts()
    eligible = txns_per_user[txns_per_user >= 2].index.to_numpy()

    fraction = float(loc_cfg.get("impossible_jump_user_fraction", 0.02))
    n_injected = max(int(len(eligible) * fraction), len(eligible)) if len(eligible) else 0
    injected_users = (
        set(rng.choice(eligible, size=min(n_injected, len(eligible)), replace=False))
        if n_injected else set()
    )

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
        # home_idx is positional, target_rows are index labels -- translate,
        # rather than assuming the two happen to line up.
        target_pos = df.index.get_indexer(target_rows)
        df.loc[target_rows, "_city_idx"] = farthest_from[home_idx[target_pos]]

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
    n_multi = len(eligible)
    print(f"Simulated location added (SIMULATED -- PaySim has no real geo data).")
    print(f"  Customers with 2+ transactions in this sample: {n_multi:,} "
          f"({100 * n_multi / df['nameOrig'].nunique():.2f}% -- PaySim reuses "
          f"customer IDs almost never)")
    print(f"  Impossible-travel cases created: {n_impossible:,}")
    if n_impossible == 0:
        print("  (None possible: an impossible jump needs a previous location to "
              "travel FROM. This rule is demonstrated live in Phase 7 instead.)")

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

    # Tidy up any stray infinities/NaNs before they reach the model.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def add_account_profiles(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    BEHAVIOURAL PROFILES -- "is this normal?"

    The idea: 5,000 rupees is nothing for a business and enormous for a student,
    so we compare each amount against that account's own past, and we count how
    busy the account suddenly became ("velocity"). Fraudsters drain accounts in
    bursts, so a spike in activity is a real signal.

    >>> RUN THIS ON THE FULL DATASET, BEFORE SAMPLING. <<<
    History is a relationship between rows; sampling first destroys it.

    CRITICAL: we only ever look at EARLIER transactions (an "expanding" window
    that excludes the current row). A real bank cannot see the future, so
    neither can we. Doing otherwise is data leakage.

      >>> MEASURED REALITY OF PAYSIM -- IMPORTANT FOR THE REPORT <<<
      We counted how often each account actually reappears in the full
      6,362,620-row file:

                             senders (nameOrig)   receivers (nameDest)
        appear exactly once        99.85%               83.1%
        max appearances              3                   113
        rows on a repeat account    0.29%                64.4%

      PaySim generates a fresh customer ID for nearly every transaction, so
      SENDER history barely exists -- sender profile and sender velocity are
      inert on this dataset no matter how we sample. RECEIVER history is real
      and rich, so that is where the model's behavioural features come from.

      >>> BUT DO NOT OVERSELL WHAT THE RECEIVER FEATURES DETECT. <<<
      An earlier version of this comment claimed the receiver side was "the
      better signal anyway, because a mule account collecting a burst of
      transfers is one of the clearest fraud patterns there is". That is true
      of real banking and FALSE of this dataset. Measured over the full file:

          prior payments to this receiver     legitimate    fraud
            within 1 hour                         0.189     0.036
            within 24 hours                       1.132     0.338
          receiver reused at any point            57.2%     37.0%

      PaySim fraud targets fresh, QUIET receivers, so the model learned --
      correctly, for this data -- that a busy receiver is LESS suspicious.
      What the receiver features genuinely contribute is amount-based:
      dest_amount_ratio separates 0.885 (legitimate) from 8.201 (fraud).

      The mule-burst typology is handled in rules.py as an explicit rule
      instead, because a signal that is absent from the data cannot be learned
      from it.

      We still COMPUTE the sender-side profile, because in the live system
      (Phases 5-7) a real customer does have history and those numbers become
      meaningful for the explanation text and the OTP step-up rules. They are
      simply kept out of the trained model -- see MODEL_FEATURES below.
    """
    print("\nBuilding account history profiles on the FULL dataset...")
    min_history = int(cfg["data"].get("min_user_history", 3))
    window = int(cfg["data"]["velocity_window_steps"])

    # Global fallback values, saved to disk so the live API scores a brand-new
    # account exactly the way training did.
    global_mean = float(df["amount"].mean())
    global_std = float(df["amount"].std())

    # Sender side: computed for the live layer, NOT fed to the model on PaySim.
    df = _add_account_profile(
        df, key="nameOrig", prefix="user",
        min_history=min_history, window=window,
        global_mean=global_mean, global_std=global_std,
    )

    # Receiver side: this is what the model actually learns from.
    df = _add_account_profile(
        df, key="nameDest", prefix="dest",
        min_history=min_history, window=window,
        global_mean=global_mean, global_std=global_std,
    )

    df["is_new_receiver"] = (df["dest_txn_count"] == 0).astype(int)

    for prefix, label in [("user", "Sender"), ("dest", "Receiver")]:
        thin = df[f"{prefix}_has_thin_history"]
        active = df[f"{prefix}_velocity_count"] > 0
        print(f"  {label} profile: {100 * thin.mean():5.1f}% of rows have thin "
              f"history (fall back to global averages), "
              f"{100 * active.mean():5.1f}% show recent activity")

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


def _group_codes(sorted_keys: np.ndarray) -> np.ndarray:
    """
    Turn an already-sorted column of account IDs into compact integer codes:
    ["C1","C1","C7","C9","C9"] -> [0, 0, 1, 2, 2]

    Grouping on small integers is far faster than grouping on 6.3 million
    strings, which matters because we now run this over the whole file.
    """
    n = len(sorted_keys)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    is_new_account = np.empty(n, dtype=bool)
    is_new_account[0] = True
    is_new_account[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return np.cumsum(is_new_account) - 1


def _add_account_profile(
    df: pd.DataFrame,
    key: str,
    prefix: str,
    min_history: int,
    window: int,
    global_mean: float,
    global_std: float,
) -> pd.DataFrame:
    """
    Build a behavioural profile for one side of the transaction.

    `key` is the column identifying the account ("nameOrig" for the sender,
    "nameDest" for the receiver). `prefix` names the output columns
    ("user_..." or "dest_...").

    Produces, for every row:
      {prefix}_txn_count        how many times we saw this account BEFORE now
      {prefix}_mean_amount      its average past amount (global avg if too new)
      {prefix}_std_amount       how much its amounts normally vary
      {prefix}_has_thin_history 1 if we do not have enough history to trust it
      {prefix}_amount_deviation how many standard deviations above normal this is
      {prefix}_amount_ratio     how many times its usual amount this is
      {prefix}_velocity_count   how many transactions in the last `window` hours

    Everything is computed on PRIOR rows only -- the current row is excluded --
    so no information from the future ever reaches the model.

    MUST be run on the FULL dataset, before any sampling. If you sample first,
    the other half of an account's history usually gets thrown away and every
    account looks brand new.
    """
    # Sort by account, then time, so "previous" means the right thing.
    # mergesort is stable, which keeps ties in their original order.
    ordered = df.sort_values([key, "step"], kind="mergesort")
    codes = _group_codes(ordered[key].to_numpy())
    amt = ordered["amount"].to_numpy(dtype=np.float64)
    steps = ordered["step"].to_numpy(dtype=np.int64)

    grouper = pd.Series(codes, index=ordered.index)
    amt_series = pd.Series(amt, index=ordered.index)
    sq_series = pd.Series(amt ** 2, index=ordered.index)

    # --- Running count / sum / sum-of-squares of everything BEFORE this row ---
    # (cumsum includes the current row, so we subtract it back off.)
    prior_count = amt_series.groupby(grouper, sort=False).cumcount().to_numpy()
    prior_sum = amt_series.groupby(grouper, sort=False).cumsum().to_numpy() - amt
    prior_sq_sum = sq_series.groupby(grouper, sort=False).cumsum().to_numpy() - amt ** 2

    # Standard deviation from running totals:  std = sqrt(E[x^2] - E[x]^2)
    divisor = np.maximum(prior_count, 1)
    prior_mean = np.where(prior_count > 0, prior_sum / divisor, np.nan)
    prior_var = np.where(prior_count > 0, prior_sq_sum / divisor - prior_mean ** 2, np.nan)
    prior_std = np.sqrt(np.clip(prior_var, 0.0, None))

    # Assigning Series back to df -- pandas aligns them by index, so the
    # original row order is preserved even though `ordered` was sorted.
    df[f"{prefix}_txn_count"] = pd.Series(prior_count, index=ordered.index)
    df[f"{prefix}_mean_amount"] = pd.Series(prior_mean, index=ordered.index)
    df[f"{prefix}_std_amount"] = pd.Series(prior_std, index=ordered.index)

    # --- Fall back to global averages when history is too thin to trust ---
    thin = df[f"{prefix}_txn_count"] < min_history
    df[f"{prefix}_has_thin_history"] = thin.astype(int)
    df[f"{prefix}_mean_amount"] = df[f"{prefix}_mean_amount"].fillna(global_mean)
    df[f"{prefix}_std_amount"] = df[f"{prefix}_std_amount"].fillna(global_std)
    df.loc[thin, f"{prefix}_mean_amount"] = global_mean
    df.loc[thin, f"{prefix}_std_amount"] = global_std

    # Guard against a zero standard deviation (an account that always moves
    # exactly the same amount) -- that would make the division below blow up.
    safe_std = df[f"{prefix}_std_amount"].where(
        df[f"{prefix}_std_amount"] > 1.0, global_std
    )

    df[f"{prefix}_amount_deviation"] = (
        (df["amount"] - df[f"{prefix}_mean_amount"]) / safe_std
    ).clip(-20, 20)

    df[f"{prefix}_amount_ratio"] = (
        df["amount"] / df[f"{prefix}_mean_amount"].clip(lower=1.0)
    ).clip(upper=100.0)

    # --- Velocity: prior transactions within the last `window` steps (hours) ---
    df[f"{prefix}_velocity_count"] = pd.Series(
        _count_recent(codes, steps, window), index=ordered.index
    )

    return df


def _count_recent(codes: np.ndarray, steps: np.ndarray, window: int) -> np.ndarray:
    """
    For each row: how many EARLIER transactions did this same account have
    within the last `window` steps (hours)?

    `codes` and `steps` must already be sorted by (account, step).

    THE TRICK: we want a binary search per account, but looping over 6.3 million
    accounts in Python would take minutes. Instead we place every account on its
    own stretch of one long number line:

        position = account_code * span + step        (span > any possible step)

    Because each account occupies a disjoint stretch, a single vectorised
    binary search over the whole array can never wander into a neighbouring
    account -- so we get the answer for all rows in one shot.
    """
    n = len(codes)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    span = int(steps.max()) + int(window) + 2
    line = codes * span + steps                 # every row's spot on the line
    earliest_allowed = codes * span + (steps - window)

    first_in_window = np.searchsorted(line, earliest_allowed, side="left")

    # A negative (step - window) could push the search back into the previous
    # account's stretch, so clamp it to where this account's own block starts.
    account_start = np.searchsorted(codes, codes, side="left")
    first_in_window = np.maximum(first_in_window, account_start)

    # Everything from there up to (but not including) this row.
    return (np.arange(n) - first_in_window).astype(np.int32)


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
    # Receiver behavioural profile.
    #
    # NOTE: the SENDER-side equivalents (user_txn_count, user_mean_amount,
    # user_amount_deviation, user_velocity_count, ...) are deliberately absent.
    # We measured PaySim: 99.85% of senders appear exactly once, so on this
    # dataset those columns are either constant or a rescaled copy of `amount`.
    # Feeding them in would pad the feature count without adding information,
    # and worse, `user_amount_deviation` would mean something DIFFERENT at live
    # time (where customers do have history) than it did during training --
    # a train/serve mismatch that would silently corrupt live scores.
    # They are still computed and available for the live explanation layer.
    "dest_txn_count",
    "dest_mean_amount",
    "dest_std_amount",
    "dest_has_thin_history",
    "dest_amount_deviation",
    "dest_amount_ratio",
    "dest_velocity_count",
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
    # --- Receiver profile (used by the model) ---
    "dest_txn_count": "how often this receiving account has been used before",
    "dest_mean_amount": "the amount this receiving account usually collects",
    "dest_std_amount": "how much the amounts sent to this receiver normally vary",
    "dest_has_thin_history": "this receiving account is almost brand new",
    "dest_amount_deviation": "the amount is far above what this receiver normally collects",
    "dest_amount_ratio": "the amount is many times what this receiver normally collects",
    "dest_velocity_count": "this account is collecting money unusually rapidly (possible mule account)",
    "is_new_receiver": "the money is going to a receiver never used before",
    # --- Sender profile (computed for the live layer, not used by the model
    #     on PaySim -- see the note in MODEL_FEATURES) ---
    "user_txn_count": "how many past transactions this customer has",
    "user_mean_amount": "this customer's typical transaction amount",
    "user_std_amount": "how much this customer's amounts normally vary",
    "user_has_thin_history": "this customer has almost no transaction history yet",
    "user_amount_deviation": "the amount is far above what is normal for this customer",
    "user_amount_ratio": "the amount is many times this customer's usual payment",
    "user_velocity_count": "this customer made an unusual number of transactions in a short time",
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
    #
    # ORDER MATTERS. History features are relationships BETWEEN rows, so they
    # must be built on the full 6.3M rows. Only then do we sample down.
    # Sampling first would leave almost every account looking brand new.
    _print_header("4. FEATURE ENGINEERING")
    df_full = add_simulated_location(df_raw, cfg)
    df_full = add_account_profiles(df_full, cfg)

    print()
    df_sample = sample_data(df_full, cfg)
    del df_full                                  # free the 6.3M-row frame

    df_feat = engineer_features(df_sample, cfg)

    X, y, feature_cols = build_feature_matrix(df_feat, cfg)
    print(f"\nEngineered feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Features used by the model ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {col}")

    _print_header("5. SAMPLE OF THE ENGINEERED FEATURES")
    preview_cols = [
        "amount", "type", "errorBalanceOrig", "errorBalanceDest",
        "orig_account_emptied", "dest_amount_deviation", "dest_velocity_count",
        "is_new_receiver", "location", "impossible_travel", "isFraud",
    ]
    preview_cols = [c for c in preview_cols if c in df_feat.columns]
    print(df_feat[preview_cols].head(10).to_string(float_format=lambda v: f"{v:,.2f}"))

    _print_header("6. DO THE FEATURES ACTUALLY SEPARATE FRAUD FROM LEGIT?")
    print("(Average value of each clue, for legitimate vs fraudulent rows.\n"
          " Big gaps mean the clue is genuinely useful.)\n")
    comparison = df_feat.groupby("isFraud")[feature_cols].mean().T
    comparison.columns = ["legitimate (0)", "fraud (1)"]

    # A feature that never varies teaches the model nothing. Flag any such
    # column loudly rather than quietly shipping dead weight into Phase 2.
    variation = df_feat[feature_cols].std()
    comparison["USEFUL?"] = [
        "DEAD (no variation)" if variation[c] == 0 else "" for c in comparison.index
    ]
    print(comparison.to_string(float_format=lambda v: f"{v:,.3f}"))

    dead = [c for c in feature_cols if variation[c] == 0]
    if dead:
        print(f"\n*** WARNING: {len(dead)} feature(s) are constant and teach the "
              f"model nothing: {dead}")
    else:
        print(f"\nAll {len(feature_cols)} model features vary across the dataset.")

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
