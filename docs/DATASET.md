# Getting the PaySim Dataset

The model trains on **PaySim** — "Synthetic Financial Datasets For Fraud Detection".
It is a simulated (not real) set of ~6.3 million mobile-money transactions, built
by researchers specifically so that fraud detection can be studied without
exposing anyone's real bank data.

**Source:** https://www.kaggle.com/datasets/ealaxi/paysim1
**License:** CC BY-SA 4.0 — free to use, just cite it in your report.
**Download size:** ~180 MB zipped, ~470 MB as CSV.

---

## Option A — Download in the browser (easiest)

1. Go to **https://www.kaggle.com/datasets/ealaxi/paysim1**
2. Sign in to Kaggle (a free account is required — sign up with Google in ~30 seconds).
3. Click the **Download** button near the top right.
4. You will get a file called `archive.zip` in your `~/Downloads` folder.
5. Now run these two commands in Terminal — they unzip it and put the CSV in
   exactly the right place with exactly the right name:

```bash
cd "/Users/lucky/Documents/fraud detection system"
unzip -o ~/Downloads/archive.zip -d ml-service/data/
```

6. The unzipped file is named `PS_20174392719_1491204439457_log.csv`.
   Rename it to `transactions.csv`:

```bash
cd "/Users/lucky/Documents/fraud detection system/ml-service/data"
mv PS_*.csv transactions.csv
```

---

## Option B — Download from the command line (Kaggle CLI)

Only worth it if you already use Kaggle a lot.

```bash
# One-time setup
cd "/Users/lucky/Documents/fraud detection system/ml-service"
./venv/bin/pip install kaggle

# Get an API token: kaggle.com -> your avatar -> Settings -> API -> "Create New Token"
# That downloads kaggle.json. Install it:
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download and unzip in one step
./venv/bin/kaggle datasets download -d ealaxi/paysim1 -p data --unzip
mv data/PS_*.csv data/transactions.csv
```

---

## Confirm it landed correctly

```bash
cd "/Users/lucky/Documents/fraud detection system/ml-service"
ls -lh data/transactions.csv
head -2 data/transactions.csv
```

You should see a file of roughly **470 MB**, and this header line:

```
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
```

If the header does not match, you downloaded a different fraud dataset —
there are several on Kaggle with similar names. Make sure it is `ealaxi/paysim1`.

---

## Then run Phase 1

```bash
cd "/Users/lucky/Documents/fraud detection system/ml-service"
./venv/bin/python features.py
```

Expected: around **6,362,620 rows**, of which **8,213 are fraud (0.129%)**, and
fraud appearing **only** in `TRANSFER` and `CASH_OUT` rows.

The script automatically samples down to 200,000 rows for fast training while
keeping **every single fraud row** — you can change `data.sample_size` in
`config.yaml` if you want more.

---

## Notes on this dataset (worth putting in your report)

| Quirk | How we handle it |
|---|---|
| Fraud occurs **only** in `TRANSFER` and `CASH_OUT` | Kept as the `is_risky_type` feature; all types are still used for training |
| `isFlaggedFraud` is the dataset's own naive rule | **Dropped.** Using it would be data leakage — the model would copy the answer |
| Extremely imbalanced (0.129% fraud) | SMOTE in Phase 2, applied **only** to the training split |
| Balance columns don't always add up | Turned into `errorBalanceOrig` / `errorBalanceDest` — the strongest signals in the data |
| Most users appear only a few times | Thin per-user history falls back to global averages |
| **No location / geo column at all** | Location is **simulated** for the demo and clearly labelled; impossible-travel runs as a live rule in Phase 7, not as a trained feature |

---

# Using your own data

*Phase 14. PaySim is the reference dataset, not a requirement.*

The pipeline reads through a **column mapping**. You tell it which of your
columns mean what, and everything downstream keeps working — it never learns
that your file called them something else.

## The `schema:` block

In `config.yaml`. Canonical name on the left, **your** column name on the right:

```yaml
schema:
  step: trans_date_trans_time   # an hour index OR a timestamp
  type: category
  amount: amt
  nameOrig: cc_num              # who paid
  nameDest: merchant            # who was paid
  isFraud: is_fraud             # the label; needed to train
  oldbalanceOrg: null           # this dataset has no balances
  newbalanceOrig: null
  oldbalanceDest: null
  newbalanceDest: null
```

Left as shipped, every field maps to itself — which is exactly PaySim, so
nothing changes for the existing setup.

| Field | Required | Notes |
|---|---|---|
| `step` | yes | Integer hour index, **or** a timestamp column, converted to hours since the earliest row |
| `type`, `amount` | yes | |
| `nameOrig`, `nameDest` | yes | Account identifiers for payer and payee |
| the four balances | no | `null` if absent — see below |
| `isFraud` | to train | |

The loader refuses to start on an unknown canonical name, two fields mapped to
the same column, a required field missing from the file, or a `step` that is
neither an integer nor a parseable timestamp. It reports **every** problem at
once rather than one per run.

## Feature groups, and what your data switches on

`MODEL_FEATURES` is not a fixed list. It is resolved from what you supplied:

| Group | Features | Needs |
|---|---|---|
| A core transaction | 9 | amount, type, step |
| B balance | 11 | **all four** balance columns |
| C receiver behaviour | 8 | nameDest + step |
| D sender behaviour | 7 | nameOrig + step, **and repeat senders** |

Group B is all-or-nothing on purpose: `errorBalanceOrig` is
`oldbalanceOrg - amount - newbalanceOrig`, so one missing column makes the
arithmetic meaningless rather than merely incomplete.

## If your data has no balances

Most public transaction datasets don't. You are told what it costs:

```
DATA CAPABILITY REPORT
  Mapped        step, type, amount, nameOrig, nameDest, isFraud
  Not supplied  oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest

  COST OF THE MISSING BALANCE COLUMNS
    On PaySim these carried 68.3% of model importance.
    Balance-error signals will be inert; account-drain
    detection is unavailable.
    Encoded as 'not recorded' (zero), NOT as reconciling
    balances -- a reconciling balance is the fraud signature
    in 99.5% of cases.
```

It is printed at load time and exposed at `GET /health` under
`data_capability`. **This is the point of the feature.** Missing balances crash
nothing — eleven features simply become constant, the forest never splits on
them, and you have trained a 17-feature model believing it is 28. On PaySim
that degradation took fraud blocking from 149/150 to **1/150**.

Four ways to get group B back, best first:

1. **Map them.** A bank has balances under some name.
2. **Reconstruct from the ledger** — a causal running balance per account. Only
   valid if the transaction stream is complete; one missing row corrupts every
   balance after it.
3. **Join an accounts table** — transactions plus an account master with EOD
   balance.
4. **Enable group D instead**, when your data has repeat senders: 17 + 7 = 24
   features. Not the same 28, but earned rather than assumed.

## Why group D is a gate, not a switch

The sender-side features are off for PaySim for a **measured** reason: 99.85% of
its senders appear exactly once, so those columns are constant or a rescaled
copy of `amount`, and `user_amount_deviation` would mean something different at
serving time than it did in training — a train/serve mismatch that silently
corrupts live scores.

That reasoning is about the data, not about PaySim. On a dataset with repeat
customers the same measurement comes out the other way. Phase 15's
`calibrate.py` performs that measurement and decides; until then the group is
off by default.

## What is not supported

A dataset with no accounts at all. The ULB `creditcard.csv` set is PCA'd into
`V1`–`V28` with no payer or payee, so there is nothing to map, and the loader
says so rather than half-working.

Changing the feature set also means **retraining** — the model on disk expects
its own 28 columns in its own order. Calibration and training are part of
adopting your own data, not optional extras.
