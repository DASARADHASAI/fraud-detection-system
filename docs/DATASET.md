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
