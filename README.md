# Explainable Real-Time Fraud Detection Using a Hybrid Machine Learning Model

B.Tech final-year capstone project.

Detects fraudulent bank transactions in real time, **explains every decision in
plain language**, and decides **fully automatically** — approve, block, or confirm
with the customer via OTP. There is no human reviewer anywhere in the pipeline.

## Architecture

Two services talking over REST:

| Service | Language | Port | Job |
|---|---|---|---|
| `ml-service/` | Python (Flask) | 5001 | Feature engineering, hybrid ML scoring, SHAP explanations, GenAI narration |
| `backend/` | Java (Spring Boot) | 8080 | Transaction intake, H2 database, automatic routing, dashboard |

Why split? The ML ecosystem is Python; the application layer is Java. Keeping them
separate means either side can be replaced without touching the other.

## The hybrid model

Two models vote on every transaction:

- **Isolation Forest** (unsupervised) — catches *brand-new* fraud patterns it has
  never been taught, by spotting anything that looks statistically odd.
- **Random Forest** (supervised) — catches *known* fraud patterns learned from
  labelled history.

Their scores are blended into a single 0–1 fraud score. When the two models
**disagree**, that itself is a signal: the transaction is pushed into the
step-up (OTP) path rather than being silently approved or blocked.

## Build status

- [x] **Phase 1** — Data loading + feature engineering (`ml-service/features.py`)
- [ ] Phase 2 — Hybrid model + evaluation
- [ ] Phase 3 — SHAP explainability
- [ ] Phase 4 — GenAI agent (Ollama + LangChain, with template fallback)
- [ ] Phase 5 — Flask API
- [ ] Phase 6 — Spring Boot backend + automatic routing
- [ ] Phase 7 — Dashboard
- [ ] Phase 8 — Testing

---

## Setup (macOS, Apple Silicon)

### Prerequisites

Already verified on this machine: Java 21, Maven 3.9, Python 3.14, Ollama 0.32
with `llama3.2` pulled.

```bash
# If setting up on a different Mac:
brew install openjdk maven python@3.12
brew install ollama && ollama serve      # then, in another terminal:
ollama pull llama3.2
```

### 1. Python environment

```bash
cd "/Users/lucky/Documents/fraud detection system/ml-service"
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

> Every command below uses `./venv/bin/python` directly, so you never have to
> remember to activate the environment. If you prefer activating it:
> `source venv/bin/activate`, then just use `python`.

### 2. Get the dataset

See **[docs/DATASET.md](docs/DATASET.md)** for step-by-step instructions.
Short version: download from
[kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1),
unzip, and place the CSV at `ml-service/data/transactions.csv`.

### 3. Phase 1 — build the features

```bash
cd "/Users/lucky/Documents/fraud detection system/ml-service"
./venv/bin/python features.py
```

Prints the data shape, class balance, and first rows; saves the engineered
dataset to `ml-service/data/processed.csv`.

---

## Configuration

Everything tunable lives in **`config.yaml`** at the project root — ports,
decision thresholds, model hyperparameters, file paths. No setting is hard-coded
in the source. Both the Python and Java services read from it.

Key thresholds:

```yaml
decision:
  approve_below: 0.35    # below this  -> APPROVE automatically
  block_above:   0.75    # above this  -> BLOCK automatically
                         # in between, or models disagree -> OTP step-up
```

---

## Honest notes for the report

- **Location data is simulated.** PaySim has no geo column at all. We generate a
  plausible city per transaction so the impossible-travel feature can be
  demonstrated, and we exclude it from the trained model — inventing evidence
  and then reporting accuracy on it would be dishonest. Impossible travel runs
  as a hard rule in the live layer instead (Phase 7).
- **`isFlaggedFraud` is dropped.** It is the dataset's own built-in naive rule.
  Training on it would be data leakage.
- **SMOTE is applied only to the training split**, after the train/test split,
  never before — otherwise synthetic copies of test rows leak into training and
  the reported accuracy becomes meaningless.
- **The scaler is fitted on training data only**, for the same reason.
- **No paid APIs.** The GenAI layer is local Ollama, and falls back to a plain
  template sentence if Ollama is not running. The core system works without it.
