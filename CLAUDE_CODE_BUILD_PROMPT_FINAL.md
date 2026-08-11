# Build Prompt (FINAL) — Fraud Detection Capstone
*(Keep this file in the root of the project folder. In Antigravity, tell Claude Code: "Read this file and start with Phase 1." Build phase by phase and confirm each runs before continuing.)*

---

You are helping me build my final-year B.Tech capstone. Build it **step by step**; after each phase, run it and confirm it works before moving on. Ask me if anything is unclear. Keep code clean, well-commented, and beginner-friendly — my team is strong in **Java**, less in **Python**.

## 1. Project goal
Build **"Explainable Real-Time Fraud Detection Using a Hybrid Machine Learning Model."**
It detects fraudulent bank transactions in real time, **explains every decision in plain language**, and **decides fully automatically** (no human reviewer) whether to approve, block, or confirm with the user.
Two services over a REST API: a **Java Spring Boot** app (backend + dashboard) and a **Python (Flask)** AI service (ML + explanation).

## 2. Folder name & structure
Root folder **`fraud-detection-system`**:
```
fraud-detection-system/
├── CLAUDE_CODE_BUILD_PROMPT_FINAL.md
├── README.md
├── config.yaml                 # thresholds, ports (so nothing is hard-coded)
├── ml-service/
│   ├── data/                   # <-- dataset CSV goes here
│   ├── models/                 # saved model files
│   ├── knowledge_base/         # past fraud cases for RAG
│   ├── features.py
│   ├── train_model.py
│   ├── explain.py
│   ├── genai_agent.py
│   ├── app.py                  # Flask API
│   └── requirements.txt
├── backend/                    # Spring Boot
│   ├── pom.xml
│   └── src/main/
│       ├── java/com/klu/fraud/
│       └── resources/
│           ├── application.properties
│           └── static/index.html
└── docs/
```

## 3. Tools & setup (I'm on a Mac, Apple Silicon M4 — give exact macOS commands)
- **Java JDK 17+**, **Maven**
- **Python 3.11+** (virtual environment inside `ml-service/`)
- **Ollama** — already installed and running; I've already pulled `llama3.2`.
- **Python libs** (`requirements.txt`): `pandas numpy scikit-learn imbalanced-learn shap flask flask-cors joblib pyyaml langchain langchain-community langchain-ollama langgraph chromadb requests`
- **Database:** Spring Boot **file-based H2** (so data survives restarts; no install). Easy switch to MySQL later via config only.

## 4. Dataset — where to get it, where to put it
Use **PaySim** — "Synthetic Financial Datasets For Fraud Detection":
**https://www.kaggle.com/datasets/ealaxi/paysim1**
Place the CSV at `ml-service/data/transactions.csv`. If it's too big to train fast, take a random sample (~200k rows) but keep ALL fraud rows.

**⚠️ Important — PaySim's real columns and quirks (handle these, or the model will be wrong):**
- Columns: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`.
- **Fraud only happens in `TRANSFER` and `CASH_OUT` types.** Focus modelling on these (or keep type as a feature).
- **Do NOT use `isFlaggedFraud` as a feature** — it's the dataset's own naive rule (a leak). Drop it.
- **There is NO location/geo column.** So the "impossible-travel / new-location" feature cannot come from this data — see the note in Phase 1.
- Most `nameOrig` users appear only a few times, so per-user history is thin — when a user has little history, fall back to global averages.
- **Strong known feature:** balance errors. Compute `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg` and `errorBalanceDest = oldbalanceDest + amount - newbalanceDest`. These catch fraud well in PaySim — engineer them.

## 5. Build phases (in order; confirm each runs)

**Phase 1 — Data + feature engineering (`features.py`)**
Load, clean, and engineer: amount, type (encoded), the two **balance-error** features above, a **per-user profile** (user's mean/typical amount and how far this amount deviates; fall back to global stats if the user is new), and **velocity** (count of this user's transactions in a recent window using `step`).
For the **location / impossible-travel** feature: PaySim has no location, so **add a simulated `location` field** (assign each transaction a city; for a few users create an impossible jump) — clearly comment that this is simulated for demonstration, and note the real impossible-travel check also runs in the live demo layer (Phase 7).
Do **NOT** apply SMOTE here (see Phase 2). Set `random_state=42` everywhere for reproducibility. Print the shape, class balance, and first rows.

**Phase 2 — Hybrid model + evaluation (`train_model.py`)**
- **Train/test split FIRST**, then scale (fit scaler on train only), then apply **SMOTE only to the training set** (never before the split — that causes data leakage).
- Train **Isolation Forest** (unsupervised, set a sensible `contamination`) and **Random Forest** (supervised).
- **Combine** into one 0–1 fraud score: normalise the Isolation Forest anomaly score to 0–1, take the Random Forest probability, and combine (e.g. weighted average). Also output a **`models_disagree`** flag when one says fraud and the other says safe.
- Evaluate on the untouched test set: **accuracy, precision, recall, F1, confusion matrix** — print them and save to `docs/results.md`. Save models + scaler to `models/` with joblib.

**Phase 3 — Explainable AI (`explain.py`)**
Use **SHAP** on the Random Forest to return, for one transaction, the top 3–5 features that pushed it toward fraud, as short human phrases (e.g. "amount much higher than this user's normal", "balance mismatch", "new receiver").

**Phase 4 — GenAI agent (`genai_agent.py`)** — *build this after the core works*
Using **Ollama (llama3.2) + LangChain (LangGraph for the agent)**:
- Turn the SHAP reasons into a **plain-language explanation** + a **recommended action**.
- **RAG:** seed `knowledge_base/` with ~10 example past cases, store them in **Chroma**, retrieve similar ones to support the explanation.
- A `chat(question)` function for analyst follow-ups.
- **Do this cleanly so it can be turned off** — if Ollama is unavailable, fall back to a simple template sentence built from the SHAP reasons.

**Phase 5 — Flask API (`app.py`)** — Flask on **port 5001** (port 5000 is used by macOS AirPlay). Enable CORS.
- **Load the models once at startup**, not per request.
- `POST /predict` → returns `{ fraud_score, models_disagree, decision, shap_reasons }` **fast** (do NOT wait for the LLM here — real-time must stay ~1s).
- `POST /explain` → returns the GenAI plain-language explanation for a given transaction (called separately / lazily, since the LLM is slower).
- `POST /chat` → analyst question → answer.

**Phase 6 — Java Spring Boot backend (`backend/`)** — **fully automatic routing, NO human step:**
- Receive a transaction, store it in H2, call Python `POST /predict` (with a timeout + fallback if the service is down).
- **Routing:**
  - **low score → APPROVE** automatically
  - **high score → BLOCK** automatically
  - **middle score OR `models_disagree` → auto step-up:** first check quick signals (trusted device flag, receiver on a known-fraud list, matches a recent payee) to resolve it; if still unsure, **auto-send an OTP to the user** ("Did you make this payment? Yes/No"). The user's reply decides — **confirm → APPROVE, deny/timeout → BLOCK**. Simulate the OTP step in the demo. **No human reviewer anywhere.**
- Keep a **review log** (a record of auto-decisions the bank *could* inspect later) — but the decision is automatic.
- Endpoints to list transactions/alerts and to record the final outcome (feedback saved for future model improvement = future scope).

**Phase 7 — Dashboard (`static/index.html`)**
Served by Spring Boot: submit/simulate a transaction; live list with fraud highlighted, showing the plain-language reason and the automatic decision (APPROVE / OTP-CONFIRM / BLOCK); a simulated OTP popup for middle cases; a Chart.js summary. Also implement the **impossible-travel check here** using the simulated location (two transactions too far apart in too little time → auto-block).

**Phase 8 — Testing (`docs/`)**
Create test transactions for: R1 huge amount vs normal → block; R2 impossible travel → block; R3 unusual-for-user → flagged; R5 brand-new pattern → caught by Isolation Forest; R6 genuine-but-unusual on trusted device → OTP-confirm (not blocked); R7 rapid transfers → block. Confirm each gives the expected automatic decision. Put metrics + results in `docs/results.md`.

## 6. Common errors to avoid (check these)
- Don't apply SMOTE before the train/test split (data leakage).
- Don't fit the scaler on the whole dataset — train only.
- Don't use `isFlaggedFraud` as a feature.
- Don't block the `/predict` response on the LLM — keep scoring fast; explain separately.
- Use Flask **port 5001** on Mac, not 5000.
- Load models once at startup, not per request.
- Handle the case where Ollama isn't running (fallback template).
- Set `random_state=42` so results are reproducible.

## 7. How to run (put in README)
Exact commands to: start Ollama + confirm the model; set up the venv, install requirements, train the model, start Flask (port 5001); start Spring Boot; open the dashboard.

## 8. Style & order
- ML in Python, app in Java, over REST — keep them separate.
- **No paid APIs** — local Ollama only, with a template fallback.
- **Build the CORE fully first** (data → hybrid model → Flask → Spring Boot → dashboard, working end-to-end). **Only then** add the Ollama GenAI layer. If GenAI causes trouble or time is short, keep it as future scope — the core must always run.
- Commit to Git after each working phase.

Start with **Phase 1** now: create the folder structure, the Python virtual environment, `requirements.txt`, and `config.yaml`. Then give me the exact steps to download the PaySim dataset and where to place it. After I confirm the CSV is in place, load it and print its shape, class balance, and first rows so we confirm the data reads correctly. Do not move past Phase 1 until I confirm.
