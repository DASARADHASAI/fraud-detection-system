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
- [x] **Phase 2** — Hybrid model + evaluation (`ml-service/train_model.py`)
- [x] **Phase 3** — SHAP explainability (`ml-service/explain.py`)
- [x] **Phase 4** — GenAI agent, optional (`ml-service/genai_agent.py`)
- [x] **Phase 5** — Flask API on port 5001 (`ml-service/app.py`)
- [x] **Phase 6** — Spring Boot backend + automatic routing (`backend/`)
- [x] **Phase 7** — Dashboard (`backend/src/main/resources/static/index.html`)
- [x] **Phase 8** — Acceptance tests (`docs/acceptance-tests.md`)
- [x] **Phase 9** — Demonstration: UniPay, `demo.sh`, run-sheet
- [x] **Phase 10** — UPI adapter + integration contract (`ml-service/upi_adapter.py`, [docs/INTEGRATION.md](docs/INTEGRATION.md))
- [x] **Phase 11** — UniPay's state moved into the bank (`backend/.../LedgerService.java`)
- [x] **Phase 12** — Razorpay webhook receiver (`ml-service/webhook_razorpay.py`)
- [x] **Phase 13** — Live wiring: tunnel, self-test, runbook ([docs/RAZORPAY-LIVE.md](docs/RAZORPAY-LIVE.md))

## Two applications, not one

| | URL | What it is |
|---|---|---|
| **UniPay** | `localhost:8080/pay.html` | The **customer's** UPI payment app |
| **FraudDesk** | `localhost:8080` | The **bank's** fraud console |

The console on its own is a closed loop — you click a button in the bank's screen
and the bank's screen answers. UniPay is a separate application that holds a
balance, knows its payees, and calls `POST /api/transactions` at the
authorisation step exactly as a bank's core system would. It cannot see a score,
a model or a rule; it learns only what it is allowed to learn — paid, held, or
declined.

**We simulate the bank. We do not simulate the fraud detection.**

Two deliberate details:

- **A declined customer is never told why.** Telling a fraudster which signal
  caught them is how they learn to avoid it next time. The full reason sits on
  the console against the same reference number.
- **The fraud signals come from the app's own state**, not from hard-coded test
  payloads. Paying your entire balance to a never-seen UPI ID *is* the
  account-drain shape; the model is not being fed a scenario.
- **That state belongs to the bank, not to the page.** The balance, the payee
  list and whether the device is known are all fetched from
  `/api/account/*` and `/api/device/seen`. See below.

### And a third door, for everyone else

UniPay is one caller. The point of the system is that it is not the only
possible one.

`POST /predict/upi` accepts the payload a **UPI switch actually emits** — VPAs,
a rupee amount object, a payee type — maps it onto the schema the model was
trained on, and returns the same score, the same decision and the same reasons.
The mapping lives in `ml-service/upi_adapter.py`, which imports no model and
holds no scoring logic. A payment platform adopts this engine by writing an
adapter, not a model.

Measured: the same payment expressed in UPI's language and in PaySim's produces
an identical score and an identical decision.

The one thing a real bank does not have is the *beneficiary's* balance — it sits
in another bank's core, behind NPCI. We measured what that blindness costs:
across 300 payments the decision was unchanged 96.0% of the time, and of the 7
frauds that stopped being blocked, **all 7 moved to OTP confirmation and none
was silently approved**. The step-up path absorbs exactly the uncertainty that
missing evidence creates.

The contract, the field mapping, the measurements and the limits are in
**[docs/INTEGRATION.md](docs/INTEGRATION.md)**.

### A real payment gateway, and the number that justifies the whole design

`POST /webhook/razorpay` takes a genuine Razorpay webhook — HMAC-signed, paise
not rupees, `payment.authorized` treated as a decision and `payment.captured` as
a report. Retries are deduplicated; an unsigned body is a 401; and with no
secret configured it refuses everything rather than running unauthenticated.

Building it surfaced the sharpest result in the project. A merchant cannot see
the **payer's** balance — and that is where 68.3% of the model's importance
lives. Measured against the beneficiary blindness of the bank-side route:

| 300 payments, scored twice | beneficiary blind | **payer blind** |
|---|---|---|
| Decision unchanged | 96.0% | **44.7%** |
| Fraud BLOCKed | 143/150 | **1/150** |
| Fraud silently APPROVEd | **0** | **57/150** |

**Losing the beneficiary's balance is nearly free. Losing the payer's balance
collapses detection** — 38% of fraud silently approved. That is the quantified
reason this system is built to sit at the remitter bank rather than at a
merchant: not a preference, a 149-to-1 difference.

One further limit, stated before anyone asks: **PaySim contains no merchant
fraud at all.** All 8,213 frauds are TRANSFER or CASH_OUT; `PAYMENT` has zero.
The webhook proves the integration — real payloads, real signatures. It cannot
prove detection on this data.

### Actually going live

```bash
export RAZORPAY_WEBHOOK_SECRET='whsec_pick_something_long'

cd ml-service                                    # terminal 1
RAZORPAY_WEBHOOK_SECRET="$RAZORPAY_WEBHOOK_SECRET" ./venv/bin/python app.py

brew install cloudflared                         # once
./razorpay-live.sh                               # terminal 2
```

`razorpay-live.sh` refuses to raise a tunnel until the local path is proved,
then prints the exact URL and event to paste into the Razorpay dashboard. The
full runbook, including the four ways this goes wrong and how to tell them
apart, is in **[docs/RAZORPAY-LIVE.md](docs/RAZORPAY-LIVE.md)**.

The secret is a value **you invent** and export — it is never issued to you and
never lives in this repository. With no secret configured the endpoint returns
503 to everything rather than running unauthenticated.

### UniPay's state is the bank's, not the page's

UniPay used to open with `balance: 1000000` and a hard-coded contact list, both
sitting in JavaScript where anyone could read them in View Source. Three things
moved to the server:

| | Was | Is |
|---|---|---|
| **Balance** | a JS literal | the `accounts` table, read over REST |
| **Payees** | a JS array of three names | derived from payments that actually completed |
| **Known device** | a switch on the page | a fingerprint the bank has seen before |

**This was a security fix, not cosmetics.** `amount_to_balance_ratio` and
`orig_account_emptied` are the model's 3rd and 4th strongest features, and both
are computed from the sender's balance. While that number arrived in the
request, the payer chose the evidence used to judge it — claim a huge balance
and a full account drain stops looking like one. `TransactionController`
now overwrites the balances from the ledger for any account this bank holds:

```
claimed oldbalanceOrg  99,999,999   ->  stored 99,500  ->  BLOCK (0.7983)
```

Accounts the bank does *not* hold — every counterparty at another bank, and
every account the test suites invent — are left exactly as sent, so the change
is additive.

Two consequences worth noticing on a fresh database:

- **You start with no payees.** The first payment to anybody is a payment to a
  new receiver, which is a genuine signal rather than a penalty. A payee appears
  because a payment to them went through.
- **Your browser is not trusted on its first payment.** Trust is earned by
  having been seen before, so it means *"the bank has a record of this device"*
  — something an attacker on a fresh machine does not have. Clear the site data
  and you are a new device again.

Two conditions still cannot come from a browser, and the panel on the right says
so: the **clock** (a server cannot be time-travelled, and `hour_of_day` is a real
feature) and a **journey** (you cannot be in two cities an hour apart — which is
the entire point of the impossible-travel rule). Location otherwise comes from
the device's real GPS fix, and when permission is denied nothing is claimed.

## Running it for a demonstration

```bash
./demo.sh                # cold machine -> demo-ready, then opens the dashboard
./demo.sh --no-reset     # keep the existing database (mid-session rehearsal)
./demo.sh --stop         # stop both services
```

Resets the H2 database, starts both services **polling `/health` until each is
actually ready** rather than sleeping a fixed duration, warms Ollama so the
first narration is not slow in front of an audience, and seeds ~15 ordinary
payments so the feed and chart are not empty.

It never runs `features.py` — that is a ~50 s, ~4.5 GB pass and has no place on
demo day. If the model files are missing it says so and stops, rather than
failing halfway through.

The demonstration itself is designed in
[docs/superpowers/specs/2026-08-28-demonstration-plan-design.md](docs/superpowers/specs/2026-08-28-demonstration-plan-design.md)
— the arc, the pre-flight checklist, failure fallbacks, and the answers to the
questions a panel actually asks.

### The clock is a model feature, so the dashboard pins it

`hour_of_day` and `is_night` are real features, so an identical payment scores
0.2954 at 3am and 0.2206 at 2pm — either side of the 0.35 approve threshold.
Sending no `step` makes the ML service fall back to wall-clock **UTC** hours, and
because IST is UTC+5:30 that silently stops R6 raising an OTP challenge after
11:30 local time. Every scenario button now pins `step`, exactly as
`acceptance_tests.py` does. The payment form exposes it as a **Clock** dropdown,
so "what if this were 3am?" is answered on screen in one click.

## Running the ML service

```bash
cd "/Users/lucky/Documents/fraud detection system/ml-service"

# one-time: build features, then train (Phase 1 ~51s, Phase 2 ~2min)
./venv/bin/python features.py
./venv/bin/python train_model.py

# start the API
./venv/bin/python app.py            # http://localhost:5001
```

Optional, for the GenAI narration — the system runs fine without it:

```bash
ollama serve                        # in another terminal
ollama pull llama3.2
```

### Endpoints

| Method | Path | Speed | Purpose |
|---|---|---|---|
| GET | `/health` | instant | Service, model and Ollama status |
| GET | `/config` | instant | Decision thresholds, so Java need not duplicate them |
| POST | `/predict` | **~28 ms** | Score a transaction — called on every payment |
| POST | `/predict/upi` | **~28 ms** | Same engine, UPI-shaped payload — see [INTEGRATION.md](docs/INTEGRATION.md) |
| POST | `/webhook/razorpay` | ~30 ms | A real Razorpay webhook, signature-verified |
| GET | `/webhook/razorpay/recent` | instant | What actually arrived — the wiring diagnostic |
| POST | `/explain` | **~3.7 s** | Plain-language write-up — called only when someone opens a transaction |
| POST | `/chat` | ~3 s | Analyst question, grounded in `knowledge_base/` |
| POST | `/reset` | instant | Clear demo history (testing only) |

```bash
curl -X POST localhost:5001/predict -H 'Content-Type: application/json' -d '{
  "step": 400, "type": "TRANSFER", "amount": 3244067.13,
  "nameOrig": "C111", "oldbalanceOrg": 3244067.13, "newbalanceOrig": 0,
  "nameDest": "C999", "oldbalanceDest": 0, "newbalanceDest": 0
}'
```

**`/predict` and `/explain` are separate on purpose.** Scoring takes 28 ms;
llama3.2 narration takes 3.7 s — a **135× difference**. A payment cannot wait
three seconds for prose nobody may ever read, so `/predict` returns the score
plus the SHAP reasons (already human-readable) and the narration is fetched
lazily. Measured: with three `/explain` calls in flight, `/predict`'s median
rises only from 28 ms to 111 ms.

### Tests

```bash
./venv/bin/python test_explain.py     # 13 checks — SHAP correctness
./venv/bin/python test_genai.py       # 36 checks — GenAI guards and fallbacks
./venv/bin/python test_api.py         # 44 checks — features, validation, HTTP
./venv/bin/python test_upi.py         # 66 checks — the UPI adapter
./venv/bin/python test_ledger.py      # 34 checks — balances, payees, device trust
./venv/bin/python test_webhook.py     # 74 checks — Razorpay signatures, mapping, diagnostics
./venv/bin/python test_backend.py     # 33 checks — the Java backend over HTTP
./venv/bin/python acceptance_tests.py # 7 end-to-end cases (R0–R7)
```

The last one needs **both** services running. It posts payments exactly as the
dashboard does and writes [docs/acceptance-tests.md](docs/acceptance-tests.md)
from the real run — the document is generated, so it cannot drift from the code.

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

## What we measured about PaySim (and why the features look the way they do)

Before trusting any feature, we counted how often each account actually
reappears across all 6,362,620 rows:

| | senders (`nameOrig`) | receivers (`nameDest`) |
|---|---|---|
| appear exactly once | **99.85%** | 83.1% |
| max appearances | **3** | **113** |
| rows belonging to a repeat account | **0.29%** | **64.4%** |

PaySim mints a fresh customer ID for almost every transaction, so **sender
history effectively does not exist**. A per-sender profile and per-sender
velocity are inert on this dataset no matter how you sample — the columns come
out constant, and `amount_deviation` collapses into a rescaled copy of `amount`.

So the model's behavioural features are built on the **receiver** side, where
history is real. What they genuinely contribute is **amount**-based, not
activity-based — the separation is clear:

| feature | legitimate | fraud |
|---|---|---|
| `dest_amount_ratio` | 0.885 | **8.201** |
| `dest_amount_deviation` | 0.089 | **2.979** |
| `is_new_receiver` | 0.427 | 0.630 |

Do **not** claim the model detects mule accounts collecting rapid transfers.
We measured it: fraud receivers are *less* busy than legitimate ones (0.036 vs
0.189 prior payments per hour). That typology is real in banking and absent
from PaySim, so it is handled by the **rule layer**, not the model — see below.

The sender-side profile is still computed — in the live system a real customer
*does* have history — but it is kept out of the trained model. Feeding it in
would be worse than useless: `user_amount_deviation` is a constant during
training but varies at prediction time, so the model would be reading a feature
whose meaning silently changed between training and serving.

### Order of operations matters

Account history is a relationship *between* rows, so `features.py` engineers it
on the **full 6.3M rows and only then samples down**. Sampling first keeps ~3%
of rows, which throws away the earlier transactions that gave each account its
history — we measured that mistake making 99.5% of receivers look brand new,
against a true figure of 56%.

This is why Phase 1 takes ~50 seconds and peaks around 4.5 GB of RAM. If your
Mac is short on memory, lower `data.sample_size` in `config.yaml` — the
full-file pass still runs, only the output gets smaller.

## Honest notes for the report

- **Location data is simulated.** PaySim has no geo column at all. We generate a
  plausible city per transaction so the impossible-travel feature can be
  demonstrated, and we exclude it from the trained model — inventing evidence
  and then reporting accuracy on it would be dishonest. Impossible travel runs
  as a hard rule in the live layer instead (Phase 7). Only 109 impossible-travel
  cases exist in the whole file, because a jump needs a customer with two
  transactions and PaySim has only 9,298 of those.
- **Sender-side behavioural features are excluded from the model** — not by
  preference but because they are provably inert on this dataset. See the table
  above.
- **`isFlaggedFraud` is dropped.** It is the dataset's own built-in naive rule.
  Training on it would be data leakage.
- **SMOTE is applied only to the training split**, after the train/test split,
  never before — otherwise synthetic copies of test rows leak into training and
  the reported accuracy becomes meaningless.
- **The scaler is fitted on training data only**, for the same reason.
- **No paid APIs.** The GenAI layer is local Ollama, and falls back to a plain
  template sentence if Ollama is not running. The core system works without it.

## Running the Spring Boot backend

```bash
cd "/Users/lucky/Documents/fraud detection system/backend"
mvn spring-boot:run                 # http://localhost:8080
```

Start the Python service first — the backend works without it, but in
**degraded mode** (see below).

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/transactions` | Submit a payment — comes back **already decided** |
| POST | `/api/transactions/{id}/otp?confirmed=true` | Customer answers the challenge |
| GET | `/api/transactions` | Recent payments |
| GET | `/api/alerts` | Everything not simply approved |
| GET | `/api/transactions/{id}` | One payment plus its full audit trail |
| GET | `/api/transactions/{id}/explanation` | Plain-language write-up (lazy, ~3.7 s) |
| POST | `/api/transactions/{id}/feedback?wasFraud=true` | Record the real outcome |
| GET | `/api/stats` | Dashboard counts |
| GET | `/api/health` | Both services + the thresholds in force |
| GET | `/api/account/me` | The customer's real balance, from the ledger |
| GET | `/api/account/{h}/payees` | Who they have actually paid before |
| POST | `/api/device/seen` | Has this bank seen this device before? |

Browse the database at `http://localhost:8080/h2-console`
(JDBC URL `jdbc:h2:file:./data/frauddb`, user `sa`, no password).

### The automatic routing

```
                         ┌─ score < 0.35, models agree ──────→ APPROVE
  payment → ML score  ───┼─ score > 0.75 ────────────────────→ BLOCK
                         └─ in between, OR models disagree ──→ step up:
                                1. known-fraud receiver?  → BLOCK
                                2. established payee + known device? → APPROVE
                                3. otherwise → ask the customer by OTP
                                     YES → APPROVE
                                     NO or no reply → BLOCK
```

**No human reviewer exists anywhere.** The `review_log` table is an audit
record a bank *could* inspect afterwards, not a queue of work. `/api/stats`
reports `humanReviewers: 0`.

Three details that are easy to get backwards:

- **Silence is a denial.** An unanswered OTP blocks the payment. The opposite
  would hand an attacker a free win: pay, stay quiet, wait out the timer.
- **A trusted device is never enough on its own.** It proves *identity*, not
  *intent* — see knowledge-base case FRAUD-006, where the victim authorises the
  payment themselves. It only counts alongside an established payee.
- **Nothing may soften a decision**, except a customer answering YES to an OTP —
  the one piece of evidence no other component has.

### When the Python service is down

The backend keeps running. It does **not** approve everything (the bank would be
unprotected) and does **not** block everything (the bank has DDoSed itself).
Instead a crude local check errs toward asking the customer, and every such
decision is flagged `degradedMode: true` all the way to the dashboard. Verified
with Python killed:

| Payment | Degraded decision |
|---|---|
| Small merchant payment | APPROVE |
| Full-balance transfer to an empty account | BLOCK |
| Large transfer, balance left over | OTP_CONFIRM |

### Backend tests

```bash
./venv/bin/python test_backend.py           # 33 checks over HTTP
./venv/bin/python test_backend.py --slow    # adds the OTP-timeout check
```
