# Bring Your Own Data — Open-Source Release Design

*Design spec. Written 2026-08-29. Phases 1–13 are complete and verified; this
covers the release work: letting **anyone run this engine on their own
transaction data**, and telling them honestly what it can and cannot detect
given what they supplied.*

## 1. Why this document exists

The system currently detects fraud on exactly one dataset. Every threshold in
`config.yaml` was measured from PaySim, every column name is PaySim's, and the
trained model on disk expects PaySim's 28 features in PaySim's order.

That is fine for a capstone and useless as a release. The question this design
answers is: *what has to be true for a stranger to point this at their own data
and get a trustworthy answer?*

The answer is not "make it accept any CSV". It is:

> **Map your columns, calibrate on your own data, and the system tells you what
> it can and cannot detect given what you gave it.**

The third clause is the one that matters. Anything can be made to *run* on
foreign data. The contribution here is that it refuses to pretend.

## 2. The decision, and what was rejected

| Question | Decision |
|---|---|
| Who is the release for? | **Open source — anyone runs it on their own data** |
| What does "their own data" mean? | **Same conceptual schema, their column names** — a mapping layer, not arbitrary-schema generalisation |
| What happens to PaySim-measured thresholds? | **A `calibrate` command derives thresholds from *their* data**, printing the evidence for each |

Rejected, with reasons:

- **Arbitrary-schema generalisation.** Every one of ~480 PaySim column references
  becomes indirect, all 13 files change, and the four anti-leakage guards that
  keep the reported metrics honest are put at risk for a capability nobody
  asked for. YAGNI.
- **Shipping PaySim's constants as defaults with a warning.** "We warned you" is
  a weaker position than "we measured yours", and it silently hands users
  thresholds that are wrong for their data.
- **Shipping the rule layer disabled on foreign data.** Hands the user a hard
  problem with no tools, and undersells one of the system's strongest pieces.

## 3. The coupling, measured

PaySim's column names appear roughly **480 times across 13 Python files**:

| Name | Occurrences |
|---|---|
| `step` | 141 |
| `nameOrig` | 54 |
| `oldbalanceOrg` / `newbalanceOrig` | 47 each |
| `nameDest` | 39 |
| `oldbalanceDest` / `newbalanceDest` | 38 / 39 |
| `isFraud` | 11 |

Including the core — `features.py`, `rules.py`, `explain.py`, `live_features.py`
— and every test file.

**But none of that has to change.** There is a single boundary:
`features.py::load_raw_data()` (line 95) does one `pd.read_csv` (line 114)
followed by one `PAYSIM_COLUMNS` membership check.

Normalise at that boundary — rename the user's columns to canonical names the
moment the file is read — and every line downstream keeps working unchanged.
PaySim's names stop being *PaySim's names* and simply become **our canonical
vocabulary**. A 13-file rewrite becomes one new module and one config block.

This is the standard pattern for exactly this problem, and it is the reason the
work is two phases rather than two months.

## 4. Canonical vocabulary

| Canonical | Meaning | Required |
|---|---|---|
| `step` | integer hour index | yes — derivable from a timestamp column |
| `type` | transaction type | yes |
| `amount` | value moved | yes |
| `nameOrig` | sender account id | yes |
| `nameDest` | receiver account id | yes |
| `oldbalanceOrg` | sender balance before | optional |
| `newbalanceOrig` | sender balance after | optional |
| `oldbalanceDest` | receiver balance before | optional |
| `newbalanceDest` | receiver balance after | optional |
| `isFraud` | ground-truth label | training only; absent when scoring |

`live_features.py::REQUIRED_FIELDS` (line 57) already encodes the runtime
half of this contract and stays the authority for live scoring.

### 4a. When the balance columns are absent

Most public transaction datasets have no balance columns. They are encoded with
**PaySim's own "not recorded" convention — `oldbalanceOrg = 0` and
`newbalanceOrig = 0`** — never with a reconciling balance.

This is not a guess. It is the same choice already justified and measured in
`webhook_razorpay.py` for the merchant route, where a merchant likewise cannot
see the payer's balance. Measured across the full 6.3M rows:

    errorBalanceOrig == 0    in  99.5%  of fraud
                             in  45.7%  of legitimate merchant payments

A perfectly reconciling balance is close to the fraud signature itself. The
zero convention is the shape of 36.0% of legitimate `PAYMENT` rows, and under
it `errorBalanceOrig` degenerates to `amount`, while `orig_account_emptied` and
`amount_to_balance_ratio` go to zero.

**We are not inventing a balance. We are using the dataset's existing
convention for "this balance is not known."**

## 5. Feature groups — the core of this design

`MODEL_FEATURES` (features.py:665) is today one fixed list of 28. It becomes
**four groups, each switched on or off by what the data can support.**

| Group | Features | Requires | Count |
|---|---|---|---|
| **A · Core transaction** | `amount`, `type_CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER`, `is_risky_type`, `hour_of_day`, `is_night` | amount, type, step | 9 |
| **B · Balance** | 4 raw balances, `errorBalanceOrig/Dest`, `absErrorBalanceOrig/Dest`, `orig_account_emptied`, `dest_was_empty`, `amount_to_balance_ratio` | the 4 balance columns | 11 |
| **C · Receiver behaviour** | `dest_txn_count`, `dest_mean_amount`, `dest_std_amount`, `dest_has_thin_history`, `dest_amount_deviation`, `dest_amount_ratio`, `dest_velocity_count`, `is_new_receiver` | nameDest + step | 8 |
| **D · Sender behaviour** | `user_txn_count`, `user_mean_amount`, `user_std_amount`, `user_has_thin_history`, `user_amount_deviation`, `user_amount_ratio`, `user_velocity_count` | nameOrig + step **and repeat senders** | 7 |

A + B + C = the current 28. Group D is **already built** in
`add_account_profiles()` and deliberately excluded from `MODEL_FEATURES`.

### 5a. Group D and why it is gated, not merely optional

Group D is excluded on PaySim for a measured reason: **99.85% of senders appear
exactly once**, so those columns are either constant or a rescaled copy of
`amount`. Worse, `user_amount_deviation` would mean something *different* at
live time (where customers do have history) than during training — a
train/serve mismatch that would silently corrupt live scores.

That reasoning is about the data, not about PaySim. So it becomes a measurement
the system performs on whatever it is given:

    repeat-sender rate = share of nameOrig values appearing more than once

    PaySim   0.15%   → Group D OFF  (reproduces the original decision)
    Sparkov  ~100%   → Group D ON

**The same rule the project applied to itself, applied automatically to
everyone.** This is the single most defensible idea in the release: the
methodology ships, not just the model.

### 5b. What losing Group B costs, and how to get it back

Without balances, 11 of 28 features — **39% of the feature set, carrying 68.3%
of model importance** (`errorBalanceOrig` 0.2105, `absErrorBalanceOrig` 0.1771,
`amount_to_balance_ratio` 0.1543, `orig_account_emptied` 0.1411) — become
constant zero. Nothing crashes; the forest simply never splits on them. The
measured consequence is already on record from the merchant route: fraud
blocking falls from **149/150 to 1/150**.

Four recovery routes, documented in order of preference:

1. **Map them.** A bank has balances under some name. This is what the mapping
   layer is for.
2. **Reconstruct from the ledger** — a causal cumulative balance per account.
   Valid only if the transaction stream is *complete*; one missing row corrupts
   every balance after it. Must use the expanding-window discipline already
   enforced in `add_account_profiles()`.
3. **Join an accounts table** — transactions + account master with EOD balance.
4. **Enable Group D instead**, when the repeat-sender measurement permits:
   17 + 7 = **24 features**. Not the same 28, but an honestly-earned
   substitute rather than a silent degradation.

## 6. Components

| Component | Status | Responsibility |
|---|---|---|
| `ml-service/schema.py` | new | Canonical field table; validate mapping; rename at load; derive `step`; apply the not-recorded encoding; emit the capability report |
| `config.yaml` → `schema:` | new block | User declares which of their columns mean what. **Defaults to identity mapping**, so PaySim behaviour is unchanged |
| `features.py::load_raw_data` | modified | The `PAYSIM_COLUMNS` check becomes `schema.normalize()`. One function |
| `features.py::FEATURE_GROUPS` | new | The four groups of section 5, declared once |
| `features.py::MODEL_FEATURES` | modified | **Keeps its name.** Becomes the resolver's output for the active config — `test_explain.py:27` imports it and iterates it, so renaming it breaks the regression gate |
| `explain.py::PHRASE_BUILDERS` | modified | **+7 entries for Group D** — see section 6a |
| `ml-service/calibrate.py` | new | Measure their data; print evidence; write derived thresholds |
| `config.yaml` → `calibrated_on:` | new key | Provenance stamp; mismatched provenance warns loudly |

Each unit is independently testable and has one job. `schema.py` knows nothing
about models; `calibrate.py` knows nothing about serving.

### 6a. Group D needs seven new phrase builders

Measured during spec review: `explain.py::PHRASE_BUILDERS` has **exactly 28
entries, matching `MODEL_FEATURES` one-for-one. None of the seven Group D
features has one.**

This is enforced, not merely untidy — `test_explain.py:57` asserts that every
model feature has a phrase builder:

    missing = [f for f in MODEL_FEATURES if f not in PHRASE_BUILDERS]

So enabling Group D without this work fails the test suite. And the failure the
test is guarding against is the one that matters most here: a SHAP reason with
no human-readable sentence behind it. The explanation layer *is* the project;
a feature that can influence a decision but cannot be explained is exactly what
this system exists not to produce.

Seven phrases are required, in the register the existing 28 use:

| Feature | Sense of the phrase |
|---|---|
| `user_txn_count` | how many past transactions this customer has |
| `user_mean_amount` | this customer's typical transaction amount |
| `user_std_amount` | how much this customer's amounts normally vary |
| `user_has_thin_history` | this customer has almost no history yet |
| `user_amount_deviation` | the amount is far above normal for this customer |
| `user_amount_ratio` | the amount is many times this customer's usual payment |
| `user_velocity_count` | an unusual number of transactions in a short time |

**This moves into Phase 14**, not Phase 15 — the regression gate cannot go green
without it once `FEATURE_GROUPS` exists.

## 7. The capability report

Printed at train time, written into `model_metadata.json`, and exposed on
`GET /health`:

Illustrative — the numbers below are the report's *shape*, not measurements:

```
DATA CAPABILITY REPORT
  Source          data/sparkov.csv
  Mapped          amount, nameOrig, nameDest, type, step, isFraud
  Not supplied    oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest

  Feature groups
    A core transaction     ON    9 features
    B balance              OFF  11 features   no balance columns supplied
    C receiver behaviour   ON    8 features
    D sender behaviour     ON    7 features   repeat-sender rate high

  Active feature count     24 of 35 possible

  COST OF GROUP B BEING OFF
    On PaySim these 11 features carried 68.3% of model importance.
    Balance-error signals will be inert. Account-drain detection
    (orig_account_emptied, amount_to_balance_ratio) is unavailable.
    Balances encoded as 'not recorded' (zero), NOT as reconciling
    balances -- see section 4a.
```

Nobody can accidentally claim a capability they do not have, and the number
that would otherwise be silent is the loudest thing on the screen.

## 8. Calibration

`calibrate.py` measures the user's data the way PaySim was measured, and writes
each threshold with the evidence beside it. It **refuses to emit any threshold
it could not measure** rather than falling back to PaySim's value.

| Config key | Derived from |
|---|---|
| `rules.rapid_sender_transfers.max_allowed` | percentile of prior sender payments per window |
| `rules.rapid_receiver_collection.max_allowed` | percentile of prior receiver collections per window |
| `model.isolation_forest.contamination` | observed fraud base rate |
| `decision.approve_below` / `block_above` | score distribution on a held-out split |
| `data.velocity_window_steps` | retained unless the time unit differs |
| *(gate)* Group D on/off | repeat-sender rate |

Output is a new config file, never an in-place edit — the user diffs it against
their current one before adopting.

## 9. Honesty guards

This release must not become a way to launder unmeasured claims. The four
existing guards are unchanged and are protected by the identity-mapping
regression gate in section 10:

- Simulated location columns stay out of the model.
- Per-account profile stats stay causal (expanding window, excludes current row).
- `isFlaggedFraud`-style leaky columns stay dropped; SMOTE stays train-split
  only; the scaler stays fitted on train only.

Three new guards:

- **No silent degradation.** A missing feature group is reported at train time,
  in metadata, and on `/health`.
- **No inherited thresholds.** A threshold that was not measured on this data is
  not written.
- **No asserted capability.** Group D turns on because the repeat-sender rate
  says so, never because the user claims their data has history.

## 10. Testing

| Suite | Covers |
|---|---|
| `test_schema.py` *(new)* | Mapping validation; unmapped required field; duplicate mapping; column absent from file; dtype coercion; `step` derived from a timestamp; the not-recorded encoding; capability report contents |
| `test_calibrate.py` *(new)* | Thresholds derived from synthetic data with known percentiles; refusal when a quantity cannot be measured; Group D gate at boundary values |
| **Identity-mapping regression gate** | **All five existing suites must stay green with the default identity mapping.** This is the proof that PaySim behaviour did not move, and it is what protects the four guards |

The regression gate is not optional and not a formality. If `test_api`,
`test_upi`, `test_webhook`, `test_explain` or `test_genai` changes behaviour
under identity mapping, the boundary normalisation is wrong.

## 11. Proving it

A release that claims "works on your data" and was only ever run on one dataset
has not earned the claim. The full pipeline runs end-to-end on a second real
public dataset: **Sparkov** (Kaggle credit-card transactions) — cardholder,
merchant, amount, timestamp, `is_fraud`, and **no balance columns**.

That absence is the point. It exercises the mapping layer, forces the capability
report to do its job on real data, and triggers the Group D gate in the opposite
direction from PaySim.

The model will be materially worse there. **That is a finding, not a failure** —
it reproduces the merchant-blindness result on a second, independent dataset,
which is considerably stronger evidence than measuring it once.

## 12. What this does not claim

- **Not arbitrary-schema.** Data must be conceptually transactional: a payer, a
  payee, an amount, a time. A PCA'd feature matrix such as the ULB credit-card
  set has no accounts and cannot be mapped — and the loader will say so rather
  than half-work.
- **Not a pretrained model you can point at anything.** Changing the feature set
  requires retraining; the shipped PaySim model cannot score a 24-feature
  matrix. Calibration and training are part of adoption.
- **Not a claim that detection quality transfers.** Only that the *method*
  transfers, and that the system reports what it lost.

## 13. Scope

| Phase | Deliverable |
|---|---|
| **14** | `schema.py`, `schema:` config block, `load_raw_data` normalisation, `FEATURE_GROUPS` + resolver (`MODEL_FEATURES` retained as its output), **7 Group D phrase builders**, capability report, `test_schema.py`, identity-mapping regression gate green |
| **15** | `calibrate.py`, Group D measurement gate, `calibrated_on:` provenance, `test_calibrate.py`, the Sparkov end-to-end run, release docs |

Untouched: model architecture, rule logic, routing, OTP, the Java backend,
UniPay, the Razorpay route, and every reported PaySim metric.

**Risk:** concentrated almost entirely in section 10's regression gate. If the
existing suites stay green under identity mapping, PaySim results cannot have
moved, because the code they exercise is byte-identical.
