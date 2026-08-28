# Integration — how a payment platform adopts this system

*Companion to the [README](../README.md). Everything measured here was produced
by `ml-service/test_upi.py` and the sampling run described in §6, against the
running service.*

---

## 1. The claim

> A payment platform adopts this fraud engine by **writing an adapter, not a model**.

The scorer has always spoken PaySim's language — `nameOrig`, `oldbalanceOrg`,
`newbalanceDest`. That is a research dataset's schema. No real payment system
will ever send it. So the schema is not the interface; **the adapter is**.

`ml-service/upi_adapter.py` is the proof. It takes the payload a UPI switch
actually emits, maps it onto what the model was trained on, and hands it to the
same scorer, the same thresholds and the same rule layer. It imports no model
and contains no scoring logic — it is a pure function from one dict to another,
about 300 lines including the reasoning.

**Measured**: the same payment expressed in UPI's language and in PaySim's
produces an identical fraud score, an identical decision and an identical
disagreement flag (`test_upi.py` check B2). If that ever drifts, the adapter has
stopped being a translation and become a second model.

---

## 2. Where this actually sits

The most common misconception is that fraud detection lives inside the payment
app. It does not, and it cannot — a PSP app is a front end that collects a payee
and a PIN. The risk decision happens where the money leaves.

```
   Customer's app   (PhonePe / GPay / Paytm / UniPay)
        │   collects payee + amount + PIN
        ▼
   PSP  /  payment gateway
        │
        ▼
   NPCI switch
        │
        ▼
   Remitter bank core  ──────►  FRM system   ◄── THIS PROJECT
        │                       (RBI mandates every bank operate one)
        ▼
   Beneficiary bank
```

Consequences worth stating plainly, because a panel will ask:

- **The customer's app is never modified.** It learns only the verdict.
- **We are not "inside UPI".** We are the risk brain a UPI participant calls.
- **This is not UPI-specific.** Cards, net banking, wallets and IMPS all have the
  same authorisation moment. Only the adapter changes.

---

## 3. The contract — one call

```
POST /predict/upi        →  score + decision + reasons        ~28 ms
```

The caller must obey three rules. They are not implementation details; they are
the security model.

| Rule | Why |
|---|---|
| **Honour the verdict.** `BLOCK` means do not move the money. | A recommendation nobody is obliged to follow is not a control. |
| **Never tell the customer the reason.** | Tell a fraudster which signal caught them and you have trained them to avoid it. The full reason belongs on the bank's console against the same reference number. |
| **Silence on an OTP is a denial.** | Otherwise the attack is: pay, say nothing, wait out the timer. |

Three possible verdicts, and what the platform does with each:

| `decision` | The platform | The customer sees |
|---|---|---|
| `APPROVE` | settle the payment | *Paid* |
| `OTP_CONFIRM` | hold, challenge the customer, settle only on an explicit YES | *Held — did you make this payment?* |
| `BLOCK` | refuse | *Not paid* + a reference number, and nothing else |

---

## 4. Request and response

### Request — the shape a PSP switch emits

```json
{
  "txnId":  "AXI4f9c2b1e7a3d",
  "amount": { "value": "3244067.13", "currency": "INR" },
  "payer":  { "vpa": "ravi.kumar@oksbi",
              "accountBalance": "3244067.13",
              "balanceAfter":   "0.00" },
  "payee":  { "vpa": "quickcash9931@ybl", "type": "PERSON" },
  "device": { "trusted": false, "location": { "city": "Mumbai" } },
  "step":   400
}
```

Only four things are genuinely required: `amount`, `payer.vpa`,
`payer.accountBalance`, `payee.vpa`. Everything else sharpens the answer.

### Response — a real one, not a mock-up

```jsonc
{
  "fraud_score": 0.9333,
  "models_disagree": false,
  "decision": "BLOCK",
  "model_decision": "BLOCK",
  "rule_applied": null,
  "shap_reasons": [
    { "feature": "errorBalanceOrig", "impact": 0.094, "direction": "increases_risk",
      "reason": "the sender's balance reconciles to the exact rupee, the signature of an automated account drain" },
    { "feature": "orig_account_emptied", "impact": 0.0805, "direction": "increases_risk",
      "reason": "this payment empties the sender's account completely" }
    // ...
  ],
  "upi": {
    "upi_txn_id": "AXI4f9c2b1e7a3d",
    "payer_vpa":  "ravi.kumar@oksbi",
    "payee_vpa":  "quickcash9931@ybl",
    "payee_class": "person",
    "mapped_type": "TRANSFER",
    "beneficiary_balance_visible": false,
    "assumptions": [
      "beneficiary balance not visible to the remitter bank (off-us payment); imputed as a clean credit so it adds no fraud signal"
    ]
  }
}
```

The response is byte-for-byte the `/predict` response plus one `upi` block. That
block exists so the caller can see **what the adapter assumed** — a score
computed from partly imputed evidence has to say so.

---

## 5. The field mapping

| UPI field | Internal field | Note |
|---|---|---|
| `payer.vpa` | `nameOrig` | lowercased and trimmed, so history keys stay stable |
| `payee.vpa` | `nameDest` | the receiver-history key — VPAs are stable identifiers |
| `amount.value` | `amount` | accepts `"48,500.00"`, `"48500.00"`, `48500.0` or the wrapper object |
| `payer.accountBalance` | `oldbalanceOrg` | **required** — see below |
| `payer.balanceAfter` | `newbalanceOrig` | computed as `balance − amount` when absent |
| `payee.accountBalance` | `oldbalanceDest` | usually absent — see §6 |
| `payee.type` / `payee.mcc` | `type` | `PERSON`→`TRANSFER`, `MERCHANT`→`PAYMENT` |
| `device.trusted` | `trusted_device` | rule layer only, never a model feature |
| `device.location` | `location`, `location_lat/lon` | feeds the impossible-travel rule |
| `step` / `hour` | `step` | left unset if absent — the adapter never invents an hour |

Two mappings that are decisions, not plumbing:

**P2P → TRANSFER, P2M → PAYMENT.** `is_risky_type` is the model's 5th strongest
feature (0.0640). Mapping every UPI payment to `TRANSFER` would quietly inflate
the score on every legitimate shop purchase in India. A payee carrying a real
merchant category code is treated as a merchant whatever the payload declared;
MCC `0000` is UPI's placeholder for "not a merchant" and does not count.

**`payer.accountBalance` is required, never defaulted to zero.** The remitter
bank always holds its own customer's balance, and four of the model's five
strongest features are derived from it. A missing balance silently read as zero
would produce a confident score computed from nothing.

---

## 6. The honest problem: a bank cannot see the other side's balance

PaySim gives every row four balance columns — both sides, before and after. Real
UPI does not, and cannot. The remitter bank sees its own customer's account; the
beneficiary's balance sits in a different bank's core, behind NPCI, and is never
disclosed. It arrives only for an **on-us** payment, where both accounts happen
to be at the same bank.

**What the adapter does.** When the payee balance is absent it imputes a *clean
credit* — `oldbalanceDest = 0`, `newbalanceDest = amount` — so the destination
balance equation reconciles exactly and the imputation contributes **no fraud
signal in either direction**. Imputing `0 / 0` would have been much worse: in
PaySim an unchanged destination balance is part of the drain signature, so every
off-us payment would have been handed a free push toward `BLOCK`.

**Why the loss is survivable.** Of the model's top ten features by importance,
the four largest are all sender-side:

| feature | importance |
|---|---|
| `errorBalanceOrig` | 0.2105 |
| `absErrorBalanceOrig` | 0.1771 |
| `amount_to_balance_ratio` | 0.1543 |
| `orig_account_emptied` | 0.1411 |
| **combined** | **0.6830** |

68.3% of the model's importance comes from evidence a remitter bank genuinely
holds. The evidence a real bank is missing is not the evidence this model leans
on.

### What it actually costs — measured

300 PaySim `TRANSFER`/`PAYMENT` payments (150 fraud, 150 legitimate), each scored
twice against the same engine: once with the true destination balances, once with
the beneficiary balance withheld. History was reset before every score, so the
only difference between the arms is the destination evidence.

| | with full evidence | beneficiary blind |
|---|---|---|
| Decision unchanged | — | **288 / 300 (96.0%)** |
| Fraud `BLOCK`ed | 150 / 150 | 143 / 150 |
| Fraud silently `APPROVE`d | 0 | **0** |
| Legitimate `BLOCK`ed | 0 / 150 | **0 / 150** |
| Legitimate sent to OTP | 4 / 150 | 9 / 150 |

Median score movement: **−0.0217**. Largest single movement: 0.3468.

**The finding that matters:** all 7 frauds that stopped being blocked moved to
`OTP_CONFIRM` — **not one was silently approved**. Losing the beneficiary's
balance degrades the system's *confidence*, and the step-up path absorbs exactly
that: uncertainty is routed to the customer instead of being resolved by
guessing. This is the routing design earning its keep under real-world evidence
loss, which is a stronger result than the accuracy table.

The price is 5 extra legitimate customers challenged per 150 payments (2.7% →
6.0%). Whether that trade is acceptable is a business decision, and
`config.yaml` is where it is tuned.

---

## 7. The gateway route: a real Razorpay webhook

`POST /webhook/razorpay` accepts a genuine Razorpay webhook — the events a real
payment company sends a real merchant. It verifies the signature, maps the
payload through the same UPI adapter, and scores it with the same engine.

In production this receiver would sit on the **merchant's** server, not inside
the fraud engine; it lives here because that is where the mapping code is.

### Getting one to arrive

```bash
export RAZORPAY_WEBHOOK_SECRET='whsec_...'   # from the Razorpay dashboard
export RAZORPAY_MERCHANT_VPA='yourstore@razorpay'
./venv/bin/python app.py

cloudflared tunnel --url http://localhost:5001    # Razorpay cannot reach localhost
```

Then add the tunnel's URL + `/webhook/razorpay` as a webhook in the Razorpay
dashboard (test mode), subscribed to `payment.authorized`.

### Three things this route gets right

| | Why it matters |
|---|---|
| **The signature is checked against the raw bytes, first** | Re-serialising the JSON changes byte order and breaks the HMAC. Everything before the check treats the body as hostile; everything after treats it as trusted. Compared in constant time, because `==` leaks how many bytes matched. |
| **No secret means refuse everything** (503) | An unverified webhook endpoint is an open door — anyone with the URL could post payments and read the verdicts. |
| **Paise are not rupees** | Razorpay sends integer paise. Treating 4,850,000 as rupees is a silent 100× error in every amount and every threshold comparison. A float or string amount is refused rather than guessed. |

### `authorized` decides; `captured` only reports

A gateway sends several events per payment, and only one of them is a decision:

- **`payment.authorized`** — funds held, nobody has taken them. With *manual
  capture* the merchant still chooses capture or void. This is a point in the
  money path, so the response is marked `enforceable: true`.
- **`payment.captured`** / **`payment.failed`** — already moved, or already
  dead. Scored, but returned with `enforceable: false` and a note. A verdict
  here can only prompt a refund.

Retries are deduplicated on Razorpay's payment id: a replay answers with the
original verdict rather than scoring the payment into its own history twice.

---

## 7a. The merchant is blind to the payer's balance — and it is expensive

Section 6 showed a bank cannot see the *beneficiary's* balance. A merchant has
the mirror problem, and it is far worse: it cannot see the **payer's** balance.
Not "usually does not" — cannot. The payer banks elsewhere, and no gateway
discloses it. That is where 68.3% of this model's importance lives.

**The imputation, chosen by measurement.** The tempting choice is a balance that
reconciles exactly. Measured on all 6,362,620 rows, that would have been a
serious mistake:

| `errorBalanceOrig == 0` | share |
|---|---|
| among **fraud** | **99.5%** |
| among legitimate merchant payments | 45.7% |

A perfectly reconciling sender balance is close to the fraud signature itself.
So the adapter uses PaySim's own encoding for a balance that was never recorded
— `oldbalanceOrg = 0, newbalanceOrig = 0` — the shape of **36.0%** of legitimate
`PAYMENT` rows. Under it `errorBalanceOrig` lands on `amount` rather than on 0,
`orig_account_emptied` stays false, and `amount_to_balance_ratio` is 0. We are
not inventing a balance; we are using the dataset's existing convention for
"not known", which is exactly the merchant's situation.

### What it costs — measured

300 `TRANSFER`/`CASH_OUT` payments (150 fraud, 150 legitimate), each scored
twice against the same engine, history reset before every score. Set beside the
section 6 result, the contrast is the finding:

| | beneficiary blind (§6) | **payer blind** |
|---|---|---|
| Decision unchanged | 96.0% | **44.7%** |
| Fraud BLOCKed | 143/150 | **1/150** |
| Fraud silently APPROVEd | **0** | **57/150 (38%)** |
| Legitimate BLOCKed | 0/150 | 4/150 |

**Losing the beneficiary's balance is nearly free. Losing the payer's balance
collapses detection.** Fraud blocking falls from 149/150 to 1/150, and 38% of
fraud is silently approved — the failure the whole system exists to prevent. The
step-up path catches 92 of the remaining 150 as `OTP_CONFIRM`, but a merchant
cannot send a bank's OTP, so in practice that is not a rescue.

This is the quantified argument for the architecture: **fraud detection belongs
where the payer's balance is, which is the remitter bank.** Not as a preference
— as a 149-to-1 difference.

### And PaySim has no merchant fraud at all

| type | legitimate | fraud |
|---|---|---|
| TRANSFER | 528,812 | 4,097 |
| CASH_OUT | 2,233,384 | 4,116 |
| **PAYMENT** | **2,151,495** | **0** |
| CASH_IN | 1,399,284 | 0 |
| DEBIT | 41,432 | 0 |

Every one of PaySim's 8,213 frauds is a `TRANSFER` or a `CASH_OUT`. There is no
merchant-directed fraud in the dataset, so the gateway route's effectiveness
against real merchant fraud **cannot be evaluated here at all**. The webhook
proves the *integration* — real payloads, real signatures, real events. It does
not, and cannot, prove detection on this data.

---

## 8. Adopting it for something other than UPI

Write a second adapter. It needs to answer four questions, and nothing else:

1. Who is paying, and what is their balance before and after?
2. Who is being paid, and have we seen them before?
3. How much, and at what hour?
4. Is this a person or a merchant?

For cards that is the ISO 8583 authorisation message; for net banking, the core
banking debit advice. Neither touches the model, the thresholds or the rule
layer. `upi_adapter.py` is the worked example — copy its structure, including
the part where it reports what it had to assume.

---

## 9. What this project does *not* claim

- **We are not connected to live UPI, and cannot be.** That requires NPCI
  membership, a PSP licence or a sponsor bank, and NPCI certification — a
  regulated banking entity, not a student project. Any capstone claiming
  otherwise is bluffing.
- **We simulate the bank. We do not simulate the fraud detection.** UniPay plays
  the customer app and the payload shapes are the industry's; the scoring, the
  rules and the routing are real code making real decisions.
- **The measurements above are PaySim's**, and PaySim's balance columns follow
  mechanical rules. The 96.0% agreement figure describes how gracefully this
  pipeline degrades, not what a real bank would see.

---

## 10. Verifying it yourself

```bash
cd "/Users/lucky/Documents/fraud detection system/ml-service"
./venv/bin/python app.py          # in one terminal
./venv/bin/python test_upi.py     # in another — 66 checks
```

48 mapping checks run offline with no server. The remaining 18 need the service
up, and include the equivalence check that keeps the adapter honest.
