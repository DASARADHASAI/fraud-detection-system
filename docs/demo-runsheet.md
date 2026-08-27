# Demonstration run-sheet

*Read this while recording the backup video, and again before the viva. Beat
timings are targets, not limits — the whole arc is ~7 minutes, leaving room for
questions in a 10-minute slot.*

Design and rationale: [superpowers/specs/2026-08-28-demonstration-plan-design.md](superpowers/specs/2026-08-28-demonstration-plan-design.md)

---

## Before you press record

```bash
./demo.sh                      # then wait for READY
```

- All three health dots green — backend, ML service, GenAI.
- Feed already shows ~15 approved payments. If it is empty, `demo.sh` did not seed.
- A second terminal open and visible, sitting in `ml-service/`, for beat 5.
- Notifications off. Browser zoom set so the decision feed is readable from the back.
- **One take.** If you stumble, keep going — a cut invites "what happened there?"

---

## Beat 1 — Control (~30 s)

**Do:** In *New payment*, leave the defaults (Rs 2,500, C1001 → M-GROCER). Send.

**Say:**
> "An ordinary payment. Approved automatically in about thirty milliseconds, and
> the system says why: it does not drain the sender's account. I am starting here
> deliberately — if I only showed you fraud being blocked, you could not tell
> whether the system was thinking or just blocking everything."

---

## Beat 2 — Obvious fraud (~60 s)

**Do:** Click scenario **R1**. Then click the blocked row to open it.

**Say:**
> "The whole balance, moved in one transfer, to an account with no history.
> Blocked, score 0.94. The reason it gives is that the sender's balance
> reconciles to the exact rupee — the signature of an automated drain, not a
> person deciding what to send."

Point at the SHAP contributions. Do not read them all out; name the top one.

---

## Beat 3 — What a model cannot learn (~60 s)

**Do:** Click scenario **R2**. Two payments land: Mumbai, then Kolkata.

**Say:**
> "Two payments, 1,650 kilometres apart, inside the same hour. Blocked — but
> notice *what* blocked it. The row says `impossible_travel`, a rule, not the
> model. PaySim has no location column at all, so we could not train this. A
> model cannot learn what is not in the data. We simulate the location, we
> exclude it from the trained model, and we handle it as an explicit rule
> instead — because reporting accuracy on invented evidence would be dishonest."

**This is the beat that buys you credibility for everything after it.** Do not rush it.

---

## Beat 4 — Uncertainty (~90 s)

**Do:** Click scenario **R6**. The OTP modal appears. Pause on it. Click **"Yes, that was me"**.

**Say, while the modal is open:**
> "Rs 4,80,000 at three in the morning, but from a device this customer has used
> before. The two models disagree, and disagreement is itself a signal — so
> instead of guessing, the system asks the customer."

**After clicking Yes:**
> "Approved. Three things worth stating, because they are easy to get backwards.
> Silence is a denial — an unanswered challenge blocks the payment, otherwise an
> attacker just stays quiet and waits out the timer. A trusted device is never
> enough on its own; it proves who is paying, not whether they meant to. And
> nothing may soften a decision except the customer answering yes — that is the
> one piece of evidence no other component has."

> *If asked why 3am:* the clock is a real model feature. The Clock dropdown on
> the form changes it live — the same payment at 2pm scores 0.22 and approves.

---

## Beat 5 — Kill the ML service (~60 s)

**Do:** In the second terminal, `./demo.sh --stop` is too blunt — instead find and
kill only Python:

```bash
lsof -ti tcp:5001 | xargs kill
```

Watch the ML health dot go red. Then send an ordinary payment from the form.

**Say:**
> "I have just killed the machine learning service. The bank's payment traffic
> does not stop because a model server crashed. Watch — it still decides, and it
> flags every such decision as degraded, all the way to this screen. It does not
> fail open, which would leave the bank unprotected, and it does not fail closed,
> which would mean the bank has DDoSed itself. A crude local check errs toward
> asking the customer."

**Restart it:**
```bash
cd ml-service && ./venv/bin/python app.py &
```
Wait for the dot to go green again before moving on.

---

## Beat 6 — Why two models (~60 s)

**Do:** Switch to `docs/results.md` (or your slide of the same table).

**Say:**
> "The obvious objection is that the Random Forest alone scores the same F1, so
> why carry a second model. We tested it properly: we hid *every* transfer fraud
> from the supervised model during training, then showed it those frauds. It
> caught 96%. The unsupervised model caught 47.7%. Together, 99.9% — and the
> unsupervised model rescued 32 frauds the supervised one missed completely.
> That is what the hybrid buys.
>
> The cost, honestly: on fraud types it already knows, the anomaly detector
> contributes only false alarms — about 2% of customers get an extra challenge.
> Whether that trade is worth it is a business decision, and the thresholds sit
> in a config file so a risk team can move them without a redeploy."

---

## Beat 7 — Deployment and integration (~60 s)

**Say:**
> "It runs as two services over REST, and one script brings both up from cold
> with health-gated startup. It is not on a public URL, deliberately — a bank's
> fraud engine sits inside the bank's network, called by the core banking system
> at the authorisation step. Exposing it publicly would be a design error, not a
> milestone.
>
> To integrate it: the core banking system calls one endpoint. Swap the embedded
> database for the bank's own, the simulated OTP for the SMS channel they already
> run, and the simulated location for the device and IP data they already hold.
>
> One thing would have to change, and it is the most interesting part. We
> excluded all sender-side behavioural features from the model, because 99.85% of
> senders in this dataset appear exactly once — a per-customer profile is
> mathematically a constant here. A real bank customer has years of history. Those
> features are already computed and waiting; they were withheld because a feature
> that is constant in training and variable in serving is worse than no feature
> at all."

---

## If something breaks

| What | Do | Say |
|---|---|---|
| ML service down unexpectedly | Nothing — carry on | "Conveniently, this is exactly what I was about to show you." Then run beat 5 early. |
| Ollama down | Ignore it | "The narration layer is optional by design — the decision never waits on it." |
| A scenario gives an unexpected decision | Open the row and read the actual reason | "Let's look at why it decided that" — the explanation is the product. |
| Dashboard unresponsive | Reload the page | State lives in the database, not the browser. |
| Total loss | Play the backup video | Narrate over it exactly as above. |

---

## Recording it

```bash
./record-demo.sh                 # silent — you narrate live over it
./record-demo.sh --with-audio    # capture narration now
```

First run needs Screen Recording permission for Terminal
(System Settings → Privacy & Security → Screen & System Audio Recording),
then a full quit and reopen of Terminal. **Test this today, not on the day.**

Watch it back before trusting it. Check the feed is legible, the OTP modal is on
screen long enough to read, and the degraded-mode badge in beat 5 is visible.
