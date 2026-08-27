# Demonstration Plan — Live Viva Design

*Design spec. Written 2026-08-28. Phases 1–8 are complete; this covers Phase 9,
which the original build prompt never defined: **showing that it works.***

## 1. Why this document exists

`CLAUDE_CODE_BUILD_PROMPT_FINAL.md` defines eight phases and stops at testing.
Nothing in the repository is a runbook. The README is setup instructions written
for a developer with unlimited time and no audience; it is not usable while a
panel is watching.

The raw materials for a demonstration already exist and are good:

| Asset | Location | Role in the demo |
|---|---|---|
| Six one-click scenario buttons | `index.html:795` | Fire whole payment paths live |
| Seven end-to-end acceptance cases | `acceptance_tests.py` | Proof, regenerates `docs/acceptance-tests.md` |
| Blinded-holdout experiment | `docs/results.md` | The single strongest number: 32 frauds rescued |
| Degraded-mode behaviour | `MlServiceClient` + README table | The most distinctive beat in the demo |
| Latency measurements | README | Answers "is it really real-time?" |

What is missing is everything that turns those into a performance that survives
contact with a projector, a panel, and nerves.

## 2. The setting

Confirmed by the user:

- **Live viva to a panel**, presented **on the user's own MacBook**.
- The panel is expected to probe **deployment** and **integration with a real bank**.
- **The UI must carry the demonstration** — it is the thing being shown.

Assumed, because they were not specified. Any of these can change without
invalidating the design; they only affect how much of section 5 gets built:

- **Time remaining: ~2 weeks.** The plan is ordered so that stopping early still
  leaves a working demonstration — section 5.1 alone is enough to be safe.
- **Slot length: unknown.** The arc is therefore built as a **10-minute core**
  with optional bolt-on modules, rather than a fixed 20-minute script.
- **"Deploy" means "runs reliably and is packaged so anyone could run it"**, not
  "is live on a public URL". Section 8 argues this is the correct reading and
  arms the user to defend it either way.

## 3. The defect that would have broken the demonstration

**The R6 scenario button stops producing an OTP challenge after roughly
11:30 AM IST, and fails silently.**

The chain, verified in the code:

1. `index.html:831` — the scenario sender builds its payload without `step`.
2. `MlServiceClient.java:76` — `if (request.getStep() != null)` means nothing is
   forwarded to Python.
3. `live_features.py:134` — the service falls back to `int(time.time() // 3600)`,
   which is **UTC** epoch hours.
4. `live_features.py:375–376` — `hour_of_day = step % 24`, and `is_night` is true
   only for hours 0–5.

Because IST is UTC+5:30, `hour_of_day` crosses out of the night band at
11:30 AM local time:

| Viva time (IST) | `hour_of_day` | `is_night` | R6 outcome |
|---|---|---|---|
| 09:00 | 3 | yes | **OTP_CONFIRM** — modal appears |
| 11:00 | 5 | yes | **OTP_CONFIRM** — modal appears |
| 12:00 | 6 | no | APPROVE — no modal |
| 14:00 | 8 | no | APPROVE — no modal |
| 16:00 | 10 | no | APPROVE — no modal |

`docs/acceptance-tests.md` already documents the underlying sensitivity: the same
payment scores 0.287–0.295 at night with the two models disagreeing, and
0.211–0.237 by day with them agreeing — straddling the 0.35 approve threshold.

**The test suite is immune and the dashboard is not.** `acceptance_tests.py:60`
pins `step`, and its docstring states the lesson plainly: *"A test whose result
depends on the clock is not a test."* That lesson was never carried across to
`index.html`.

This matters more than a normal bug because the OTP modal — a live
"Did you make this payment? Yes / No" that visibly changes the outcome — is the
most persuasive interactive moment available. In an afternoon viva it simply
would not happen, with no error message to explain why.

## 4. Approach

Three options were weighed.

**A. Harden what exists — chosen.** Pin the clock, make the demo resettable,
write a runbook and pre-flight checklist, record a fallback video, rehearse the
defence. Adds no new subsystems; removes failure modes from a system that
already works.

**B. A, plus Docker Compose.** Deferred to optional stretch. It answers
"is it deployable?" concretely, but Ollama containerises badly, the models and
~4.5 GB working set are awkward, and the demo would still be given from
localhost — new risk for a benefit largely obtainable by explaining the
architecture well.

**C. A, plus a cloud deployment.** Rejected. Campus wifi becomes a single point
of failure, free tiers cannot hold the models in memory, cold starts stall the
narrative, and **Ollama cannot run there**, so the GenAI narration would
silently fall back to templates mid-demo. It is also architecturally wrong: no
bank exposes its fraud engine to the public internet, and an examiner may well
say so. Preparing a confident *answer* about cloud deployment is a stronger
position than owning a fragile one.

## 5. Design

### 5.1 Code changes (do these first — they are the safety net)

**Pin the clock in the dashboard scenarios.** Add a `step` field to every entry
in `SCENARIOS` (`index.html:795`) and forward it in the payload built at
`index.html:831`. Mirror `acceptance_tests.py` exactly, since those values are
already proven to produce the documented outcomes:

- `STEP_AFTERNOON = 254` (254 % 24 = 14, an ordinary 2pm) for R1, R2, R3, R5, R7.
- `STEP_NIGHT = 243` (243 % 24 = 3, which the model treats as riskier) for R6.
- R2's two payments and R7's five payments keep the **same** step as each other —
  that is what makes the travel physically impossible and the burst a burst.

The manual payment form should also send a step. Default it to `254` and expose
it as an optional field, so a panel member asking "what if this were 3am?" can be
answered by changing one number on screen.

**Add a demo reset.** The H2 database is file-backed
(`jdbc:h2:file:./data/frauddb`), so every rehearsal accumulates in the feed and
skews `/api/stats`. There is no backend reset endpoint and one should not be
added to production code for this. Instead a script performs: stop backend →
delete `backend/data/frauddb*` → start backend → `POST localhost:5001/reset` →
seed.

**Add a seed script.** Post ~15 ordinary payments so the decision feed, the
"where payments ended up" chart, and the totals are populated before the panel
sees the screen. An empty dashboard reads as an unfinished project.

**Add `demo.sh`.** One command that does the whole start-up: reset, launch both
services, poll `/health` on each until green rather than sleeping a fixed
duration, warm Ollama with a throwaway `/explain` call, seed, and open the
browser. This script *is* the deployment answer for section 8.

*Note, not a blocker:* `index.html:769` uses `alert()` on a rejected payment.
It is survivable in a human-driven demo, but a rejected payment during the viva
would freeze the page behind a modal until dismissed.

### 5.2 The arc — seven beats, about ten minutes

Ordered so that each beat answers the question the previous one raises, and so
that the two beats a competing project could not show come late, when attention
is highest.

1. **Control** (~30 s) — an ordinary payment approves in ~35 ms. Establishes that
   the system is not simply blocking everything. Without this, nothing after it
   means anything.
2. **Obvious fraud** (~60 s) — R1: the balance moved in one transfer to an empty
   account. Blocks. Open the row to show the plain-language reason and the SHAP
   contributions.
3. **What a model cannot learn** (~60 s) — R2: Mumbai then Kolkata within the
   hour. *"PaySim has no location column at all. A model cannot learn what is not
   in the data, so this is a hard rule, and the dashboard shows which mechanism
   decided."* This is the honesty beat and it sets up the whole Q&A.
4. **Uncertainty** (~90 s) — R6: large payment, trusted device, 3am. The OTP
   modal appears; answer **Yes**; it approves. Then state the two rules that are
   easy to get backwards: silence is a denial, and a trusted device proves
   identity, never intent.
5. **Kill the ML service live** (~60 s) — `Ctrl-C` the Python process in front of
   the panel, submit a payment, show it still decides and is flagged
   `degradedMode`. *"It does not fail open, which would leave the bank
   unprotected, and it does not fail closed, which would mean the bank has
   DDoSed itself."* Restart it; the health dot goes green again.
6. **Why hybrid** (~60 s) — the blinded holdout from `docs/results.md`: every
   TRANSFER fraud hidden from the Random Forest, which then caught 96.0% while
   the Isolation Forest caught 47.7% — together 99.9%, with **32 frauds rescued**
   that the supervised model missed entirely. State the cost honestly too: on
   known fraud the Isolation Forest contributes only false alarms.
7. **Deployment and integration** (~60 s) — sections 8 and 9 below.

**Bolt-on modules** if the panel lets it run long: R3 and R5, the GenAI
narration, the `/chat` analyst endpoint, the H2 console, the review-log audit
trail.

### 5.3 Pre-flight checklist (T-30 minutes)

- **Never run `features.py` on demo day.** ~50 s and a ~4.5 GB peak. Models must
  already be in `ml-service/models/`.
- Run `demo.sh`; confirm all three health dots are green.
- Warm Ollama with one throwaway `/explain` — the first call after idle is slow.
- Run one full silent rehearsal of the arc, then reset.
- Browser zoom set for the projector; the decision feed must be legible from the
  back of the room.
- Notifications off, laptop on mains, screen-sleep disabled.
- Backup video open in a background tab.

### 5.4 Failure fallbacks

| What fails | Recovery | What to say |
|---|---|---|
| Python service down | `demo.sh` restarts it; meanwhile beat 5 becomes the story | "Conveniently, this is exactly the failure I was about to demonstrate." |
| Ollama down / slow | Ignore — template fallback is automatic | "The narration layer is optional by design; the decision does not wait on it." |
| Backend won't start | Port 8080 in use — kill and retry | — |
| Browser wedged | Reload; state lives in H2, not the page | — |
| Total loss | Backup video | — |

The recorded video is a genuine deliverable, not a formality. Record it once the
arc is rehearsed, in one take, at full screen resolution.

## 6. Out of scope

Cloud deployment, Docker (unless time remains after rehearsal), retraining,
new model work, new dashboard features, and any change to the four anti-leakage
guards. **Nothing in this plan may weaken a decision path to make the demo look
better.**

## 7. Success criteria

- All six scenario buttons produce their documented outcome **regardless of the
  machine's wall clock**. Verify by clicking all six while the laptop clock reads
  afternoon IST — the exact condition that silently broke R6 — and confirming R6
  still raises the OTP challenge.
- `acceptance_tests.py` still passes 7/7 after the changes.
- `demo.sh` takes the machine from cold to demo-ready with one command and no
  manual waiting.
- The arc has been rehearsed end to end at least three times, once with the
  Python service killed at an unplanned moment.
- A backup video exists.

## 8. The deployment answer

> "It runs as two services over REST, and `demo.sh` brings both up from cold
> with health-gated startup. It is not on a public URL, and that is deliberate —
> a bank's fraud engine sits inside the bank's network, called by the core
> banking system at the authorisation step. Exposing it publicly would be a
> design error, not a milestone. What *is* required for deployability is that it
> installs and runs cleanly on a machine that has never seen it, which is what
> the script demonstrates."

If pressed on containers: the honest answer is that Docker Compose is
straightforward for the two application services, and the only awkward part is
Ollama, which is optional to the system by design.

## 9. The bank-integration answer

The architecture already supports this and the answer should be concrete, not
hand-waved:

- **Entry point:** the bank's core banking system (Finacle, FLEXCUBE, T24) calls
  `POST /api/transactions` at the authorisation step. The request DTO carries no
  PaySim-specific coupling.
- **Database:** swap H2 for the bank's own store; the repository layer is the
  only thing that changes.
- **OTP:** replace the simulated challenge with the bank's existing SMS or
  push-notification channel. The step-up logic does not change.
- **Location:** replace simulated cities with the device and IP geolocation the
  bank already holds. The impossible-travel rule already consumes lat/lon.
- **Tuning:** `config.yaml` holds every threshold, so a risk team retunes the
  approve/block boundaries without a redeploy.
- **Retraining is mandatory.** PaySim numbers do not transfer to real
  transactions and should never be quoted as if they did.

The strongest point to make last: **the sender-side behavioural features that are
provably inert on PaySim become live and valuable in a real bank.** 99.85% of
PaySim senders appear exactly once, so a per-sender profile is a constant; a real
customer has years of history. The features are already computed and deliberately
withheld from the trained model — because a feature that is constant in training
and variable in serving is worse than no feature at all. That is a limitation of
the dataset that the design already anticipates.

## 10. Q&A defence pack

The answers exist in the repository; what they need is rehearsal.

1. **"99.97% accuracy — really?"** No, and `results.md` says so first: predicting
   "never fraud" scores 95.9% on this data. Read precision, recall and PR-AUC.
   PaySim's balance columns follow mechanical rules, so every model scores far
   higher here than on real transactions. These numbers show the pipeline is
   correct; they are not a claim about the real world.
2. **"Where is the human reviewer?"** There isn't one, anywhere. `/api/stats`
   reports `humanReviewers: 0`. The review log is an audit record, not a queue.
3. **"Why two models when the Random Forest alone scores the same F1?"** Because
   the blinded holdout shows the Isolation Forest rescuing 32 frauds the
   supervised model missed entirely, and because model disagreement is itself
   routed to OTP rather than silently resolved.
4. **"Isn't 2.1% of customers getting OTP-challenged too many?"** It is a
   business decision, and the thresholds are in `config.yaml`. State the cost
   plainly rather than defending it.
5. **"Why is the location data fake?"** PaySim has no geo column. It is simulated
   for demonstration and **excluded from the trained model** — reporting accuracy
   on invented evidence would be dishonest. It runs as a hard rule instead.
6. **"What if the OTP is never answered?"** Blocked. The opposite would hand an
   attacker a free win: pay, stay silent, wait out the timer.
7. **"A trusted device — isn't that enough?"** No. It proves identity, not
   intent. Knowledge-base case FRAUD-006 is the victim authorising the payment
   themselves.
8. **"Is it real-time?"** Scoring is ~28 ms. Narration is ~3.7 s and is fetched
   lazily, because a payment cannot wait on prose nobody may read. With three
   explanations in flight, the median score time rises only to 111 ms.
9. **"What would you do differently?"** The honest answer, and a strong one:
   train on real labelled transactions, because every limitation in this project
   traces back to the dataset rather than the architecture.
