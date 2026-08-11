# Knowledge Base — Past Fraud Cases

These ten cases are the retrieval corpus for the RAG layer (Phase 4). When the
system explains a decision, it searches this folder for the most similar past
case and cites it, so the explanation is grounded in precedent rather than
invented.

**These cases are illustrative examples written for this project.** They are not
real customer incidents, and no real bank data was used anywhere in this system.
They are modelled on well-documented public fraud typologies.

Each `case-*.json` file has the same shape:

| field | meaning |
|---|---|
| `case_id` | stable identifier |
| `outcome` | `fraud` or `legitimate` — what it turned out to be |
| `title` | one-line label |
| `summary` | what happened, in plain English |
| `signals` | the warning signs, worded to match our SHAP reasons |
| `decision` | what the system did: APPROVE / OTP_CONFIRM / BLOCK |
| `resolution` | how it ended |
| `lesson` | the transferable rule |

To add a case, drop in another JSON file with the same fields and re-run
`genai_agent.py` — the vector store rebuilds automatically.

The `legitimate` cases matter as much as the fraud ones. Without them the
retriever would match every transaction to a fraud precedent and the
explanations would be biased toward alarm.

## Which of these the model can actually detect

Not all of them, and the difference is worth stating plainly in the report.

| Case | Detected by |
|---|---|
| FRAUD-001 account takeover, balance drained | **the model** — this is its strongest pattern |
| FRAUD-004 overnight staged withdrawals | **the model** (partly) + the rapid-sender rule |
| FRAUD-005 small test payment, then a large one | the rapid-sender rule |
| FRAUD-002 mule collecting rapid transfers | **a rule, not the model** |
| FRAUD-003 impossible travel | **a rule, not the model** |
| FRAUD-006 authorised push payment | *neither* — see below |

**FRAUD-002 and FRAUD-003 are rule-detected because the data cannot support
them.** PaySim has no location column at all, and it does not contain the
mule-burst pattern — measured across the full file, fraud receivers are *less*
busy than legitimate ones (0.036 vs 0.189 prior payments per hour). A model
cannot learn a pattern that is absent and points backwards, so `rules.py`
encodes both explicitly.

**FRAUD-006 is caught by nothing, by design.** The customer authorises the
payment themselves from their own device, so there is no technical signal to
find. It is in the knowledge base precisely as a reminder that OTP confirms
identity, never intent — the one case where the step-up path fails.
