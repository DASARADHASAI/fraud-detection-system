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
