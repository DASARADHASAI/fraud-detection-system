"""
=============================================================================
PHASE 4 -- The GenAI layer (Ollama + LangChain + LangGraph + Chroma RAG)
=============================================================================

WHAT THIS ADDS:
Phase 3 already produces correct, honest explanations -- but they read like a
list of clauses stitched together. This layer hands those same facts to a local
language model and asks it to write them up the way a person would, backed by
similar past cases retrieved from knowledge_base/.

>>> THIS LAYER IS COMPLETELY OPTIONAL, BY DESIGN. <<<
Every path through this file has a fallback. If Ollama is not running, is slow,
returns nonsense, or invents a number, the system quietly uses the Phase 3
template sentence instead. The core product NEVER depends on it. You can delete
this file and everything else still works -- that is the test of whether the
optional layer is really optional.

WHY IT IS NOT ON THE /predict PATH:
Measured on this machine, llama3.2 runs at about 58 tokens/second, so a short
explanation takes 2-3 seconds. Fraud scoring has to answer in about a second.
So /predict returns the score and the SHAP reasons immediately, and the
narration is fetched separately from /explain when someone actually wants to
read it.

THE FOUR PIECES:
  1. Chroma vector store over knowledge_base/  -> find similar past cases (RAG)
  2. LangGraph agent                           -> retrieve, then write, then advise
  3. Ollama (llama3.2) via LangChain           -> the writing itself
  4. Template fallback from Phase 3            -> when any of the above is absent

RUN IT:
    ./venv/bin/python genai_agent.py
=============================================================================
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import numpy as np
import requests

from explain import build_summary_sentence
from features import load_config, resolve_path


# =============================================================================
# SECTION 1 -- Is Ollama actually there?
# =============================================================================

def ollama_is_available(cfg: dict) -> tuple[bool, str]:
    """
    Check quickly whether Ollama is running AND has the model we need.

    Deliberately uses a short timeout. This runs at startup, and a hung network
    call must never delay the fraud service booting.

    Returns (available, human_readable_reason).
    """
    g = cfg["genai"]
    if not g.get("enabled", True):
        return False, "disabled in config.yaml (genai.enabled = false)"

    url = g["base_url"].rstrip("/")
    wanted = g["model"]
    try:
        resp = requests.get(f"{url}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException as e:
        return False, f"Ollama not reachable at {url} ({type(e).__name__})"

    installed = [m["name"] for m in resp.json().get("models", [])]
    # Ollama reports "llama3.2:latest" when you asked for "llama3.2".
    if not any(name == wanted or name.startswith(f"{wanted}:") for name in installed):
        return False, (f"model '{wanted}' not pulled. Run:  ollama pull {wanted}\n"
                       f"    (installed: {', '.join(installed) or 'none'})")
    return True, f"Ollama ready with '{wanted}'"


# =============================================================================
# SECTION 2 -- Turning text into vectors, without extra downloads
# =============================================================================

class TfidfEmbedder:
    """
    Offline embeddings using scikit-learn's TF-IDF.

    WHY NOT A NEURAL EMBEDDING MODEL?
    llama3.2 cannot produce embeddings at all -- Ollama answers HTTP 501,
    because it is a generation-only model. The usual alternatives either
    download a model from the internet on first use (Chroma's bundled MiniLM) or
    require pulling another Ollama model. Both add a setup step and a network
    dependency to what is meant to be a fully offline system.

    For a corpus of ten short, keyword-dense case files, TF-IDF is genuinely
    adequate: fraud cases are matched on words like "emptied", "mule",
    "impossible travel", which is exactly what TF-IDF is good at. It is also
    deterministic, which makes the demo reproducible.

    Swap in OllamaEmbedder (below) by setting genai.use_ollama_embeddings: true
    after pulling a real embedding model.
    """

    name = "tf-idf (offline, scikit-learn)"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.fitted = False

    def fit_transform(self, documents: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.fit_transform(documents).toarray().astype(float)
        self.fitted = True
        return matrix.tolist()

    def transform(self, text: str) -> list[float]:
        if not self.fitted:
            raise RuntimeError("fit_transform must be called before transform")
        return self.vectorizer.transform([text]).toarray().astype(float)[0].tolist()


class OllamaEmbedder:
    """Neural embeddings, if the user has pulled a dedicated embedding model."""

    def __init__(self, cfg: dict):
        g = cfg["genai"]
        self.url = g["base_url"].rstrip("/")
        self.model = g.get("embedding_model", "nomic-embed-text")
        self.timeout = int(g.get("request_timeout_seconds", 30))
        self.name = f"ollama:{self.model}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def fit_transform(self, documents: list[str]) -> list[list[float]]:
        return self._embed(documents)

    def transform(self, text: str) -> list[float]:
        return self._embed([text])[0]


def choose_embedder(cfg: dict):
    """
    Pick an embedder, checking that the neural one really works before
    committing to it. Asking and then failing at query time would be worse than
    not asking at all.
    """
    if cfg["genai"].get("use_ollama_embeddings", False):
        try:
            embedder = OllamaEmbedder(cfg)
            embedder._embed(["health check"])       # prove it before relying on it
            return embedder
        except Exception as e:
            print(f"  Ollama embeddings unavailable ({type(e).__name__}), "
                  f"falling back to TF-IDF")
    return TfidfEmbedder()


# =============================================================================
# SECTION 3 -- RAG: retrieve similar past cases from knowledge_base/
# =============================================================================

class CaseKnowledgeBase:
    """
    Loads the case files, indexes them in Chroma, and finds the closest matches
    for a new transaction.

    The point of RAG here is grounding. Without it the language model would
    write plausible-sounding fraud commentary out of thin air. With it, the
    explanation can say "this resembles case FRAUD-002, where a mule account
    collected rapid transfers" -- a claim anyone can go and check.
    """

    COLLECTION = "fraud-case-history"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kb_dir = resolve_path(cfg, cfg["paths"]["knowledge_base_dir"])
        self.top_k = int(cfg["genai"].get("rag_top_k", 3))
        self.cases: list[dict] = []
        self.embedder = None
        self.collection = None
        self.ready = False

    # -------------------------------------------------------------------------
    def load(self) -> None:
        """Read every case-*.json and index it. Safe to call repeatedly."""
        import chromadb

        files = sorted(self.kb_dir.glob("case-*.json"))
        if not files:
            print(f"  No case files found in {self.kb_dir} -- RAG disabled.")
            return

        self.cases = []
        for path in files:
            try:
                self.cases.append(json.loads(path.read_text()))
            except json.JSONDecodeError as e:
                print(f"  Skipping malformed case file {path.name}: {e}")

        if not self.cases:
            return

        documents = [self._case_to_text(c) for c in self.cases]
        self.embedder = choose_embedder(self.cfg)
        vectors = self.embedder.fit_transform(documents)

        # An in-memory store is the right choice: the corpus is ten small files,
        # rebuilding takes milliseconds, and a stale index on disk that silently
        # disagrees with knowledge_base/ would be a nasty bug to chase.
        #
        # >>> WHY THE COLLECTION NAME IS UNIQUE PER INSTANCE <<<
        # chromadb caches clients by their settings, so EphemeralClient() hands
        # back the SAME underlying system every time. With a fixed name, a
        # second agent calling delete_collection() destroyed the first agent's
        # collection, and the first agent's queries then failed with
        # NotFoundError. A unique name per instance keeps them isolated.
        client = chromadb.EphemeralClient()
        self.collection_name = f"{self.COLLECTION}-{uuid4().hex[:8]}"
        self.collection = client.create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.collection.add(
            ids=[c["case_id"] for c in self.cases],
            documents=documents,
            embeddings=vectors,
            metadatas=[
                {"case_id": c["case_id"], "outcome": c["outcome"],
                 "title": c["title"], "decision": c["decision"],
                 "lesson": c["lesson"]}
                for c in self.cases
            ],
        )
        self.ready = True
        n_fraud = sum(1 for c in self.cases if c["outcome"] == "fraud")
        print(f"  Indexed {len(self.cases)} cases "
              f"({n_fraud} fraud, {len(self.cases) - n_fraud} legitimate) "
              f"using {self.embedder.name}")

    # -------------------------------------------------------------------------
    @staticmethod
    def _case_to_text(case: dict) -> str:
        """Flatten a case into the text that gets embedded."""
        return " ".join([
            case.get("title", ""),
            case.get("summary", ""),
            " ".join(case.get("signals", [])),
            case.get("lesson", ""),
        ])

    # -------------------------------------------------------------------------
    def find_similar(self, reasons: list[str], k: int | None = None) -> list[dict]:
        """
        Find past cases resembling the current transaction.

        We build the query from the SHAP reasons rather than the raw numbers,
        because the case files are written in the same vocabulary ("empties the
        sender's account", "receiver never used before"). Matching explanation
        to explanation is what makes the retrieval land.

        >>> CALLERS: for a payment the model CLEARED, pass the protective
        factors, not the risk factors. <<<
        Querying with risk factors always retrieves fraud precedents, because
        that is the vocabulary those cases are written in. We saw an ordinary
        approved payment cite "Account takeover" as its closest match, which
        would be alarming and wrong on a customer's screen. What a clean payment
        resembles is a clean past case.
        """
        if not self.ready or not reasons:
            return []

        query = " ".join(reasons)
        try:
            result = self.collection.query(
                query_embeddings=[self.embedder.transform(query)],
                n_results=min(k or self.top_k, len(self.cases)),
            )
        except Exception as e:
            print(f"  Retrieval failed ({type(e).__name__}) -- continuing without it")
            return []

        # Below this, a "match" is just the nearest of ten documents rather than
        # anything genuinely similar. Citing a weak match as precedent is worse
        # than citing nothing.
        min_similarity = float(self.cfg["genai"].get("rag_min_similarity", 0.15))

        matches = []
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            # Cosine distance 0 = identical. Report similarity, which reads
            # more naturally, and clamp so it never shows a negative.
            similarity = round(max(0.0, 1.0 - float(distance)), 3)
            if similarity < min_similarity:
                continue
            matches.append({
                "case_id": meta["case_id"],
                "title": meta["title"],
                "outcome": meta["outcome"],
                "decision": meta["decision"],
                "lesson": meta["lesson"],
                "similarity": similarity,
            })
        return matches


# =============================================================================
# SECTION 4 -- Guarding against invented facts
# =============================================================================

# Matches money figures the way our phrases write them: "Rs 3,244,067.13"
MONEY_PATTERN = re.compile(r"Rs\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)


def _normalise_amount(text: str) -> str:
    """'Rs 1,234.50' -> '1234.5' so we can compare figures fairly."""
    digits = re.sub(r"[^\d.]", "", text)
    try:
        return f"{float(digits):.2f}"
    except ValueError:
        return digits


# Words that assert a specific crime. The model has no evidence for any of
# these -- it sees a score and a list of feature descriptions. When it wrote
# "suggesting a possible case of money laundering" about a payment scoring 0.28,
# that was pure invention, and the kind of statement that causes real harm if it
# reaches a customer or an audit log.
SPECULATION_TERMS = [
    "money laundering", "laundering", "terrorist", "criminal record",
    "stolen card", "identity theft", "hacker", "scam artist",
]

# Language that contradicts the decision the measured model already made.
# An explanation saying "this payment should be blocked" under a decision of
# APPROVE is worse than no explanation at all.
CONTRADICTIONS = {
    "APPROVE": ["should be blocked", "was blocked", "we blocked", "high risk",
                "highly suspicious", "must be stopped", "recommend blocking",
                "should be declined", "flagged for review", "requires review",
                # Our own dest_was_empty phrase mentions mule accounts. Quoting
                # it about a payment we cleared would alarm a customer for no
                # reason, so it is banned here even though it is not invented.
                "mule", "suspicious", "potential fraud", "possible fraud"],
    "BLOCK": ["was approved", "we approved", "is legitimate", "appears legitimate",
              "low risk", "no cause for concern", "safe to proceed",
              "should be allowed",
              # llama3.2 wrote "due to its extremely LOW fraud score of 0.96".
              # A number is not enough; small models mislabel its direction.
              "low fraud score", "low score", "score is low"],
    "OTP_CONFIRM": ["was approved", "was blocked", "we blocked", "we approved",
                    "appears ordinary", "appears normal", "looks ordinary",
                    "was cleared", "has been cleared", "is cleared",
                    "is legitimate", "confirmed fraud", "is fraudulent"],
}


def find_contradictions(generated: str, decision: str) -> list[str]:
    """
    Catch narration that argues against the decision actually taken.

    The language model is told the decision but small models drift, especially
    when the score sits near a threshold. This is a cheap, deterministic check:
    if the prose and the routing disagree, throw the prose away.
    """
    lowered = generated.lower()
    return [p for p in CONTRADICTIONS.get(decision, []) if p in lowered]


def find_speculation(generated: str) -> list[str]:
    """Catch invented claims about what kind of crime this supposedly is."""
    lowered = generated.lower()
    return [t for t in SPECULATION_TERMS if t in lowered]


def find_invented_amounts(generated: str, source_facts: str) -> list[str]:
    """
    Return any money figure in the model's output that was NOT in the facts.

    WHY THIS MATTERS:
    A language model asked to write about a transaction will cheerfully produce
    "the customer transferred Rs 50,000" when the real figure was Rs 3,244,067.
    That number then appears on a customer's screen, or in a bank's audit trail,
    as though the system asserted it. A fluent lie is worse than a clumsy truth,
    so when this fires we discard the generated text entirely and fall back to
    the deterministic Phase 3 sentence.
    """
    allowed = {_normalise_amount(m) for m in MONEY_PATTERN.findall(source_facts)}
    invented = []
    for match in MONEY_PATTERN.findall(generated):
        if _normalise_amount(match) not in allowed:
            invented.append(match)
    return invented


# =============================================================================
# SECTION 5 -- The LangGraph agent
# =============================================================================

class AgentState(TypedDict, total=False):
    """The data that flows between nodes of the graph."""
    fraud_score: float
    risk_band: str      # "high risk" / "uncertain" / "low risk"
    decision: str
    reasons: list[str]
    protective: list[str]
    facts: str
    # MUST be declared here. LangGraph passes through ONLY the keys defined on
    # the state schema -- an undeclared key is silently dropped. This one held
    # the template fallback text, so omitting it made every fallback produce an
    # EMPTY explanation, and only on the rare path where a guard fired.
    facts_template: str
    similar_cases: list[dict]
    narrative: str
    recommended_action: str
    source: str            # "ollama" or "template"
    notes: list[str]       # anything worth telling the operator
    latency_ms: float


class FraudGenAIAgent:
    """
    The optional narration layer.

    Build it once at startup (Phase 5 does this). If Ollama is missing it still
    constructs successfully and simply operates in template mode -- callers do
    not need to check anything or handle a different type.
    """

    def __init__(self, cfg: dict | None = None, verbose: bool = True):
        self.cfg = cfg or load_config()
        self.g = self.cfg["genai"]
        self.verbose = verbose

        self.available, self.status = ollama_is_available(self.cfg)
        if verbose:
            print(f"  Ollama: {self.status}")

        self.llm = None
        if self.available:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=self.g["model"],
                    base_url=self.g["base_url"],
                    temperature=float(self.g.get("temperature", 0.2)),
                    num_predict=int(self.g.get("max_tokens", 260)),
                )
            except Exception as e:
                self.available = False
                self.status = f"LangChain could not reach Ollama: {e}"
                if verbose:
                    print(f"  {self.status}")

        self.knowledge_base = CaseKnowledgeBase(self.cfg)
        try:
            self.knowledge_base.load()
        except Exception as e:
            if verbose:
                print(f"  Knowledge base failed to load ({type(e).__name__}: {e})")

        self.graph = self._build_graph()

    # -------------------------------------------------------------------------
    def _build_graph(self):
        """
        Wire the three steps into a LangGraph state machine:

            START -> retrieve -> narrate -> advise -> END

        Why a graph rather than three function calls in a row? Because each node
        reads and writes one shared state object, so adding a step later (say, a
        node that checks the customer's recent complaints) means adding a node
        and an edge, not rewriting the flow. It also makes the pipeline
        inspectable, which matters when explaining the design in a report.
        """
        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("narrate", self._node_narrate)
        builder.add_node("advise", self._node_advise)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "narrate")
        builder.add_edge("narrate", "advise")
        builder.add_edge("advise", END)
        return builder.compile()

    # ------------------------------------------------------------------ node 1
    def _node_retrieve(self, state: AgentState) -> AgentState:
        """
        Pull the most similar past cases out of the knowledge base.

        Which factors we search with depends on the verdict. A cleared payment
        is searched for using what made it look SAFE, so it matches the
        legitimate cases; a suspicious one is searched using the warning signs.
        Searching a clean payment with its (weak) risk factors made it retrieve
        account-takeover precedents, which is both wrong and alarming.
        """
        risk_band = state.get("risk_band", "uncertain")
        if risk_band == "low risk":
            query_terms = state.get("protective", []) or state.get("reasons", [])
        else:
            query_terms = state.get("reasons", []) or state.get("protective", [])
        cases = self.knowledge_base.find_similar(query_terms)

        # Precedents must point the same way as the decision. Showing
        # "Account takeover: entire balance transferred" beside a payment we
        # just approved would alarm a customer over a 0.2-similarity match.
        if risk_band == "low risk":
            cases = [c for c in cases if c["outcome"] == "legitimate"]
        return {"similar_cases": cases}

    # ------------------------------------------------------------------ node 2
    def _node_narrate(self, state: AgentState) -> AgentState:
        """
        Write the plain-language explanation.

        Falls back to the Phase 3 template on: no Ollama, an exception, an empty
        answer, or an invented money figure.
        """
        notes = list(state.get("notes", []))
        template = state["facts_template"] if "facts_template" in state else ""

        if not self.available or self.llm is None:
            notes.append(f"used template ({self.status})")
            return {"narrative": template, "source": "template", "notes": notes}

        # Routine approvals are ~94% of traffic. Spending 3 seconds of model time
        # on each one buys nothing: nobody reads the explanation for a payment
        # that simply went through, and the small model's weakest output is
        # exactly here -- it kept writing "cleared BECAUSE [warning sign]".
        # The Phase 3 template is instant and correct. Configurable via
        # genai.narrate_decisions.
        narrate_for = self.g.get("narrate_decisions",
                                 ["BLOCK", "OTP_CONFIRM"])
        if state.get("decision") not in narrate_for:
            notes.append(f"used template ({state.get('decision')} is not in "
                         f"genai.narrate_decisions)")
            return {"narrative": template, "source": "template", "notes": notes}

        prompt = self._build_prompt(state)
        try:
            response = self.llm.invoke(prompt)
            text = (response.content or "").strip()
        except Exception as e:
            notes.append(f"used template (Ollama call failed: {type(e).__name__})")
            return {"narrative": template, "source": "template", "notes": notes}

        # --- Validate before trusting it. Any failure falls back silently. ---
        if len(text) < 30:
            notes.append("used template (model returned an empty or trivial answer)")
            return {"narrative": template, "source": "template", "notes": notes}

        if self.g.get("reject_hallucinated_amounts", True):
            invented = find_invented_amounts(text, state["facts"])
            if invented:
                notes.append(
                    f"used template (model invented amounts not in the data: "
                    f"{', '.join(invented[:3])})"
                )
                return {"narrative": template, "source": "template", "notes": notes}

        speculation = find_speculation(text)
        if speculation:
            notes.append(
                f"used template (model speculated about crimes it has no evidence "
                f"for: {', '.join(speculation[:3])})"
            )
            return {"narrative": template, "source": "template", "notes": notes}

        contradictions = find_contradictions(text, state.get("decision", ""))
        if contradictions:
            notes.append(
                f"used template (narration contradicted the "
                f"{state.get('decision')} decision: {', '.join(contradictions[:3])})"
            )
            return {"narrative": template, "source": "template", "notes": notes}

        return {"narrative": text, "source": "ollama", "notes": notes}

    # ------------------------------------------------------------------ node 3
    def _node_advise(self, state: AgentState) -> AgentState:
        """
        State the action.

        >>> THE LANGUAGE MODEL DOES NOT GET A VOTE HERE. <<<
        The action is derived deterministically from the decision the hybrid
        model already made. If the LLM were allowed to recommend, a fluent
        hallucination could contradict the actual routing -- the screen would
        say "we have approved this payment" while the backend blocked it. The
        model writes prose; the model that was measured makes decisions.
        """
        actions = {
            "BLOCK": ("Block the payment and notify the customer. "
                      "No further action is required from the customer unless "
                      "they wish to dispute it."),
            "OTP_CONFIRM": ("Hold the payment and ask the customer to confirm it "
                            "by one-time passcode. Approve on confirmation, "
                            "block on denial or timeout."),
            "APPROVE": ("Allow the payment to proceed. No customer contact needed."),
        }
        decision = state.get("decision", "OTP_CONFIRM")
        return {"recommended_action": actions.get(decision, actions["OTP_CONFIRM"])}

    # -------------------------------------------------------------------------
    def _build_prompt(self, state: AgentState) -> str:
        """
        Build the instruction. The constraints matter more than the politeness:
        we are asking a small model to rephrase given facts, not to reason about
        fraud, because it has no access to the data and no reason to be right.
        """
        cases_text = "\n".join(
            f"  - {c['case_id']} ({c['outcome']}): {c['title']}. Lesson: {c['lesson']}"
            for c in state.get("similar_cases", [])
        ) or "  (no similar past cases found)"

        risk_band = state.get("risk_band", "uncertain")

        # Telling the model the verdict IN WORDS, and instructing it to match
        # that tone, is what stops it writing an alarmed paragraph about a
        # payment that scored 0.28 and was cleared. Small models anchor on the
        # warning signs in the list and ignore the number unless you spell out
        # the conclusion for them.
        tone = {
            "high risk": "This payment IS considered high risk. Explain why it "
                         "looked wrong.",
            "uncertain": "This payment is BORDERLINE. It is NOT confirmed fraud. "
                         "State plainly which specific details looked unusual, "
                         "then say that these are not conclusive on their own. "
                         "Never write that the payment 'appears ordinary' or "
                         "'appears normal' -- it did not.",
            "low risk": "This payment is considered NORMAL and was cleared. "
                        "Explain what made it look ordinary. Do NOT describe it "
                        "as risky or suspicious.",
        }[risk_band]

        return f"""You are writing a short fraud-review note for a bank customer.

THE VERDICT HAS ALREADY BEEN DECIDED: {risk_band}.
{tone}

RULES -- these matter more than style:
- Use ONLY the facts listed below. Do not add details, amounts, dates, names or
  locations that do not appear here.
- Never invent a number. If a figure is not listed, do not state one.
- Do NOT name a type of crime. You have no evidence of money laundering, theft,
  hacking or any other specific offence. Describe only what the data shows.
- Do NOT recommend an action and do NOT state whether the payment was approved
  or blocked. That is written separately.
- Your tone must match the verdict above. Do not argue for a different outcome.
- Write 2 to 3 sentences, plain English, no jargon, no bullet points.
- Write about "this payment", not "you".

FACTS ABOUT THIS PAYMENT
{state['facts']}

SIMILAR PAST CASES (background only -- do NOT claim this payment IS one of them)
{cases_text}

Write the explanation now."""

    # -------------------------------------------------------------------------
    def explain_transaction(
        self,
        explanation: dict,
        fraud_score: float,
        decision: str,
        models_disagree: bool = False,
    ) -> dict:
        """
        THE MAIN ENTRY POINT. Phase 5's /explain endpoint calls this.

        `explanation` is whatever FraudExplainer.explain() returned in Phase 3.

        Always returns a dict, never raises. Check the "source" field to see
        whether you got a language-model narration or the template.
        """
        started = time.perf_counter()

        reasons = [r["reason"] for r in explanation.get("reasons", [])]
        protective = [p["reason"] for p in explanation.get("protective_factors", [])]

        # Derive the band from the DECISION, never from the score alone.
        # models_disagree can escalate a low-scoring payment to OTP_CONFIRM; if
        # the band came from the score we would tell the model "this was cleared
        # and is normal" about a payment the system actually held. It then wrote
        # exactly that.
        risk_band = {"BLOCK": "high risk",
                     "OTP_CONFIRM": "uncertain",
                     "APPROVE": "low risk"}.get(decision, "uncertain")

        facts = self._format_facts(fraud_score, decision, reasons,
                                   protective, models_disagree, risk_band)
        template_sentence = build_summary_sentence(explanation, fraud_score, decision)

        state: dict[str, Any] = {
            "fraud_score": fraud_score,
            "risk_band": risk_band,
            "decision": decision,
            "reasons": reasons,
            "protective": protective,
            "facts": facts,
            "facts_template": template_sentence,
            "notes": [],
        }

        try:
            result = self.graph.invoke(state)
        except Exception as e:
            # The graph itself failing is not a reason to fail the request.
            return {
                "explanation": template_sentence,
                "recommended_action": self._node_advise({"decision": decision})[
                    "recommended_action"],
                "similar_cases": [],
                "source": "template",
                "notes": [f"agent graph failed: {type(e).__name__}: {e}"],
                "latency_ms": round(1000 * (time.perf_counter() - started), 1),
            }

        # `or` rather than a .get default: a key that EXISTS but is empty would
        # sail past .get(). The explanation is the whole product here; it must
        # never come back blank.
        return {
            "explanation": result.get("narrative") or template_sentence,
            "recommended_action": result.get("recommended_action", ""),
            "similar_cases": result.get("similar_cases", []),
            "source": result.get("source", "template"),
            "notes": result.get("notes", []),
            "latency_ms": round(1000 * (time.perf_counter() - started), 1),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _format_facts(fraud_score, decision, reasons, protective, disagree,
                      risk_band: str = "uncertain") -> str:
        """
        Lay out the facts in the order that matches the verdict.

        >>> ORDER AND LABELLING CHANGE THE OUTPUT MORE THAN THE PROMPT DOES. <<<
        Originally every fact was listed as a flat "Warning sign" / "Reassuring
        sign" list regardless of the outcome. Given a cleared payment, the model
        read the warning signs at the top and wrote "there may be a mule account
        involved" about a transaction scoring 0.09. The warnings were real -- they
        were simply outweighed -- but nothing in the input said so.
        """
        lines = [f"- Fraud score: {fraud_score:.2f} out of 1.00 "
                 f"(this counts as {risk_band})"]
        if disagree:
            lines.append("- The two detection models disagreed about this payment")

        if risk_band == "low risk":
            # Lead with why it was cleared; demote the warnings and say plainly
            # that they lost the argument.
            for p in protective:
                lines.append(f"- Reason it was cleared: {p}")
            for r in reasons[:2]:
                lines.append(f"- Minor point, NOT enough to raise concern "
                             f"and already outweighed: {r}")
        else:
            for r in reasons:
                lines.append(f"- Warning sign: {r}")
            for p in protective:
                lines.append(f"- Point in the payment's favour: {p}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def chat(self, question: str, context: dict | None = None) -> dict:
        """
        Answer an analyst's follow-up question, grounded in the knowledge base.

        Note this is an ANALYST tool, not part of the automatic decision path.
        Nothing it says changes any routing -- it exists so a human reviewing
        the audit log afterwards can interrogate the system in plain English.
        """
        started = time.perf_counter()
        cases = self.knowledge_base.find_similar([question], k=3)

        if not self.available or self.llm is None:
            # Still useful without a language model: hand back the retrieved
            # cases so the analyst has something concrete to read.
            if cases:
                body = "\n".join(
                    f"  - {c['case_id']} ({c['outcome']}): {c['title']}\n"
                    f"    Lesson: {c['lesson']}" for c in cases
                )
                answer = ("The AI assistant is offline, so here are the most "
                          f"relevant past cases instead:\n{body}")
            else:
                answer = ("The AI assistant is offline and no relevant past "
                          "cases were found. Start Ollama with 'ollama serve' "
                          "to enable full answers.")
            return {"answer": answer, "similar_cases": cases, "source": "template",
                    "notes": [self.status],
                    "latency_ms": round(1000 * (time.perf_counter() - started), 1)}

        cases_text = "\n".join(
            f"  - {c['case_id']} ({c['outcome']}): {c['title']}. {c['lesson']}"
            for c in cases
        ) or "  (no relevant past cases)"

        context_text = ""
        if context:
            context_text = ("\nTHE PAYMENT CURRENTLY BEING REVIEWED\n"
                            + self._format_facts(
                                context.get("fraud_score", 0.0),
                                context.get("decision", "UNKNOWN"),
                                context.get("reasons", []),
                                context.get("protective", []),
                                context.get("models_disagree", False)))

        prompt = f"""You are assisting a bank fraud analyst.

RULES:
- Answer only from the information below. If it does not contain the answer,
  say so plainly instead of guessing.
- Never invent amounts, names, dates or case numbers.
- Be concise: at most 4 sentences.

PAST CASES FROM THE KNOWLEDGE BASE
{cases_text}
{context_text}

ANALYST'S QUESTION: {question}

Answer:"""

        try:
            text = (self.llm.invoke(prompt).content or "").strip()
        except Exception as e:
            return {"answer": f"The AI assistant could not be reached "
                              f"({type(e).__name__}). The retrieved past cases "
                              f"are still shown.",
                    "similar_cases": cases, "source": "template",
                    "notes": [str(e)],
                    "latency_ms": round(1000 * (time.perf_counter() - started), 1)}

        return {"answer": text or "No answer produced.", "similar_cases": cases,
                "source": "ollama", "notes": [],
                "latency_ms": round(1000 * (time.perf_counter() - started), 1)}


# =============================================================================
# SECTION 6 -- Demo
# =============================================================================

def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    import pandas as pd
    from explain import load_explainer
    from features import build_feature_matrix
    from train_model import route_decisions, score_transactions

    cfg = load_config()

    _print_header("PHASE 4 -- GENAI EXPLANATION LAYER")
    agent = FraudGenAIAgent(cfg)

    if not agent.available:
        print("\nRunning in TEMPLATE MODE. This is a supported state, not an error.")
        print("To enable the language model:  ollama serve  &&  ollama pull llama3.2")

    # ---- Load a few real transactions to explain -----------------------------
    explainer, models = load_explainer(cfg)
    df = pd.read_csv(resolve_path(cfg, cfg["paths"]["processed_data"]))
    X, y, _ = build_feature_matrix(df, cfg)

    scores = score_transactions(models["rf"], models["iso"], models["scaler"],
                                X, cfg, models["calibration"])
    decisions = route_decisions(scores, cfg)
    fraud_scores = scores["fraud_score"]
    y_arr = y.to_numpy()

    picks = []
    for label, mask in [
        ("a fraud that was blocked", (y_arr == 1) & (decisions == "BLOCK")),
        ("an uncertain payment sent to OTP", decisions == "OTP_CONFIRM"),
        ("an ordinary payment approved", (y_arr == 0) & (decisions == "APPROVE")),
    ]:
        idx = np.where(mask)[0]
        if len(idx):
            picks.append((int(idx[0]), label))

    _print_header("EXPLAINING REAL DECISIONS")
    for idx, label in picks:
        row = X.iloc[[idx]]
        explanation = explainer.explain(row)
        result = agent.explain_transaction(
            explanation=explanation,
            fraud_score=float(fraud_scores[idx]),
            decision=str(decisions[idx]),
            models_disagree=bool(scores["models_disagree"][idx]),
        )

        print(f"\n--- {label} ---")
        print(f"score {fraud_scores[idx]:.3f} | decision {decisions[idx]} | "
              f"source: {result['source']} | {result['latency_ms']:.0f} ms")
        print(f"\n  EXPLANATION:\n    {result['explanation']}")
        print(f"\n  RECOMMENDED ACTION:\n    {result['recommended_action']}")
        if result["similar_cases"]:
            print(f"\n  SIMILAR PAST CASES (retrieved from knowledge_base/):")
            for c in result["similar_cases"]:
                print(f"    - {c['case_id']} ({c['outcome']}, "
                      f"similarity {c['similarity']:.2f}): {c['title']}")
        if result["notes"]:
            print(f"\n  NOTES: {'; '.join(result['notes'])}")

    # ---- Analyst chat --------------------------------------------------------
    _print_header("ANALYST CHAT")
    for question in [
        "Why do we treat a brand-new receiver as risky?",
        "Does confirming by OTP guarantee the payment is safe?",
    ]:
        answer = agent.chat(question)
        print(f"\nQ: {question}")
        print(f"A ({answer['source']}, {answer['latency_ms']:.0f} ms): {answer['answer']}")
        if answer["similar_cases"]:
            print(f"   grounded in: "
                  f"{', '.join(c['case_id'] for c in answer['similar_cases'])}")

    _print_header("PHASE 4 COMPLETE")
    print("The GenAI layer is optional everywhere:")
    print("  - Ollama missing/slow/broken -> Phase 3 template sentence")
    print("  - Model invents an amount    -> answer discarded, template used")
    print("  - Knowledge base empty       -> explanation still works, no citations")
    print("  - The LLM never decides      -> routing comes from the measured model")
    print("\nNext: Phase 5 -- the Flask API (/predict fast, /explain lazy, /chat).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
