"""One function per graph node. Each takes the state and returns only the fields it changes."""

import re
from datetime import date

from app.config import get_settings
from app.graph import prompts
from app.graph.edges import kb_score_gate
from app.graph.state import GraphState
from app.services import embeddings, llm, vector_store
from app.services import web_search as tavily

# Employees may only retrieve chunks marked all_employees (the HR runbook is hr_only)
EMPLOYEE_FILTER = {"visibility": {"$eq": "all_employees"}}


def _t(node: str, **detail) -> dict:
    """One audit trace entry."""
    return {"node": node, **detail}

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thx|bye|goodbye|good (morning|afternoon|evening))\b[\s!.,?]*$",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_NUMBER = re.compile(r"\d{6,}")


def sanitize_for_web(text: str) -> str:
    """Only a cleaned question may leave the company. Mask emails and ID-like numbers."""
    text = _EMAIL.sub("[email]", text)
    text = _LONG_NUMBER.sub("[number]", text)
    return text.strip()[:300]

def clean_text(text: str) -> str:
    """Normalise characters the LLM sometimes emits (non-breaking hyphens and spaces)."""
    for bad, good in {"\u2011": "-", "\u2010": "-", "\u00a0": " ", "\u202f": " "}.items():
        text = text.replace(bad, good)
    return text


# ---------------------------------------------------------------- [1] route

def route_question(state: GraphState) -> dict:
    if _GREETING.match(state["question"]):
        return {
            "route": "chitchat",
            "web_allowed": False,
            "trace": [_t("route_question", route="chitchat", web_allowed=False, rule="greeting")],
        }
    s = get_settings()
    out = llm.generate_json(
        prompts.router_prompt(state["question"]), s.router_model, system=prompts.ROUTER_SYSTEM
    )
    route = out.get("route")
    if route not in ("chitchat", "hr", "out_of_scope"):
        route = "hr"  # router failed: fail toward retrieval, never toward a free-text answer
    web_allowed = route == "hr" and out.get("web_allowed") is True
    return {
        "route": route,
        "web_allowed": web_allowed,
        "trace": [_t("route_question", route=route, web_allowed=web_allowed, parsed=bool(out))],
    }


def direct_answer(state: GraphState) -> dict:
    if state.get("route") == "out_of_scope":
        answer = prompts.OUT_OF_SCOPE_REPLY
    else:
        s = get_settings()
        answer = llm.generate(state["question"], s.router_model, system=prompts.CHITCHAT_SYSTEM)
        answer = answer or "Hello! I can help with questions about company HR policies."
    return {
        "answer": answer,
        "source_type": "direct",
        "citations": [],
        "trace": [_t("direct_answer", route=state.get("route"))],
    }


# ---------------------------------------------------------------- [2] retrieve

def retrieve_private(state: GraphState) -> dict:
    s = get_settings()
    vec = embeddings.embed_query(state["query"])
    chunks = vector_store.query(vec, top_k=s.top_k, filter=EMPLOYEE_FILTER)
    return {
        "chunks": chunks,
        "trace": [
            _t(
                "retrieve_private",
                query=state["query"],
                hits=[(c["metadata"].get("heading_path"), round(c["score"], 3)) for c in chunks],
            )
        ],
    }


# ---------------------------------------------------------------- [3] grade

def grade_private(state: GraphState) -> dict:
    s = get_settings()
    chunks = state.get("chunks", [])

    # Cheap gate first: a low top score means weak, with no LLM call
    if not kb_score_gate(chunks, s.kb_score_threshold):
        top = round(max((c["score"] for c in chunks), default=0.0), 3)
        return {
            "kb_grade": "weak",
            "used_chunks": [],
            "trace": [_t("grade_private", grade="weak", reason="score_gate", top_score=top)],
        }

    out = llm.generate_json(
        prompts.grader_prompt(state["question"], chunks), s.grader_model, system=prompts.GRADER_SYSTEM
    )
    sufficient = out.get("sufficient") is True
    relevant = out.get("relevant", [])
    picked = []
    if isinstance(relevant, list):
        for x in relevant:
            if str(x).isdigit() and 1 <= int(x) <= len(chunks) and int(x) not in picked:
                picked.append(int(x))

    if sufficient and picked:
        used = [chunks[i - 1] for i in picked]
        return {
            "kb_grade": "good",
            "used_chunks": used,
            "trace": [_t("grade_private", grade="good", reason="llm", used=[c["id"] for c in used])],
        }

    reason = "llm_unparsed" if not out else "llm_insufficient"
    return {
        "kb_grade": "weak",
        "used_chunks": [],
        "trace": [_t("grade_private", grade="weak", reason=reason)],
    }


# ---------------------------------------------------------------- generate (KB)

def generate_from_kb(state: GraphState) -> dict:
    s = get_settings()
    used = state["used_chunks"]
    answer = clean_text(llm.generate(
        prompts.kb_answer_prompt(state["question"], used),
        s.generator_model,
        system=prompts.KB_ANSWER_SYSTEM,
    ))
    if not answer:
        return {
            "answer": prompts.INSUFFICIENT_REPLY,
            "source_type": "none",
            "citations": [],
            "trace": [_t("generate_from_kb", error="empty_answer")],
        }

    citations, seen = [], set()
    for c in used:
        m = c["metadata"]
        key = (m.get("doc_id"), m.get("heading_path"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "doc_id": m.get("doc_id"),
                "title": m.get("title"),
                "section": m.get("heading_path"),
                "version": m.get("version"),
                "score": round(c["score"], 3),
            }
        )
    return {
        "answer": answer,
        "source_type": "kb",  # set by the branch taken, never by the LLM
        "citations": citations,
        "trace": [_t("generate_from_kb", sources=[c["section"] for c in citations])],
    }


# ---------------------------------------------------------------- [6] rewrite

def rewrite_query(state: GraphState) -> dict:
    s = get_settings()
    raw = llm.generate(
        prompts.rewrite_prompt(state["question"], state["query"]),
        s.router_model,
        system=prompts.REWRITER_SYSTEM,
    )
    lines = raw.strip().strip('"').splitlines() if raw else []
    new_query = lines[0].strip().strip('"') if lines else ""
    if not new_query:
        new_query = state["query"]
    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": [_t("rewrite_query", old=state["query"], new=new_query)],
    }


# ---------------------------------------------------------------- fallback

def insufficient_evidence(state: GraphState) -> dict:
    return {
        "answer": prompts.INSUFFICIENT_REPLY,
        "source_type": "none",
        "citations": [],
        # The audit layer will turn this flag into a knowledge_gaps row
        "trace": [_t("insufficient_evidence", knowledge_gap=True, question=state["question"])],
    }

# ---------------------------------------------------------------- [4] web search

def web_search(state: GraphState) -> dict:
    # RULE: only the sanitized question/query goes out, never internal document text
    q = sanitize_for_web(state["query"])
    if not re.search(r"\b20\d{2}\b", q):
        q = f"{q} {date.today().year}"
    try:
        results = tavily.search(q, max_results=5)
    except Exception as e:  # Tavily down or out of credits: treat as no evidence
        return {
            "web_results": [],
            "trace": [_t("web_search", query=q, error=type(e).__name__)],
        }
    return {
        "web_results": results,
        "trace": [_t("web_search", query=q, urls=[r["url"] for r in results])],
    }


# ---------------------------------------------------------------- [5] grade web

def grade_web(state: GraphState) -> dict:
    s = get_settings()
    results = state.get("web_results", [])
    if not results:
        return {
            "web_grade": "weak",
            "used_web": [],
            "trace": [_t("grade_web", grade="weak", reason="no_results")],
        }

    out = llm.generate_json(
        prompts.web_grader_prompt(state["question"], results, date.today().isoformat()),
        s.grader_model,
        system=prompts.WEB_GRADER_SYSTEM,
    )
    picked = []
    relevant = out.get("relevant", [])
    if isinstance(relevant, list):
        for x in relevant:
            if str(x).isdigit() and 1 <= int(x) <= len(results) and int(x) not in picked:
                picked.append(int(x))

    if out.get("sufficient") is True and picked:
        used = [results[i - 1] for i in picked]
        return {
            "web_grade": "good",
            "used_web": used,
            "trace": [_t("grade_web", grade="good", used=[r["url"] for r in used])],
        }
    reason = "llm_unparsed" if not out else "llm_insufficient"
    return {
        "web_grade": "weak",
        "used_web": [],
        "trace": [_t("grade_web", grade="weak", reason=reason)],
    }


# ---------------------------------------------------------------- generate (web)

def generate_from_web(state: GraphState) -> dict:
    s = get_settings()
    used = state["used_web"]
    answer = clean_text(llm.generate(
        prompts.web_answer_prompt(state["question"], used),
        s.generator_model,
        system=prompts.WEB_ANSWER_SYSTEM,
    ))
    if not answer:
        return {
            "answer": prompts.INSUFFICIENT_REPLY,
            "source_type": "none",
            "citations": [],
            "trace": [_t("generate_from_web", error="empty_answer")],
        }
    return {
        "answer": f"{answer}\n\n{prompts.WEB_DISCLAIMER}",  # disclaimer is added in code
        "source_type": "web",                                # set by the branch, not the LLM
        "citations": [{"title": r["title"], "url": r["url"]} for r in used],
        "trace": [_t("generate_from_web", sources=[r["url"] for r in used])],
    }
    