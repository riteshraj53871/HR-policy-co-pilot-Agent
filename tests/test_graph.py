import pytest

from app.config import get_settings
from app.graph import builder, nodes, prompts

EMPLOYEE_FILTER = {"visibility": {"$eq": "all_employees"}}

WEB_RESULTS = [
    {
        "title": "Public holidays in Bangladesh 2026",
        "url": "https://example.gov.bd/holidays",
        "content": "List of 2026 public holidays.",
        "score": 0.9,
    }
]


def kb_chunk(score: float) -> dict:
    return {
        "id": "company_policy-annual-leave-0",
        "score": score,
        "text": "Full-time employees receive 20 days of paid annual leave per calendar year.",
        "metadata": {
            "doc_id": "company_policy",
            "title": "Handbook",
            "heading_path": "Annual Leave",
            "version": "n/a",
        },
    }


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    for key in ("GROQ_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.setenv(key, "test")
    monkeypatch.setenv("MAX_RETRIES", "2")
    monkeypatch.setenv("KB_SCORE_THRESHOLD", "0.55")
    monkeypatch.setenv("TOP_K", "3")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class Spy:
    def __init__(self):
        self.kb_filters = []
        self.web_queries = []
        self.systems = []


def install(monkeypatch, *, route="hr", web_allowed=False, kb_score=0.84,
            kb_sufficient=True, web_sufficient=True) -> Spy:
    """Replace every external service with a fake and record how it was called."""
    spy = Spy()
    monkeypatch.setattr(nodes.embeddings, "embed_query", lambda q: [0.0] * 384)

    def fake_query(vector, top_k=None, filter=None):
        spy.kb_filters.append(filter)
        return [kb_chunk(kb_score)]

    def fake_json(prompt, model, system=None):
        spy.systems.append(system)
        if system == prompts.ROUTER_SYSTEM:
            return {"route": route, "web_allowed": web_allowed}
        if system == prompts.GRADER_SYSTEM:
            return {"sufficient": kb_sufficient, "relevant": [1] if kb_sufficient else []}
        if system == prompts.WEB_GRADER_SYSTEM:
            return {"sufficient": web_sufficient, "relevant": [1] if web_sufficient else []}
        return {}

    def fake_generate(prompt, model, system=None, temperature=0.0):
        spy.systems.append(system)
        return "rewritten query" if system == prompts.REWRITER_SYSTEM else "generated answer"

    def fake_search(query, max_results=5):
        spy.web_queries.append(query)
        return WEB_RESULTS

    monkeypatch.setattr(nodes.vector_store, "query", fake_query)
    monkeypatch.setattr(nodes.llm, "generate_json", fake_json)
    monkeypatch.setattr(nodes.llm, "generate", fake_generate)
    monkeypatch.setattr(nodes.tavily, "search", fake_search)
    return spy


# ---------------------------------------------------------------- KB path

def test_kb_answer_never_touches_web(monkeypatch):
    spy = install(monkeypatch)
    state = builder.ask("How many annual leave days do employees receive?")
    assert state["source_type"] == "kb"
    assert state["citations"][0]["section"] == "Annual Leave"
    assert spy.web_queries == []


def test_employee_filter_is_always_applied(monkeypatch):
    spy = install(monkeypatch)
    builder.ask("How many sick days do I get?")
    assert spy.kb_filters == [EMPLOYEE_FILTER]


def test_high_score_but_insufficient_chunk_is_weak(monkeypatch):
    """The 'first day of work' case: a good score, but the chunk does not answer it."""
    spy = install(monkeypatch, kb_score=0.65, kb_sufficient=False)
    state = builder.ask("What is the process for a new employee's first day?")
    assert state["source_type"] == "none"
    assert spy.web_queries == []          # company-specific: never the web
    assert state["retry_count"] == 2


# ---------------------------------------------------------------- web path

def test_public_question_weak_kb_uses_web_with_disclaimer(monkeypatch):
    spy = install(monkeypatch, web_allowed=True, kb_score=0.40)
    state = builder.ask("What are the latest public holiday rules in Bangladesh?")
    assert state["source_type"] == "web"
    assert prompts.WEB_DISCLAIMER in state["answer"]
    assert state["citations"][0]["url"] == WEB_RESULTS[0]["url"]
    assert len(spy.web_queries) == 1
    assert state["retry_count"] == 0


def test_company_specific_weak_kb_never_uses_web(monkeypatch):
    spy = install(monkeypatch, web_allowed=False, kb_score=0.40)
    state = builder.ask("What is our offboarding checklist?")
    assert state["source_type"] == "none"
    assert spy.web_queries == []


def test_weak_web_retries_then_gives_up(monkeypatch):
    spy = install(monkeypatch, web_allowed=True, kb_score=0.40, web_sufficient=False)
    state = builder.ask("What are the latest public holiday rules in Bangladesh?")
    assert state["source_type"] == "none"
    assert state["retry_count"] == 2
    assert len(spy.web_queries) == 3      # first attempt + one per retry, then it stops


def test_web_query_is_sanitized(monkeypatch):
    spy = install(monkeypatch, web_allowed=True, kb_score=0.40)
    builder.ask("I am ali@company.com, ID 12345678. What are the public holidays in Bangladesh?")
    assert "ali@company.com" not in spy.web_queries[0]
    assert "12345678" not in spy.web_queries[0]


# ---------------------------------------------------------------- fallback and routing

def test_insufficient_evidence_flags_a_knowledge_gap(monkeypatch):
    install(monkeypatch, kb_score=0.40)
    state = builder.ask("What is the dress code for client visits?")
    last = state["trace"][-1]
    assert last["node"] == "insufficient_evidence" and last["knowledge_gap"] is True
    assert state["answer"] == prompts.INSUFFICIENT_REPLY


def test_chitchat_skips_retrieval(monkeypatch):
    spy = install(monkeypatch, route="chitchat")
    state = builder.ask("how are you doing today")
    assert state["source_type"] == "direct"
    assert spy.kb_filters == [] and spy.web_queries == []


def test_out_of_scope_uses_the_fixed_message(monkeypatch):
    install(monkeypatch, route="out_of_scope")
    state = builder.ask("How many leave days do I have left?")
    assert state["answer"] == prompts.OUT_OF_SCOPE_REPLY


@pytest.mark.parametrize("text", ["Hello!", "hi", "Thanks!", "good morning"])
def test_greeting_never_reaches_the_router_llm(monkeypatch, text):
    spy = install(monkeypatch, route="out_of_scope")   # a bad router would reject this
    state = builder.ask(text)
    assert state["route"] == "chitchat"
    assert prompts.ROUTER_SYSTEM not in spy.systems