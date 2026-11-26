"""All branching decisions live here. Pure functions: state in, next node name out.
No LLM calls, no network, no settings imports."""

from app.graph.state import GraphState


def kb_score_gate(chunks: list[dict], threshold: float) -> bool:
    """Cheap pre-check before the LLM grader. False means the KB is weak, with no LLM call."""
    if not chunks:
        return False
    return max(c.get("score", 0.0) for c in chunks) >= threshold


def _can_retry(state: GraphState) -> bool:
    return state.get("retry_count", 0) < state.get("max_retries", 0)


def _retry_or_stop(state: GraphState) -> str:
    return "rewrite_query" if _can_retry(state) else "insufficient_evidence"


def route_after_router(state: GraphState) -> str:
    # Only an explicit chitchat/out_of_scope skips retrieval. Anything else is treated as HR.
    if state.get("route") in ("chitchat", "out_of_scope"):
        return "direct_answer"
    return "retrieve_private"


def route_after_kb_grade(state: GraphState) -> str:
    if state.get("kb_grade") == "good":
        return "generate_from_kb"
    # KB is weak (or the grade is missing or unknown)
    if state.get("web_allowed", False):
        return "web_search"
    # Company-specific question with weak evidence: never answer it from the public web
    return _retry_or_stop(state)


def route_after_web_grade(state: GraphState) -> str:
    if state.get("web_grade") == "good":
        return "generate_from_web"
    return _retry_or_stop(state)