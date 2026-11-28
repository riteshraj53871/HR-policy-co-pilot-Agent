from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.graph import edges, nodes
from app.graph.state import GraphState, initial_state


@lru_cache
def get_graph():
    g = StateGraph(GraphState)

    g.add_node("route_question", nodes.route_question)
    g.add_node("direct_answer", nodes.direct_answer)
    g.add_node("retrieve_private", nodes.retrieve_private)
    g.add_node("grade_private", nodes.grade_private)
    g.add_node("generate_from_kb", nodes.generate_from_kb)
    g.add_node("web_search", nodes.web_search)
    g.add_node("grade_web", nodes.grade_web)
    g.add_node("generate_from_web", nodes.generate_from_web)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("insufficient_evidence", nodes.insufficient_evidence)

    g.add_edge(START, "route_question")
    g.add_conditional_edges(
        "route_question",
        edges.route_after_router,
        {"direct_answer": "direct_answer", "retrieve_private": "retrieve_private"},
    )
    g.add_edge("retrieve_private", "grade_private")
    g.add_conditional_edges(
        "grade_private",
        edges.route_after_kb_grade,
        {
            "generate_from_kb": "generate_from_kb",
            "web_search": "web_search",
            "rewrite_query": "rewrite_query",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    g.add_edge("web_search", "grade_web")
    g.add_conditional_edges(
        "grade_web",
        edges.route_after_web_grade,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    g.add_edge("rewrite_query", "retrieve_private")  # the retry loop

    g.add_edge("direct_answer", END)
    g.add_edge("generate_from_kb", END)
    g.add_edge("generate_from_web", END)
    g.add_edge("insufficient_evidence", END)
    return g.compile()


def ask(question: str) -> GraphState:
    """Run one question through the graph and return the final state."""
    s = get_settings()
    state = initial_state(question, s.max_retries)
    # Each retry loop is up to ~6 node visits, so scale the limit with max_retries
    return get_graph().invoke(
        state, config={"recursion_limit": 10 + 6 * (s.max_retries + 1)}
    )