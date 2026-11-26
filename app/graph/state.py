import operator
from typing import Annotated, Literal, TypedDict

Route = Literal["chitchat", "hr", "out_of_scope"]
Grade = Literal["good", "weak"]
SourceType = Literal["kb", "web", "none", "direct"]


class GraphState(TypedDict, total=False):
    # Input
    question: str                 # what the employee asked (never changes)
    query: str                    # current search query (changes after a rewrite)

    # Router decision
    route: Route
    web_allowed: bool             # True only for public-information questions

    # Private KB branch
    chunks: list[dict]            # [{"id", "score", "text", "metadata"}], best first
    kb_grade: Grade
    used_chunks: list[dict]

    # Web branch
    web_results: list[dict]       # [{"title", "url", "content", "score"}]
    web_grade: Grade

    # Retry control
    retry_count: int              # rewrites done so far
    max_retries: int              # copied from settings at the start of the request

    # Output
    answer: str
    source_type: SourceType       # set by the branch taken, never by the LLM
    citations: list[dict]

    # Audit: nodes append to this, LangGraph merges the lists
    trace: Annotated[list[dict], operator.add]


def initial_state(question: str, max_retries: int) -> GraphState:
    return {
        "question": question,
        "query": question,
        "chunks": [],
        "used_chunks":[],
        "web_results": [],
        "citations": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "trace": [],
    }