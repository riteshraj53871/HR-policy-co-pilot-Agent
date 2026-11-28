import pytest

from app.graph.edges import (
    kb_score_gate,
    route_after_router,
    route_after_kb_grade,
    route_after_web_grade,
)
from app.graph.state import initial_state


# ---------- score gate ----------

@pytest.mark.parametrize(
    "scores, expected",
    [
        ([0.84, 0.78, 0.69], True),    # clear match
        ([0.55], True),                # exactly on the threshold passes
        ([0.54, 0.50], False),         # just under
        ([0.53, 0.51, 0.50], False),   # the Bangladesh holidays case
        ([], False),                   # nothing retrieved
    ],
)
def test_kb_score_gate(scores, expected):
    chunks = [{"score": s} for s in scores]
    assert kb_score_gate(chunks, threshold=0.55) is expected


# ---------- router ----------

@pytest.mark.parametrize(
    "route, expected",
    [
        ("chitchat", "direct_answer"),
        ("out_of_scope", "direct_answer"),
        ("hr", "retrieve_private"),
        (None, "retrieve_private"),         # router failed: fail toward retrieval
        ("something_odd", "retrieve_private"),
    ],
)
def test_route_after_router(route, expected):
    state = {} if route is None else {"route": route}
    assert route_after_router(state) == expected


# ---------- after KB grade ----------

def test_good_kb_never_uses_web():
    state = {"kb_grade": "good", "web_allowed": True}
    assert route_after_kb_grade(state) == "generate_from_kb"


def test_weak_kb_public_question_goes_to_web():
    state = {"kb_grade": "weak", "web_allowed": True, "retry_count": 0, "max_retries": 2}
    assert route_after_kb_grade(state) == "web_search"


def test_weak_kb_company_question_never_goes_to_web():
    state = {"kb_grade": "weak", "web_allowed": False, "retry_count": 0, "max_retries": 2}
    assert route_after_kb_grade(state) == "rewrite_query"


def test_weak_kb_company_question_stops_when_retries_exhausted():
    state = {"kb_grade": "weak", "web_allowed": False, "retry_count": 2, "max_retries": 2}
    assert route_after_kb_grade(state) == "insufficient_evidence"


@pytest.mark.parametrize("grade", [None, "", "maybe", "GOOD"])
def test_unknown_kb_grade_is_treated_as_weak(grade):
    state = {"kb_grade": grade, "web_allowed": False, "retry_count": 2, "max_retries": 2}
    assert route_after_kb_grade(state) == "insufficient_evidence"


# ---------- after web grade ----------

def test_good_web_generates_from_web():
    assert route_after_web_grade({"web_grade": "good"}) == "generate_from_web"


def test_weak_web_retries_while_allowed():
    state = {"web_grade": "weak", "retry_count": 1, "max_retries": 2}
    assert route_after_web_grade(state) == "rewrite_query"


def test_weak_web_stops_when_retries_exhausted():
    state = {"web_grade": "weak", "retry_count": 2, "max_retries": 2}
    assert route_after_web_grade(state) == "insufficient_evidence"


def test_missing_max_retries_means_no_retry():
    assert route_after_web_grade({"web_grade": "weak"}) == "insufficient_evidence"


# ---------- the loop must always end ----------

@pytest.mark.parametrize("max_retries", [0, 1, 2, 5])
def test_loop_always_terminates(max_retries):
    state = initial_state("question", max_retries=max_retries)
    state.update(kb_grade="weak", web_grade="weak", web_allowed=True)

    node, steps = route_after_kb_grade(state), 0
    while node != "insufficient_evidence":
        if node == "web_search":
            node = route_after_web_grade(state)
        elif node == "rewrite_query":
            state["retry_count"] += 1          # what the rewrite node will do
            node = route_after_kb_grade(state)
        else:
            pytest.fail(f"unexpected node {node}")
        steps += 1
        assert steps < 50, "loop did not terminate"

    assert state["retry_count"] == max_retries