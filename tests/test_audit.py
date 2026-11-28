import sqlite3

import pytest

from app import audit
from app.config import get_settings

STATE = {
    "route": "hr",
    "source_type": "none",
    "retry_count": 2,
    "answer": "Please email ali@company.com",
    "trace": [
        {"node": "route_question", "route": "hr"},
        {"node": "insufficient_evidence", "knowledge_gap": True,
         "question": "Mail ali@company.com about the dress code"},
    ],
}


@pytest.fixture(autouse=True)
def db(monkeypatch, tmp_path):
    for key in ("GROQ_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.setenv(key, "test")
    path = tmp_path / "audit.db"
    monkeypatch.setenv("AUDIT_DB_PATH", str(path))
    get_settings.cache_clear()
    audit.init_db()
    yield path
    get_settings.cache_clear()


def rows(db, sql):
    con = sqlite3.connect(db)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def log(request_id="r1", question="My ID is 12345678, what is the dress code?"):
    audit.log_request(request_id=request_id, question=question, session_id="s1",
                      state=STATE, latency_ms=120, status="ok")


def test_mask_pii():
    assert audit.mask_pii("ali@company.com 12345678") == "[email] [number]"
    assert audit.mask_pii("20 days of leave") == "20 days of leave"   # short numbers stay


def test_log_request_masks_everything_it_stores(db):
    log()
    dump = str(rows(db, "SELECT * FROM requests")) + str(rows(db, "SELECT * FROM node_events")) \
        + str(rows(db, "SELECT * FROM knowledge_gaps"))
    assert "12345678" not in dump
    assert "ali@company.com" not in dump


def test_trace_and_gap_are_stored(db):
    log()
    assert [r[0] for r in rows(db, "SELECT node FROM node_events ORDER BY seq")] == \
        ["route_question", "insufficient_evidence"]
    assert len(rows(db, "SELECT * FROM knowledge_gaps")) == 1
    assert len(audit.list_gaps()) == 1


def test_crash_is_logged_without_state(db):
    audit.log_request(request_id="r2", question="q", session_id=None,
                      state=None, latency_ms=5, status="error")
    assert rows(db, "SELECT status FROM requests WHERE id='r2'") == [("error",)]


def test_feedback_requires_a_real_request(db):
    log()
    assert audit.add_feedback("r1", "down", "wrong") is True
    assert audit.add_feedback("nope", "up", None) is False


def test_purge_removes_old_requests_and_their_events(db):
    log()
    con = sqlite3.connect(db)
    con.execute("UPDATE requests SET created_at = '2020-01-01T00:00:00+00:00'")
    con.commit()
    con.close()
    assert audit.purge_older_than(30) == 1
    assert rows(db, "SELECT * FROM node_events") == []
    assert len(rows(db, "SELECT * FROM knowledge_gaps")) == 1   # gaps survive until resolved