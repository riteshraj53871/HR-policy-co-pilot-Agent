import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import get_settings


def fake_state():
    return {
        "route": "hr", "source_type": "kb", "retry_count": 0, "answer": "20 days.",
        "citations": [{"doc_id": "company_policy", "title": "Handbook",
                       "section": "Annual Leave", "version": "n/a", "score": 0.84}],
        "trace": [{"node": "route_question", "route": "hr"}],
    }


@pytest.fixture
def env(monkeypatch, tmp_path):
    for key in ("GROQ_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.setenv(key, "test")
    path = tmp_path / "audit.db"
    monkeypatch.setenv("AUDIT_DB_PATH", str(path))
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def client(env, monkeypatch):
    monkeypatch.setattr(main.builder, "ask", lambda q: fake_state())
    with TestClient(main.app) as c:
        yield c


def test_chat_returns_answer_and_logs_it(client, env):
    r = client.post("/api/chat", json={"question": "How many annual leave days?"})
    assert r.status_code == 200
    body = r.json()
    assert body["source_type"] == "kb"
    assert body["citations"][0]["section"] == "Annual Leave"
    con = sqlite3.connect(env)
    assert con.execute("SELECT status FROM requests WHERE id=?", (body["request_id"],)).fetchone() == ("ok",)
    con.close()


@pytest.mark.parametrize("payload", [{"question": ""}, {}, {"question": "x" * 1001}])
def test_bad_input_is_rejected(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_graph_failure_returns_503_and_is_logged(env, monkeypatch):
    def boom(q):
        raise RuntimeError("groq down")
    monkeypatch.setattr(main.builder, "ask", boom)
    with TestClient(main.app) as c:
        r = c.post("/api/chat", json={"question": "hi there"})
    assert r.status_code == 503
    con = sqlite3.connect(env)
    assert con.execute("SELECT status FROM requests").fetchone() == ("error",)
    con.close()


def test_feedback(client):
    rid = client.post("/api/chat", json={"question": "q"}).json()["request_id"]
    assert client.post("/api/feedback", json={"request_id": rid, "rating": "up"}).status_code == 200
    assert client.post("/api/feedback", json={"request_id": "nope", "rating": "up"}).status_code == 404
    assert client.post("/api/feedback", json={"request_id": rid, "rating": "meh"}).status_code == 422


def test_admin_needs_the_token(client):
    assert client.get("/api/admin/gaps").status_code == 403
    assert client.get("/api/admin/gaps", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.get("/api/admin/gaps", headers={"X-Admin-Token": "secret"}).status_code == 200


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}