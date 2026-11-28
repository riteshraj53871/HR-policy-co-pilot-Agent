import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import get_settings

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_NUMBER = re.compile(r"\d{6,}")


def mask_pii(text: str) -> str:
    """Hide emails and ID-like numbers before anything is written to disk."""
    return _LONG_NUMBER.sub("[number]", _EMAIL.sub("[email]", text or ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    session_id  TEXT,
    question    TEXT NOT NULL,
    route       TEXT,
    source_type TEXT,
    retries     INTEGER,
    status      TEXT NOT NULL,
    latency_ms  INTEGER,
    answer      TEXT
);
CREATE TABLE IF NOT EXISTS node_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    node       TEXT NOT NULL,
    detail     TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    rating     TEXT NOT NULL,
    comment    TEXT
);
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT REFERENCES requests(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    question   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS idx_events_request ON node_events(request_id);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
"""


@contextmanager
def _conn():
    path = Path(get_settings().audit_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.execute("PRAGMA journal_mode = WAL")
        con.executescript(SCHEMA)


def log_request(
    *,
    request_id: str,
    question: str,
    session_id: str | None,
    state: dict | None,
    latency_ms: int,
    status: str,
) -> None:
    """Write one request, its node trace, and any knowledge gap. `state` is None on a crash."""
    state = state or {}
    masked_q = mask_pii(question)
    trace = state.get("trace", [])
    with _conn() as con:
        con.execute(
            "INSERT INTO requests (id, created_at, session_id, question, route, source_type,"
            " retries, status, latency_ms, answer) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                request_id, _now(), session_id, masked_q, state.get("route"),
                state.get("source_type"), state.get("retry_count"), status, latency_ms,
                mask_pii(state.get("answer", "")) or None,
            ),
        )
        con.executemany(
            "INSERT INTO node_events (request_id, seq, node, detail) VALUES (?,?,?,?)",
            [
                (
                    request_id, i, t.get("node", "?"),
                    mask_pii(json.dumps({k: v for k, v in t.items() if k != "node"},
                                        ensure_ascii=False, default=str)),
                )
                for i, t in enumerate(trace)
            ],
        )
        if any(t.get("knowledge_gap") for t in trace):
            con.execute(
                "INSERT INTO knowledge_gaps (request_id, created_at, question) VALUES (?,?,?)",
                (request_id, _now(), masked_q),
            )


def add_feedback(request_id: str, rating: str, comment: str | None) -> bool:
    """Returns False if the request id does not exist."""
    with _conn() as con:
        if not con.execute("SELECT 1 FROM requests WHERE id = ?", (request_id,)).fetchone():
            return False
        con.execute(
            "INSERT INTO feedback (request_id, created_at, rating, comment) VALUES (?,?,?,?)",
            (request_id, _now(), rating, mask_pii(comment or "") or None),
        )
    return True


def list_gaps(status: str = "open", limit: int = 100) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, request_id, created_at, question, status FROM knowledge_gaps"
            " WHERE status = ? ORDER BY id DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def purge_older_than(days: int) -> int:
    """Delete requests (and their events and feedback) older than `days`. Returns the count.
    Knowledge gaps are kept until HR resolves them."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with _conn() as con:
        return con.execute("DELETE FROM requests WHERE created_at < ?", (cutoff,)).rowcount