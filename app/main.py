import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles

from app import audit
from app.config import get_settings
from app.graph import builder
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest

logger = logging.getLogger("hr_copilot")
FRONTEND_DIR = Path("frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    audit.init_db()
    yield


app = FastAPI(title="HR Policy Copilot", lifespan=lifespan)


def require_admin(x_admin_token: str = Header(default="")) -> None:
    token = get_settings().admin_token
    if not token or not secrets.compare_digest(x_admin_token.encode(), token.encode()):
        raise HTTPException(status_code=403, detail="Forbidden")


def _safe_log(**kwargs) -> None:
    """An audit failure must never stop an employee from getting an answer."""
    try:
        audit.log_request(**kwargs)
    except Exception:
        logger.exception("audit logging failed")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    request_id = uuid.uuid4().hex
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")

    start = time.perf_counter()
    try:
        state = builder.ask(question)
    except Exception:
        logger.exception("graph failed")
        _safe_log(
            request_id=request_id, question=question, session_id=req.session_id,
            state=None, latency_ms=int((time.perf_counter() - start) * 1000), status="error",
        )
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again or contact HR.",
        )

    _safe_log(
        request_id=request_id, question=question, session_id=req.session_id, state=state,
        latency_ms=int((time.perf_counter() - start) * 1000), status="ok",
    )
    return ChatResponse(
        request_id=request_id,
        answer=state["answer"],
        source_type=state["source_type"],
        citations=state.get("citations", []),
    )


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if not audit.add_feedback(req.request_id, req.rating, req.comment):
        raise HTTPException(status_code=404, detail="Unknown request_id")
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/admin/gaps", dependencies=[Depends(require_admin)])
def knowledge_gaps(status: str = "open", limit: int = 100):
    return {"gaps": audit.list_gaps(status=status, limit=min(limit, 500))}


# Must stay last, so it doesn't shadow the /api routes
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")