from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=64)


class Citation(BaseModel):
    title: str | None = None
    section: str | None = None
    doc_id: str | None = None
    version: str | None = None
    url: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    source_type: Literal["kb", "web", "none", "direct"]  # the UI maps this to a badge
    citations: list[Citation] = []


class FeedbackRequest(BaseModel):
    request_id: str = Field(max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)