from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    chunk_index: int | None = None
    text: str
    source_file: str | None = None
    page_number: int | None = None
    parent_id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source_file: str | None = None
    page_number: int | None = None
    chunk_id: str
    quote: str


class ChatRequest(BaseModel):
    document_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
