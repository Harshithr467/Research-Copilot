from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    parent_id: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    chunk_id: str
    quote: str


class ChatRequest(BaseModel):
    document_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
