import re
from typing import Any, Iterable, List, Mapping, Optional, TypedDict

try:
    from .rag_types import Citation, RetrievedChunk
except ImportError:
    from rag_types import Citation, RetrievedChunk


FALLBACK_ANSWER = "I cannot find enough evidence in the uploaded document to answer that."


class CitationDict(TypedDict):
    source_file: Optional[str]
    page_number: Optional[int]
    chunk_id: str
    quote: str


class ValidatedCitationResponse(TypedDict):
    answer: str
    citations: List[CitationDict]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _quote_in_chunk(quote: str, chunk_text: str) -> bool:
    normalized_quote = _normalize_text(quote)
    if not normalized_quote:
        return False
    return normalized_quote in _normalize_text(chunk_text)


def _short_quote(text: str, max_chars: int = 300) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:max_chars]


def _dump_citation(citation: Citation) -> CitationDict:
    return {
        "source_file": citation.source_file,
        "page_number": citation.page_number,
        "chunk_id": citation.chunk_id,
        "quote": citation.quote,
    }


def validate_citations(
    llm_response: Mapping[str, Any],
    retrieved_chunks: Iterable[RetrievedChunk],
) -> ValidatedCitationResponse:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    valid_citations: List[Citation] = []

    raw_citations_value = llm_response.get("citations", [])
    raw_citations: List[Any] = (
        raw_citations_value if isinstance(raw_citations_value, list) else []
    )

    for raw_citation in raw_citations:
        if not isinstance(raw_citation, Mapping):
            continue
        chunk_id_value = raw_citation.get("chunk_id")
        quote_value = raw_citation.get("quote", "")
        chunk_id = chunk_id_value if isinstance(chunk_id_value, str) else None
        quote = quote_value if isinstance(quote_value, str) else None
        if not isinstance(chunk_id, str) or not isinstance(quote, str):
            continue
        chunk = chunks_by_id.get(chunk_id)
        if not chunk or not _quote_in_chunk(quote, chunk.text):
            continue

        valid_citations.append(
            Citation(
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                quote=quote,
            )
        )

    raw_answer = llm_response.get("answer", "")
    answer = raw_answer.strip() if isinstance(raw_answer, str) else ""
    if answer and valid_citations:
        return {
            "answer": answer,
            "citations": [_dump_citation(citation) for citation in valid_citations],
        }

    fallback_citations: List[Citation] = []
    for chunk in list(chunks_by_id.values())[:2]:
        if chunk.text:
            fallback_citations.append(
                Citation(
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    quote=_short_quote(chunk.text),
                )
            )

    return {
        "answer": FALLBACK_ANSWER,
        "citations": [_dump_citation(citation) for citation in fallback_citations],
    }
