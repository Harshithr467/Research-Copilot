import re

from retrieval.searcher import SearchResult, search

from .models import RetrievedChunk


def extract_page_number(question: str) -> int | None:
    """Extract natural-language page references like 'page 54' or 'p. 54'."""
    patterns = [
        r"\bpages?\s*(?:number\s*)?(\d{1,5})\b",
        r"\bp\.\s*(\d{1,5})\b",
        r"\bpg\.\s*(\d{1,5})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _chunk_id(result: SearchResult) -> str:
    parent_id = result.get("parent_id", "")
    chunk_index = result.get("chunk_id", 0)
    if parent_id:
        return f"{parent_id}_chunk_{chunk_index}"
    return str(chunk_index)


def retrieve_chunks(
    question: str,
    document_id: str,
    page_number: int | None = None,
    top_k: int = 5,
    es: object | None = None,
) -> list[RetrievedChunk]:
    results = search(
        question,
        top_k=top_k,
        document_id=document_id,
        page_number=page_number,
        es=es,
    )
    return [
        RetrievedChunk(
            chunk_id=_chunk_id(result),
            chunk_index=result.get("chunk_id"),
            text=result.get("content", ""),
            source_file=result.get("source_file"),
            page_number=result.get("page_number"),
            parent_id=result.get("parent_id"),
            score=result.get("score"),
            metadata={
                "author": result.get("author"),
                "year": result.get("year"),
            },
        )
        for result in results
    ]
