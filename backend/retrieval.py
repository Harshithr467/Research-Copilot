import os
import re
from typing import Callable, Dict, List, Optional

try:
    from .rag_types import RetrievedChunk
except ImportError:
    from rag_types import RetrievedChunk


INDEX_NAME = os.getenv("ES_INDEX_NAME", "research_papers")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")


def extract_page_number(question: str) -> Optional[int]:
    """Extract a natural-language page reference like 'page 54' or 'p. 54'."""
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


def build_retrieval_query(
    question: str,
    document_id: str,
    query_embedding: Optional[List[float]] = None,
    page_number: Optional[int] = None,
    top_k: int = 5,
) -> Dict:
    """Build an Elasticsearch hybrid retrieval request."""
    filters = [{"term": {"metadata.parent_id": document_id}}]
    should = [
        {
            "multi_match": {
                "query": question,
                "fields": ["content^3", "title"],
                "type": "best_fields",
            }
        }
    ]

    if page_number is not None:
        should.extend(
            [
                {"term": {"metadata.page_number": {"value": page_number, "boost": 8}}},
                {"term": {"metadata.page_number": {"value": page_number - 1, "boost": 2}}},
                {"term": {"metadata.page_number": {"value": page_number + 1, "boost": 2}}},
            ]
        )

    body = {
        "size": top_k,
        "_source": {"excludes": ["embedding"]},
        "query": {
            "bool": {
                "filter": filters,
                "should": should,
                "minimum_should_match": 0,
            }
        },
    }

    if query_embedding is not None:
        body["knn"] = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": max(top_k * 3, 10),
            "num_candidates": max(top_k * 10, 50),
            "filter": filters,
        }

    return body


def get_default_es_client():
    from elasticsearch import Elasticsearch

    return Elasticsearch([ES_URL], verify_certs=False, request_timeout=30)


def normalize_hit(hit: Dict) -> RetrievedChunk:
    source = hit.get("_source", {})
    metadata = source.get("metadata", {}) or {}
    parent_id = metadata.get("parent_id")
    chunk_index = metadata.get("chunk_id")
    fallback_chunk_id = (
        f"{parent_id}_chunk_{chunk_index}"
        if parent_id is not None and chunk_index is not None
        else hit.get("_id", "")
    )
    return RetrievedChunk(
        chunk_id=hit.get("_id") or fallback_chunk_id,
        text=source.get("content", ""),
        source_file=metadata.get("source_file"),
        page_number=metadata.get("page_number"),
        parent_id=parent_id,
        score=hit.get("_score"),
        metadata=metadata,
    )


def retrieve_chunks(
    question: str,
    document_id: str,
    page_number: Optional[int] = None,
    top_k: int = 5,
    es_client=None,
    embedding_fn: Optional[Callable[[str], List[float]]] = None,
) -> List[RetrievedChunk]:
    query_embedding = embedding_fn(question) if embedding_fn else None
    body = build_retrieval_query(
        question=question,
        document_id=document_id,
        query_embedding=query_embedding,
        page_number=page_number,
        top_k=top_k,
    )
    es = es_client or get_default_es_client()
    response = es.search(index=INDEX_NAME, body=body)
    return [normalize_hit(hit) for hit in response.get("hits", {}).get("hits", [])]
