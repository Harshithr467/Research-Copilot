import os
from typing import Any, TypedDict, cast

from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "research_papers"

_client: Any | None = None

def _get_client() -> Any:
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client


class SearchResult(TypedDict):
    title: str
    content: str
    score: float
    author: str
    year: int
    source_file: str
    chunk_id: int
    parent_id: str
    page_number: int | None


def _embed_query(text: str) -> list[float]:
    result = _get_client().models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={
            "task_type": "retrieval_query",
            "output_dimensionality": 768,
        },
    )
    embeddings = result.embeddings
    if not embeddings:
        raise RuntimeError("Google GenAI did not return an embedding.")
    values = embeddings[0].values
    if values is None:
        raise RuntimeError("Google GenAI did not return an embedding.")
    return cast(list[float], values)


def search(
    query: str,
    *,
    top_k: int = 5,
    document_id: str | None = None,
    page_number: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    es: object | None = None,
) -> list[SearchResult]:
    """
    Run a hybrid search over the research_papers index.

    Args:
        query:     Natural-language question or keyword string.
        top_k:     Number of results to return (default 5).
        document_id: Optional metadata.parent_id filter for one uploaded document.
        page_number: Optional page reference to strongly boost exact/nearby pages.
        year_from:   Optional lower bound on metadata.year (inclusive).
        year_to:     Optional upper bound on metadata.year (inclusive).
        es:        Optional pre-built Elasticsearch client (used in tests to
                   inject a mock without touching the network).

    Returns:
        List of SearchResult dicts ordered by RRF score (best first).

    Raises:
        ValueError: if top_k < 1.
        RuntimeError: if Elasticsearch returns an error response.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    if es is None:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            ["http://localhost:9200"],
            verify_certs=False,
            request_timeout=30,
        )

    query_vector = _embed_query(query)

    # --- optional year filter (applied to both legs via post_filter) ----------
    filters: list[dict] = []
    if document_id is not None:
        filters.append({"term": {"metadata.parent_id": document_id}})
    if year_from is not None:
        filters.append({"range": {"metadata.year": {"gte": year_from}}})
    if year_to is not None:
        filters.append({"range": {"metadata.year": {"lte": year_to}}})

    filter_clause: dict = {"bool": {"filter": filters}} if filters else {"match_all": {}}
    page_boosts: list[dict] = []
    if page_number is not None:
        page_boosts = [
            {"term": {"metadata.page_number": {"value": page_number, "boost": 8}}},
            {"term": {"metadata.page_number": {"value": page_number - 1, "boost": 2}}},
            {"term": {"metadata.page_number": {"value": page_number + 1, "boost": 2}}},
        ]

    body = {
        "size": top_k,
        "_source": {"excludes": ["embedding"]},   # don't ship 768 floats back
        "retriever": {
            "rrf": {
                "retrievers": [
                    # Leg 1: BM25 keyword search across title + content
                    {
                        "standard": {
                            "query": {
                                "bool": {
                                    "must": {
                                        "multi_match": {
                                            "query": query,
                                            "fields": ["title^2", "content"],
                                        }
                                    },
                                    "filter": filters,
                                    "should": page_boosts,
                                    "minimum_should_match": 0,
                                }
                            }
                        }
                    },
                    # Leg 2: kNN semantic vector search
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_vector,
                            "num_candidates": 100,
                            "k": top_k,
                            "filter": filter_clause,
                        }
                    },
                ],
                "rank_window_size": 50,
                "rank_constant": 60,
            }
        },
    }

    response = es.search(index=INDEX_NAME, body=body)

    hits = response["hits"]["hits"]
    results: list[SearchResult] = []
    for hit in hits:
        src = hit["_source"]
        meta = src.get("metadata", {})
        results.append(
            SearchResult(
                title=src.get("title", ""),
                content=src.get("content", ""),
                score=hit["_score"],
                author=meta.get("author", ""),
                year=meta.get("year", 0),
                source_file=meta.get("source_file", ""),
                chunk_id=meta.get("chunk_id", 0),
                parent_id=meta.get("parent_id", ""),
                page_number=meta.get("page_number"),
            )
        )

    return results
