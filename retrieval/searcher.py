from typing import TypedDict

from dotenv import load_dotenv


from indexing.ingester import get_embedding_model

load_dotenv()

INDEX_NAME = "research_papers"


class SearchResult(TypedDict):
    title: str
    content: str
    score: float
    author: str
    year: int
    source_file: str
    chunk_id: int


def _embed_query(text: str) -> list[float]:
    """
    Embed a query using the same local all-MiniLM-L6-v2 model used to embed
    documents at ingestion time (see indexing/ingester.py). This blocks until
    the background model-loading thread in ingester.py has finished, if it
    hasn't already.
    """
    model = get_embedding_model()
    return model.embed_query(text)


def search(
    query: str,
    *,
    top_k: int = 5,
    year_from: int | None = None,
    year_to: int | None = None,
    es: object | None = None,
) -> list[SearchResult]:
    """
    Run a hybrid search over the research_papers index.

    Args:
        query:     Natural-language question or keyword string.
        top_k:     Number of results to return (default 5).
        year_from: Optional lower bound on metadata.year (inclusive).
        year_to:   Optional upper bound on metadata.year (inclusive).
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
    if year_from is not None:
        filters.append({"range": {"metadata.year": {"gte": year_from}}})
    if year_to is not None:
        filters.append({"range": {"metadata.year": {"lte": year_to}}})

    filter_clause: dict = {"bool": {"filter": filters}} if filters else {"match_all": {}}

    body = {
        "size": top_k,
        "_source": {"excludes": ["embedding"]},   # don't ship the raw vector back
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
            )
        )

    return results