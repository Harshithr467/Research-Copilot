from typing import Any, Protocol, TypedDict, cast

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
    parent_id: str


class ElasticsearchClient(Protocol):
    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        ...


def _embed_query(text: str) -> list[float]:
    """
    Embed a query using the same local all-MiniLM-L6-v2 model used to embed
    documents at ingestion time.
    """
    model = get_embedding_model()
    return cast(list[float], model.embed_query(text))


def search(
    query: str,
    *,
    top_k: int = 5,
    year_from: int | None = None,
    year_to: int | None = None,
    es: ElasticsearchClient | None = None,
) -> list[SearchResult]:
    """
    Run hybrid search over the research_papers index.

    The request combines BM25 keyword matching with kNN vector search. It does
    not interpret page numbers from the query; document locations are handled
    later in the answer/citation layer.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    if es is None:
        from elasticsearch import Elasticsearch

        es = cast(
            ElasticsearchClient,
            Elasticsearch(
                ["http://localhost:9200"],
                verify_certs=False,
                request_timeout=30,
            ),
        )

    query_vector = _embed_query(query)

    filters: list[dict[str, Any]] = []
    if year_from is not None:
        filters.append({"range": {"metadata.year": {"gte": year_from}}})
    if year_to is not None:
        filters.append({"range": {"metadata.year": {"lte": year_to}}})

    body: dict[str, Any] = {
        "size": top_k,
        "_source": {"excludes": ["embedding"]},
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
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "num_candidates": 100,
            "k": top_k,
        },
    }

    if filters:
        body["knn"]["filter"] = {"bool": {"filter": filters}}

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
            )
        )

    return results
