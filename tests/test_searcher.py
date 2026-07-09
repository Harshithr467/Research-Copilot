from unittest.mock import MagicMock, patch

import pytest

from retrieval.searcher import SearchResult, search


# ── shared fixtures ────────────────────────────────────────────────────────────

def _fake_es_response(hits: list[dict]) -> dict:
    return {
        "hits": {
            "total": {"value": len(hits)},
            "hits": hits,
        }
    }


def _fake_hit(
    title="Test Paper",
    content="Some content.",
    score=1.0,
    author="A. Author",
    year=2024,
    source_file="paper.pdf",
    chunk_id=0,
) -> dict:
    return {
        "_id": f"paper_abc_{chunk_id}",
        "_score": score,
        "_source": {
            "title": title,
            "content": content,
            "metadata": {
                "author": author,
                "year": year,
                "source_file": source_file,
                "chunk_id": chunk_id,
            },
        },
    }


@pytest.fixture()
def fake_embedding():
    # 384-dim to match the all-MiniLM-L6-v2 model used at ingestion time
    # and the "dims": 384 mapping in indexing/setup_index.py.
    return [0.0] * 384


@pytest.fixture()
def mock_embedder(fake_embedding):
    mock_model = MagicMock()
    mock_model.embed_query.return_value = fake_embedding

    with patch("retrieval.searcher.get_embedding_model", return_value=mock_model):
        yield mock_model


@pytest.fixture()
def mock_es():
    return MagicMock()


# ── happy-path tests ───────────────────────────────────────────────────────────

def test_search_returns_correct_number_of_results(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response(
        [_fake_hit(title=f"Paper {i}", chunk_id=i) for i in range(3)]
    )
    results = search("neural rendering", top_k=3, es=mock_es)
    assert len(results) == 3


def test_search_result_fields_are_mapped_correctly(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response(
        [_fake_hit(title="Gaussian Splatting", content="Dense scene rep.", score=0.92,
                   author="Kerbl et al.", year=2023, source_file="kerbl.pdf", chunk_id=1)]
    )
    results = search("3D scene", es=mock_es)
    r = results[0]
    assert r["title"] == "Gaussian Splatting"
    assert r["content"] == "Dense scene rep."
    assert r["score"] == pytest.approx(0.92)
    assert r["author"] == "Kerbl et al."
    assert r["year"] == 2023
    assert r["source_file"] == "kerbl.pdf"
    assert r["chunk_id"] == 1


def test_search_embeds_query_with_shared_local_model(mock_embedder, mock_es):
    """
    The query text must be embedded with the *same* model instance used at
    ingestion time (indexing/ingester.get_embedding_model), not a separate
    embedding backend, or query vectors and document vectors will live in
    different spaces.
    """
    mock_es.search.return_value = _fake_es_response([])
    search("attention mechanisms", es=mock_es)
    mock_embedder.embed_query.assert_called_once_with("attention mechanisms")


def test_search_sends_query_vector_matching_index_dims(mock_embedder, mock_es, fake_embedding):
    mock_es.search.return_value = _fake_es_response([])
    search("attention mechanisms", es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    knn_leg = body["knn"]
    assert knn_leg["query_vector"] == fake_embedding
    assert len(knn_leg["query_vector"]) == 384


def test_search_excludes_embedding_from_response(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("transformers", es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    assert "embedding" in body["_source"]["excludes"]


def test_search_empty_index_returns_empty_list(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    results = search("anything", es=mock_es)
    assert results == []


def test_search_results_ordered_by_score(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response([
        _fake_hit(title="Best", score=0.99, chunk_id=0),
        _fake_hit(title="Middle", score=0.75, chunk_id=1),
        _fake_hit(title="Worst", score=0.50, chunk_id=2),
    ])
    results = search("query", es=mock_es)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ── year filter tests ──────────────────────────────────────────────────────────

def test_year_from_is_included_in_query_body(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("transformers", year_from=2023, es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    assert "gte" in str(body)
    assert "2023" in str(body)


def test_year_to_is_included_in_query_body(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("diffusion models", year_to=2024, es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    assert "lte" in str(body)
    assert "2024" in str(body)


def test_no_year_filter_uses_match_all(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("attention", es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == []
    assert "filter" not in body["knn"]


def test_search_does_not_add_page_number_boosts(mock_embedder, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("what happened on page 54?", es=mock_es)

    body = mock_es.search.call_args.kwargs["body"]
    query_bool = body["query"]["bool"]

    assert "should" not in query_bool
    assert "metadata.page_number" not in str(body)


# ── validation tests ───────────────────────────────────────────────────────────

def test_top_k_zero_raises_value_error(mock_embedder, mock_es):
    with pytest.raises(ValueError, match="top_k"):
        search("anything", top_k=0, es=mock_es)


def test_top_k_negative_raises_value_error(mock_embedder, mock_es):
    with pytest.raises(ValueError, match="top_k"):
        search("anything", top_k=-1, es=mock_es)
