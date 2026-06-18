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
    return [0.0] * 768


@pytest.fixture()
def mock_gemini(fake_embedding):
    embedding_obj = MagicMock()
    embedding_obj.values = fake_embedding

    result_obj = MagicMock()
    result_obj.embeddings = [embedding_obj]

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = result_obj

    with patch("retrieval.searcher._get_client", return_value=mock_client):
        yield mock_client


@pytest.fixture()
def mock_es():
    return MagicMock()


# ── happy-path tests ───────────────────────────────────────────────────────────

def test_search_returns_correct_number_of_results(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response(
        [_fake_hit(title=f"Paper {i}", chunk_id=i) for i in range(3)]
    )
    results = search("neural rendering", top_k=3, es=mock_es)
    assert len(results) == 3


def test_search_result_fields_are_mapped_correctly(mock_gemini, mock_es):
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


def test_search_uses_retrieval_query_task_type(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("attention mechanisms", es=mock_es)
    call_kwargs = mock_gemini.models.embed_content.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs.args[2]
    assert config["task_type"] == "retrieval_query"


def test_search_excludes_embedding_from_response(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("transformers", es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    assert "embedding" in body["_source"]["excludes"]


def test_search_empty_index_returns_empty_list(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    results = search("anything", es=mock_es)
    assert results == []


def test_search_results_ordered_by_score(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response([
        _fake_hit(title="Best", score=0.99, chunk_id=0),
        _fake_hit(title="Middle", score=0.75, chunk_id=1),
        _fake_hit(title="Worst", score=0.50, chunk_id=2),
    ])
    results = search("query", es=mock_es)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ── year filter tests ──────────────────────────────────────────────────────────

def test_year_from_is_included_in_query_body(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("transformers", year_from=2023, es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    assert "gte" in str(body)
    assert "2023" in str(body)


def test_year_to_is_included_in_query_body(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("diffusion models", year_to=2024, es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    assert "lte" in str(body)
    assert "2024" in str(body)


def test_no_year_filter_uses_match_all(mock_gemini, mock_es):
    mock_es.search.return_value = _fake_es_response([])
    search("attention", es=mock_es)
    body = mock_es.search.call_args.kwargs["body"]
    knn_leg = body["retriever"]["rrf"]["retrievers"][1]["knn"]
    assert knn_leg["filter"] == {"match_all": {}}


# ── validation tests ───────────────────────────────────────────────────────────

def test_top_k_zero_raises_value_error(mock_gemini, mock_es):
    with pytest.raises(ValueError, match="top_k"):
        search("anything", top_k=0, es=mock_es)


def test_top_k_negative_raises_value_error(mock_gemini, mock_es):
    with pytest.raises(ValueError, match="top_k"):
        search("anything", top_k=-1, es=mock_es)