"""
Tests for the /chat endpoint in api_server.py.

These mock both retrieval.searcher.search (so no real Elasticsearch or
embedding model is needed) and the Gemini client (so no real API key or
network call is needed). They verify the endpoint's wiring and contract,
not the actual quality of retrieval or generation.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api_server
from api_server import app, ChatAnswer, CitationOut


@pytest.fixture()
def client():
    return TestClient(app)


def _fake_search_result(
    source_file="paper.pdf",
    chunk_id=0,
    page_number=1,
    content="Some text.",
    title="A Paper",
):
    return {
        "title": title,
        "content": content,
        "score": 1.0,
        "author": "Someone",
        "year": 2024,
        "source_file": source_file,
        "chunk_id": chunk_id,
        "page_number": page_number,
    }


def test_chat_returns_insufficient_when_no_search_results(client):
    with patch("api_server.search", return_value=[]):
        res = client.post("/chat", json={"query": "anything"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["insufficient"] is True


def test_chat_calls_search_with_query_and_top_k(client):
    with patch("api_server.search", return_value=[]) as mock_search:
        client.post("/chat", json={"query": "what is attention", "top_k": 3})

    mock_search.assert_called_once_with("what is attention", top_k=3)


def test_chat_defaults_top_k_to_five(client):
    with patch("api_server.search", return_value=[]) as mock_search:
        client.post("/chat", json={"query": "what is attention"})

    mock_search.assert_called_once_with("what is attention", top_k=5)


def test_chat_generates_answer_from_search_results(client):
    results = [_fake_search_result(source_file="attention.pdf", chunk_id=2)]

    mock_llm_response = MagicMock()
    mock_llm_response.parsed = ChatAnswer(
        answer="Attention uses scaled dot-products [1].",
        citations=[CitationOut(id=1, doc="attention.pdf", page=2)],
        insufficient=False,
    )

    with patch("api_server.search", return_value=results), \
         patch.object(api_server.client.models, "generate_content", return_value=mock_llm_response):
        res = client.post("/chat", json={"query": "how does attention work"})

    assert res.status_code == 200
    body = res.json()
    assert body["insufficient"] is False
    assert body["answer"] == "Attention uses scaled dot-products [1]."
    assert body["citations"] == [{
        "id": 1,
        "doc": "attention.pdf",
        "page": 1,
        "author": "Someone",
        "year": 2024,
        "formatted": "Someone. (2024). A Paper. attention.pdf, p. 1.",
    }]


def test_chat_rebuilds_citations_from_real_results_not_llm_output(client):
    """
    Even if the LLM hallucinates a wrong doc/page in its citation output, the
    endpoint should overwrite it with the real source_file/chunk_id from our
    own retrieval results, keyed by citation id.
    """
    results = [_fake_search_result(source_file="real_source.pdf", chunk_id=7, page_number=12)]

    mock_llm_response = MagicMock()
    mock_llm_response.parsed = ChatAnswer(
        answer="Some answer [1].",
        citations=[CitationOut(id=1, doc="hallucinated_source.pdf", page=999)],
        insufficient=False,
    )

    with patch("api_server.search", return_value=results), \
         patch.object(api_server.client.models, "generate_content", return_value=mock_llm_response):
        res = client.post("/chat", json={"query": "test"})

    body = res.json()
    assert body["citations"][0]["doc"] == "real_source.pdf"
    assert body["citations"][0]["page"] == 12
    assert body["citations"][0]["formatted"] == "Someone. (2024). A Paper. real_source.pdf, p. 12."


def test_chat_appends_apa_locations_when_requested(client):
    results = [
        _fake_search_result(
            title="Example Study",
            source_file="example.pdf",
            chunk_id=4,
            page_number=9,
        )
    ]

    mock_llm_response = MagicMock()
    mock_llm_response.parsed = ChatAnswer(
        answer="The document supports the claim [1].",
        citations=[CitationOut(id=1, doc="example.pdf", page=4)],
        insufficient=False,
    )

    with patch("api_server.search", return_value=results), \
         patch.object(api_server.client.models, "generate_content", return_value=mock_llm_response):
        res = client.post("/chat", json={"query": "Explain this in APA format"})

    body = res.json()
    assert body["insufficient"] is False
    assert "APA location(s):" in body["answer"]
    assert "Example Study" in body["answer"]
    assert "example.pdf, p. 9" in body["answer"]
    assert body["citations"][0]["page"] == 9
    assert body["citations"][0]["author"] == "Someone"
    assert body["citations"][0]["formatted"] == "Someone. (2024). Example Study. example.pdf, p. 9."


def test_chat_appends_mla_locations_when_requested(client):
    results = [
        _fake_search_result(
            title="Example Study",
            source_file="example.pdf",
            page_number=3,
        )
    ]

    mock_llm_response = MagicMock()
    mock_llm_response.parsed = ChatAnswer(
        answer="The document supports the claim [1].",
        citations=[CitationOut(id=1, doc="example.pdf", page=3)],
        insufficient=False,
    )

    with patch("api_server.search", return_value=results), \
         patch.object(api_server.client.models, "generate_content", return_value=mock_llm_response):
        res = client.post("/chat", json={"query": "Explain this in MLA format"})

    body = res.json()
    assert "MLA location(s):" in body["answer"]
    assert 'Someone. "Example Study." example.pdf, 2024, p. 3.' in body["answer"]
    assert body["citations"][0]["formatted"] == 'Someone. "Example Study." example.pdf, 2024, p. 3.'


def test_chat_drops_citation_ids_with_no_matching_result(client):
    results = [_fake_search_result(source_file="only_one.pdf", chunk_id=0)]

    mock_llm_response = MagicMock()
    mock_llm_response.parsed = ChatAnswer(
        answer="Some answer [1][2].",
        citations=[
            CitationOut(id=1, doc="only_one.pdf", page=0),
            CitationOut(id=2, doc="made_up.pdf", page=5),  # no result #2 exists
        ],
        insufficient=False,
    )

    with patch("api_server.search", return_value=results), \
         patch.object(api_server.client.models, "generate_content", return_value=mock_llm_response):
        res = client.post("/chat", json={"query": "test"})

    body = res.json()
    assert len(body["citations"]) == 1
    assert body["citations"][0]["doc"] == "only_one.pdf"


def test_chat_falls_back_to_insufficient_when_llm_keeps_failing(client):
    results = [_fake_search_result()]

    with patch("api_server.search", return_value=results), \
         patch.object(api_server.client.models, "generate_content", side_effect=Exception("boom")), \
         patch("api_server.time.sleep"):  # skip real retry delays in tests
        res = client.post("/chat", json={"query": "test"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["insufficient"] is True
