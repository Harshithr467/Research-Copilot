from rag.citations import FALLBACK_ANSWER, validate_citations
from rag.models import RetrievedChunk
from rag.prompting import build_grounded_prompt
from rag.retrieval import extract_page_number


def test_extract_page_number_from_question():
    assert extract_page_number("What does the author say on page 54?") == 54
    assert extract_page_number("summarize p. 7") == 7
    assert extract_page_number("No specific page here") is None


def test_prompt_construction_requires_grounding_and_json():
    chunks = [
        RetrievedChunk(
            chunk_id="paper_123_chunk_17",
            text="The author states that retrieval must be grounded.",
            source_file="paper.pdf",
            page_number=54,
        )
    ]

    prompt = build_grounded_prompt("What is said on page 54?", chunks)

    assert "Do not use outside knowledge" in prompt
    assert "Return only valid JSON" in prompt
    assert "paper_123_chunk_17" in prompt
    assert "The author states that retrieval must be grounded." in prompt


def test_validate_citations_rejects_missing_chunk_and_bad_quote():
    chunks = [
        RetrievedChunk(
            chunk_id="paper_123_chunk_17",
            text="Exact supporting text from the uploaded paper.",
            source_file="paper.pdf",
            page_number=54,
        )
    ]
    response = {
        "answer": "The paper includes exact supporting text.",
        "citations": [
            {
                "source_file": "paper.pdf",
                "page_number": 54,
                "chunk_id": "paper_123_chunk_17",
                "quote": "Exact supporting text from the uploaded paper.",
            },
            {
                "source_file": "paper.pdf",
                "page_number": 54,
                "chunk_id": "paper_123_chunk_999",
                "quote": "This chunk was not retrieved.",
            },
            {
                "source_file": "paper.pdf",
                "page_number": 54,
                "chunk_id": "paper_123_chunk_17",
                "quote": "This quote is not present.",
            },
        ],
    }

    validated = validate_citations(response, chunks)

    assert validated["answer"] == response["answer"]
    assert len(validated["citations"]) == 1
    assert validated["citations"][0]["chunk_id"] == "paper_123_chunk_17"


def test_validate_citations_falls_back_when_no_valid_citations():
    chunks = [
        RetrievedChunk(
            chunk_id="paper_123_chunk_17",
            text="Available retrieved evidence.",
            source_file="paper.pdf",
            page_number=None,
        )
    ]
    response = {
        "answer": "Unsupported answer.",
        "citations": [
            {
                "chunk_id": "paper_123_chunk_17",
                "quote": "Missing quote.",
            }
        ],
    }

    validated = validate_citations(response, chunks)

    assert validated["answer"] == FALLBACK_ANSWER
    assert validated["citations"][0]["quote"] == "Available retrieved evidence."
