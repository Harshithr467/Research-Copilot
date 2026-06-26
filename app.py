from fastapi import FastAPI, HTTPException

from rag.citations import FALLBACK_ANSWER, validate_citations
from rag.llm_provider import GeminiProvider
from rag.models import ChatRequest, ChatResponse, Citation, RetrievedChunk
from rag.prompting import build_grounded_prompt
from rag.retrieval import extract_page_number, retrieve_chunks


app = FastAPI(title="Research Copilot RAG API")


def build_extractive_fallback(chunks: list[RetrievedChunk]) -> ChatResponse:
    citations = [
        Citation(
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            quote=" ".join(chunk.text.split())[:300],
        )
        for chunk in chunks[:3]
        if chunk.text.strip()
    ]
    if not citations:
        return ChatResponse(answer=FALLBACK_ANSWER, citations=[])

    answer = (
        "The LLM service is temporarily unavailable, so this is an extractive "
        "answer from the retrieved document chunks. The retrieved evidence says: "
        + " ".join(citation.quote for citation in citations)
    )

    return ChatResponse(answer=answer, citations=citations)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    page_number = extract_page_number(request.question)

    try:
        chunks = retrieve_chunks(
            question=request.question,
            document_id=request.document_id,
            page_number=page_number,
            top_k=5,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        return ChatResponse(answer=FALLBACK_ANSWER, citations=[])

    prompt = build_grounded_prompt(request.question, chunks)

    try:
        llm_response = GeminiProvider().generate_json(prompt)
    except Exception:
        return build_extractive_fallback(chunks)

    validated_response = validate_citations(llm_response, chunks)
    return ChatResponse(
        answer=validated_response["answer"],
        citations=[
            Citation(**citation)
            for citation in validated_response["citations"]
        ],
    )
