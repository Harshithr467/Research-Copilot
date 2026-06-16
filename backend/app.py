import os

from fastapi import FastAPI, HTTPException

try:
    from .citations import FALLBACK_ANSWER, validate_citations
    from .embeddings import embed_query
    from .llm_provider import GeminiProvider
    from .prompting import build_grounded_prompt
    from .rag_types import ChatRequest, ChatResponse, Citation
    from .retrieval import extract_page_number, retrieve_chunks
except ImportError:
    from citations import FALLBACK_ANSWER, validate_citations
    from embeddings import embed_query
    from llm_provider import GeminiProvider
    from prompting import build_grounded_prompt
    from rag_types import ChatRequest, ChatResponse, Citation
    from retrieval import extract_page_number, retrieve_chunks


app = FastAPI(title="Research Copilot RAG API")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    page_number = extract_page_number(request.question)
    embedding_fn = embed_query if os.getenv("GOOGLE_API_KEY") else None

    try:
        chunks = retrieve_chunks(
            question=request.question,
            document_id=request.document_id,
            page_number=page_number,
            top_k=5,
            embedding_fn=embedding_fn,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    if not chunks:
        return ChatResponse(answer=FALLBACK_ANSWER, citations=[])

    prompt = build_grounded_prompt(request.question, chunks)

    try:
        llm_response = GeminiProvider().generate_json(prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    validated_response = validate_citations(llm_response, chunks)
    return ChatResponse(
        answer=validated_response["answer"],
        citations=[
            Citation(**citation)
            for citation in validated_response["citations"]
        ],
    )
