import os
import shutil
import time
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import uvicorn
from contextlib import asynccontextmanager
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

# Import real extraction, ingestion, and retrieval functions
from extraction.extractor import extract
from indexing.ingester import ingest_paper, ResearchPaper
from retrieval.searcher import search

# Load environment variables
load_dotenv()

# Configure Google GenAI Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class PaperMetadata(BaseModel):
    title: str
    author: str
    year: int

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class CitationOut(BaseModel):
    id: int
    doc: str

    page: int

class ChatAnswer(BaseModel):
    answer: Optional[str]
    citations: List[CitationOut]
    insufficient: bool

UPLOAD_DIR = "uploaded_files"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the upload directory exists on startup."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        print(f"Created directory: {UPLOAD_DIR}")
    yield

app = FastAPI(title="Research-Copilot API", lifespan=lifespan)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_metadata_with_llm(text: str, filename: str) -> PaperMetadata:
    """
    Use gemini-2.5-flash to extract structured metadata with retries.
    """
    prompt = f"""
    Extract the title, primary author, and publication year from the following research paper text.
    If multiple authors exist, combine them into a single string.
    If the publication year is not found, return 0.
    
    Text:
    {text[:4000]}
    """

    for _ in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': PaperMetadata,
                }
            )
            return response.parsed
        except Exception as e:
            print(f"API Error/Rate Limit: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    
    raise e

@app.get("/")
async def root():
    """Welcome message and basic health check."""
    return {
        "message": "Welcome to the Research-Copilot API",
        "docs": "Visit /docs for the interactive API documentation",
        "status": "online"
    }

def generate_grounded_answer(query: str, results: List[dict]) -> ChatAnswer:
    """
    Given retrieved chunks, ask gemini-2.5-flash to produce an answer that is
    grounded ONLY in those chunks, with citations mapped back to the chunk
    numbers we gave it. Retries a few times on transient API errors, same
    pattern as extract_metadata_with_llm above.
    """
    context_blocks = [
        f"[{i}] (source: {r['source_file']}, chunk {r['chunk_id']})\n{r['content']}"
        for i, r in enumerate(results, start=1)
    ]
    context = "\n\n".join(context_blocks)

    prompt = f"""
    You are a research assistant. Answer the user's question using ONLY the
    numbered context excerpts below - do not use outside knowledge.

    Cite every claim inline using the excerpt number it came from, e.g. [1].
    In your structured "citations" output, each entry's "id" must be one of
    the excerpt numbers above, and "doc" must be copied exactly from that
    excerpt's "source" value.

    If the excerpts do not contain enough information to answer the
    question, set "insufficient" to true and "answer" to null.

    Question: {query}

    Context:
    {context}
    """

    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ChatAnswer,
                },
            )
            parsed: ChatAnswer = response.parsed

           
            fixed_citations: List[CitationOut] = []
            for c in parsed.citations:
                idx = c.id - 1
                if 0 <= idx < len(results):
                    fixed_citations.append(CitationOut(
                        id=c.id,
                        doc=results[idx]["source_file"],
                        page=results[idx]["chunk_id"],
                    ))
            parsed.citations = fixed_citations
            return parsed
        except Exception as e:
            last_error = e
            print(f"Chat LLM error: {e}. Retrying in 3 seconds...")
            time.sleep(3)

    # If the LLM call kept failing, fail safe rather than crash the request.
    print(f"Chat LLM permanently failed: {last_error}")
    return ChatAnswer(answer=None, citations=[], insufficient=True)

@app.post("/chat", response_model=ChatAnswer)
async def chat(req: ChatRequest):
    """
    Search the index for relevant chunks, then generate a grounded answer
    with citations. Runs the (blocking) search + embedding call in a
    threadpool so it doesn't block the event loop.
    """
    results = await run_in_threadpool(search, req.query, top_k=req.top_k)

    if not results:
        return ChatAnswer(answer=None, citations=[], insufficient=True)

    return await run_in_threadpool(generate_grounded_answer, req.query, results)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, extract text, parse metadata, and ingest into Elasticsearch.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 1. Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Extract text chunks from the file
        extracted_chunks = extract(file_path)
        
        # 3. Sort chunks by page number and concatenate text
        extracted_chunks.sort(key=lambda x: x.get("page") if x.get("page") is not None else 0)
        
        full_text_string = "".join(chunk["text"] for chunk in extracted_chunks)
        
        # 4. Prepare metadata text from page 1 and 2
        intro_text = "".join(
            chunk["text"] for chunk in extracted_chunks 
            if chunk.get("page") in [1, 2]
        )
        if not intro_text and extracted_chunks:
            intro_text = "".join(chunk["text"] for chunk in extracted_chunks[:2])

        # 5. Extract metadata (with retries)
        metadata = extract_metadata_with_llm(intro_text, file.filename)
        
        # 6. Assemble into ResearchPaper dictionary format
        paper_data: ResearchPaper = {
            "title": metadata.title,
            "content": full_text_string,
            "author": metadata.author,
            "year": metadata.year,
            "source_file": file.filename
        }
        
        # 7. Pass directly to the ingester
        # The ingester now has its own internal retry logic for embeddings.
        ingestion_result = ingest_paper(paper_data)
        
        return {
            "status": "success",
            "filename": file.filename,
            "metadata_source": "llm" if metadata.year != 0 or metadata.author != "Unknown Author" else "fallback",
            "pipeline_results": ingestion_result
        }

    except Exception as e:
        return {
            "status": "error",
            "filename": file.filename,
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)