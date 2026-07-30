import os
import re
import shutil
import time
from typing import Any, List, Mapping, Optional
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
    page: Optional[int] = None
    author: Optional[str] = None
    year: Optional[int] = None
    formatted: Optional[str] = None

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

    last_error: Exception | None = None
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
            last_error = e
            print(f"API Error/Rate Limit: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    raise RuntimeError("Metadata extraction failed after 3 attempts") from last_error


def requested_citation_style(query: str) -> Optional[str]:
    """Detect whether the user explicitly asked for APA or MLA formatting."""
    lowered = query.lower()
    if re.search(r"\bapa\b", lowered):
        return "APA"
    if re.search(r"\bmla\b", lowered):
        return "MLA"
    return None


def _safe_author(result: Mapping[str, Any]) -> str:
    author = str(result.get("author") or "").strip()
    return author if author else "Unknown Author"


def _format_single_author_apa(author: str) -> str:
    author = author.strip()
    if not author:
        return ""
    if "et al" in author.lower():
        return author

    # Already looks like APA initials, e.g. "Soto, C. J."
    if "," in author and re.search(r"\b[A-Z]\.", author):
        return author

    parts = author.split()
    if len(parts) < 2:
        return author

    suffixes = {"Jr.", "Sr.", "II", "III", "IV"}
    suffix = ""
    if parts[-1] in suffixes:
        suffix = f", {parts.pop()}"

    last_name = parts[-1]
    initials = " ".join(f"{part[0]}." for part in parts[:-1] if part)
    return f"{last_name}, {initials}{suffix}".strip()


def _split_author_names(author: str) -> List[str]:
    normalized = re.sub(r"\s+(?:and|&)\s+", ", ", author.strip())
    names = [name.strip() for name in normalized.split(",") if name.strip()]

    # Preserve already-APA author strings such as "Soto, C. J., & John, O. P."
    if len(names) > 1 and any(re.fullmatch(r"(?:[A-Z]\.\s*)+", name) for name in names[1::2]):
        rebuilt: List[str] = []
        i = 0
        while i < len(names):
            if i + 1 < len(names) and re.fullmatch(r"(?:[A-Z]\.\s*)+", names[i + 1]):
                rebuilt.append(f"{names[i]}, {names[i + 1]}")
                i += 2
            else:
                rebuilt.append(names[i])
                i += 1
        return rebuilt

    return names


def format_authors_apa(author: str) -> str:
    authors = [_format_single_author_apa(name) for name in _split_author_names(author)]
    authors = [author for author in authors if author]

    if not authors:
        return "Unknown Author"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]}, & {authors[1]}"
    return f"{', '.join(authors[:-1])}, & {authors[-1]}"


def _format_single_author_mla_first(author: str) -> str:
    author = author.strip()
    if not author:
        return ""
    if "et al" in author.lower():
        return author

    # Already inverted, e.g. "Bhabha, Homi K."
    if "," in author:
        return author

    parts = author.split()
    if len(parts) < 2:
        return author

    suffixes = {"Jr.", "Sr.", "II", "III", "IV"}
    suffix = ""
    if parts[-1] in suffixes:
        suffix = f", {parts.pop()}"

    last_name = parts[-1]
    rest_of_name = " ".join(parts[:-1])
    return f"{last_name}, {rest_of_name}{suffix}".strip()


def _format_single_author_mla_following(author: str) -> str:
    author = author.strip()
    if not author:
        return ""

    # Convert simple APA initials back to a readable following author form.
    # Example: "John, O. P." -> "O. P. John"
    if "," in author:
        last_name, rest = [part.strip() for part in author.split(",", 1)]
        return f"{rest} {last_name}".strip()

    return author


def format_authors_mla(author: str) -> str:
    authors = [name for name in _split_author_names(author) if name]

    if not authors:
        return "Unknown Author"
    if len(authors) == 1:
        return _format_single_author_mla_first(authors[0])
    if len(authors) == 2:
        first = _format_single_author_mla_first(authors[0])
        second = _format_single_author_mla_following(authors[1])
        return f"{first}, and {second}"

    return f"{_format_single_author_mla_first(authors[0])}, et al."


def _safe_year(result: Mapping[str, Any]) -> str:
    year = result.get("year")
    return str(year) if isinstance(year, int) and year > 0 else "n.d."


def _safe_title(result: Mapping[str, Any]) -> str:
    title = str(result.get("title") or "").strip()
    return title if title else str(result.get("source_file") or "Uploaded document")


def _safe_source(result: Mapping[str, Any]) -> str:
    source = str(result.get("source_file") or "").strip()
    return source if source else "uploaded document"


def _page_number(result: Mapping[str, Any]) -> Optional[int]:
    page_number = result.get("page_number")
    if isinstance(page_number, int) and page_number > 0:
        return page_number
    return None


def _page_locator(result: Mapping[str, Any]) -> str:
    page_number = _page_number(result)
    if page_number is not None:
        return f"p. {page_number}"
    return f"chunk {result.get('chunk_id', 0)}"


def format_location(result: Mapping[str, Any], style: Optional[str]) -> str:
    """Format the retrieved source location in APA or MLA style."""
    author = _safe_author(result)
    year = _safe_year(result)
    title = _safe_title(result)
    source = _safe_source(result)
    locator = _page_locator(result)

    if style == "APA":
        return f"{format_authors_apa(author)}. ({year}). {title}. {source}, {locator}."
    if style == "MLA":
        mla_author = format_authors_mla(author)
        author_period = "" if mla_author.endswith(".") else "."
        return f'{mla_author}{author_period} "{title}." {source}, {year}, {locator}.'
    return f"{author}. {title}. {source}, {locator}."


def build_citation(result: Mapping[str, Any], citation_id: int, style: Optional[str]) -> CitationOut:
    year = result.get("year")
    citation_style = style or "APA"
    return CitationOut(
        id=citation_id,
        doc=_safe_source(result),
        page=_page_number(result),
        author=_safe_author(result),
        year=year if isinstance(year, int) and year > 0 else None,
        formatted=format_location(result, citation_style),
    )


def _citation_key(result: Mapping[str, Any]) -> tuple[str, int | str]:
    source = _safe_source(result)
    page_number = _page_number(result)
    if page_number is not None:
        return (source, page_number)
    return (source, f"chunk:{result.get('chunk_id', 0)}")


def _repair_answer_citation_ids(answer: Optional[str], id_map: Mapping[int, int]) -> Optional[str]:
    if not answer or not id_map:
        return answer

    def replace_bracket(match: re.Match[str]) -> str:
        raw_ids = [int(value) for value in re.findall(r"\d+", match.group(1))]
        repaired_ids: List[int] = []
        for raw_id in raw_ids:
            repaired_id = id_map.get(raw_id, raw_id)
            if repaired_id not in repaired_ids:
                repaired_ids.append(repaired_id)
        return "[" + ", ".join(str(value) for value in repaired_ids) + "]"

    repaired = re.sub(r"\[([\d,\s]+)\]", replace_bracket, answer)
    duplicate_adjacent = re.compile(r"(\[(\d+)\])(?:\s*\[\2\])+")
    while duplicate_adjacent.search(repaired):
        repaired = duplicate_adjacent.sub(r"\1", repaired)
    return repaired


def append_requested_locations(
    answer: Optional[str],
    citations: List[CitationOut],
    results: List[dict],
    style: Optional[str],
) -> Optional[str]:
    if not answer or style is None or not citations:
        return answer

    lines: List[str] = []
    for citation in citations:
        if citation.formatted:
            lines.append(f"[{citation.id}] {citation.formatted}")

    if not lines:
        return answer

    return f"{answer}\n\n{style} location(s):\n" + "\n".join(lines)

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
        (
            f"[{i}] (source: {r['source_file']}, title: {r.get('title', '')}, "
            f"author: {r.get('author', '')}, year: {r.get('year', 0)}, "
            f"page: {r.get('page_number')}, chunk {r['chunk_id']})\n{r['content']}"
        )
        for i, r in enumerate(results, start=1)
    ]
    context = "\n\n".join(context_blocks)
    citation_style = requested_citation_style(query)

    prompt = f"""
    You are a research assistant. Answer the user's question using ONLY the
    numbered context excerpts below - do not use outside knowledge.

    Cite every claim inline using the excerpt number it came from, e.g. [1].
    In your structured "citations" output, each entry's "id" must be one of
    the excerpt numbers above, and "doc" must be copied exactly from that
    excerpt's "source" value.

    If the user asks for APA or MLA format, answer the question normally first.
    The backend will add exact formatted document locations from verified
    retrieval metadata.

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
            seen_citations: dict[tuple[str, int | str], int] = {}
            citation_id_map: dict[int, int] = {}
            for c in parsed.citations:
                idx = c.id - 1
                if 0 <= idx < len(results):
                    citation_key = _citation_key(results[idx])
                    if citation_key in seen_citations:
                        citation_id_map[c.id] = seen_citations[citation_key]
                        continue

                    citation_id_map[c.id] = c.id
                    seen_citations[citation_key] = c.id
                    fixed_citations.append(build_citation(results[idx], c.id, citation_style))
            parsed.citations = fixed_citations
            parsed.answer = _repair_answer_citation_ids(parsed.answer, citation_id_map)
            parsed.answer = append_requested_locations(
                parsed.answer,
                fixed_citations,
                results,
                citation_style,
            )
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
            "source_file": file.filename,
            "pages": [
                {"text": chunk["text"], "page": chunk.get("page")}
                for chunk in extracted_chunks
            ],
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
