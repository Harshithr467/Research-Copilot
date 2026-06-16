import os
import shutil
import time
from typing import List
from fastapi import FastAPI, UploadFile, File
import uvicorn
from contextlib import asynccontextmanager
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

# Import real extraction and ingestion functions
from extraction.extractor import extract
from indexing.ingester import ingest_paper, ResearchPaper

# Load environment variables
load_dotenv()

# Configure Google GenAI Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class PaperMetadata(BaseModel):
    title: str
    author: str
    year: int

UPLOAD_DIR = "uploaded_files"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the upload directory exists on startup."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        print(f"Created directory: {UPLOAD_DIR}")
    yield

app = FastAPI(title="Research-Copilot API", lifespan=lifespan)

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
