import os
import shutil
from fastapi import FastAPI, UploadFile, File
import uvicorn

from contextlib import asynccontextmanager

# Installation instructions:
# pip install fastapi uvicorn python-multipart

UPLOAD_DIR = "temp_uploads"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the upload directory exists on startup."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        print(f"Created directory: {UPLOAD_DIR}")
    yield

app = FastAPI(title="Research-Copilot API", lifespan=lifespan)

def mock_text_extraction(file_path: str) -> dict:
    """
    Mock function to simulate text extraction from a PDF.
    Matches the ResearchPaper structure from ingester.py.
    """
    print(f"Processing file for extraction: {file_path}")
    
    # Dummy ResearchPaper dictionary
    return {
        "title": "Mocked Research Paper Title",
        "content": "This is some extracted text from the mock function. In a real scenario, this would be the actual content of the PDF.",
        "author": "Mock Author",
        "year": 2024,
        "source_file": os.path.basename(file_path)
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, save it locally, and trigger (mock) text extraction.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Securely save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Mock text extraction call
    # Note: This is where actual text extraction and ingest_paper functions 
    # from ingester.py will be called later.
    extracted_data = mock_text_extraction(file_path)
    
    return {
        "filename": file.filename,
        "status": "success",
        "mock_extracted_data": extracted_data
    }

# Testing Instructions:
# 1. Run the server: python api_server.py
# 2. Open your browser and navigate to http://127.0.0.1:8000/docs
# 3. Use the interactive Swagger UI to test the /upload-pdf endpoint by uploading a PDF file.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
