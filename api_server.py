import os
import shutil
from fastapi import FastAPI, UploadFile, File
import uvicorn
from contextlib import asynccontextmanager

# Import real extraction and ingestion functions
from extraction.extractor import extract
from indexing.ingester import ingest_paper

# Installation instructions:
# pip install fastapi uvicorn python-multipart pymupdf python-docx

UPLOAD_DIR = "uploaded_files"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the upload directory exists on startup."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        print(f"Created directory: {UPLOAD_DIR}")
    yield

app = FastAPI(title="Research-Copilot API", lifespan=lifespan)

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
    Upload a PDF file, save it locally, and trigger real text extraction.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Securely save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Call the real text extraction function
    try:
        extracted_chunks = extract(file_path)
        
        # Note: ingest_paper from indexing.ingester will be integrated 
        # in the next phase after verification.
        
        return {
            "filename": file.filename,
            "status": "success",
            "extracted_chunks_count": len(extracted_chunks),
            "data": extracted_chunks
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "status": "error",
            "message": str(e)
        }

# Testing Instructions:
# 1. Run the server: python api_server.py
# 2. Open your browser and navigate to http://127.0.0.1:8000/docs
# 3. Use the interactive Swagger UI to test the /upload-pdf endpoint by uploading a PDF file.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
