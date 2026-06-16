# Research Copilot Backend

This project uses Elasticsearch and LangChain to provide a research assistance backend.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.10 or higher

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Harshithr467/Research-Copilot.git
cd Research-Copilot
```

### 2. Start Elasticsearch
We use Docker Compose to run a local Elasticsearch instance (v8.12.0) with security disabled for development.

```sh
docker compose -f indexing/docker-compose.yml up -d
```
*Wait a few seconds for the service to initialize.*

### 3. Setup Python Environment
Create a virtual environment and install the required dependencies:

```sh
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/macOS
# .\venv\Scripts\activate # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Verify Connection
Run the health check script to ensure everything is working correctly:

```bash
python indexing/check_es_health.py
```

## Local Architecture
- **Elasticsearch:** `http://localhost:9200`
- **Security:** Disabled (`xpack.security.enabled=false`)
- **Version:** 8.12.0 (Matches client library version)

## Backend API Server
To run the local API server for testing uploads:

1.  Ensure your virtual environment is active.
2.  Run `pip install -r requirements.txt` to get the latest server libraries.
3.  Start the server by running: `uvicorn api_server:app --reload`
4.  Open your browser and navigate to `http://localhost:8000/docs` to use the interactive upload UI.

The endpoint for PDF uploads is `POST /upload-pdf`. Uploaded files are stored in the `uploaded_files/` directory, and text is automatically extracted using the `extraction` module.

- **Configuration:** All Elasticsearch-related files are located in the `indexing/` directory.
- **Extraction:** PDF and DOCX text extraction logic is in the `extraction/` directory.

## Embedding & Indexing Strategy

### Local Embeddings (No Rate Limits)
The ingestion pipeline has been optimized to use a local, open-source embedding model (**all-MiniLM-L6-v2**) via `langchain-huggingface`. 
- **Bypasses API Quotas:** Unlike the Google GenAI API, there are no rate limits (e.g., 100 RPM) or costs for indexing.
- **Dimensions:** This model produces **384-dimensional** vectors.
- **Background Loading:** The model is initialized in a separate thread when the server starts. This allows the API server to be available **instantly**, while the heavy model loading happens in the background. If you attempt an upload before loading is complete, the process will gracefully wait for the model to be ready.

### Index Configuration
If you are setting up the index for the first time or moving from the previous 768-dim model, you **must** re-run the index setup script to align with the 384-dim mapping:

```bash
# Re-create the index with 384 dimensions
python3 indexing/setup_index.py
```
