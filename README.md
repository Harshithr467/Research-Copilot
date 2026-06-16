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
docker compose -f backend/docker-compose.yml up -d
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
python backend/check_es_health.py
```

## Local Architecture
- **Elasticsearch:** `http://localhost:9200`
- **Security:** Disabled (`xpack.security.enabled=false`)
- **Version:** 8.12.0 (Matches client library version)
- **Configuration:** All backend and Elasticsearch-related files are located in the `backend/` directory.

## RAG Chat Endpoint

The first backend RAG endpoint is available at `POST /chat`. It retrieves chunks for a `document_id`, builds a grounded prompt, asks Gemini for structured JSON, and validates that returned citations point to retrieved chunks with quotes present in the chunk text.

```json
{
  "document_id": "paper_01",
  "question": "What key elements does the author introduce on page 1?"
}
```

Example local run:

```sh
docker compose -f backend/docker-compose.yml up -d
python backend/setup_index.py
python backend/ingest_chunked_dummy_data.py
cd backend
uvicorn app:app --reload
```

Then call:

```sh
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"document_id\":\"paper_01\",\"question\":\"What key elements does the author introduce on page 1?\"}"
```

Notes:
- Set `GOOGLE_API_KEY` before ingestion and chat if you want vector embeddings and LLM answers.
- `metadata.page_number` is optional. Exact page citations are returned only when ingestion receives real page-level metadata; the generic raw-content ingester does not guess page numbers.

