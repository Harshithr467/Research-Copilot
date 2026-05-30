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
docker compose up -d
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
python check_es_health.py
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

The endpoint for PDF uploads is `POST /upload-pdf`.
