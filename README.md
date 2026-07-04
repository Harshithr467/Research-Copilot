# Research Copilot

Research Copilot is a full-stack AI-powered research assistant. It allows users to upload research articles (PDFs, DOCX, TXT, MD), extracts and indexes their contents into a local Elasticsearch vector database using local embeddings (via Hugging Face `all-MiniLM-L6-v2`), and enables chatting with the documents using Google Gemini models. The backend will query the vector database and provide answers grounded strictly in the context of the uploaded documents.

---

## Architecture Overview

The system consists of three main components:
1. **Frontend (Next.js):** A modern React-based user interface styled with Tailwind CSS, running at `http://localhost:3000`.
2. **Backend (FastAPI):** A high-performance API server running at `http://localhost:8000`. It processes document uploads, runs text extraction pipelines, performs metadata extraction using Gemini, and handles semantic retrieval.
3. **Database (Elasticsearch):** A search engine running at `http://localhost:9200` to index text chunks and their corresponding 384-dimensional dense vectors.

---

## Prerequisites

Before starting, ensure you have the following installed on your machine:
- **Docker & Docker Compose** (for running Elasticsearch)
- **Node.js** (v18+ recommended)
- **Python 3** (3.10+ recommended)
- **Virtual Environment Tool** (e.g., `venv` package)

In addition, you will need a Google Gemini API Key configured in your environment.

---

## Setup Instructions

Follow these steps to set up and run the entire stack.

### 1. Environment Setup

Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Infrastructure Setup (Elasticsearch)

Start the local Elasticsearch instance using Docker Compose:

```bash
docker compose -f indexing/docker-compose.yml up -d
```

> [!WARNING]
> **Initialize the Index Only Once:**
> Run the setup script to create the index schema and map the 384-dimensional vector field. **Do not run this script repeatedly**, as running it again will delete the existing index and wipe all your uploaded documents.
>
> To initialize the index, run:
> ```bash
> # Make sure your Python environment is active (see Backend Setup below)
> python3 indexing/setup_index.py
> ```

### 3. Backend Setup

Set up the Python virtual environment and install the required backend dependencies:

```bash
# 1. Create the virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
# .\venv\Scripts\activate # On Windows

# 3. Install requirements
pip install -r requirements.txt
```

### 4. Frontend Setup

Install the Node.js packages and dependencies for the Next.js frontend:

```bash
npm install
```

---

## Running the Application

Once the steps above are completed, you can start both the Next.js frontend and the FastAPI backend concurrently using a single command:

```bash
npm run dev
```

This runs:
- The Next.js development server at [http://localhost:3000](http://localhost:3000)
- The FastAPI API server with hot-reload enabled at [http://localhost:8000](http://localhost:8000)

Open your browser and navigate to [http://localhost:3000](http://localhost:3000) to begin using Research Copilot.
