# Knowledge-base Search Engine (RAG)

Objective: Search across documents and provide synthesized answers using LLM-based Retrieval-Augmented Generation (RAG).

## Features
- Ingest multiple text/PDF documents
- Query endpoint returning synthesized answer (placeholder synthesis)
- Simple vector retrieval via Sentence-Transformers + FAISS (fallback to linear search if FAISS not available)

## Tech
- FastAPI backend
- Sentence-Transformers for embeddings
- FAISS for vector search (optional; falls back to linear search)

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Then:
- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs

## Ingest Documents
Provide file paths to TXT or PDF files:

```powershell
curl -X POST http://localhost:8000/ingest \ 
  -H "Content-Type: application/json" \ 
  -d '{"file_paths": ["docs/sample.pdf", "docs/notes.txt"]}'
```

## Query
Ask a question about the ingested documents:

```powershell
curl -X POST http://localhost:8000/query \ 
  -H "Content-Type: application/json" \ 
  -d '{"question": "Summarize the key points", "top_k": 5}'
```

## LLM Usage Guidance
Prompt example: "Using these documents, answer the user’s question succinctly."  
Note: The current implementation returns a placeholder synthesis from retrieved context. You can integrate your preferred LLM (e.g., OpenAI, Azure OpenAI, local models) by replacing the placeholder in `backend/app/main.py` with a call that uses an API key from environment variables.

## Repository Structure
- `backend/app/main.py` – FastAPI app with `/ingest` and `/query`
- `backend/app/ingest.py` – Ingestion utilities (PDF/TXT) + chunking
- `backend/app/retriever.py` – Embedding + vector store wrapper
- `backend/app/models/schemas.py` – Pydantic request/response models
- `docs/` – Place demo assets, notes, and any evaluation materials

## Evaluation Focus
- Retrieval accuracy
- Synthesis quality (once an LLM is integrated)
- Code structure
- LLM integration

## Environment Variables
Copy `.env.example` to `.env` and populate secrets as needed.
