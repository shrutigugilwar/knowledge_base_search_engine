from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models.schemas import IngestRequest, QueryRequest, QueryResponse
from .retriever import DocumentStore
from .ingest import ingest_files

app = FastAPI(title="Knowledge-base Search Engine", version="0.1.0")

store = DocumentStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.file_paths:
        raise HTTPException(status_code=400, detail="file_paths is required")
    count = ingest_files(req.file_paths, store)
    return {"ingested_documents": count}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="question is required")
    texts, scores, metas = store.search(req.question, top_k=req.top_k or 5)
    if not texts:
        return QueryResponse(answer="No documents indexed yet or no relevant results found.", sources=[])

    context = "\n\n".join(texts)
    # Placeholder synthesis: truncate concatenated context
    answer = f"Using indexed docs, a succinct answer might be: {context[:800]}"
    sources = [m.get("source", "unknown") for m in metas]
    return QueryResponse(answer=answer, sources=sources)
