from pydantic import BaseModel, Field
from typing import List, Optional


class IngestRequest(BaseModel):
    file_paths: List[str] = Field(default_factory=list, description="Absolute or relative file paths to ingest (PDF/TXT)")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
