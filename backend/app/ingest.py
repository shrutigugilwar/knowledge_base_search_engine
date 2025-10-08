from __future__ import annotations

from typing import List, Dict
from pathlib import Path

from pypdf import PdfReader


def _read_text_from_file(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            text = []
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""
    if path.suffix.lower() in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def ingest_files(file_paths: List[str], store) -> int:
    total = 0
    for p in file_paths:
        path = Path(p).resolve()
        raw = _read_text_from_file(path)
        chunks = _chunk_text(raw)
        if chunks:
            metas: List[Dict] = [{"source": str(path)} for _ in chunks]
            total += store.add_texts(chunks, metas)
    return total
