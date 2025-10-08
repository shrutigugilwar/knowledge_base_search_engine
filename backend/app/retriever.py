from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None

from sentence_transformers import SentenceTransformer


class DocumentStore:
    """
    Minimal in-memory vector store using FAISS (if available) with
    SentenceTransformer embeddings for retrieval.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = None  # Fallback to linear search

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype("float32")

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] | None = None) -> int:
        if not texts:
            return 0
        if metadatas is None:
            metadatas = [{} for _ in texts]
        embs = self._embed(texts)

        if self.index is not None:
            self.index.add(embs)
        # For linear fallback search
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        if not query:
            return [], [], []
        q = self._embed([query])

        if self.index is not None and len(self.texts) > 0:
            D, I = self.index.search(q, min(top_k, len(self.texts)))
            idxs = I[0].tolist()
            scores = D[0].tolist()
            texts = [self.texts[i] for i in idxs]
            metas = [self.metadatas[i] for i in idxs]
            return texts, scores, metas

        # Linear fallback if FAISS isn't available
        if len(self.texts) == 0:
            return [], [], []
        all_embs = self._embed(self.texts)
        # cosine similarity since embeddings are normalized -> dot product
        sims = (q @ all_embs.T)[0]
        idxs = np.argsort(-sims)[:top_k].tolist()
        scores = sims[idxs].tolist()
        texts = [self.texts[i] for i in idxs]
        metas = [self.metadatas[i] for i in idxs]
        return texts, scores, metas
