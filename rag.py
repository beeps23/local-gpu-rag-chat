"""Document ingestion, vector retrieval, and reranking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "vectors.faiss"
METADATA_PATH = DATA_DIR / "chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunk size is measured in words.
# all-MiniLM-L6-v2 only reads the first 256 tokens (roughly 190 words) of any text.
# The previous 700-word chunks meant most of every chunk was ignored during search.
# 150 words (about 200 tokens) fits fully inside that limit.
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

TOP_K = 8    # how many candidate chunks FAISS retrieves
FINAL_K = 4  # how many chunks are kept and sent to the model


def ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def split_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word chunks.

    Overlap means the end of one chunk is repeated at the start of the next, so an
    answer that falls on a boundary still appears whole in at least one chunk.
    """
    words = text.replace("\n", " ").split()
    if not words:
        return []
    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


class RAGPipeline:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        device = "cuda"
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        # Loaded up front so the first question's rerank timing isn't inflated by loading.
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)
        self.metadata: list[dict[str, Any]] = []
        # IndexFlatIP + normalized embeddings = exact cosine-similarity search.
        self.index: faiss.IndexFlatIP | None = None
        self._load_saved_index()

    def _load_saved_index(self) -> None:
        if INDEX_PATH.exists() and METADATA_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(INDEX_PATH))
        METADATA_PATH.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def add_document(self, path: Path) -> int:
        if any(item["source"] == path.name for item in self.metadata):
            raise ValueError(f"{path.name} is already indexed. Clear all documents first to re-index it.")

        text = extract_text(path)
        chunks = split_text(text)
        if not chunks:
            raise ValueError("No readable text was found in this file.")

        vectors = self.embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.metadata.extend(
            {"source": path.name, "chunk_id": chunk_id, "text": chunk} for chunk_id, chunk in enumerate(chunks)
        )
        self._save()
        return len(chunks)

    def clear(self) -> None:
        """Remove every indexed document."""
        self.index = None
        self.metadata = []
        INDEX_PATH.unlink(missing_ok=True)
        METADATA_PATH.unlink(missing_ok=True)

    def retrieve_and_rerank(
        self,
        question: str,
        top_k: int = TOP_K,
        final_k: int = FINAL_K,
        use_rerank: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Return the best chunks for a question, plus separate retrieval and rerank timings."""
        timings = {"retrieval_ms": 0.0, "rerank_ms": 0.0}
        if self.index is None or not self.metadata:
            return [], timings

        # Stage 1: fast vector search finds a broad set of similar chunks.
        started = time.perf_counter()
        query = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        scores, indices = self.index.search(np.asarray(query, dtype="float32"), min(top_k, len(self.metadata)))
        candidates = [
            {**self.metadata[idx], "retrieval_rank": rank, "retrieval_score": float(score), "rerank_score": None}
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1)
            if idx >= 0
        ]
        timings["retrieval_ms"] = ms_since(started)

        if not use_rerank:
            return candidates[:final_k], timings

        # Stage 2: the cross-encoder reads question + chunk together and re-scores them precisely.
        started = time.perf_counter()
        rerank_scores = self.reranker.predict([(question, item["text"]) for item in candidates])
        for item, score in zip(candidates, rerank_scores):
            item["rerank_score"] = float(score)
        results = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)[:final_k]
        timings["rerank_ms"] = ms_since(started)
        return results, timings
