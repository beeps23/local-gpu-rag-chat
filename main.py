"""FastAPI service: upload documents and ask the local RAG assistant questions."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .llm import LocalLLM
from .rag import BASE_DIR, RAGPipeline


UPLOAD_DIR = BASE_DIR / "data" / "uploads"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "requests.jsonl"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Local GPU RAG Chat API", version="1.1.0")
rag: RAGPipeline | None = None
llm = LocalLLM()


class ChatRequest(BaseModel):
    question: str
    use_rerank: bool = True  # lets you compare results with and without reranking


def get_rag() -> RAGPipeline:
    global rag
    if rag is None:
        rag = RAGPipeline()
    return rag


def ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def write_log(event: dict) -> None:
    """Append one JSON record per request to logs/requests.jsonl."""
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "local-gpu-rag"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    # Keep only the file name, so an upload can never be written outside the uploads folder.
    filename = Path(file.filename or "document.txt").name
    if Path(filename).suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise HTTPException(400, "Upload a .txt, .md, or .pdf file.")
    destination = UPLOAD_DIR / filename
    destination.write_bytes(await file.read())
    try:
        chunks = get_rag().add_document(destination)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"filename": destination.name, "chunks_indexed": chunks}


@app.delete("/documents")
def clear_documents() -> dict:
    get_rag().clear()
    for path in UPLOAD_DIR.iterdir():
        if path.is_file():
            path.unlink()
    return {"status": "cleared"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        sources, timings = get_rag().retrieve_and_rerank(request.question, use_rerank=request.use_rerank)
        context = "\n\n".join(f"[Source: {item['source']}]\n{item['text']}" for item in sources)
        result = llm.answer(request.question, context)
        event = {
            "request_id": request_id,
            "timestamp": timestamp,
            "question": request.question,
            "use_rerank": request.use_rerank,
            **timings,
            **result["metrics"],
            "total_ms": ms_since(started),
            "sources": [f"{item['source']}#chunk{item.get('chunk_id', '?')}" for item in sources],
            "status": "success",
        }
        write_log(event)
        return {"request_id": request_id, "answer": result["answer"], "sources": sources, "metrics": event}
    except Exception as exc:
        # Failed requests are logged too, so errors are visible in the LLM Ops log.
        write_log({
            "request_id": request_id,
            "timestamp": timestamp,
            "question": request.question,
            "use_rerank": request.use_rerank,
            "total_ms": ms_since(started),
            "status": "error",
            "error": str(exc),
        })
        raise HTTPException(500, f"Chat request failed: {exc}") from exc
