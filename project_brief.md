# Local GPU RAG Chat Assistant

## Problem statement

Build a chat application that runs a Hugging Face language model locally on a GPU. The system must use a Streamlit frontend, a FastAPI backend, a Retrieval-Augmented Generation (RAG) pipeline, reranking, contextual prompting, and LLM Ops observability.

## Solution overview

This project is a local document question-answering assistant. A user uploads a TXT, Markdown, or PDF document and asks a question. Instead of relying only on the language model's general knowledge, the application retrieves relevant parts of the uploaded material and gives them to the model as context. This helps answers stay grounded in the supplied documents.

## Architecture

1. **Streamlit frontend** provides the chat interface and a document uploader.
2. **FastAPI backend** accepts document uploads and chat requests.
3. **Ingestion** extracts document text and divides it into overlapping chunks.
4. **Embedding model** converts each chunk and each user question into numerical vectors. The project uses `all-MiniLM-L6-v2`.
5. **FAISS retrieval** searches the vector index for the most similar chunks.
6. **Cross-encoder reranking** scores the retrieved candidates more precisely and retains the strongest evidence.
7. **Context construction** combines the selected chunks with source labels.
8. **Prompting and generation** send the question plus context to `Qwen/Qwen2.5-1.5B-Instruct`, a Hugging Face model running locally on the NVIDIA GPU.
9. **LLM Ops logging** records a request ID, source files, retrieval latency, generation latency, total latency, and request status.

## Why retrieval and reranking are both used

Vector retrieval is fast and identifies a broad set of semantically related chunks. It can occasionally include chunks that are similar but not directly useful. Reranking uses a cross-encoder that reads the question and candidate chunk together, allowing it to make a more accurate relevance decision. The highest-ranked chunks form the final context for the model.

## Local deployment

The chatbot uses an NVIDIA GeForce RTX 5060 Laptop GPU through CUDA-enabled PyTorch. The model is downloaded from Hugging Face and runs on the laptop rather than calling a paid cloud API. This makes the architecture private, self-contained, and suitable for an offline-first deployment after the initial downloads.

## Testing approach

The application is tested by uploading this brief and asking questions whose answers appear in it. The UI exposes the retrieved source file, rerank score, and timing metrics. A correct answer paired with the matching source demonstrates that the response was grounded in retrieved context.

## Limitations and future work

The current version uses a compact model for practical local inference. Future improvements could include conversational memory, support for more file types, a persistent database for logs, user authentication, evaluation datasets, and a larger model for improved response quality.
