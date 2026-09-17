"""Streamlit user interface for the local GPU RAG chat application."""

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Local GPU RAG Chat", page_icon="🤖", layout="wide")
st.title("🤖 Local GPU RAG Chat")
st.caption("FastAPI • Streamlit • Hugging Face • GPU inference • retrieval + reranking • LLM Ops")


def error_detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def format_details(payload: dict) -> str:
    """Build the sources + metrics line shown under each answer."""
    metrics = payload["metrics"]
    parts = []
    for source in payload["sources"]:
        label = f"{source['source']} (chunk {source.get('chunk_id', '?')}"
        if source.get("rerank_score") is not None:
            label += f", rerank {source['rerank_score']:.2f})"
        else:
            label += f", similarity {source['retrieval_score']:.2f})"
        parts.append(label)
    sources_line = "**Sources:** " + (", ".join(parts) if parts else "none found")

    timing_line = (
        f"⏱️ Total {metrics['total_ms'] / 1000:.2f}s · "
        f"retrieval {metrics['retrieval_ms']:.0f} ms · "
        f"rerank {metrics['rerank_ms']:.0f} ms · "
        f"generation {metrics['generation_ms'] / 1000:.2f}s · "
        f"{metrics['completion_tokens']} tokens · "
        f"{metrics['tokens_per_second']:.1f} tok/s · "
        f"peak VRAM {metrics['peak_vram_mb']:.0f} MB"
    )
    if metrics.get("model_load_ms"):
        timing_line += f" · first-run model load {metrics['model_load_ms'] / 1000:.1f}s"
    return f"{sources_line}  \n{timing_line}"


with st.sidebar:
    st.header("Add knowledge")
    file = st.file_uploader("Upload a TXT, Markdown, or PDF file", type=["txt", "md", "pdf"])
    if file and st.button("Index document"):
        with st.spinner("Extracting text and creating embeddings…"):
            try:
                response = requests.post(
                    f"{API_URL}/documents", files={"file": (file.name, file.getvalue())}, timeout=180
                )
                if response.ok:
                    st.success(f"Indexed {response.json()['chunks_indexed']} chunks from {file.name}.")
                else:
                    st.error(error_detail(response))
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend. Start FastAPI first. Details: {exc}")

    st.divider()
    st.header("Settings")
    use_rerank = st.toggle(
        "Use cross-encoder reranking",
        value=True,
        help="Turn off to compare answers using vector search alone.",
    )
    if st.button("Clear all documents"):
        try:
            response = requests.delete(f"{API_URL}/documents", timeout=60)
            if response.ok:
                st.session_state.messages = []
                st.success("All documents and chat history cleared.")
            else:
                st.error(error_detail(response))
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend. Details: {exc}")

    st.divider()
    st.markdown("**How it works**\n\nQuestion → retrieve → rerank → context → local model → answer")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("details"):
            st.caption(message["details"])

question = st.chat_input("Ask a question about your uploaded documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating locally on GPU…"):
            try:
                response = requests.post(
                    f"{API_URL}/chat", json={"question": question, "use_rerank": use_rerank}, timeout=600
                )
                if not response.ok:
                    st.error(error_detail(response))
                else:
                    payload = response.json()
                    answer = payload["answer"]
                    details = format_details(payload)
                    st.markdown(answer)
                    st.caption(details)
                    if payload["metrics"].get("hit_token_limit"):
                        st.warning("This answer reached the token limit and may be cut off.")
                    with st.expander("Full LLM Ops record"):
                        st.json(payload["metrics"])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "details": details}
                    )
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend. Start FastAPI first. Details: {exc}")
