"""Diagnostic: finds which part of the app slows down generation.

It runs the same 100-token generation under different conditions, changing one
thing at a time. Stop the backend before running this.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from backend.llm import SYSTEM_PROMPT, LocalLLM

N_TOKENS = 100
QUESTION = "What are the limitations and future work?"
SHORT_CONTEXT = "The project uses a compact local model and could add conversational memory later."

print("=" * 60)
print("Loading the language model (same code as the app)...")
llm = LocalLLM()
llm._load()
tokenizer, model = llm.tokenizer, llm.model


def measure(context: str) -> str:
    user = f"Context:\n{context}\n\nQuestion: {QUESTION}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=N_TOKENS,
            min_new_tokens=N_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return f"{N_TOKENS / seconds:5.1f} tokens/sec   (prompt: {inputs.input_ids.shape[1]} tokens)"


def wake_gpu(seconds: float = 5.0) -> None:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        a @ a
        torch.cuda.synchronize()


# Tiny warm-up so first-call setup doesn't count.
with torch.inference_mode():
    model.generate(**tokenizer("Hi", return_tensors="pt").to(model.device), max_new_tokens=3)

print("\nWaiting 15 seconds so the GPU goes idle, like between chat questions...")
time.sleep(15)
print("\nTEST 1  GPU idle first, short prompt:       ", measure(SHORT_CONTEXT))

print("\nWaking the GPU with 5 seconds of heavy math...")
wake_gpu()
print("TEST 2  GPU awake, short prompt:            ", measure(SHORT_CONTEXT))

brief = Path("project_brief.md")
words = brief.read_text(encoding="utf-8").split() if brief.exists() else SHORT_CONTEXT.split() * 60
long_context = " ".join(words[:600])
print("TEST 3  Long prompt (like the app):         ", measure(long_context))

print("\nLoading search + reranker models (same code as the app)...")
from backend.rag import RAGPipeline  # noqa: E402

rag = RAGPipeline()
sources, _ = rag.retrieve_and_rerank(QUESTION)
rag_context = "\n\n".join(f"[Source: {item['source']}]\n{item['text']}" for item in sources)
if not rag_context:
    rag_context = long_context
print("TEST 4  After search + rerank, real context:", measure(rag_context))

with ThreadPoolExecutor(max_workers=1) as pool:
    result = pool.submit(measure, rag_context).result()
print("TEST 5  Same, in a background thread:       ", result)

print("\nWaiting 15 seconds so the GPU goes idle again...")
time.sleep(15)
print("TEST 6  Idle again, real context:           ", measure(rag_context))
print("=" * 60)
