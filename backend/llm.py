"""Local Hugging Face model loader and answer generator."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Maximum number of tokens the model may write in one answer.
# 350 was cutting answers off mid-sentence, so this is raised to 512.
MAX_NEW_TOKENS = 512

NOT_FOUND_MESSAGE = "I couldn't find that in the uploaded documents."

SYSTEM_PROMPT = (
    "You are a helpful document assistant.\n"
    "Rules:\n"
    "1. Answer using ONLY the information in the context.\n"
    "2. Do not use outside knowledge, even if you know the answer.\n"
    f"3. If the context does not contain the answer, reply with exactly: {NOT_FOUND_MESSAGE}\n"
    "4. When you do answer, end with a line in the form 'Source: <file name>'.\n"
    "5. Keep the answer clear and concise."
)


def ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


class LocalLLM:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None

    def _load(self) -> float:
        """Load the model onto the GPU the first time it is needed.

        Returns how long loading took in milliseconds (0 if already loaded), so the
        one-time loading cost is reported separately from normal generation time.
        """
        if self.model is not None:
            return 0.0
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map="cuda",
        )
        self.model.eval()
        return ms_since(started)

    def answer(self, question: str, context: str) -> dict:
        """Generate an answer and return it together with inference metrics."""
        model_load_ms = self._load()
        assert self.tokenizer is not None and self.model is not None

        # No documents matched, so skip the model and give the not-found reply directly.
        if not context:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "metrics": {
                    "model_load_ms": model_load_ms,
                    "generation_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0.0,
                    "peak_vram_mb": 0.0,
                    "hit_token_limit": False,
                },
            }

        user = f"Context:\n{context}\n\nQuestion: {question}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs.input_ids.shape[1])

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        with torch.inference_mode():
            # do_sample=False makes answers deterministic: the same question and
            # context always produce the same answer, which makes testing fair.
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        if use_cuda:
            torch.cuda.synchronize()
        generation_ms = ms_since(started)

        generated = output[0][prompt_tokens:]
        completion_tokens = int(generated.shape[0])
        seconds = generation_ms / 1000
        # Peak GPU memory used by PyTorch in the backend process during this answer.
        # This includes the LLM, embedding model, and reranker, which share the process.
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if use_cuda else 0.0

        return {
            "answer": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "metrics": {
                "model_load_ms": model_load_ms,
                "generation_ms": generation_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_second": round(completion_tokens / seconds, 1) if seconds > 0 else 0.0,
                "peak_vram_mb": peak_vram_mb,
                "hit_token_limit": completion_tokens >= MAX_NEW_TOKENS,
            },
        }
