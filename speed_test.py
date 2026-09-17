"""Speed test: finds out what is slowing down local generation.

Run it with the backend STOPPED, so the GPU isn't shared with the app.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def sync() -> None:
    torch.cuda.synchronize()


print("=" * 60)
print("1. SETUP")
print("   PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("   GPU:", torch.cuda.get_device_name(0))

print("\n2. BIG GPU MATH (tests the GPU itself)")
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(3):
    a @ b
sync()
start = time.perf_counter()
for _ in range(50):
    a @ b
sync()
print(f"   50 large matrix multiplies: {time.perf_counter() - start:.2f}s")

print("\n3. TINY GPU OPERATIONS (tests how fast the CPU can send work to the GPU)")
x = torch.ones(16, device="cuda")
for _ in range(200):
    x = x + 1
sync()
start = time.perf_counter()
for _ in range(5000):
    x = x + 1
sync()
seconds = time.perf_counter() - start
print(f"   5000 tiny operations: {seconds:.2f}s ({5000 / seconds:,.0f} ops/sec)")

print("\n4. MODEL SETUP")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print("   Parameters are on:", sorted({str(p.device) for p in model.parameters()}))
print("   Data type:", next(model.parameters()).dtype)
print("   Attention:", getattr(model.config, "_attn_implementation", "unknown"))

messages = [{"role": "user", "content": "Explain in detail how a computer processor works."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


def generate(n_tokens: int) -> tuple[int, float]:
    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            min_new_tokens=n_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    sync()
    return output.shape[1] - inputs.input_ids.shape[1], time.perf_counter() - started


print("\n5. GENERATION SPEED")
generate(10)  # warm-up, not measured
tokens, seconds = generate(100)
print(f"   {tokens} tokens in {seconds:.2f}s = {tokens / seconds:.1f} tokens/sec")
print("=" * 60)
