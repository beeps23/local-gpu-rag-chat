# Evaluation summary

30 questions, each asked with reranking ON and OFF.

| Metric | Rerank ON | Rerank OFF |
|---|---|---|
| Correct answers (answerable) | 25/25 (100%) | 25/25 (100%) |
| Correct refusals (unanswerable) | 4/5 (80%) | 4/5 (80%) |
| Answer found in retrieved context | 25/25 (100%) | 25/25 (100%) |
| Answer is in the #1 ranked chunk | 17/25 (68%) | 16/25 (64%) |
| Average position of the answer chunk | 1.48 | 1.80 |
| Top source is the right document | 22/25 (88%) | 22/25 (88%) |
| Median total latency | 4.18 s | 4.53 s |
| Average retrieval time | 27 ms | 31 ms |
| Average rerank time | 46 ms | 0 ms |
| Average generation speed | 4.5 tokens/s | 4.2 tokens/s |
| Peak VRAM | 3.23 GB | 3.22 GB |

## Where reranking changed the position of the answer chunk

- Position 3 without reranking -> 2 with reranking: Which embedding model does the chat assistant use?
- Position 2 without reranking -> 3 with reranking: What GPU does the local RAG chatbot run on?
- Position 4 without reranking -> 2 with reranking: What does the chat assistant's LLM Ops logging record?
- Position 1 without reranking -> 2 with reranking: Why does the chat assistant run the model locally instead of calling a cloud API?
- Position 4 without reranking -> 1 with reranking: What future improvements are planned for the chat assistant?
- Position 4 without reranking -> 2 with reranking: What GPU do Northwind engineers get in their laptops?
- Position 1 without reranking -> 2 with reranking: How much is the Northwind learning budget per year?
- Position 4 without reranking -> 3 with reranking: Which password manager does Northwind use?
- Position 2 without reranking -> 1 with reranking: How quickly must the primary Orbit on-call engineer acknowledge a page?
- Position 2 without reranking -> 1 with reranking: What is the maximum number of inference workers in Orbit production?

## Answers marked incorrect (check these by hand)

- [ON] What version of Kubernetes does the Orbit platform run? -> 'The Orbit platform uses Kubernetes versions compatible with the deployment environment described in the provided documentation. Specifically:\n\n- The p'
- [OFF] What version of Kubernetes does the Orbit platform run? -> 'The Orbit platform uses Kubernetes versions compatible with the deployment environment described in the provided documentation. Specifically:\n\n- The p'
