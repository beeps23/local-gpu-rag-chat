"""Evaluation: asks the app a fixed set of questions and scores the results.

Every question is asked twice, with reranking ON and OFF. Results are saved to
eval/results.csv (open it in Excel) and a summary is saved to eval/summary.md.

The backend must be running before you start this script.
"""

from __future__ import annotations

import csv
import re
import statistics
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
DOCS_DIR = EVAL_DIR / "docs"

# ---------------------------------------------------------------------------
# Test documents. These are fictional, so the model cannot answer questions
# about them from general knowledge: correct answers must come from retrieval.
# Some details deliberately resemble the project brief (a different GPU, request
# IDs, latency) to act as distractors for retrieval.
# ---------------------------------------------------------------------------

HANDBOOK = """
# Northwind Robotics Employee Handbook

## Working hours
Core collaboration hours at Northwind Robotics are 10:00 to 15:00 local time. Outside core hours, employees may arrange their schedules freely, provided they work a total of 40 hours per week. Team meetings should not be scheduled on Fridays after 13:00, which the company reserves as focus time.

## Remote work
Employees may work remotely up to three days per week. Fully remote arrangements require written approval from both the employee's manager and the head of people operations. Remote employees receive a one-time home office stipend of 450 euros, which can be used for a desk, chair, monitor, or lighting.

## Leave
Full-time employees receive 26 days of paid annual leave, plus public holidays. Unused leave of up to 5 days may be carried into the next calendar year; any remaining days expire on 31 March. Parental leave is 20 weeks at full pay for all new parents. Sick leave does not require a doctor's note for absences of three days or fewer.

## Equipment
Engineers receive a laptop with an NVIDIA RTX 4070 GPU and 32 GB of RAM. Design and operations staff receive a 14-inch ultrabook. Laptops are replaced every three years. Lost or damaged equipment must be reported to the IT help desk within 24 hours through the ticket system, not by email.

## Learning budget
Every employee has an annual learning budget of 1,200 euros for courses, certifications, books, and conference tickets. Conference travel is funded separately from the learning budget, but requires manager approval at least six weeks before the event. Employees who complete a cloud or Kubernetes certification receive an additional bonus of 300 euros.

## Expenses
Expenses must be submitted within 30 days using the Ledgerly app. Meals during business travel are reimbursed up to 55 euros per day. Taxi rides are reimbursed only when public transport is unavailable or the trip happens between 23:00 and 06:00. Alcohol is never reimbursed.

## Security
Passwords must be at least 14 characters long and are stored only in the company password manager, Vaultkeep. Multi-factor authentication is mandatory for all company accounts. Customer data may never be copied to personal devices or uploaded to external AI tools without approval from the security team.

## Onboarding
New employees are paired with an onboarding buddy for their first 60 days. During the first week, every new hire completes a security training module and a robotics safety course. The onboarding process ends with a check-in conversation with the employee's manager at the end of month two.
"""

RUNBOOK = """
# Orbit Platform Runbook

## Overview
Orbit is the internal platform that serves machine learning models for the Northwind Robotics fleet. It runs a FastAPI gateway in front of a pool of inference workers. Each request receives a request ID at the gateway, and that ID is attached to every log line the request produces.

## Clusters
Orbit runs on two Kubernetes clusters. The production cluster, called orbit-prod, has 12 GPU worker nodes, each with four NVIDIA L4 GPUs. The staging cluster, orbit-stage, has 3 GPU worker nodes with one L4 GPU each. All deployments reach staging first and must run there for at least 24 hours before promotion to production.

## Service level objectives
The p95 latency target for the gateway is 800 milliseconds for embedding requests and 4 seconds for text generation requests. The monthly availability target is 99.9 percent. If the error rate stays above 2 percent for 10 minutes, the on-call engineer is paged automatically.

## Logging and metrics
Logs are shipped to a central log store and kept for 30 days. Metrics include request count, error rate, queue time, time to first token, and GPU memory usage. Dashboards for Orbit live in the observability workspace under the folder named Orbit Serving.

## Deployments
Model deployments use a canary strategy. A new model version first receives 5 percent of traffic for one hour. If error rate and latency stay within the service level objectives, traffic moves to 50 percent and then to 100 percent. Rollbacks are performed with the command orbitctl rollback followed by the service name.

## On-call
The on-call rotation changes every Monday at 09:00. The primary on-call engineer must acknowledge a page within 10 minutes. If the primary does not respond, the page escalates to the secondary engineer after 15 minutes. Incidents rated severity 1 require a written postmortem within five working days.

## Capacity and scaling
Inference workers scale horizontally based on queue depth. When the average queue time exceeds 500 milliseconds for five minutes, the autoscaler adds workers, up to a maximum of 40 workers in production. GPU memory usage above 90 percent on any node triggers a warning alert.
"""

# ---------------------------------------------------------------------------
# Questions: (question, type, expected source file, phrase that must appear in
# the retrieved context, answer keywords).
# Answer keywords: ";" separates words that must ALL appear in the answer,
# "|" separates accepted alternatives.
# ---------------------------------------------------------------------------

BRIEF = "project_brief.md"
HB = "northwind_handbook.md"
RB = "orbit_platform_runbook.md"

QUESTIONS = [
    # Project brief
    ("Which embedding model does the chat assistant use?", "answerable", BRIEF, "all-MiniLM-L6-v2", "MiniLM"),
    ("Which language model generates answers in the chat assistant?", "answerable", BRIEF, "Qwen2.5-1.5B-Instruct", "Qwen"),
    ("What GPU does the local RAG chatbot run on?", "answerable", BRIEF, "RTX 5060", "5060"),
    ("What does the cross-encoder reranker do in the chat assistant?", "answerable", BRIEF, "reads the question and candidate chunk together", "together|jointly|pair"),
    ("What does the chat assistant's LLM Ops logging record?", "answerable", BRIEF, "retrieval latency", "latency"),
    ("What file types can users upload to the chat assistant?", "answerable", BRIEF, "TXT, Markdown, or PDF", "PDF"),
    ("Why does the chat assistant use vector retrieval before reranking?", "answerable", BRIEF, "Vector retrieval is fast", "fast|quick|speed"),
    ("Why does the chat assistant run the model locally instead of calling a cloud API?", "answerable", BRIEF, "paid cloud API", "priva|offline|self-contained|cost|paid"),
    ("How is the chat assistant tested?", "answerable", BRIEF, "uploading this brief", "upload"),
    ("What future improvements are planned for the chat assistant?", "answerable", BRIEF, "conversational memory", "memory"),
    # Handbook
    ("How many days of paid annual leave do Northwind employees get?", "answerable", HB, "26 days of paid annual leave", "26"),
    ("What GPU do Northwind engineers get in their laptops?", "answerable", HB, "RTX 4070", "4070"),
    ("How much is the Northwind learning budget per year?", "answerable", HB, "learning budget of 1,200 euros", "1200"),
    ("How long is parental leave at Northwind?", "answerable", HB, "Parental leave is 20 weeks", "20 weeks|twenty weeks"),
    ("What is the daily meal reimbursement limit during Northwind business travel?", "answerable", HB, "55 euros per day", "55"),
    ("Which password manager does Northwind use?", "answerable", HB, "Vaultkeep", "Vaultkeep"),
    ("How many days per week can Northwind employees work remotely?", "answerable", HB, "three days per week", "three|3"),
    ("How long does the Northwind onboarding buddy program last?", "answerable", HB, "first 60 days", "60"),
    # Runbook
    ("How many GPU worker nodes does the orbit-prod cluster have?", "answerable", RB, "12 GPU worker nodes", "12|twelve"),
    ("What is the p95 latency target for text generation requests in Orbit?", "answerable", RB, "4 seconds for text generation", "4 seconds|four seconds|4s|4 s"),
    ("How long are Orbit logs kept?", "answerable", RB, "kept for 30 days", "30"),
    ("What percentage of traffic does a new model version receive first in an Orbit canary deployment?", "answerable", RB, "5 percent of traffic", "5 percent|5%|five percent"),
    ("Which command rolls back an Orbit deployment?", "answerable", RB, "orbitctl rollback", "orbitctl"),
    ("How quickly must the primary Orbit on-call engineer acknowledge a page?", "answerable", RB, "within 10 minutes", "10 minutes|ten minutes"),
    ("What is the maximum number of inference workers in Orbit production?", "answerable", RB, "maximum of 40 workers", "40|forty"),
    # Not answerable from any document
    ("What is the WiFi password for the Northwind office?", "unanswerable", "", "", ""),
    ("Who is the CEO of Northwind Robotics?", "unanswerable", "", "", ""),
    ("What version of Kubernetes does the Orbit platform run?", "unanswerable", "", "", ""),
    ("How much does the RTX 5060 laptop cost?", "unanswerable", "", "", ""),
    ("What is the capital of France?", "unanswerable", "", "", ""),
]

NOT_FOUND_PATTERNS = ("couldn't find", "could not find", "not in the uploaded documents", "not found in the uploaded documents")


def normalize(text: str) -> str:
    """Lowercase and remove formatting so matching isn't fooled by commas or markdown."""
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[`*_,]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def keywords_ok(answer: str, keywords: str) -> bool:
    text = normalize(answer)
    return all(any(normalize(alt) in text for alt in group.split("|")) for group in keywords.split(";"))


def is_refusal(answer: str) -> bool:
    text = normalize(answer)
    return any(normalize(pattern) in text for pattern in NOT_FOUND_PATTERNS)


def yes_no(value: bool | None) -> str:
    return "" if value is None else ("yes" if value else "no")


def check_backend() -> None:
    try:
        requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
    except requests.RequestException:
        raise SystemExit("The backend is not running. Start it first, then run this script again.")


def prepare_documents() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / HB).write_text(HANDBOOK.strip() + "\n", encoding="utf-8")
    (DOCS_DIR / RB).write_text(RUNBOOK.strip() + "\n", encoding="utf-8")

    with (EVAL_DIR / "questions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "type", "expected_source", "context_phrase", "answer_keywords"])
        writer.writerows(QUESTIONS)

    print("Clearing old documents and indexing the test set...")
    requests.delete(f"{API_URL}/documents", timeout=60).raise_for_status()
    for path in [ROOT / BRIEF, ROOT / "sample_knowledge.txt", DOCS_DIR / HB, DOCS_DIR / RB]:
        if not path.exists():
            print(f"  Skipped {path.name} (file not found)")
            continue
        response = requests.post(
            f"{API_URL}/documents", files={"file": (path.name, path.read_bytes())}, timeout=300
        )
        response.raise_for_status()
        print(f"  Indexed {path.name}: {response.json()['chunks_indexed']} chunks")


def ask(question: str, use_rerank: bool) -> dict:
    response = requests.post(
        f"{API_URL}/chat", json={"question": question, "use_rerank": use_rerank}, timeout=600
    )
    response.raise_for_status()
    return response.json()


def score(question: tuple, use_rerank: bool) -> dict:
    text, qtype, expected_source, phrase, keywords = question
    row = {"rerank": "ON" if use_rerank else "OFF", "question": text, "type": qtype, "expected_source": expected_source}
    try:
        payload = ask(text, use_rerank)
    except requests.RequestException as exc:
        row.update({"status": "error", "answer": str(exc)})
        return row

    answer, sources, metrics = payload["answer"], payload["sources"], payload["metrics"]
    context = normalize(" ".join(item["text"] for item in sources))
    top_source = sources[0]["source"] if sources else ""

    answer_rank = None
    if qtype == "answerable":
        answer_ok = keywords_ok(answer, keywords) and not is_refusal(answer)
        context_hit = normalize(phrase) in context
        source_ok = top_source == expected_source
        # Position (1 = first) of the first chunk that contains the answer.
        for position, item in enumerate(sources, start=1):
            if normalize(phrase) in normalize(item["text"]):
                answer_rank = position
                break
    else:
        answer_ok, context_hit, source_ok = is_refusal(answer), None, None

    row.update({
        "status": "ok",
        "answer_correct": answer_ok,
        "context_hit": context_hit,
        "top_source_correct": source_ok,
        "answer_rank": answer_rank,
        "top_chunk_has_answer": None if qtype != "answerable" else answer_rank == 1,
        "top_source": top_source,
        "answer": answer,
        **{key: metrics.get(key) for key in (
            "total_ms", "retrieval_ms", "rerank_ms", "generation_ms",
            "completion_tokens", "tokens_per_second", "peak_vram_mb",
        )},
    })
    return row


def rate(rows: list[dict], key: str) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return "n/a"
    return f"{sum(values)}/{len(values)} ({100 * sum(values) / len(values):.0f}%)"


def avg_rank(rows: list[dict]) -> str:
    ranks = [row["answer_rank"] for row in rows if row.get("answer_rank")]
    return f"{statistics.mean(ranks):.2f}" if ranks else "n/a"


def summarize(rows: list[dict]) -> str:
    lines = [
        "# Evaluation summary",
        "",
        f"{len(QUESTIONS)} questions, each asked with reranking ON and OFF.",
        "",
        "| Metric | Rerank ON | Rerank OFF |",
        "|---|---|---|",
    ]

    def by_mode(mode: str, qtype: str | None = None) -> list[dict]:
        return [r for r in rows if r["rerank"] == mode and r["status"] == "ok" and (qtype is None or r["type"] == qtype)]

    def stat(mode: str, key: str, fn) -> str:
        values = [r[key] for r in by_mode(mode) if isinstance(r.get(key), (int, float))]
        return fn(values) if values else "n/a"

    table = [
        ("Correct answers (answerable)", lambda m: rate(by_mode(m, "answerable"), "answer_correct")),
        ("Correct refusals (unanswerable)", lambda m: rate(by_mode(m, "unanswerable"), "answer_correct")),
        ("Answer found in retrieved context", lambda m: rate(by_mode(m, "answerable"), "context_hit")),
        ("Answer is in the #1 ranked chunk", lambda m: rate(by_mode(m, "answerable"), "top_chunk_has_answer")),
        ("Average position of the answer chunk", lambda m: avg_rank(by_mode(m, "answerable"))),
        ("Top source is the right document", lambda m: rate(by_mode(m, "answerable"), "top_source_correct")),
        ("Median total latency", lambda m: stat(m, "total_ms", lambda v: f"{statistics.median(v) / 1000:.2f} s")),
        ("Average retrieval time", lambda m: stat(m, "retrieval_ms", lambda v: f"{statistics.mean(v):.0f} ms")),
        ("Average rerank time", lambda m: stat(m, "rerank_ms", lambda v: f"{statistics.mean(v):.0f} ms")),
        ("Average generation speed", lambda m: stat(m, "tokens_per_second", lambda v: f"{statistics.mean([x for x in v if x > 0] or [0]):.1f} tokens/s")),
        ("Peak VRAM", lambda m: stat(m, "peak_vram_mb", lambda v: f"{max(v) / 1024:.2f} GB")),
    ]
    for label, fn in table:
        lines.append(f"| {label} | {fn('ON')} | {fn('OFF')} |")

    errors = [r for r in rows if r["status"] != "ok"]
    if errors:
        lines += ["", f"{len(errors)} requests failed; see results.csv."]

    on = {r["question"]: r for r in by_mode("ON", "answerable")}
    off = {r["question"]: r for r in by_mode("OFF", "answerable")}
    lines += ["", "## Where reranking changed the position of the answer chunk", ""]
    changed = False
    for q in on:
        if q not in off or on[q]["answer_rank"] == off[q]["answer_rank"]:
            continue
        changed = True
        before = off[q]["answer_rank"] or "not retrieved"
        after = on[q]["answer_rank"] or "not retrieved"
        lines.append(f"- Position {before} without reranking -> {after} with reranking: {q}")
    if not changed:
        lines.append("- Reranking did not change the position of the answer chunk for any question.")

    wrong = [r for r in rows if r["status"] == "ok" and not r["answer_correct"]]
    lines += ["", "## Answers marked incorrect (check these by hand)", ""]
    lines += [f"- [{r['rerank']}] {r['question']} -> {r['answer'][:150]!r}" for r in wrong] or ["- None."]
    return "\n".join(lines) + "\n"


def main() -> None:
    check_backend()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    prepare_documents()

    print("\nWarm-up question (not scored, loads the model if needed)...")
    ask("What is this project about?", True)

    rows: list[dict] = []
    started = time.perf_counter()
    total = len(QUESTIONS) * 2
    count = 0
    for use_rerank in (True, False):
        for question in QUESTIONS:
            count += 1
            row = score(question, use_rerank)
            rows.append(row)
            if row["status"] == "ok":
                result = "CORRECT" if row["answer_correct"] else "WRONG  "
                seconds = (row["total_ms"] or 0) / 1000
                print(f"[{count:2}/{total}] rerank {row['rerank']:3}  {result}  {seconds:5.1f}s  {question[0]}")
            else:
                print(f"[{count:2}/{total}] rerank {row['rerank']:3}  ERROR    {question[0]}  ({row['answer']})")

    columns = [
        "rerank", "question", "type", "answer_correct", "context_hit", "answer_rank", "top_chunk_has_answer", "top_source_correct",
        "expected_source", "top_source", "answer", "total_ms", "retrieval_ms", "rerank_ms",
        "generation_ms", "completion_tokens", "tokens_per_second", "peak_vram_mb", "status",
    ]
    with (EVAL_DIR / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: yes_no(row.get(key)) if key in ("answer_correct", "context_hit", "top_chunk_has_answer", "top_source_correct") else row.get(key, "")
                for key in columns
            })

    summary = summarize(rows)
    (EVAL_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Finished in {(time.perf_counter() - started) / 60:.1f} minutes.")
    print("Saved eval/results.csv and eval/summary.md")


if __name__ == "__main__":
    main()
