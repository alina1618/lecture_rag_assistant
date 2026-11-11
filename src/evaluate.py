"""Measure how often retrieval finds the right lecture page."""
import csv
from pathlib import Path

from retrieve_bm25 import BM25Retriever, load_chunks

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
QUESTIONS_FILE = Path("eval/questions.csv")
K = 5                          # how many results we look at
STEM_LENGTHS = [0, 5, 6, 7]    # BM25 settings to compare


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_rank(results: list[dict], doc: str, page: int) -> int | None:
    for rank, hit in enumerate(results, start=1):
        if hit["doc"] == doc and hit["page"] == page:
            return rank
    return None


def evaluate(retriever, questions: list[dict], k: int = K) -> list[dict]:
    rows = []
    for q in questions:
        results = retriever.search(q["question"], k=k)
        rank = find_rank(results, q["source_doc"].strip(), int(q["source_page"]))
        rows.append({
            "question": q["question"],
            "type": (q.get("type") or "direct").strip(),
            "rank": rank,
        })
    return rows


def summarize(rows: list[dict]) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    hit_rate = sum(r["rank"] is not None for r in rows) / len(rows)
    mrr = sum(1 / r["rank"] for r in rows if r["rank"] is not None) / len(rows)
    return hit_rate, mrr


def main() -> None:
    chunks = load_chunks(CHUNKS_FILE)
    questions = load_questions(QUESTIONS_FILE)
    print(f"{len(questions)} questions, {len(chunks)} chunks, k={K}\n")
    print(f"{'setting':<16}{'group':<12}{'n':>4}{'hit@' + str(K):>9}{'MRR':>7}")

    for stem_length in STEM_LENGTHS:
        retriever = BM25Retriever(chunks, stem_length=stem_length)
        rows = evaluate(retriever, questions)
        setting = f"bm25 stem={stem_length}"
        for group in ["all", "direct", "paraphrase"]:
            selected = rows if group == "all" else [r for r in rows if r["type"] == group]
            hit_rate, mrr = summarize(selected)
            print(f"{setting:<16}{group:<12}{len(selected):>4}{hit_rate:>9.2f}{mrr:>7.2f}")
        print()

        missed = [r["question"] for r in rows if r["rank"] is None]
        for question in missed:
            print(f"   missed: {question}")
        if missed:
            print()


if __name__ == "__main__":
    main()