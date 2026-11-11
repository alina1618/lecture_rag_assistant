"""Keyword search over lecture chunks with BM25."""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
TOP_K = 5        # how many chunks to return
STEM_LENGTH = 5  # 0 = off; e.g. 6 keeps only the first 6 letters of each word


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    if STEM_LENGTH > 0:
        words = [word[:STEM_LENGTH] for word in words]
    return words


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        best_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best_ids]


def main() -> None:
    retriever = BM25Retriever(load_chunks(CHUNKS_FILE))
    print(f"Loaded {len(retriever.chunks)} chunks. Ask a question (empty line to exit).")

    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            break
        for rank, hit in enumerate(retriever.search(query), start=1):
            print(f"\n{rank}. {hit['doc']}, page {hit['page']} (score {hit['score']:.2f})")
            print(f"   {hit['text'][:200]}...")


if __name__ == "__main__":
    main()