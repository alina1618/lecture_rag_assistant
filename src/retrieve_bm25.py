"""Keyword search over lecture chunks with BM25."""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
TOP_K = 5        # how many chunks to return
STEM_LENGTH = 6  


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenize(text: str, stem_length: int = STEM_LENGTH) -> list[str]:
    words = re.findall(r"\w+", text.lower())
    if stem_length > 0:
        words = [word[:stem_length] for word in words]
    return words

class BM25Retriever:
    def __init__(self, chunks: list[dict], stem_length: int = STEM_LENGTH):
        self.chunks = chunks
        self.stem_length = stem_length
        tokenized_chunks = [tokenize(chunk["text"], stem_length) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query, self.stem_length))
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