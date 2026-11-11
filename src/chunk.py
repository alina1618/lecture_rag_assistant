"""Split extracted text into overlapping chunks."""
import json
from pathlib import Path

IN_FILE = Path("data/processed/pages.jsonl")
OUT_FILE = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 150  # max number of characters in one chunk
OVERLAP = 30  # number of words shared by neighboring chunks

def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks

def load_pages(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main() -> None:
    pages = load_pages(IN_FILE)

    chunks = []
    for page in pages:
        page_chunks = split_text(page["text"], CHUNK_SIZE, OVERLAP)
        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": f"{page['doc']}_p{page['page']}_c{i}",
                "doc": page["doc"],
                "page": page["page"],
                "text": chunk_text,
            })

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = [len(c["text"].split()) for c in chunks]
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Words per chunk: min {min(lengths)}, "
          f"avg {sum(lengths) / len(lengths):.0f}, max {max(lengths)}")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()