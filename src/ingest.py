"""Extract text from PDF documents page by page."""
import json
from pathlib import Path
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/pages.jsonl")

def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF file page by page."""
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "doc": pdf_path.name,
                "page": page_number,
                "text": text,
            })
    return pages

def main() -> None:
    """Extract text from all PDF files in the RAW_DIR and save to OUT_FILE."""
    all_pages = []
    for pdf_file in sorted(RAW_DIR.glob("*.pdf")):
        pages = extract_pages(pdf_file)
        print(f"{pdf_file.name}: {len(pages)} pages with text")
        all_pages.extend(pages)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_pages)} pages to {OUT_FILE}")


if __name__ == "__main__":
    main()