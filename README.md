# lecture_rag_assistant
RAG assistant that answers questions about university lecture materials, with retrieval evaluation.

## Project structure
```
data/raw/          lecture PDFs (not published)
data/processed/    extracted pages and chunks
src/ingest.py      extracts text from PDFs page by page
src/chunk.py       splits pages into overlapping chunks
eval/              evaluation questions
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. Put lecture PDFs into `data/raw/`
2. Extract text: `python src/ingest.py`
3. Split into chunks: `python src/chunk.py`