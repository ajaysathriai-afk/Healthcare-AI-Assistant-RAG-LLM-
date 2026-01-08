from pathlib import Path
from typing import List
from pypdf import PdfReader
import csv

def load_text_from_file(path: Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    elif path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    elif path.suffix.lower() == ".csv":
        rows = []
        with path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            for r in reader:
                rows.append(" ".join(r))
        return "\n".join(rows)
    else:
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
