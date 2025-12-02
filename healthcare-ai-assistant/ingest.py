# ingest.py
"""
FAISS-based ingestion pipeline
Compatible with main.py and Step 7 tests.

- Loads PDF/TXT/CSV files from data/
- Extracts text
- Chunks text
- Embeds with SBERT ("all-MiniLM-L6-v2")
- Builds FAISS index
- Saves:
    data/embeddings_faiss/faiss.index
    data/embeddings_faiss/meta.jsonl

- Exposes load_and_index() so pytest can call it
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from utils.preprocess import load_text_from_file, chunk_text


# ----------------------------- CONFIG -----------------------------
DATA_DIR = "data"
EMB_DIR = "data/embeddings_faiss"
MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(EMB_DIR, exist_ok=True)


# ------------------------ EMBEDDING MODEL ------------------------
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# ------------------------- FILE LOADING --------------------------
def load_all_documents(source_dir: str = DATA_DIR) -> List[Dict[str, Any]]:
    """
    Load all PDF/TXT/CSV files and chunk them.
    Returns list of dicts: [{"id":..., "text":..., "source":...}, ...]
    """
    source = Path(source_dir)
    docs = []

    for path in source.rglob("*"):
        if path.suffix.lower() not in {".pdf", ".txt", ".csv"}:
            continue

        text = load_text_from_file(path)
        if not text:
            continue

        chunks = chunk_text(text, chunk_size=1000, overlap=200)

        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": f"{path.stem}-{i}",
                    "text": chunk,
                    "source": str(path),
                    "chunk": i,
                }
            )

    return docs


# ------------------------- BUILD FAISS ---------------------------
def build_faiss_index(docs: List[Dict[str, Any]]):
    """
    Builds FAISS index and returns (index, meta_list)
    """
    if not docs:
        raise ValueError("No documents found to index.")

    model = get_embedding_model()

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Metadata stored in JSONL: each line contains {"id":..., "text":..., "source":...}
    meta_list = []
    for d in docs:
        meta_list.append(
            {
                "id": d["id"],
                "text": d["text"],
                "source": d["source"],
            }
        )

    return index, meta_list


# ------------------------- SAVE FILES ----------------------------
def save_faiss_index(index, meta: List[Dict[str, Any]]):
    """
    Saves faiss.index and meta.jsonl
    """
    idx_path = os.path.join(EMB_DIR, "faiss.index")
    meta_path = os.path.join(EMB_DIR, "meta.jsonl")

    faiss.write_index(index, idx_path)

    with open(meta_path, "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    return idx_path, meta_path


# ---------------------- MAIN ENTRY POINT -------------------------
def load_and_index(source_dir: str = DATA_DIR):
    """
    Called by pytest AND by CLI.
    Runs the full ingestion pipeline.
    """
    print("🔍 Loading documents...")
    docs = load_all_documents(source_dir)

    print(f"📄 Loaded {len(docs)} chunks.")

    print("⚙️ Building FAISS index...")
    index, meta = build_faiss_index(docs)

    print("💾 Saving FAISS index + metadata...")
    save_faiss_index(index, meta)

    print("✅ Ingestion complete.")
    return True


# CLI helper
if __name__ == "__main__":
    load_and_index()
