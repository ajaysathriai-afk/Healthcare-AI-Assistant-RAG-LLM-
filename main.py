# main.py — FastAPI backend for Healthcare RAG (PATCHED & STABLE)

import os
import json
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_DIR = os.path.join(BASE_DIR, "data", "embeddings_faiss")

INDEX_FILE = "faiss.index"
META_FILE = "meta.jsonl"

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found — set it in your env")

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= RAG DATA =================
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faiss_index = None
metadata: List[Dict[str, Any]] = []

# ================= LOADERS =================
def load_faiss_and_meta() -> Tuple[Any, List[Dict[str, Any]]]:
    index_path = os.path.join(EMBED_DIR, INDEX_FILE)
    meta_path = os.path.join(EMBED_DIR, META_FILE)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    index = faiss.read_index(index_path)

    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    return index, meta

# ================= SEARCH =================
def search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    if faiss_index is None or not metadata:
        return []

    q_emb = embed_model.encode([query]).astype("float32")
    _, indices = faiss_index.search(q_emb, k)

    docs = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            docs.append(metadata[idx])

    return docs

# ================= PROMPT =================
def build_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(d["text"] for d in retrieved_docs)

    return f"""
You are a helpful medical assistant.

Use the context below if it helps.
If the context is weak or incomplete, answer normally using your medical knowledge.
Do NOT say "I don't know" unless the question is completely unclear.

Context:
{context}

Question: {question}

Answer in a clear, natural paragraph:
""".strip()


# ================= OPENAI =================
def call_llm(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=r.text)

    return r.json()["choices"][0]["message"]["content"]

# ================= API =================
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask_llm")
def ask(req: AskRequest):
    docs = search(req.question, req.top_k)
    answer = call_llm(build_prompt(req.question, docs))

    return {
        "answer": answer,
        "sources": sorted(set(d.get("source", "unknown") for d in docs)),
    }

@app.post("/ask_llm_stream")
async def ask_stream(req: AskRequest):
    docs = search(req.question, req.top_k)
    prompt = build_prompt(req.question, docs)

    def stream():
        yield json.dumps({
            "type": "sources",
            "sources": sorted(set(d.get("source", "unknown") for d in docs)),
        }) + "\n"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "stream": True,
        }

        with requests.post(OPENAI_URL, headers=headers, json=payload, stream=True) as r:
            if r.status_code != 200:
                yield json.dumps({"type": "error", "error": r.text}) + "\n"
                return

            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                chunk = line.replace("data:", "").strip()
                if chunk == "[DONE]":
                    break

                try:
                    obj = json.loads(chunk)
                    token = obj["choices"][0]["delta"].get("content")
                    if token:
                        yield json.dumps({"type": "token", "token": token}) + "\n"
                except Exception:
                    continue

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/jsonl")

# ================= STARTUP =================
@app.on_event("startup")
def startup():
    global faiss_index, metadata
    try:
        faiss_index, metadata = load_faiss_and_meta()
        print(f"[FAISS] Loaded {len(metadata)} chunks from {EMBED_DIR}")
    except Exception as e:
        print(f"[FAISS ERROR] {e}")
        faiss_index = None
        metadata = []
