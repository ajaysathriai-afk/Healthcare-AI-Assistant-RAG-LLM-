# main.py — FastAPI backend for Healthcare RAG (STRICT RAG MODE)

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

# ================= STRICT RAG SETTINGS =================
STRICT_RAG = True

# If retrieved context is too small, treat as "no context"
MIN_CONTEXT_CHARS = 250
MIN_DOCS = 1

STRICT_RAG_FALLBACK_MSG = (
    "I don't have enough information in my knowledge base to answer this safely.\n\n"
    "✅ Try asking using a topic present in the uploaded medical documents, "
    "or add more documents to expand coverage."
)

# ================= APP =================
app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok", "message": "Healthcare AI Assistant API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


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

    # Safety: keep k bounded
    if k is None:
        k = 3
    k = max(1, min(int(k), 10))

    q_emb = embed_model.encode([query]).astype("float32")
    _, indices = faiss_index.search(q_emb, k)

    docs = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            docs.append(metadata[idx])

    return docs


def build_context(docs: List[Dict[str, Any]]) -> str:
    # Join retrieved chunks
    return "\n\n".join(d.get("text", "") for d in docs if d.get("text"))


def has_sufficient_context(docs: List[Dict[str, Any]]) -> bool:
    if not docs or len(docs) < MIN_DOCS:
        return False
    ctx = build_context(docs)
    if len(ctx.strip()) < MIN_CONTEXT_CHARS:
        return False
    return True

# ================= PROMPT =================
def build_prompt(question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    context = build_context(retrieved_docs)

    if STRICT_RAG:
        # STRICT MODE: MUST NOT use general medical knowledge
        return f"""
You are a medical assistant that MUST answer ONLY using the given context.

RULES (STRICT):
- Use ONLY the context to answer.
- If the context does not contain enough information, respond exactly with:
  "I don't know based on the provided documents."
- Do NOT use general knowledge.
- Do NOT guess.
- Keep answer short and clear.

Context:
{context}

Question: {question}

Answer:
""".strip()

    # Non-strict mode (not recommended)
    return f"""
You are a helpful medical assistant.

Use the context below if it helps.
If the context is weak or incomplete, answer normally using your medical knowledge.

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

    # STRICT RAG: block answering if no/weak context
    if STRICT_RAG and not has_sufficient_context(docs):
        return {
            "answer": STRICT_RAG_FALLBACK_MSG,
            "sources": [],
        }

    answer = call_llm(build_prompt(req.question, docs))
    sources = sorted(set(d.get("source", "unknown") for d in docs))

    return {
        "answer": answer,
        "sources": sources,
    }


@app.post("/ask_llm_stream")
async def ask_stream(req: AskRequest):
    docs = search(req.question, req.top_k)

    # STRICT RAG: streaming fallback immediately
    if STRICT_RAG and not has_sufficient_context(docs):
        def fallback_stream():
            yield json.dumps({"type": "sources", "sources": []}) + "\n"
            for part in STRICT_RAG_FALLBACK_MSG.split():
                yield json.dumps({"type": "token", "token": part + " "}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(fallback_stream(), media_type="application/jsonl")

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

