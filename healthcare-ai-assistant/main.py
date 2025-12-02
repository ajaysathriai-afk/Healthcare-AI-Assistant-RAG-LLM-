# main.py
# Healthcare AI Assistant — FAISS RAG + simple LLM glue
# - Loads FAISS index + metadata from data/embeddings_faiss when needed
# - Endpoints: /health, /healthcheck, /ask, /ask_llm, /ask_llm_stream

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import os
import json
import traceback
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ---------- OpenAI client (optional) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI()  # reads key from OPENAI_API_KEY env var
    except Exception:
        openai_client = None

# ---------- FastAPI app ----------
app = FastAPI(title="Healthcare AI Assistant — FAISS RAG")

# Allow local Streamlit dev UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request schema ----------
class Query(BaseModel):
    question: str
    top_k: int = 3

# ---------- FAISS globals ----------
EMB_DIR = "data/embeddings_faiss"
faiss_index = None
faiss_meta: List[Dict[str, Any]] = []
sbert: SentenceTransformer | None = None

def load_faiss_and_meta(emb_dir: str = EMB_DIR):
    """
    Load FAISS index and metadata and SBERT model from disk.
    Safe to call multiple times.
    """
    global faiss_index, faiss_meta, sbert
    try:
        if faiss_index is not None and sbert is not None and len(faiss_meta) > 0:
            return  # already loaded

        idx_path = os.path.join(emb_dir, "faiss.index")
        meta_path = os.path.join(emb_dir, "meta.jsonl")

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS index or meta not found in {emb_dir}")

        # load index
        faiss_index = faiss.read_index(idx_path)

        # load meta.jsonl
        with open(meta_path, "r", encoding="utf-8") as f:
            faiss_meta = [json.loads(line) for line in f if line.strip()]

        # load sentence-transformer for query embedding
        # choose a small model that you used during ingest
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"[FAISS] Loaded {len(faiss_meta)} vectors and SBERT model.")
    except Exception as e:
        print("FAISS load error:", e)
        traceback.print_exc()
        faiss_index = None
        faiss_meta = []
        sbert = None

# Try to load on import/startup unless SKIP_STARTUP_LOAD set
if os.getenv("SKIP_STARTUP_LOAD") != "1":
    try:
        load_faiss_and_meta()
    except Exception:
        pass

# ---------- Retrieval ----------
def retrieve_faiss(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Return a list of retrieved items:
    [{ "rank": 1, "text": "...", "metadata": {...} }, ...]
    Loads FAISS + SBERT lazily if not loaded yet.
    """
    global faiss_index, sbert, faiss_meta

    # if not loaded, attempt to load
    if faiss_index is None or sbert is None or len(faiss_meta) == 0:
        try:
            load_faiss_and_meta()
        except Exception:
            return []

    if faiss_index is None or sbert is None or len(faiss_meta) == 0:
        return []

    # embed query robustly (sbert.encode or sbert.__call__)
    try:
        qv = None
        try:
            # prefer encode
            qv = sbert.encode([question])
        except Exception:
            try:
                qv = sbert([question])
            except Exception:
                qv = None

        if qv is None:
            return []

        qv = np.array(qv, dtype="float32")
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        qv = qv.astype("float32")
    except Exception:
        return []

    try:
        D, I = faiss_index.search(qv, top_k)
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(I[0], start=1):
        if idx < 0 or idx >= len(faiss_meta):
            continue
        m = faiss_meta[idx]
        results.append(
            {
                "rank": rank,
                "text": m.get("text"),
                "metadata": {"source": m.get("source"), "id": m.get("id")},
            }
        )
    return results

# ---------- Prompt building with safety ----------
def build_system_prompt() -> str:
    return (
        "You are a cautious healthcare information assistant.\n"
        "You provide general health information ONLY, not medical advice.\n\n"
        "At the very beginning of EVERY answer, show this disclaimer exactly:\n"
        "⚠️ Important: This assistant provides general health information only and is not a substitute "
        "for professional medical advice, diagnosis, or treatment. If you have serious or worsening symptoms, "
        "chest pain, trouble breathing, thoughts of self-harm, or any emergency, contact your doctor or "
        "local emergency services immediately.\n\n"
        "Additional rules:\n"
        "- Use ONLY the context provided to you.\n"
        "- If the context is incomplete or unclear, say you are not fully confident and advise consulting a doctor.\n"
        "- Do NOT diagnose specific individuals.\n"
        "- Do NOT prescribe medications or give exact doses.\n"
        "- Do NOT provide instructions for self-harm, overdose, suicide, or harming others.\n"
        "- Always encourage the user to consult a licensed healthcare professional for personal questions.\n"
        "- Use simple, clear language.\n"
        "- Do NOT include a 'Sources' section in your answer. The UI will show sources separately."
    )

def build_rag_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, r in enumerate(retrieved):
        meta = r.get("metadata", {}) or {}
        src = meta.get("source", f"doc_{i+1}")
        text = r.get("text", "") or ""
        context_parts.append(f"Source {i+1} ({src}):\n{text}")
    context = "\n\n".join(context_parts) if context_parts else "No retrieved context available."
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Your task:\n"
        "- Answer the question concisely using ONLY the context above.\n"
        "- If the context is incomplete or conflicting, say you are not fully confident and advise seeing a healthcare professional.\n"
        "- Remember: Do NOT list 'Sources' in your answer; the UI will show sources separately."
    )
    return user_prompt

# ---------- Helper retriever for tests / clients ----------
def get_retriever(default_top_k: int = 3):
    """
    Return a simple retriever object with method get_relevant_documents(question, top_k=None)
    Ensures FAISS loaded lazily so tests/imports get a working retriever.
    """
    # ensure faiss is loaded
    if faiss_index is None or sbert is None or len(faiss_meta) == 0:
        try:
            load_faiss_and_meta()
        except Exception:
            pass

    class SimpleRetriever:
        def __init__(self, top_k):
            self.top_k = top_k

        def get_relevant_documents(self, question, top_k=None):
            k = top_k or self.top_k
            # use retrieve_faiss() and convert to a simple list of dicts
            return retrieve_faiss(question, top_k=k)

    return SimpleRetriever(default_top_k)

# ---------- Routes ----------
@app.get("/health")
@app.get("/healthcheck")
async def health():
    return {"status": "ok", "faiss_loaded": faiss_index is not None, "vectors": len(faiss_meta)}

@app.post("/ask")
async def ask(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="question required")
    retrieved = retrieve_faiss(q.question, top_k=q.top_k)
    return {"question": q.question, "retrieved": retrieved}

@app.post("/ask_llm")
async def ask_llm(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="question required")
    retrieved = retrieve_faiss(q.question, top_k=q.top_k)

    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set or OpenAI client unavailable. Export OPENAI_API_KEY and restart server.",
        )

    system_prompt = build_system_prompt()
    user_prompt = build_rag_prompt(q.question, retrieved)

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    sources = [
        {"rank": r["rank"], "source": r["metadata"].get("source"), "id": r["metadata"].get("id")}
        for r in retrieved
    ]
    return {"question": q.question, "answer": answer, "sources": sources}

@app.post("/ask_llm_stream")
async def ask_llm_stream(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="question required")

    retrieved = retrieve_faiss(q.question, top_k=q.top_k)

    system_prompt = build_system_prompt()
    user_prompt = build_rag_prompt(q.question, retrieved)

    sources_payload = [
        {"rank": r["rank"], "source": r["metadata"].get("source"), "id": r["metadata"].get("id")}
        for r in retrieved
    ]

    def event_stream():
        # first send sources
        yield json.dumps({"type": "sources", "sources": sources_payload}) + "\n"

        # if OpenAI client available, try streaming; else fallback to a simple summary
        if openai_client is None:
            # fallback: short auto-answer using retrieved context (safe stub)
            try:
                # create a short synthesized answer from retrieved text (very simple)
                combined = " ".join([r.get("text","")[:600] for r in retrieved])[:3000].strip()
                if not combined:
                    combined = "I couldn't find relevant context. Please ask another question or consult a professional."
                # stream tokens in small chunks
                i = 0
                while i < len(combined):
                    chunk = combined[i:i+80]
                    yield json.dumps({"type": "token", "token": chunk}) + "\n"
                    i += 80
            except Exception as e:
                yield json.dumps({"type":"error","error":str(e)}) + "\n"
            yield json.dumps({"type":"done"}) + "\n"
            return

        # streaming via OpenAI client
        try:
            stream = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=512,
                stream=True,
            )
            for chunk in stream:
                try:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if not delta:
                        delta = choice.get("delta") if isinstance(choice, dict) else None
                    token = None
                    if delta:
                        token = delta.get("content") if isinstance(delta, dict) else getattr(delta, "content", None)
                    if token:
                        yield json.dumps({"type":"token","token":token}) + "\n"
                except Exception:
                    continue
        except Exception as e:
            yield json.dumps({"type":"error","error":str(e)}) + "\n"

        yield json.dumps({"type":"done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/jsonlines")
