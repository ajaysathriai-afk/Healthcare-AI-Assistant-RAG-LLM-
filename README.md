# 🏥 Healthcare AI Assistant — RAG + FAISS + FastAPI + Streamlit (Deployed)

A **Retrieval-Augmented Generation (RAG)** based Healthcare AI Assistant that answers user health questions using a **custom medical knowledge base** (TXT documents).  
It retrieves relevant context using **FAISS semantic search**, then generates responses using **OpenAI LLM**, with **source citations** and **streaming output**.

✅ End-to-end deployed on Render  
✅ Strict RAG mode (reduces hallucinations)  
✅ Streaming tokens + sources (`/ask_llm_stream`)  
✅ Evaluation benchmark + results (`eval/`)  

---

## 🚀 Live Demo (Deployed)

- **Frontend (Streamlit UI):** https://healthcare-ai-assistant-rag-llm-1.onrender.com/
- **Backend (FastAPI - Docker):** https://healthcare-ai-assistant-docker.onrender.com
- **Swagger Docs:** https://healthcare-ai-assistant-docker.onrender.com/docs

---

## 🎯 Problem

Most health chatbots hallucinate answers and cannot cite reliable sources.

Healthcare applications require:
- factual grounding
- traceable sources
- safety disclaimers
- predictable behavior (refuse when context is missing)

---

## ✅ Solution

This project implements a complete **RAG pipeline**:

1. A small **medical dataset** is stored as documents (`data/docs/`)
2. Documents are chunked + embedded
3. **FAISS** stores vectors for fast semantic retrieval
4. At query time, the system retrieves top-k relevant chunks
5. The LLM answers using only retrieved context (**strict mode**)
6. App returns answer + citations

---

## 🧠 RAG Architecture (Pipeline)

**User Question → Embeddings → FAISS Search → Context Prompt → LLM → Answer + Sources**

### Core workflow:
1. **Ingestion**
   - Load TXT docs from `data/docs/`
   - Chunk text
   - Create embeddings
   - Build FAISS index (`data/embeddings_faiss/`)

2. **Retrieval**
   - Embed user query
   - FAISS similarity search
   - select top_k chunks

3. **Generation**
   - Strict prompt forces grounded answer from context only
   - If context is weak → model replies “I don’t know”

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI + Uvicorn
- **Vector DB / Search:** FAISS
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** OpenAI API (`gpt-4o-mini`)
- **Deployment:** Render (Docker backend + Python frontend)
- **Evaluation:** Custom benchmark script (`eval/`)

---

## 📌 Key Features

✅ Strict RAG mode (reduces hallucinations)  
✅ Top-K retrieval controls results  
✅ Streaming response endpoint  
✅ Shows sources used for each answer  
✅ Safety disclaimer included  
✅ Evaluation benchmark: PASS/FAIL for correct sources  
✅ Dockerfile included for production deployment  

---

## 🔌 API Endpoints

### `POST /ask_llm`
Returns final answer + sources.

Request:
```json
{
  "question": "Symptoms of diabetes?",
  "top_k": 3
}
POST /ask_llm_stream

Streams JSONL:

sources

tokens

done

Example:

curl -N -X POST "https://healthcare-ai-assistant-docker.onrender.com/ask_llm_stream" \
-H "Content-Type: application/json" \
-d '{"question":"Symptoms of diabetes?","top_k":3}'

📊 Evaluation

This project includes a benchmark dataset + automated evaluation.

eval/eval_questions.json → test questions + expected sources

eval/eval.py → runs automated PASS/FAIL checks

eval/eval_results.json → saved output report

Run evaluation locally:
python eval/eval.py


✅ Output includes:

PASS/FAIL score

retrieved sources

saved results JSON
## 📸 Screenshots

### Streamlit UI
![Streamlit UI](assets/screenshots/01_streamlit_home.png)

### Answer + Sources
![Answer + Sources](assets/screenshots/02_answer_sources.png)

### Retrieval Settings (Top-K)
![Top-K Settings](assets/screenshots/03_topk_settings.png)

### Swagger API Docs
![Swagger Docs](assets/screenshots/04_swagger_docs.png)

### Evaluation Results
![Evaluation](assets/screenshots/05_eval_results.png)


Demo Video: <>

🛠️ Run Locally
1) Clone repo
git clone <your-repo-url>
cd healthcare-ai-assistant-clean

2) Setup virtualenv + install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3) Set OpenAI key
export OPENAI_API_KEY="your_key_here"

4) Run ingestion (build FAISS)
python ingest.py

5) Run backend
uvicorn main:app --reload --port 8000

6) Run frontend
streamlit run app.py

🐳 Docker Support

Dockerfile included.

Run locally with Docker:

docker build -t healthcare-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY="your_key_here" healthcare-rag

🔮 Future Work

Expand dataset: 200–500+ docs

Add re-ranking for higher retrieval accuracy

Add source chunk preview in UI

Add unit tests + CI workflow

Use persistent storage (Render Disk) for FAISS index

Add multi-language support (Telugu/English)

⚠️ Disclaimer

This assistant provides general health information only and is not a substitute for professional medical advice, diagnosis, or treatment.
If symptoms are severe or worsening, please consult a doctor or seek emergency care.
## 🚦 CI Status
![CI](https://github.com/ajaysathriai-afk/Healthcare-AI-Assistant-RAG-LLM-/actions/workflows/ci.yml/badge.svg)

