# ---------- Base ----------
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# ✅ IMPORTANT: Build FAISS at container start (only if missing)
CMD ["sh", "-c", "if [ ! -f data/embeddings_faiss/faiss.index ]; then echo '⚙️ FAISS missing -> running ingest...'; python ingest.py; fi; uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]


