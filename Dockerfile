# ---------- Base ----------
FROM python:3.10-slim

WORKDIR /app

# Install system deps (needed by FAISS & others)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Environment
ENV PYTHONUNBUFFERED=1

# Expose FastAPI
EXPOSE 8000

# Start backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

