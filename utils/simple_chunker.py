import pandas as pd
import os
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt", quiet=True)

INPUT="data/raw/medical_faq.csv"
OUT="data/combined/medical_chunks.csv"
CHUNK_SENTENCES = 6


def chunk_text(text: str, chunk_size: int = CHUNK_SENTENCES):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        c = " ".join(sentences[i : i + chunk_size]).strip()
        if c:
            chunks.append(c)
    return chunks


def build_chunks():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"{INPUT} not found")

    df = pd.read_csv(INPUT)
    rows = []

    for idx, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        text = (q + "\n\n" + a).strip()

        for i, chunk in enumerate(chunk_text(text)):
            rows.append({
                "id": f"{idx}_c{i}",
                "text": chunk,
                "source": row.get("source", "meddata")
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"✔ Wrote {len(rows)} chunks → {OUT}")


if __name__ == "__main__":
    build_chunks()

