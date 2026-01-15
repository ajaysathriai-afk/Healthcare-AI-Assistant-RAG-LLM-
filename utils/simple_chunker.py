# utils/simple_chunker.py
# Lightweight text chunker (NO NLTK dependency)

import re
from typing import List


def simple_sentence_split(text: str) -> List[str]:
    """
    Very lightweight sentence splitter.
    Works fine for our docs without needing nltk.
    """
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    # split on sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """
    Convert long text into chunks.
    - max_chars: chunk size
    - overlap: overlapping characters to preserve context
    """
    sentences = simple_sentence_split(text)

    chunks = []
    buf = ""

    for sent in sentences:
        if len(buf) + len(sent) + 1 <= max_chars:
            buf = f"{buf} {sent}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = sent

    if buf:
        chunks.append(buf)

    # add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev_tail = chunks[i - 1][-overlap:]
                overlapped.append((prev_tail + " " + ch).strip())
        chunks = overlapped

    return chunks

