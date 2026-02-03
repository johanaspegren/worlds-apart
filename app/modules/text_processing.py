import re

def chunk_text(text: str, max_len: int = 1200):
    # simple chunking for now
    words = text.split()
    chunks, current = [], []
    length = 0

    for w in words:
        current.append(w)
        length += len(w)
        if length > max_len:
            chunks.append(" ".join(current))
            current, length = [], 0

    if current:
        chunks.append(" ".join(current))
    return chunks
