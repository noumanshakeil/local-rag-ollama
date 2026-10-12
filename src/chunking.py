"""Chunking utilities for RAG demos."""
from __future__ import annotations

def fixed_chunks(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += max(size - overlap, 1)
    return chunks

def recursive_chunks(text: str, size: int = 400) -> list[str]:
    parts = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    out: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 1 <= size:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                out.append(buf)
            if len(p) > size:
                out.extend(fixed_chunks(p, size=size, overlap=40))
                buf = ""
            else:
                buf = p
    if buf:
        out.append(buf)
    return out or fixed_chunks(text, size=size)
