"""In-memory vector store (numpy cosine) for offline demos."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

def hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic bag-of-hash embedding (no model download required)."""
    vec = np.zeros(dim, dtype=np.float64)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec if n == 0 else vec / n

@dataclass
class Hit:
    text: str
    score: float
    meta: dict

class VectorStore:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.texts: list[str] = []
        self.meta: list[dict] = []
        self.mat: np.ndarray | None = None

    def add(self, texts: list[str], metas: list[dict] | None = None) -> None:
        metas = metas or [{} for _ in texts]
        embs = [hash_embed(t, self.dim) for t in texts]
        block = np.vstack(embs) if embs else np.zeros((0, self.dim))
        self.texts.extend(texts)
        self.meta.extend(metas)
        self.mat = block if self.mat is None else np.vstack([self.mat, block])

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if not self.texts or self.mat is None:
            return []
        q = hash_embed(query, self.dim)
        scores = self.mat @ q
        idx = np.argsort(-scores)[:k]
        return [Hit(self.texts[i], float(scores[i]), self.meta[i]) for i in idx]
