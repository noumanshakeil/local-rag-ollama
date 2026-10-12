"""CLI entry for local-rag-ollama."""
from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import recursive_chunks
from .store import VectorStore

SAMPLE = """
Retrieval Augmented Generation (RAG) grounds LLM answers in your documents.
Chunking splits long text into passages. Embeddings map text into vectors.
Hybrid search mixes keyword BM25 with dense vectors for better recall.
Always cite the passages you used so users can verify claims.
""".strip()

def build_store(path: Path | None = None) -> VectorStore:
    text = path.read_text(encoding="utf-8") if path and path.exists() else SAMPLE
    chunks = recursive_chunks(text, size=180)
    store = VectorStore()
    store.add(chunks, [{"i": i} for i in range(len(chunks))])
    return store

def answer(query: str, store: VectorStore) -> str:
    hits = store.search(query, k=3)
    if not hits:
        return "No context found."
    ctx = "\n".join(f"- ({h.score:.3f}) {h.text}" for h in hits)
    return (
        f"Query: {query}\n\nTop passages:\n{ctx}\n\n"
        "Draft answer: Based on the retrieved passages above, "
        "RAG retrieves relevant chunks then the LLM (plug in Ollama/OpenAI) "
        "would write a grounded response with citations."
    )

def main() -> None:
    p = argparse.ArgumentParser(description="Local RAG pipeline using Ollama embeddings/chat + ChromaDB for private document Q&A without cloud APIs.")
    p.add_argument("query", nargs="?", default="What is RAG?")
    p.add_argument("--file", type=Path, default=None)
    args = p.parse_args()
    store = build_store(args.file)
    print(answer(args.query, store))

if __name__ == "__main__":
    main()
