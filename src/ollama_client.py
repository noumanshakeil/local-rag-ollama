"""Optional Ollama HTTP client (graceful fallback if offline)."""
from __future__ import annotations

import httpx

def chat(prompt: str, model: str = "llama3.2") -> str:
    try:
        r = httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={{"model": model, "prompt": prompt, "stream": False}},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("response", "")
        return f"Ollama HTTP {{r.status_code}}"
    except Exception as e:
        return f"[ollama unavailable: {{e}}] Using retrieval-only mode."
