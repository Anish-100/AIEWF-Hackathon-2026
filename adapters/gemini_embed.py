"""Gemini embeddings adapter (stub). LRU-cached at Phase 2."""
from __future__ import annotations


async def embed(text: str) -> list[float]:
    raise NotImplementedError("Phase 2: implement against Gemini embeddings API.")


async def embed_batch(texts: list[str]) -> list[list[float]]:
    raise NotImplementedError("Phase 2: implement against Gemini embeddings API.")
