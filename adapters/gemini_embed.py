"""Gemini embeddings adapter.

`embed` / `embed_batch` return list[float] vectors via the configured model
(`GEMINI_EMBED_MODEL`, default `gemini-embedding-2`). Subjects we see twice
in a session should not re-embed — we keep a process-wide LRU cache keyed by
the input text.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Sequence

from google import genai
from google.genai import types

import config

log = logging.getLogger(__name__)

_client: genai.Client | None = None
_lock = asyncio.Lock()
# Tuple cache that lets us reuse the SDK call across speakers; bounded to keep
# memory predictable. 4096 entries handles a long debate easily.
_CACHE_MAX = 4096
_text_cache: dict[str, list[float]] = {}


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


async def embed(text: str) -> list[float]:
    if not text:
        return []
    cached = _text_cache.get(text)
    if cached is not None:
        return cached
    client = _get_client()
    resp = await client.aio.models.embed_content(
        model=config.GEMINI_EMBED_MODEL,
        contents=text,
    )
    # SDK returns ContentEmbedding objects under .embeddings
    emb = list(resp.embeddings[0].values)
    if len(_text_cache) >= _CACHE_MAX:
        # Drop a single arbitrary entry to keep size bounded; this is hot-path
        # so we avoid OrderedDict overhead.
        _text_cache.pop(next(iter(_text_cache)))
    _text_cache[text] = emb
    return emb


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed many texts. The Gemini embed_content endpoint is single-input
    even when handed a list, so we fan out concurrent calls. Cached entries
    return without a network call."""
    if not texts:
        return []
    # Issue all uncached embed() calls concurrently; embed() handles caching.
    return await asyncio.gather(*[embed(t) for t in texts])
