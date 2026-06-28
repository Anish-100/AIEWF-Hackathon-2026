"""Gemini 3.5 Flash claim-detection adapter (stub).

Phase 2: prompt Flash for strict JSON:
    {"is_checkworthy": bool, "subject": str, "predicate": str,
     "value": str|number|null, "unit": str|null}
Retry once on JSON parse failure; on second failure drop the sentence.
"""
from __future__ import annotations


async def detect(sentence: str) -> dict:
    raise NotImplementedError("Phase 2: implement against Gemini 3.5 Flash.")
