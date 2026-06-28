"""Stub: sentence → structured claim detection via Gemini Flash."""
from typing import Optional


async def detect_claim(sentence: str) -> Optional[dict]:
    """Returns None if not check-worthy, else {subject, predicate, value, unit}."""
    # STUB — returns fake check-worthy claim for every sentence containing a number
    import re
    if re.search(r'\d', sentence):
        return {
            "is_checkworthy": True,
            "subject": "stub subject",
            "predicate": "equals",
            "value": "0",
            "unit": None,
        }
    return None
