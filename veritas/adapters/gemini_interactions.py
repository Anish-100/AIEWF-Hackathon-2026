"""Stub: Interactions API research worker — resolves a claim via web search."""
import asyncio


async def research_claim(claim_text: str) -> dict:
    """STUB — returns unverifiable after a fake delay. Real impl uses client.interactions.create."""
    await asyncio.sleep(3)
    return {
        "verdict": "unverifiable",
        "canonical_value": None,
        "source": "stub",
        "explanation": "Research worker not yet implemented.",
    }
