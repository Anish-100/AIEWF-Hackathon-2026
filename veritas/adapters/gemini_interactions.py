"""Stub: Interactions API research worker — resolves a claim via web search."""
import asyncio


async def research_claim(claim_text: str) -> dict:
    """STUB — returns unverifiable after a fake delay. Real impl uses client.interactions.create."""
    interaction = client.interactions.create(
        model=GEMINI_FLASH_MODEL,
        input=f"Verify this claim and return JSON {{verdict, value, source, explanation}}: {claim.raw_text}",
        tools=[{"type": "google_search"}],
        background=True,
    )
    # poll interaction.id until status == "completed"
    # parse interaction.output_text → VerifiedFact → memory.put()
