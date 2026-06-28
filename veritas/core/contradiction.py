"""Stub: contradiction detector — new claim vs session history + memory."""
from .schemas import Claim, Contradiction


async def check_contradiction(
    new_claim: Claim,
    session_claims: list[Claim],
) -> Contradiction | None:
    return None
