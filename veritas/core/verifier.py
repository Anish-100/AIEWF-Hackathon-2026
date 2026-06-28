"""Stub: hot-path verifier — embed claim, search memory, return hit/miss."""
from .schemas import Claim


async def verify(claim: Claim) -> tuple[Claim, bool]:
    """Returns (claim, is_hit). Stub always returns MISS."""
    claim.status = "researching"
    return claim, False
