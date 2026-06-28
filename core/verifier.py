"""Hot-path verifier (stub). Phase 2."""
from __future__ import annotations

from core.schemas import Claim


async def verify(claim: Claim) -> tuple[Claim, bool]:
    raise NotImplementedError("Phase 2")
