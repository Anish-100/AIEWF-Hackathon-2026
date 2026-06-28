"""Stub: text → embedding vector."""
import hashlib


async def embed(text: str) -> list[float]:
    """Returns a fake deterministic 768-dim embedding (stub)."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    import random
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(768)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]
