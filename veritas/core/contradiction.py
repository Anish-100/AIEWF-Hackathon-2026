"""Contradiction detector — new claim vs session history."""
import re
import numpy as np
from .schemas import Claim, Contradiction


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _extract_number(text: str) -> float | None:
    if not text:
        return None
    text = text.replace(",", "").replace("$", "").replace("%", "")
    m = re.search(r"(\d+(?:\.\d+)?)([bBmMkK]?)", text)
    if not m:
        return None
    n = float(m.group(1))
    suffix = m.group(2).lower()
    return n * {"b": 1e9, "m": 1e6, "k": 1e3}.get(suffix, 1)


def _fmt(ts: float) -> str:
    m, s = divmod(int(ts), 60)
    return f"{m}:{s:02d}"


async def check_contradiction(
    new_claim: Claim,
    session_claims: list[Claim],
) -> Contradiction | None:
    if not new_claim.embedding or not session_claims:
        return None

    new_vec = np.array(new_claim.embedding, dtype=np.float32)
    new_num = _extract_number(str(new_claim.value or ""))

    for old in session_claims:
        if not old.embedding:
            continue

        old_vec = np.array(old.embedding, dtype=np.float32)
        if _cosine(new_vec, old_vec) < 0.85:
            continue

        old_num = _extract_number(str(old.value or ""))

        # Numeric contradiction — values differ by >15%
        if new_num is not None and old_num is not None:
            if abs(new_num - old_num) / (max(abs(old_num), 1)) > 0.15:
                return Contradiction(
                    subject=new_claim.subject,
                    claim_a_id=old.id,
                    value_a=str(old.value),
                    ts_a=old.clip_ts,
                    claim_b_id=new_claim.id,
                    value_b=str(new_claim.value),
                    ts_b=new_claim.clip_ts,
                    explanation=f"Said {old.value} at {_fmt(old.clip_ts)}, now {new_claim.value} at {_fmt(new_claim.clip_ts)}",
                )

        # String contradiction
        if new_num is None and old_num is None:
            ov = str(old.value or "").lower().strip()
            nv = str(new_claim.value or "").lower().strip()
            if ov and nv and ov != nv:
                return Contradiction(
                    subject=new_claim.subject,
                    claim_a_id=old.id,
                    value_a=str(old.value),
                    ts_a=old.clip_ts,
                    claim_b_id=new_claim.id,
                    value_b=str(new_claim.value),
                    ts_b=new_claim.clip_ts,
                    explanation=f"Said '{old.value}' at {_fmt(old.clip_ts)}, now '{new_claim.value}' at {_fmt(new_claim.clip_ts)}",
                )

    return None