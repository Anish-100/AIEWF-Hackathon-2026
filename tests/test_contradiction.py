"""Contradiction detector tests — same-speaker and cross-speaker."""
from __future__ import annotations

import numpy as np

from core.contradiction import ContradictionChecker
from core.schemas import Claim


def _make_claim(speaker: str, subject: str, value, clip_ts: float, embedding: list[float], unit: str | None = "%"):
    """Build a minimal Claim with a pre-supplied embedding so tests don't hit the network."""
    return Claim(
        session_id="t",
        speaker_id=speaker,
        clip_ts=clip_ts,
        raw_text=f"{subject} = {value}",
        subject=subject,
        predicate="equals",
        value=value,
        unit=unit,
        embedding=embedding,
        status="verified",
    )


def _emb(seed: int, dim: int = 64) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_same_speaker_value_change_fires():
    chk = ContradictionChecker()
    emb = _emb(1)
    c1 = _make_claim("alex", "US unemployment May 2025", "4.1", clip_ts=10.0, embedding=emb)
    c2 = _make_claim("alex", "US unemployment May 2025", "5.0", clip_ts=42.0, embedding=emb)
    assert chk.record_and_check(c1) is None
    out = chk.record_and_check(c2)
    assert out is not None
    assert out.kind == "same_speaker"
    assert out.speaker_a_id == "alex" and out.speaker_b_id == "alex"
    assert out.value_a == "4.1" and out.value_b == "5.0"


def test_cross_speaker_disagreement_fires():
    chk = ContradictionChecker()
    emb = _emb(2)
    c1 = _make_claim("alex", "US nonfarm payrolls March 2025", "228000", clip_ts=5.0, embedding=emb, unit="jobs")
    c2 = _make_claim("bob",  "US nonfarm payrolls March 2025", "180000", clip_ts=30.0, embedding=emb, unit="jobs")
    chk.record_and_check(c1)
    out = chk.record_and_check(c2)
    assert out is not None
    assert out.kind == "cross_speaker"
    assert out.speaker_a_id == "alex" and out.speaker_b_id == "bob"


def test_unit_aware_same_value_is_not_a_contradiction():
    """`228 thousand jobs` and `228000 jobs` are the same fact, not a conflict."""
    chk = ContradictionChecker()
    emb = _emb(3)
    c1 = _make_claim("alex", "US nonfarm payrolls March 2025", "228", clip_ts=5.0, embedding=emb, unit="thousand jobs")
    c2 = _make_claim("bob",  "US nonfarm payrolls March 2025", "228000", clip_ts=30.0, embedding=emb, unit="jobs")
    chk.record_and_check(c1)
    assert chk.record_and_check(c2) is None


def test_different_subjects_do_not_contradict():
    chk = ContradictionChecker()
    c1 = _make_claim("alex", "US unemployment May 2025", "4.1", clip_ts=5.0, embedding=_emb(4))
    c2 = _make_claim("bob",  "US healthcare jobs March 2025", "54000", clip_ts=30.0, embedding=_emb(5))
    chk.record_and_check(c1)
    assert chk.record_and_check(c2) is None


def test_missing_value_skips_contradiction():
    """An unverifiable claim (no value) shouldn't fire anything."""
    chk = ContradictionChecker()
    emb = _emb(6)
    c1 = _make_claim("alex", "US unemployment May 2025", "4.1", clip_ts=5.0, embedding=emb)
    c2 = _make_claim("bob",  "US unemployment May 2025", None, clip_ts=30.0, embedding=emb)
    chk.record_and_check(c1)
    assert chk.record_and_check(c2) is None


def test_third_claim_against_earliest_still_fires():
    """Log should retain prior claims so later contradictions still match."""
    chk = ContradictionChecker()
    emb = _emb(7)
    chk.record_and_check(_make_claim("alex", "X", "10", clip_ts=1.0, embedding=emb))
    chk.record_and_check(_make_claim("bob",  "X", "10", clip_ts=2.0, embedding=emb))  # agrees
    out = chk.record_and_check(_make_claim("carol", "X", "999", clip_ts=3.0, embedding=emb))
    assert out is not None
    assert out.kind == "cross_speaker"


def test_different_months_do_not_contradict_even_when_embeddings_are_close():
    """Regression — embeddings of 'unemployment rate March 2025' and
    'unemployment rate May 2025' are ~0.97 cosine, but these are *different
    facts* (different months), not a contradiction.

    The canonical-subject-string check should keep them apart even though we
    pass an identical fake embedding here (which would otherwise make the
    embedding-only path think they're the same)."""
    chk = ContradictionChecker()
    emb = _emb(11)
    c1 = _make_claim("banh",   "US unemployment rate March 2025", "3.5", clip_ts=33.8, embedding=emb)
    c2 = _make_claim("anis2h", "US unemployment rate May 2025",   "4.1", clip_ts=48.2, embedding=emb)
    chk.record_and_check(c1)
    out = chk.record_and_check(c2)
    assert out is None, f"different-month claims should NOT contradict, got {out}"


def test_same_value_with_and_without_unit_not_a_contradiction():
    """4.1 (no unit) vs 4.1% (with %) describe the same value — no contradiction."""
    chk = ContradictionChecker()
    emb = _emb(12)
    c1 = _make_claim("alex", "US unemployment rate May 2025", "4.1", clip_ts=10.0, embedding=emb, unit=None)
    c2 = _make_claim("bob",  "US unemployment rate May 2025", "4.1", clip_ts=20.0, embedding=emb, unit="%")
    chk.record_and_check(c1)
    assert chk.record_and_check(c2) is None


def test_subject_key_normalises_punctuation_and_case():
    """'US Unemployment Rate, May 2025' and 'us unemployment rate may 2025'
    should be treated as the same subject (so a real disagreement between
    speakers fires)."""
    chk = ContradictionChecker()
    emb = _emb(13)
    c1 = _make_claim("alex", "US Unemployment Rate, May 2025", "4.1", clip_ts=10.0, embedding=emb)
    c2 = _make_claim("bob",  "us unemployment rate may 2025",  "5.5", clip_ts=20.0, embedding=emb)
    chk.record_and_check(c1)
    out = chk.record_and_check(c2)
    assert out is not None
    assert out.kind == "cross_speaker"
