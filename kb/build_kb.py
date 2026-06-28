"""Load `kb/demo_topic_facts.yaml`, embed every subject, write VerifiedFacts
into the persistent memory store.

Run: `python -m kb.build_kb`
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adapters import gemini_embed  # noqa: E402
from core import memory  # noqa: E402
from core.schemas import VerifiedFact  # noqa: E402

log = logging.getLogger("build_kb")

KB_PATH = Path(__file__).parent / "demo_topic_facts.yaml"


def _claim_key(subject: str) -> str:
    return hashlib.sha1(subject.strip().lower().encode("utf-8")).hexdigest()[:16]


async def load_kb(yaml_path: Path = KB_PATH) -> int:
    raw = yaml.safe_load(yaml_path.read_text())
    if not isinstance(raw, list):
        raise RuntimeError(f"{yaml_path} must be a YAML list of facts")
    log.info("loading %d facts from %s", len(raw), yaml_path)

    subjects = [str(item["subject"]).strip() for item in raw]
    log.info("embedding subjects (batched)...")
    embeddings = await gemini_embed.embed_batch(subjects)

    mem = memory.get_memory()
    now = time.time()
    written = 0
    for item, emb in zip(raw, embeddings):
        if not emb:
            log.warning("skipping (empty embedding): %s", item["subject"])
            continue
        subject = str(item["subject"]).strip()
        fact = VerifiedFact(
            claim_key=_claim_key(subject),
            subject=subject,
            canonical_value=item.get("canonical_value"),
            unit=item.get("unit"),
            verdict=item.get("verdict", "true"),
            source=item.get("source", "kb"),
            explanation=str(item.get("explanation", "")),
            embedding=emb,
            first_seen_ts=now,
            times_seen=1,
        )
        mem.put(fact)
        written += 1
    log.info("KB loaded: %d facts written, store size=%d", written, mem.size())
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    asyncio.run(load_kb())


if __name__ == "__main__":
    main()
