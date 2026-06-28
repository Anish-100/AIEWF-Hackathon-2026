"""Load curated KB into memory → WARM state.

Run: `python -m scripts.seed_memory`
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kb.build_kb import load_kb  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    n = asyncio.run(load_kb())
    print(f"seeded {n} facts (WARM)")


if __name__ == "__main__":
    main()
