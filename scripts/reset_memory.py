"""Wipe the persistent memory DB → COLD state.

Run: `python -m scripts.reset_memory`
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import memory  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    memory.reset_memory()
    print("memory wiped (COLD)")


if __name__ == "__main__":
    main()
