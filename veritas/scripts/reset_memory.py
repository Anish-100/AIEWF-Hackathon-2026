"""Wipe memory store → COLD state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa
from core import memory

memory.clear()
print("Memory cleared — COLD state.")
