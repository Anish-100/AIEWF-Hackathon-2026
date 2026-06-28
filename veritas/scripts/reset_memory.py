"""Wipe memory store → COLD state."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.memory import clear, init_db

init_db()
clear()
print("Memory wiped.")
