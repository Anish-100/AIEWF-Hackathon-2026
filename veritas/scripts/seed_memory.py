"""Load curated KB into memory → WARM state."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa

def main():
    from kb.build_kb import build
    count = build()
    print(f"Seeded {count} facts into memory — WARM state.")

if __name__ == "__main__":
    main()
