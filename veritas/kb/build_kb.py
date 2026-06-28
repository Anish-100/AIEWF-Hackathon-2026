"""Load demo_clip_facts.yaml → embed each fact → write VerifiedFacts to memory. STUB."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def build() -> int:
    print("build_kb.py stub — will embed facts in Phase 2.")
    return 0


if __name__ == "__main__":
    count = build()
    print(f"Built KB with {count} facts.")
