import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)
load_dotenv(Path(__file__).parent / ".env.local", override=True)


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


GEMINI_API_KEY = _get("GEMINI_API_KEY")
GEMINI_LIVE_MODEL = _get("GEMINI_LIVE_MODEL", "gemini-3.5-live-translate-preview")
GEMINI_FLASH_MODEL = _get("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
GEMINI_EMBED_MODEL = _get("GEMINI_EMBED_MODEL", "text-embedding-004")

LIVEKIT_URL = _get("LIVEKIT_URL")
LIVEKIT_API_KEY = _get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = _get("LIVEKIT_API_SECRET")
LIVEKIT_ROOM_NAME = _get("LIVEKIT_ROOM_NAME", "veritas-demo")

ANTIGRAVITY_MODEL = _get("ANTIGRAVITY_MODEL", "antigravity-preview-05-2026")

SIM_THRESHOLD = _get_float("SIM_THRESHOLD", 0.82)
SUBJECT_MATCH_THRESHOLD = _get_float("SUBJECT_MATCH_THRESHOLD", 0.85)
VALUE_TOLERANCE = _get_float("VALUE_TOLERANCE", 0.0)

MEMORY_DB_PATH = _get("MEMORY_DB_PATH", "./veritas_memory.db")
DEMO_MODE = _get("DEMO_MODE", "warm")
USE_ANTIGRAVITY = _get_bool("USE_ANTIGRAVITY", False)
# Use the gemini-3.5-live-translate-preview model with translation_config.
# When True, the UI shows both source-language transcript and English translation,
# and the rest of the pipeline (Flash, verifier, contradiction) runs on the English.
USE_LIVE_TRANSLATE = _get_bool("USE_LIVE_TRANSLATE", True)
TRANSLATE_TARGET_LANG = _get("TRANSLATE_TARGET_LANG", "en")
