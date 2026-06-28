import os
from dotenv import load_dotenv

load_dotenv(".env.local", override=True)
load_dotenv(".env.example")

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-audio-eap")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")
LIVEKIT_ROOM_NAME = os.getenv("LIVEKIT_ROOM_NAME", "veritas-demo")

SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.82"))
SUBJECT_MATCH_THRESHOLD = float(os.getenv("SUBJECT_MATCH_THRESHOLD", "0.85"))
VALUE_TOLERANCE = float(os.getenv("VALUE_TOLERANCE", "0.0"))
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./veritas_memory.db")
DEMO_MODE = os.getenv("DEMO_MODE", "warm")
USE_INTERACTIONS = os.getenv("USE_INTERACTIONS", "false").lower() == "true"
VERITAS_SERVER_URL = os.getenv("VERITAS_SERVER_URL", "http://localhost:8000")
