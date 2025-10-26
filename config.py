"""
config.py

Centralized configuration that reads environment variables from a .env file (if present).
Expose keys used by the companion scripts.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

def getenv(key: str, default=None):
    v = os.getenv(key)
    return v if v is not None else default

# API Keys (optional)
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

# App defaults
OUTPUT_DIR = getenv("OUTPUT_DIR", str(ROOT / "outputs"))
MEMORY_DIR = getenv("MEMORY_DIR", str(ROOT / "memory"))

# Small helper export
def as_dict():
    return {
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MEMORY_DIR": MEMORY_DIR,
    }
