"""Configuration and constants for GTZAN-based pipeline."""

from pathlib import Path

# GTZAN genre order (must match folder names under data/genres_original/)
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

N_CLASSES = len(GENRES)

# Project root = parent of music_genre_classifier package (works from any cwd)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _PROJECT_ROOT / "data"
GTZAN_GENRES_DIR = DATA_DIR / "genres_original"
DEFAULT_FILLER_TRACK = str(GTZAN_GENRES_DIR / "rock" / "rock.00000.wav")
