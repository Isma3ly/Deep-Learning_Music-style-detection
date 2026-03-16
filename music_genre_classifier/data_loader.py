"""Load GTZAN dataset from data/genres_original/."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .config import GENRES, GTZAN_GENRES_DIR


def load_gtzan(data_dir: Path | str | None = None) -> List[Tuple[str, str]]:
    """
    Walk genre subfolders and return (file_path, genre) for each WAV file.

    Expects layout: data_dir/<genre>/*.wav (e.g. data/genres_original/rock/rock.00000.wav).
    """
    root = Path(data_dir) if data_dir is not None else GTZAN_GENRES_DIR
    if not root.is_dir():
        return []

    pairs: List[Tuple[str, str]] = []
    for genre in GENRES:
        genre_dir = root / genre
        if not genre_dir.is_dir():
            continue
        for p in sorted(genre_dir.glob("*.wav")):
            pairs.append((str(p.resolve()), genre))
    return pairs


def load_gtzan_dataframe(data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Return a small DataFrame with columns path, genre, and label (int index).
    """
    pairs = load_gtzan(data_dir)
    if not pairs:
        return pd.DataFrame(columns=["path", "genre", "label"])

    paths, genres = zip(*pairs)
    genre_to_idx = {g: i for i, g in enumerate(GENRES)}
    labels = [genre_to_idx[g] for g in genres]
    return pd.DataFrame({"path": paths, "genre": genres, "label": labels})
