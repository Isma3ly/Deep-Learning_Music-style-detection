from __future__ import annotations

import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import compute_features_for_file
from .model import load_trained_model


GENRE_DICT = {
    "Hip-Hop": 0,
    "Pop": 1,
    "Folk": 2,
    "Rock": 3,
    "Experimental": 4,
    "International": 5,
    "Electronic": 6,
    "Instrumental": 7,
}

INV_GENRE_DICT = {v: k for k, v in GENRE_DICT.items()}
GENRE_NAMES = [INV_GENRE_DICT[i] for i in range(len(GENRE_DICT))]


def prediction_confidence(probs: np.ndarray) -> str:
    """Map max probability to a French confidence phrase."""
    c = float(np.max(probs))
    if c < 0.25:
        return "Pas sûr mais je pense que ça soit du "
    elif c < 0.35:
        return "On sent le "
    elif c < 0.45:
        return "Je suis un peu confiant que ça soit "
    elif c < 0.60:
        return "Je suis confiant que ça soit "
    elif c < 0.70:
        return "Je suis très sûr que ça soit "
    return "100% sûr que c'est du "


def load_pca(path: str = "pca"):
    """Load the persisted PCA object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _prepare_features_for_inference(
    wav_path: str,
    filler_track_path: str,
    pca,
) -> np.ndarray:
    """Replicate the original extract_features pipeline."""
    series, _ = compute_features_for_file(wav_path)
    filler_series, _ = compute_features_for_file(filler_track_path)

    X = pd.concat([series.to_frame().T, filler_series.to_frame().T])
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca


def predict_genre_from_file(
    wav_path: str,
    model_weights: str = "DNN3.h5",
    pca_path: str = "pca",
    filler_track_path: str = "rock.wav",
) -> Tuple[str, float, str]:
    """High-level helper to predict the genre from a WAV file path."""
    pca = load_pca(pca_path)
    X = _prepare_features_for_inference(wav_path, filler_track_path, pca)

    model = load_trained_model(model_weights, input_dim=X.shape[1], n_classes=len(GENRE_DICT))
    probs = model.predict(X)[0]
    idx = int(np.argmax(probs))
    genre = INV_GENRE_DICT[idx]
    conf_text = prediction_confidence(probs)
    return genre, float(probs[idx]), conf_text

