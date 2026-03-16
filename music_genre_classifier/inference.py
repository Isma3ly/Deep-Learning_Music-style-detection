from __future__ import annotations

import pickle
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_FILLER_TRACK, GENRES, N_CLASSES
from .model import load_trained_model

# Feature extraction (moved from features.py); must match notebook 01 logic.
_MOMENT_FNS = [
    ("mean", np.mean), ("std", np.std), ("skew", stats.skew), ("kurtosis", stats.kurtosis),
    ("median", np.median), ("min", np.min), ("max", np.max),
]


def feature_columns():
    feature_sizes = dict(
        chroma_stft=12, chroma_cqt=12, chroma_cens=12, tonnetz=6, mfcc=20,
        rmse=1, zcr=1, spectral_centroid=1, spectral_bandwidth=1, spectral_contrast=7, spectral_rolloff=1,
    )
    moments = ("mean", "std", "skew", "kurtosis", "median", "min", "max")
    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            columns.extend([(name, moment, f"{i+1:02d}") for i in range(size)])
    return pd.MultiIndex.from_tuples(columns, names=("feature", "statistics", "number")).sort_values()


def compute_features_for_file(filepath: str, duration: float = 30.0) -> Tuple[pd.Series, int]:
    y, sr = librosa.load(filepath, sr=None, mono=True, duration=duration)
    features = pd.Series(index=feature_columns(), dtype=np.float64, name=filepath)
    hop = 512

    f = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["zcr", moment] = val.flat[0] if val.size == 1 else val

    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, bins_per_octave=12, n_bins=7*12, tuning=None))
    for name, feat_fn in [
        ("chroma_cqt", lambda c: librosa.feature.chroma_cqt(C=c, n_chroma=12, n_octaves=7)),
        ("chroma_cens", lambda c: librosa.feature.chroma_cens(C=c, n_chroma=12, n_octaves=7)),
    ]:
        f = feat_fn(cqt)
        for moment, fn in _MOMENT_FNS:
            val = fn(f, axis=1)
            features[name, moment] = val.flat[0] if val.size == 1 else val
    f = librosa.feature.tonnetz(chroma=f)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["tonnetz", moment] = val.flat[0] if val.size == 1 else val

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["chroma_stft", moment] = val.flat[0] if val.size == 1 else val
    f = librosa.feature.rms(S=stft)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["rmse", moment] = val.flat[0] if val.size == 1 else val
    f = librosa.feature.spectral_centroid(S=stft)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["spectral_centroid", moment] = val.flat[0] if val.size == 1 else val
    f = librosa.feature.spectral_bandwidth(S=stft)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["spectral_bandwidth", moment] = val.flat[0] if val.size == 1 else val
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["spectral_contrast", moment] = val.flat[0] if val.size == 1 else val
    f = librosa.feature.spectral_rolloff(S=stft)
    for moment, fn in _MOMENT_FNS:
        val = fn(f, axis=1)
        features["spectral_rolloff", moment] = val.flat[0] if val.size == 1 else val

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    for moment, fn in _MOMENT_FNS:
        val = fn(mfcc, axis=1)
        features["mfcc", moment] = val.flat[0] if val.size == 1 else val

    return features, sr

# GTZAN genre list as single source of truth
GENRE_DICT = {g: i for i, g in enumerate(GENRES)}
INV_GENRE_DICT = {i: g for i, g in enumerate(GENRES)}
GENRE_NAMES = GENRES.copy()


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
    filler_track_path: str | None = None,
) -> Tuple[str, float, str]:
    """High-level helper to predict the genre from a WAV file path."""
    if filler_track_path is None:
        filler_track_path = DEFAULT_FILLER_TRACK
    pca = load_pca(pca_path)
    X = _prepare_features_for_inference(wav_path, filler_track_path, pca)

    model = load_trained_model(model_weights, input_dim=X.shape[1], n_classes=N_CLASSES)
    probs = model.predict(X)[0]
    idx = int(np.argmax(probs))
    genre = INV_GENRE_DICT[idx]
    conf_text = prediction_confidence(probs)
    return genre, float(probs[idx]), conf_text

