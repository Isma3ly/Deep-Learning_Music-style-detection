from __future__ import annotations

import warnings
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
from scipy import stats


def feature_columns() -> pd.MultiIndex:
    """Return the multi-index describing the engineered audio features."""
    feature_sizes = dict(
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        tonnetz=6,
        mfcc=20,
        rmse=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_rolloff=1,
    )
    moments = ("mean", "std", "skew", "kurtosis", "median", "min", "max")

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, f"{i + 1:02d}") for i in range(size))
            columns.extend(it)

    names = ("feature", "statistics", "number")
    index = pd.MultiIndex.from_tuples(columns, names=names)
    return index.sort_values()


def _feature_stats(target: pd.Series, name: str, values: np.ndarray) -> None:
    target[name, "mean"] = np.mean(values, axis=1)
    target[name, "std"] = np.std(values, axis=1)
    target[name, "skew"] = stats.skew(values, axis=1)
    target[name, "kurtosis"] = stats.kurtosis(values, axis=1)
    target[name, "median"] = np.median(values, axis=1)
    target[name, "min"] = np.min(values, axis=1)
    target[name, "max"] = np.max(values, axis=1)


def _compute_feature_series(y: np.ndarray, sr: int, name: str) -> pd.Series:
    """Core feature-extraction routine shared by training and inference."""
    features = pd.Series(index=feature_columns(), dtype=np.float32, name=name)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings("error", module="librosa")

    # ZCR
    f = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    _feature_stats(features, "zcr", f)

    # Harmonic / pitch-related features from CQT
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=512,
            bins_per_octave=12,
            n_bins=7 * 12,
            tuning=None,
        )
    )
    assert cqt.shape[0] == 7 * 12

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    _feature_stats(features, "chroma_cqt", f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    _feature_stats(features, "chroma_cens", f)
    f = librosa.feature.tonnetz(chroma=f)
    _feature_stats(features, "tonnetz", f)

    # Spectral features from STFT
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
    _feature_stats(features, "chroma_stft", f)

    f = librosa.feature.rms(S=stft)
    _feature_stats(features, "rmse", f)

    f = librosa.feature.spectral_centroid(S=stft)
    _feature_stats(features, "spectral_centroid", f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    _feature_stats(features, "spectral_bandwidth", f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    _feature_stats(features, "spectral_contrast", f)
    f = librosa.feature.spectral_rolloff(S=stft)
    _feature_stats(features, "spectral_rolloff", f)

    # MFCCs from mel-spectrogram
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    _feature_stats(features, "mfcc", mfcc)

    return features


def compute_features_for_file(
    filepath: str, duration: float = 30.0
) -> Tuple[pd.Series, int]:
    """Compute the full feature vector for a single audio file."""
    y, sr = librosa.load(filepath, sr=None, mono=True, duration=duration)
    series = _compute_feature_series(y, sr, name=filepath)
    return series, sr

