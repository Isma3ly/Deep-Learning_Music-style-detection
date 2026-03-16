from __future__ import annotations

from typing import Optional

import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential


def build_dnn_model(input_dim: int = 215, n_classes: int = 10) -> Sequential:
    """Build the dense neural network used for genre classification."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer="nadam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(
    weights_path: str = "DNN3.weights.h5",
    input_dim: int = 215,
    n_classes: int = 10,
) -> Sequential:
    """Build the DNN model and load pre-trained weights."""
    model = build_dnn_model(input_dim=input_dim, n_classes=n_classes)
    model.load_weights(weights_path)
    return model


def forward_pass(model: Sequential, x: np.ndarray) -> np.ndarray:
    """Convenience helper for tests: run a forward pass."""
    return model.predict(x)

