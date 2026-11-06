"""LSTM de RUL (Keras 3, backend PyTorch). Importar este módulo exige o extra `train`."""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers

RUL_CAP = 125.0


def build_lstm(window: int, n_features: int) -> keras.Model:
    m = keras.Sequential(
        [
            layers.Input((window, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
            layers.Rescaling(RUL_CAP),  # treina em [0,1]; saída em ciclos
        ]
    )
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m
