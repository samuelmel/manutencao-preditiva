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
            layers.Dense(1),  # saída em [0,1] (RUL / RUL_CAP); alvo do treino escalado igual
        ]
    )
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def with_cycles_output(core: keras.Model) -> keras.Model:
    """Embrulha o modelo treinado para devolver ciclos (usado na inferência e no ONNX)."""
    return keras.Sequential([core, layers.Rescaling(RUL_CAP)])
