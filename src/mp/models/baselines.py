"""Baselines obrigatórios: média constante e Random Forest sobre features agregadas da janela."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def window_features(x: np.ndarray) -> np.ndarray:
    """(n, W, f) -> (n, 4f): último valor, média, desvio e inclinação linear da janela."""
    w = x.shape[1]
    t = np.arange(w) - (w - 1) / 2
    slope = np.einsum("t,ntf->nf", t, x - x.mean(axis=1, keepdims=True)) / (t**2).sum()
    return np.concatenate([x[:, -1], x.mean(axis=1), x.std(axis=1), slope], axis=1)


class MeanBaseline:
    def fit(self, x, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_, dtype=np.float32)


class RFBaseline:
    def __init__(self, seed: int = 42, n_estimators: int = 200):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=5, n_jobs=-1, random_state=seed
        )

    def fit(self, x, y):
        self.rf.fit(window_features(x), y)
        return self

    def predict(self, x):
        return self.rf.predict(window_features(x))
