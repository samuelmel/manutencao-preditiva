"""Métricas do C-MAPSS."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true, y_pred) -> float:
    """Score assimétrico do C-MAPSS. d = previsto - real.

    d < 0 (falha prevista cedo demais): exp(-d/13) - 1
    d >= 0 (falha prevista tarde demais): exp(d/10) - 1, penalidade maior.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
