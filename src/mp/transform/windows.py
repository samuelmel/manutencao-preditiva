"""Silver -> Gold: janelas deslizantes e divisão por motor."""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_units(unit_ids, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Divisão por motor (nunca por linha) para evitar vazamento entre treino e validação."""
    ids = np.unique(np.asarray(unit_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    return np.sort(ids[n_val:]), np.sort(ids[:n_val])


def make_windows(
    df: pd.DataFrame, features: list[str], window: int = 30, last_only: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Janelas (n, window, n_features), RUL, unit_id e flag low_confidence.

    Motor com menos de `window` ciclos: preenche o início repetindo o primeiro valor
    e marca low_confidence=True. `last_only=True` gera só a última janela por motor
    (avaliação no conjunto de teste).
    """
    xs, ys, us, lc = [], [], [], []
    for uid, g in df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        x = g[features].to_numpy(dtype=np.float32)
        y = g["rul"].to_numpy(dtype=np.float32)
        short = len(x) < window
        if short:
            pad = np.repeat(x[:1], window - len(x), axis=0)
            x = np.vstack([pad, x])
            y = np.concatenate([np.repeat(y[:1], window - len(y)), y])
        starts = [len(x) - window] if last_only else range(len(x) - window + 1)
        for s in starts:
            xs.append(x[s : s + window])
            ys.append(y[s + window - 1])
            us.append(uid)
            lc.append(short)
    return (
        np.stack(xs) if xs else np.empty((0, window, len(features)), np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(us, dtype=np.int32),
        np.asarray(lc, dtype=bool),
    )
