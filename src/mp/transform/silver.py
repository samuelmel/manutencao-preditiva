"""Bronze -> Silver: rótulo RUL, remoção de sensores constantes e normalização."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from mp.ingest.cmapss import OP_COLS, SENSOR_COLS


def add_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """RUL = ciclo_max_do_motor - ciclo_atual, com teto linear por partes em `cap`."""
    out = df.copy()
    out["rul"] = out.groupby("unit_id")["cycle"].transform("max") - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series, cap: int | None = 125) -> pd.DataFrame:
    """No teste a trajetória é truncada; RUL = RUL_final + (ciclo_max - ciclo)."""
    out = df.copy()
    ciclo_max = out.groupby("unit_id")["cycle"].transform("max")
    out["rul"] = out["unit_id"].map(final_rul) + (ciclo_max - out["cycle"])
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def constant_sensors(train: pd.DataFrame, threshold: float = 1e-6) -> list[str]:
    """Sensores com variância ≈ 0, calculada no treino (nunca uma lista fixa)."""
    var = train[SENSOR_COLS].var()
    return [c for c in SENSOR_COLS if var[c] <= threshold]


class Normalizer:
    """Min-Max por cluster de condição operacional, ajustado só no treino.

    Com n_clusters=1 (FD001/FD003) vira um Min-Max global.
    """

    def __init__(self, sensors: list[str], n_clusters: int = 1, seed: int = 42):
        self.sensors = list(sensors)
        self.n_clusters = n_clusters
        self.seed = seed
        self.centers_: np.ndarray | None = None
        self.min_: np.ndarray | None = None  # (k, n_sensores)
        self.max_: np.ndarray | None = None

    def _assign(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_clusters == 1:
            return np.zeros(len(df), dtype=int)
        d = self._dist(df[OP_COLS].to_numpy())
        return d.argmin(axis=1)

    def _dist(self, ops: np.ndarray) -> np.ndarray:
        return np.linalg.norm(ops[:, None, :] - self.centers_[None, :, :], axis=2)

    def fit(self, train: pd.DataFrame) -> Normalizer:
        if self.n_clusters > 1:
            km = KMeans(self.n_clusters, random_state=self.seed, n_init=10)
            km.fit(train[OP_COLS].to_numpy())
            self.centers_ = km.cluster_centers_
        else:
            self.centers_ = train[OP_COLS].to_numpy().mean(axis=0, keepdims=True)
        cl = self._assign(train)
        x = train[self.sensors].to_numpy()
        self.min_ = np.zeros((self.n_clusters, len(self.sensors)))
        self.max_ = np.ones((self.n_clusters, len(self.sensors)))
        for k in range(self.n_clusters):
            xk = x[cl == k]
            if len(xk):
                self.min_[k], self.max_[k] = xk.min(axis=0), xk.max(axis=0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cl = self._assign(df)
        x = df[self.sensors].to_numpy(dtype=float)
        lo, hi = self.min_[cl], self.max_[cl]
        rng = np.where(hi - lo == 0, 1.0, hi - lo)
        out = df.copy()
        out[self.sensors] = (x - lo) / rng
        out["op_cluster"] = cl.astype("int8")
        return out

    def cluster_distance(self, df: pd.DataFrame) -> np.ndarray:
        """Distância ao centroide mais próximo; valores altos = condição nunca vista (drift)."""
        return self._dist(df[OP_COLS].to_numpy()).min(axis=1)


def build_silver(
    train: pd.DataFrame,
    test: pd.DataFrame,
    final_rul: pd.Series,
    *,
    cap: int,
    n_clusters: int,
    threshold: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Normalizer]:
    train = add_rul(train, cap)
    test = add_test_rul(test, final_rul, cap)
    drop = constant_sensors(train, threshold)
    keep = [c for c in SENSOR_COLS if c not in drop]
    norm = Normalizer(keep, n_clusters, seed).fit(train)
    cols = ["unit_id", "cycle", *OP_COLS, *keep, "rul", "op_cluster"]
    return norm.transform(train)[cols], norm.transform(test)[cols], norm
