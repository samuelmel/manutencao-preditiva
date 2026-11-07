"""Drift (RF08): PSI e KS por sensor; vocabulário e distribuição de classes no texto."""
from __future__ import annotations

import re

import numpy as np
from scipy.stats import ks_2samp

_EPS = 1e-4


def psi(ref: np.ndarray, atual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index com faixas por quantis da referência."""
    ref = np.asarray(ref, dtype=float)
    atual = np.asarray(atual, dtype=float)
    if len(ref) == 0 or len(atual) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # sensor praticamente constante na referência
        return 0.0 if np.allclose(atual, ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(atual, edges)[0] / len(atual)
    p, q = np.clip(p, _EPS, None), np.clip(q, _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks(ref: np.ndarray, atual: np.ndarray) -> tuple[float, float]:
    r = ks_2samp(ref, atual)
    return float(r.statistic), float(r.pvalue)


def psi_por_sensor(ref: dict[str, np.ndarray], atual: dict[str, np.ndarray]) -> dict[str, float]:
    return {s: psi(ref[s], atual[s]) for s in ref if s in atual}


def sensores_em_drift(psis: dict[str, float], limiar: float = 0.2) -> int:
    return int(sum(v > limiar for v in psis.values()))


def drift_de_sensores(psis: dict[str, float], limiar: float = 0.2, min_sensores: int = 3) -> bool:
    """Gatilho do plano: PSI > limiar em MAIS de `min_sensores` sensores."""
    return sensores_em_drift(psis, limiar) > min_sensores


def psi_categorico(ref: dict[str, float], atual: dict[str, float]) -> float:
    """PSI entre duas distribuições de classes (proporções)."""
    total = 0.0
    for k, pk in ref.items():
        p = max(pk, _EPS)
        q = max(atual.get(k, 0.0), _EPS)
        total += (q - p) * np.log(q / p)
    return float(total)


def taxa_oov(textos: list[str], vocab: set[str]) -> float:
    """Fração de tokens alfabéticos fora do vocabulário de treino (mudança de vocabulário)."""
    toks = [t for x in textos for t in re.findall(r"[^\W\d_]+", x.lower())]
    return float(np.mean([t not in vocab for t in toks])) if toks else 0.0
