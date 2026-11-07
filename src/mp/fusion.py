"""Fusão (RF07): prioridade combinada de RUL e gravidade da OS. Pesos em configs/fusion.yaml."""
from __future__ import annotations

from mp.config import load_config


def prioridade(rul: float, p_grave: float | None, w_rul: float = 0.6, w_os: float = 0.4,
               rul_cap: float = 125.0) -> float:
    """Em [0, 1]. Sem OS conhecida (`p_grave=None`) a parcela da OS vale 0."""
    rul = min(max(float(rul), 0.0), rul_cap)
    parcela_os = 0.0 if p_grave is None else min(max(float(p_grave), 0.0), 1.0)
    return w_rul * (1.0 - rul / rul_cap) + w_os * parcela_os


def prioridade_da_config(rul: float, p_grave: float | None) -> float:
    c = load_config("fusion")
    return prioridade(rul, p_grave, c["w_rul"], c["w_os"], c["rul_cap"])
