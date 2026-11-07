import numpy as np
import pytest

from mp.drift import (
    drift_de_sensores,
    ks,
    psi,
    psi_categorico,
    psi_por_sensor,
    sensores_em_drift,
    taxa_oov,
)
from mp.fusion import prioridade, prioridade_da_config


def test_prioridade_extremos_e_formula():
    assert prioridade(0, 1.0) == pytest.approx(1.0)
    assert prioridade(125, 0.0) == pytest.approx(0.0)
    assert prioridade(50, 0.5) == pytest.approx(0.6 * (1 - 50 / 125) + 0.4 * 0.5)


def test_prioridade_limita_rul_e_probabilidade():
    assert prioridade(-10, 2.0) == pytest.approx(1.0)  # RUL negativo vira 0; p limitado a 1
    assert prioridade(500, 0.0) == pytest.approx(0.0)  # teto do RUL em 125


def test_prioridade_sem_os_e_menor_que_com_os_grave():
    assert prioridade(20, None) < prioridade(20, 0.9)


def test_prioridade_monotonica_no_rul():
    valores = [prioridade(r, 0.3) for r in range(0, 126, 5)]
    assert valores == sorted(valores, reverse=True)


def test_pesos_da_config():
    assert prioridade_da_config(0, 1.0) == pytest.approx(1.0)


def test_psi_zero_para_mesma_distribuicao_e_alto_para_deslocada():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(size=5000)) < 0.05
    assert psi(ref, rng.normal(loc=2.0, size=5000)) > 0.2


def test_psi_sensor_constante():
    assert psi(np.ones(100), np.ones(50)) == 0.0
    assert psi(np.ones(100), np.full(50, 5.0)) == float("inf")


def test_gatilho_de_drift_exige_mais_de_3_sensores():
    psis = {f"s{i}": 0.5 for i in range(3)} | {"s9": 0.01}
    assert sensores_em_drift(psis) == 3
    assert not drift_de_sensores(psis)  # exatamente 3 não dispara
    psis["s9"] = 0.5
    assert drift_de_sensores(psis)


def test_psi_por_sensor_e_ks():
    rng = np.random.default_rng(1)
    ref = {"a": rng.normal(size=2000), "b": rng.normal(size=2000)}
    atual = {"a": rng.normal(size=500), "b": rng.normal(loc=3, size=500)}
    p = psi_por_sensor(ref, atual)
    assert p["a"] < 0.1
    assert p["b"] > 0.2
    stat, pval = ks(ref["b"], atual["b"])
    assert stat > 0.8 and pval < 0.01


def test_drift_de_texto():
    ref = {"baixa": 0.25, "média": 0.25, "alta": 0.25, "crítica": 0.25}
    assert psi_categorico(ref, ref) == pytest.approx(0.0, abs=1e-6)
    assert psi_categorico(ref, {"baixa": 0.9, "média": 0.05, "alta": 0.03, "crítica": 0.02}) > 0.2
    assert taxa_oov(["vibração forte"], {"vibração", "forte"}) == 0.0
    assert taxa_oov(["the engine is overheating"], {"vibração"}) == 1.0
