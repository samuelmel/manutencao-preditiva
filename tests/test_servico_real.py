"""Teste de ponta a ponta com os artefatos reais. Pula se não foram gerados (make data + treino)."""
import time

import numpy as np
import pytest

from mp.config import ROOT

SERVING = ROOT / "artifacts" / "serving"
BRONZE = ROOT / "data" / "bronze"
pytestmark = pytest.mark.skipif(
    not (SERVING / "meta.json").exists() or not (BRONZE / "test_FD001.txt").exists(),
    reason="artefatos de serviço ou dados do C-MAPSS ausentes",
)


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("onnxruntime")
    from fastapi.testclient import TestClient

    from mp.api.app import criar_app

    return TestClient(criar_app())


@pytest.fixture(scope="module")
def motores():
    from mp.ingest.cmapss import read_raw, read_rul_file

    df = read_raw(BRONZE / "test_FD001.txt")
    rul = read_rul_file(BRONZE / "RUL_FD001.txt")
    return df, rul


def _ciclos(g):
    cols = [c for c in g.columns if c not in ("unit_id", "cycle")]
    return g.sort_values("cycle")[cols].to_dict("records")


def test_rmse_pela_api_bate_com_o_offline(client, motores):
    df, rul_real = motores
    prev, real = [], []
    for uid, g in df.groupby("unit_id"):
        r = client.post("/rul", json={"unit_id": int(uid), "ciclos": _ciclos(g)}).json()
        prev.append(r["rul"])
        real.append(min(float(rul_real.loc[uid]), 125.0))
    rmse = float(np.sqrt(np.mean((np.array(prev) - np.array(real)) ** 2)))
    assert rmse < 16.0, rmse  # offline: ~13 (LSTM); aqui o RUL real do teste também é limitado a 125


def test_fila_tem_100_motores_ordenados(client):
    itens = client.get("/prioridade").json()["itens"]
    assert len(itens) == 100
    pri = [i["prioridade"] for i in itens]
    assert pri == sorted(pri, reverse=True)


def test_latencia_rul_p95_abaixo_de_50ms(client, motores):
    df, _ = motores
    g = df[df.unit_id == 1]
    corpo = {"unit_id": 1, "ciclos": _ciclos(g)[-30:]}
    for _ in range(5):
        client.post("/rul", json=corpo)  # aquecimento
    t = []
    for _ in range(200):
        t0 = time.perf_counter()
        client.post("/rul", json=corpo)
        t.append(time.perf_counter() - t0)
    assert np.percentile(t, 95) < 0.05, np.percentile(t, 95)


def test_latencia_triagem_p95_abaixo_de_300ms(client):
    corpo = {"texto": "Motor 12: vibração forte em compressor de alta (HPC-0412)."}
    client.post("/triagem", json=corpo)
    t = []
    for _ in range(200):
        t0 = time.perf_counter()
        client.post("/triagem", json=corpo)
        t.append(time.perf_counter() - t0)
    assert np.percentile(t, 95) < 0.3, np.percentile(t, 95)


@pytest.mark.skipif(not (SERVING / "bert_int8.onnx").exists(), reason="DistilBERT não exportado")
def test_triagem_com_distilbert_real_separa_critica_de_baixa(client):
    critica = client.post("/triagem", json={
        "texto": "Motor 5: falha total em compressor de alta (HPC-0412), fumaça visível."}).json()
    baixa = client.post("/triagem", json={
        "texto": "leve ruído no rolamento BRG-0007, inspeção de rotina sem anomalia"}).json()
    assert critica["gravidade"] in {"alta", "crítica"} and critica["p_grave"] > 0.5
    assert baixa["gravidade"] in {"baixa", "média"} and baixa["p_grave"] < 0.5
    assert critica["entidades"]["codigo"] == "HPC-0412"
