import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from mp.api.app import criar_app
from mp.api.metricas import Metricas
from mp.api.servico import ServicoRul, ServicoTriagem
from mp.models.triagem import baseline
from mp.nlp.preprocess import limpar
from mp.textgen.generator import gerar_os

SENSORES = ["s2", "s3", "s4", "s5", "s6"]
CFG = {"w_rul": 0.6, "w_os": 0.4, "rul_cap": 125, "rul_critico": 15,
       "psi_limiar": 0.2, "psi_min_sensores": 3}


class FakeRul:
    """RUL = 125 * (1 - média da janela): sensores altos = motor perto de falhar."""

    def predict(self, janelas):
        return 125 * (1 - janelas.mean(axis=(1, 2)))


def ciclo(valor=0.5, op=0.0):
    return {"op1": op, "op2": 0.0, "op3": 100.0, **{s: valor for s in SENSORES}}


@pytest.fixture(scope="module")
def triagem():
    rng = np.random.default_rng(0)
    oss = [gerar_os(i % 50, i, float(rng.integers(0, 125)), rng) for i in range(1500)]
    m = baseline(0).fit([limpar(o["texto"]) for o in oss], [o["gravidade"] for o in oss])
    return ServicoTriagem(m, {"baixa": 0.25, "média": 0.25, "alta": 0.25, "crítica": 0.25})


@pytest.fixture
def client(triagem):
    norm = {"sensors": SENSORES, "centers": [[0.0, 0.0, 100.0]],
            "min": [[0.0] * 5], "max": [[1.0] * 5]}
    rng = np.random.default_rng(0)
    ref = {s: rng.uniform(0, 1, 3000) for s in SENSORES}
    rul = ServicoRul(FakeRul(), norm, ref, limite_distancia=1.0, janela=30, cfg=CFG)
    return TestClient(criar_app(rul, triagem, Metricas(), CFG))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_rul_basico_e_clip_em_zero(client):
    r = client.post("/rul", json={"unit_id": 1, "ciclos": [ciclo(0.2)] * 30}).json()
    assert r["rul"] == pytest.approx(100.0, abs=1e-3)
    assert r["low_confidence"] is False
    r = client.post("/rul", json={"unit_id": 2, "ciclos": [ciclo(1.5)] * 30}).json()
    assert r["rul"] == 0.0  # previsão negativa vira 0


def test_rul_motor_curto_marca_baixa_confianca(client):
    r = client.post("/rul", json={"unit_id": 3, "ciclos": [ciclo()] * 10}).json()
    assert r["low_confidence"] is True


def test_rul_condicao_nunca_vista_nao_prevê_em_silencio(client):
    r = client.post("/rul", json={"unit_id": 4, "ciclos": [ciclo(op=50.0)] * 30}).json()
    assert r["condicao_desconhecida"] is True


@pytest.mark.parametrize("corpo", [
    {"unit_id": 1, "ciclos": []},
    {"unit_id": -1, "ciclos": [{"op1": 0}]},
    {"unit_id": 1},
    {"unit_id": "x", "ciclos": [{"op1": 0}]},
    {"unit_id": 1, "ciclos": [{"op1": 0.0, "op2": 0.0, "op3": 100.0, "s2": 0.1}]},  # faltam sensores
])
def test_rul_entrada_invalida_devolve_422(client, corpo):
    assert client.post("/rul", json=corpo).status_code == 422


def test_triagem_classifica_e_extrai_entidades(client):
    r = client.post("/triagem", json={"texto": "Motor 5: falha total em compressor de alta (HPC-0412)."})
    assert r.status_code == 200
    j = r.json()
    assert j["gravidade"] in {"baixa", "média", "alta", "crítica"}
    assert j["entidades"]["codigo"] == "HPC-0412"
    assert j["entidades"]["componente"] == "compressor de alta"
    assert j["revisao_humana"] is False
    assert sum(j["probabilidades"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("texto", ["", "   ", "HPC-0412", "the engine is overheating badly now"])
def test_triagem_casos_de_borda_vao_para_humano(client, texto):
    j = client.post("/triagem", json={"texto": texto}).json()
    assert j["gravidade"] == "indefinida"
    assert j["revisao_humana"] is True
    assert j["p_grave"] is None


def test_triagem_entrada_invalida(client):
    assert client.post("/triagem", json={}).status_code == 422
    assert client.post("/triagem", json={"texto": "x" * 5000}).status_code == 422


def test_fila_ordena_por_prioridade_e_combina_rul_com_os(client):
    client.post("/rul", json={"unit_id": 10, "ciclos": [ciclo(0.9)] * 30})  # RUL baixo
    client.post("/rul", json={"unit_id": 11, "ciclos": [ciclo(0.1)] * 30})  # RUL alto
    client.post("/triagem", json={"texto": "falha total em turbina de alta HPT-0001 fumaça visível",
                                  "unit_id": 11})
    itens = client.get("/prioridade").json()["itens"]
    ids = [i["unit_id"] for i in itens]
    assert ids[0] == 10 and set(ids) == {10, 11}
    assert [i["prioridade"] for i in itens] == sorted((i["prioridade"] for i in itens), reverse=True)
    assert itens[0]["critico"] is True
    assert itens[1]["p_grave"] is not None
    assert len(client.get("/prioridade?top=1").json()["itens"]) == 1


def test_drift_dispara_so_com_mais_de_3_sensores_deslocados(client):
    rng = np.random.default_rng(5)

    def ciclos_ref(n):
        return [{"op1": 0.0, "op2": 0.0, "op3": 100.0,
                 **{s: float(rng.uniform(0, 1)) for s in SENSORES}} for _ in range(n)]

    for u in range(10):  # tráfego normal: mesma distribuição da referência
        client.post("/rul", json={"unit_id": u, "ciclos": ciclos_ref(30)})
    normal = client.get("/drift").json()["sensores"]
    assert normal["drift"] is False and normal["sensores_acima"] == 0

    def deslocados(n_deslocar):
        cs = ciclos_ref(30)
        for c in cs:
            for s in SENSORES[:n_deslocar]:
                c[s] = 3.0
        return cs

    for u in range(40):  # 3 de 5 sensores fora da referência: ainda NÃO passa de "mais de 3"
        client.post("/rul", json={"unit_id": u, "ciclos": deslocados(3)})
    tres = client.get("/drift").json()["sensores"]
    assert tres["sensores_acima"] == 3 and tres["drift"] is False

    for u in range(40):  # 5 de 5 sensores deslocados: dispara
        r = client.post("/rul", json={"unit_id": u, "ciclos": deslocados(5)}).json()
    assert r["alerta_drift"] is True
    assert client.get("/drift").json()["sensores"]["drift"] is True
    assert "mp_sensores_drift 5.0" in client.get("/metrics").text


def test_metricas_expostas(client):
    client.post("/rul", json={"unit_id": 1, "ciclos": [ciclo()] * 30})
    client.post("/triagem", json={"texto": "vibração forte em rolamento BRG-0001"})
    corpo = client.get("/metrics").text
    for nome in ["mp_rul_ciclos{unit_id=\"1\"}", "mp_prioridade", "mp_psi", "mp_sensores_drift",
                 "mp_os_total", "mp_texto_oov_taxa", "mp_texto_psi_classes",
                 "mp_condicao_desconhecida_total", "http_request_duration_seconds_bucket"]:
        assert nome in corpo, nome


def test_metricas_nao_usam_texto_como_rotulo(client):
    client.post("/triagem", json={"texto": "segredo-super-unico em compressor"})
    assert "segredo-super-unico" not in client.get("/metrics").text


def _textos(rul, n, seed):
    rng = np.random.default_rng(seed)
    return [gerar_os(1, 1, float(rul), rng, p_troca=0.0)["texto"] for _ in range(n)]


def test_drift_de_texto_usa_baseline_de_producao_nao_o_treino(triagem):
    # população "natural" desbalanceada (muita OS baixa): não deve alarmar contra ela mesma
    svc = ServicoTriagem(triagem.modelo, n_base=60)
    rng = np.random.default_rng(3)
    natural = [gerar_os(1, 1, float(rng.choice([100, 100, 100, 60, 30, 5])), rng)["texto"]
               for _ in range(400)]
    for t in natural[:200]:
        svc.triar(t)
    assert svc.drift_texto()["baseline_pronta"] is True
    assert svc.drift_texto()["psi_classes"] < 0.2
    for t in _textos(5, 250, seed=4):  # de repente só OS críticas: a distribuição mudou
        svc.triar(t)
    assert svc.drift_texto()["psi_classes"] > 0.2


def test_drift_de_texto_zero_antes_da_baseline(triagem):
    svc = ServicoTriagem(triagem.modelo, n_base=100)
    for t in _textos(5, 20, seed=0):
        svc.triar(t)
    assert svc.drift_texto() == {"oov_taxa": pytest.approx(svc.drift_texto()["oov_taxa"]),
                                 "psi_classes": 0.0, "baseline_pronta": False}


def _amostra_rotulada(n, seed, embaralhar=False):
    rng = np.random.default_rng(seed)
    oss = [gerar_os(i % 50, i, float(rng.integers(0, 125)), rng) for i in range(n)]
    rotulos = [o["gravidade"] for o in oss]
    if embaralhar:  # simula o modelo envelhecido: a relacao texto -> gravidade deixou de valer
        rotulos = list(rng.permutation(rotulos))
    return [{"texto": o["texto"], "gravidade": g} for o, g in zip(oss, rotulos, strict=True)]


@pytest.fixture
def client_f1(triagem):
    from sklearn.metrics import f1_score

    ref = _amostra_rotulada(400, seed=99)
    f1_ref = f1_score([o["gravidade"] for o in ref],
                      triagem.modelo.predict([limpar(o["texto"]) for o in ref]), average="macro")
    svc = ServicoTriagem(triagem.modelo, f1_ref=float(f1_ref))
    norm = {"sensors": SENSORES, "centers": [[0.0, 0.0, 100.0]],
            "min": [[0.0] * 5], "max": [[1.0] * 5]}
    rul = ServicoRul(FakeRul(), norm, {s: np.random.default_rng(0).uniform(0, 1, 500) for s in SENSORES},
                     limite_distancia=1.0, janela=30, cfg=CFG)
    return TestClient(criar_app(rul, svc, Metricas(), {**CFG, "f1_queda_pontos": 10.0,
                                                       "f1_amostra_min": 30})), svc


def test_f1_amostra_boa_nao_dispara_o_gatilho(client_f1):
    client, _ = client_f1
    r = client.post("/triagem/avaliar", json={"itens": _amostra_rotulada(200, seed=7)}).json()
    assert r["n_avaliadas"] == 200 and r["amostra_pequena"] is False
    assert abs(r["queda_pontos"]) < 10
    assert r["gatilho_retreino"] is False


def test_f1_caindo_10_pontos_dispara_o_gatilho_e_as_metricas(client_f1):
    client, _ = client_f1
    r = client.post("/triagem/avaliar",
                    json={"itens": _amostra_rotulada(200, seed=7, embaralhar=True)}).json()
    assert r["queda_pontos"] >= 10 and r["gatilho_retreino"] is True
    corpo = client.get("/metrics").text
    assert "mp_f1_macro_amostra" in corpo and "mp_f1_queda_pontos" in corpo
    assert f"mp_amostra_rotulada_n {float(r['n_avaliadas'])}" in corpo


def test_f1_amostra_pequena_e_so_informativa(client_f1):
    client, _ = client_f1
    r = client.post("/triagem/avaliar",
                    json={"itens": _amostra_rotulada(10, seed=7, embaralhar=True)}).json()
    assert r["amostra_pequena"] is True and r["gatilho_retreino"] is False


def test_f1_nao_conta_os_indefinidas_e_nao_mexe_no_drift_de_texto(client_f1):
    client, svc = client_f1
    antes = (svc.total, len(svc.recentes), len(svc.oov))
    itens = _amostra_rotulada(60, seed=3) + [{"texto": "", "gravidade": "baixa"},
                                             {"texto": "the engine is overheating badly", "gravidade": "alta"}]
    r = client.post("/triagem/avaliar", json={"itens": itens}).json()
    assert r["n_indefinidas"] == 2 and r["n_avaliadas"] == 60
    assert (svc.total, len(svc.recentes), len(svc.oov)) == antes


@pytest.mark.parametrize("corpo", [
    {"itens": []},
    {"itens": [{"texto": "vibração forte", "gravidade": "urgente"}]},
    {"itens": [{"texto": "vibração forte"}]},
    {},
])
def test_f1_entrada_invalida_devolve_422(client_f1, corpo):
    assert client_f1[0].post("/triagem/avaliar", json=corpo).status_code == 422


def test_f1_sem_referencia_devolve_409(triagem):
    svc = ServicoTriagem(triagem.modelo)  # sem f1_ref
    norm = {"sensors": SENSORES, "centers": [[0.0, 0.0, 100.0]],
            "min": [[0.0] * 5], "max": [[1.0] * 5]}
    rul = ServicoRul(FakeRul(), norm, {s: np.zeros(10) for s in SENSORES}, 1.0, 30, CFG)
    c = TestClient(criar_app(rul, svc, Metricas(), CFG))
    r = c.post("/triagem/avaliar", json={"itens": _amostra_rotulada(5, seed=1)})
    assert r.status_code == 409
