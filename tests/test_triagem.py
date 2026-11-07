import numpy as np
import pandas as pd
import pytest

from mp.models.triagem import (
    avaliar,
    baseline,
    dividir_por_motor,
    grupo_motor,
    prob_alta_ou_critica,
)
from mp.nlp.preprocess import limpar
from mp.textgen.generator import gerar_os


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    linhas = []
    for u in range(1, 61):
        for c in range(8):
            o = gerar_os(u, c, float(rng.integers(0, 125)), rng)
            linhas.append({"fd": "FD001", "unit_id": u, "texto_limpo": limpar(o["texto"]),
                           "gravidade": o["gravidade"], "gravidade_ref": o["gravidade_ref"]})
    return pd.DataFrame(linhas)


def test_divisao_por_motor_sem_vazamento(df):
    tr, te = dividir_por_motor(df, seed=0)
    assert set(grupo_motor(tr)).isdisjoint(set(grupo_motor(te)))
    assert len(tr) + len(te) == len(df)


def test_baseline_treina_e_avalia(df):
    tr, te = dividir_por_motor(df, seed=0)
    m = baseline(0).fit(tr["texto_limpo"], tr["gravidade"])
    res = avaliar(m, te, "texto_limpo")
    assert 0.3 < res["ruidoso"]["f1_macro"] < 1.0  # acima do acaso (0.25), não perfeito
    assert set(res["ruidoso"]["recall"]) == {"baixa", "média", "alta", "crítica"}
    p = prob_alta_ou_critica(m, te["texto_limpo"])
    assert p.shape == (len(te),) and ((p >= 0) & (p <= 1)).all()


def test_word2vec_acha_a_propria_os_como_mais_parecida():
    pytest.importorskip("gensim")
    from mp.nlp.topicos import matriz_vetores, os_semelhantes, treinar_word2vec

    corpus = [["vibração", "forte", "compressor"], ["falha", "total", "turbina"],
              ["leve", "ruído", "rolamento"], ["vibração", "forte", "turbina"]] * 20
    w2v = treinar_word2vec(corpus, seed=0)
    vet = matriz_vetores(w2v, corpus)
    idx, sim = os_semelhantes(w2v, vet, corpus[1], k=1)
    assert corpus[idx[0]] == corpus[1]
    assert sim[0] > 0.99
