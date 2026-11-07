import numpy as np
import pandas as pd
import pytest

from mp.nlp.ner import normalizar_codigo
from mp.textgen.generator import CODIGO_RE, amostrar_ciclos, gerar_os, gravidade_por_rul


@pytest.mark.parametrize(
    "rul,esperado",
    [(0, 3), (14.9, 3), (15, 2), (39, 2), (40, 1), (79, 1), (80, 0), (125, 0)],
)
def test_gravidade_por_rul(rul, esperado):
    assert gravidade_por_rul(rul) == esperado


def test_gerar_os_determinista():
    a = gerar_os(7, 50, 20.0, np.random.default_rng(1))
    b = gerar_os(7, 50, 20.0, np.random.default_rng(1))
    assert a == b


def test_codigo_da_peca_esta_no_texto():
    rng = np.random.default_rng(0)
    for i in range(300):
        os_ = gerar_os(i, i, float(i % 120), rng)
        achado = CODIGO_RE.search(os_["texto"])
        assert achado is not None
        assert normalizar_codigo(achado.group()) == normalizar_codigo(os_["entidades"]["codigo"])


def test_taxa_de_rotulos_trocados_perto_de_10_por_cento():
    rng = np.random.default_rng(0)
    oss = [gerar_os(1, 1, 100.0, rng) for _ in range(4000)]
    taxa = np.mean([o["gravidade"] != o["gravidade_ref"] for o in oss])
    assert 0.07 < taxa < 0.13
    assert all(o["gravidade_ref"] == "baixa" for o in oss)


def test_amostrar_ciclos_balanceia_faixas():
    rul = np.arange(200)[::-1]
    df = pd.DataFrame({"unit_id": 1, "cycle": np.arange(200), "rul": rul})
    amostra = amostrar_ciclos(df, 4, np.random.default_rng(0))
    contagem = amostra["rul"].map(gravidade_por_rul).value_counts()
    assert len(amostra) == 16
    assert (contagem == 4).all()
