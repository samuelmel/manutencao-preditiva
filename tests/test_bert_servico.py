"""Serviço com DistilBERT: lógica testada com ONNX e tokenizador falsos (sem baixar modelo)."""
import numpy as np
import pytest

from mp.api.servico import ClassificadorOnnx, ServicoTriagem
from mp.models.triagem import baseline
from mp.nlp.preprocess import limpar
from mp.textgen.generator import gerar_os
from mp.textgen.lexicon import GRAVIDADES
from mp.train_bert import Adaptador


class Enc:
    def __init__(self, tokens):
        self.ids = [hash(t) % 1000 for t in tokens]
        self.attention_mask = [1] * len(tokens)


class FakeTok:
    def __init__(self):
        self.truncation = self.padding = False

    def enable_truncation(self, max_length):
        self.truncation = max_length

    def enable_padding(self):
        self.padding = True

    def encode_batch(self, textos):
        enc = [Enc(t.split()) for t in textos]
        n = max(len(e.ids) for e in enc)  # padding até o mais longo, como o tokenizers real
        for e in enc:
            e.attention_mask = [1] * len(e.ids) + [0] * (n - len(e.ids))
            e.ids = e.ids + [0] * (n - len(e.ids))
        return enc


class FakeSess:
    """Devolve logits fixos que favorecem a classe `alvo`; registra o que recebeu."""

    def __init__(self, alvo):
        self.alvo, self.chamadas = alvo, []

    def run(self, _saidas, entradas):
        self.chamadas.append(entradas)
        lg = np.zeros((entradas["input_ids"].shape[0], len(GRAVIDADES)), dtype=np.float32)
        lg[:, GRAVIDADES.index(self.alvo)] = 6.0
        return [lg]


@pytest.fixture(scope="module")
def tfidf():
    rng = np.random.default_rng(0)
    oss = [gerar_os(i % 50, i, float(rng.integers(0, 125)), rng) for i in range(1200)]
    return baseline(0).fit([limpar(o["texto"]) for o in oss], [o["gravidade"] for o in oss])


def test_classificador_onnx_softmax_ordem_e_formato():
    sess = FakeSess("crítica")
    clf = ClassificadorOnnx(sess, FakeTok(), GRAVIDADES, max_len=32)
    p = clf.predict_proba(["falha total no compressor", "ok"])
    assert p.shape == (2, 4)
    assert p.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert p.argmax(axis=1).tolist() == [3, 3]
    ent = sess.chamadas[0]
    assert ent["input_ids"].dtype == np.int64 and ent["attention_mask"].dtype == np.int64
    assert ent["input_ids"].shape == ent["attention_mask"].shape == (2, 4)  # padding no lote


def test_triagem_usa_o_classificador_e_calcula_p_grave(tfidf):
    clf = ClassificadorOnnx(FakeSess("alta"), FakeTok(), GRAVIDADES)
    svc = ServicoTriagem(tfidf, classificador=clf)
    r = svc.triar("vibração forte em compressor de alta (HPC-0412)")
    assert r["gravidade"] == "alta"
    assert r["p_grave"] == pytest.approx(r["probabilidades"]["alta"] + r["probabilidades"]["crítica"])
    assert r["p_grave"] > 0.99
    assert r["entidades"]["codigo"] == "HPC-0412"


def test_classificador_recebe_o_texto_bruto_nao_o_limpo(tfidf):
    sess = FakeSess("baixa")
    svc = ServicoTriagem(tfidf, classificador=ClassificadorOnnx(sess, FakeTok(), GRAVIDADES))
    svc.triar("  Rolam. com TEMP alta, vibração forte  ")
    assert len(sess.chamadas[0]["input_ids"][0]) == 6  # "Rolam. com TEMP alta, vibração forte"


@pytest.mark.parametrize("texto", ["", "HPC-0412", "the engine is overheating badly now"])
def test_casos_de_borda_nao_chamam_o_classificador(tfidf, texto):
    sess = FakeSess("crítica")
    svc = ServicoTriagem(tfidf, classificador=ClassificadorOnnx(sess, FakeTok(), GRAVIDADES))
    r = svc.triar(texto)
    assert r["gravidade"] == "indefinida" and r["p_grave"] is None
    assert sess.chamadas == []


def test_sem_classificador_continua_no_baseline(tfidf):
    r = ServicoTriagem(tfidf).triar("falha total em turbina de alta HPT-0001 fumaça visível")
    assert r["gravidade"] in GRAVIDADES


def test_adaptador_devolve_rotulos_na_ordem_das_classes():
    proba = lambda ts: np.array([[0.1, 0.2, 0.3, 0.4]] * len(ts))
    assert Adaptador(proba).predict(["a", "b"]).tolist() == ["crítica", "crítica"]
