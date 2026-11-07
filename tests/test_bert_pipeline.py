"""Teste de fumaça do pipeline do DistilBERT com um modelo minúsculo de pesos aleatórios.

Roda treino -> ONNX fp32 -> INT8 -> serviço sem baixar nada. Serve para o treino real (Colab)
não falhar no meio por causa de código, e prova que o serviço lê o que o treino exporta.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

import onnxruntime as ort
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from mp.api.servico import ClassificadorOnnx
from mp.models import bert
from mp.textgen.generator import gerar_os
from mp.textgen.lexicon import GRAVIDADES

MAX_LEN = 48


def _tokenizer(textos):
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    esp = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tk.train_from_iterator(textos, trainers.WordPieceTrainer(vocab_size=400, special_tokens=esp))
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tk.token_to_id("[CLS]")),
                                                  ("[SEP]", tk.token_to_id("[SEP]"))])
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def _dados(n, seed=0):
    rng = np.random.default_rng(seed)
    oss = [gerar_os(i % 60, i, float(rng.integers(0, 125)), rng) for i in range(n)]
    return [o["texto"] for o in oss], [GRAVIDADES.index(o["gravidade"]) for o in oss]


def _modelo_minusculo(textos):
    tok = _tokenizer(textos)
    cfg = DistilBertConfig(vocab_size=tok.vocab_size, dim=32, hidden_dim=64, n_layers=2, n_heads=2,
                           max_position_embeddings=64, num_labels=4,
                           id2label=dict(enumerate(GRAVIDADES)))
    return DistilBertForSequenceClassification(cfg), tok


def test_laco_de_treino_consegue_memorizar_um_lote_pequeno(monkeypatch):
    """Um laço quebrado (rótulos desalinhados, passo sem efeito) não memoriza 96 exemplos."""
    textos, y = _dados(96, seed=1)
    modelo0, tok = _modelo_minusculo(textos)
    monkeypatch.setattr(bert, "carregar_modelo", lambda *a, **k: (modelo0, tok))
    _, _, hist = bert.treinar(textos, y, textos, y, seed=0, epocas=40, batch=32, lr=3e-3,
                              max_len=MAX_LEN, log=lambda *_: None)
    assert hist[-1]["perda_treino"] < 0.5 * hist[0]["perda_treino"]
    assert max(h["f1_macro_val"] for h in hist) > 0.7  # acaso = 0,25


@pytest.fixture(scope="module")
def treinado(tmp_path_factory, request):
    textos, y = _dados(900)
    modelo0, tok = _modelo_minusculo(textos)
    mp = pytest.MonkeyPatch()
    mp.setattr(bert, "carregar_modelo", lambda *a, **k: (modelo0, tok))
    request.addfinalizer(mp.undo)
    modelo, tok, hist = bert.treinar(textos[:700], y[:700], textos[700:], y[700:], seed=0, epocas=2,
                                     batch=32, lr=2e-3, max_len=MAX_LEN, log=lambda *_: None)
    pasta = tmp_path_factory.mktemp("bert")
    bert.exportar_onnx(modelo, tok, pasta / "bert_fp32.onnx", MAX_LEN)
    bert.quantizar_int8(pasta / "bert_fp32.onnx", pasta / "bert_int8.onnx")
    tok.backend_tokenizer.save(str(pasta / "tokenizer.json"))
    return modelo, tok, hist, pasta, textos[700:]


def test_treino_reduz_a_perda_e_guarda_historico(treinado):
    _, _, hist, _, _ = treinado
    assert [h["epoca"] for h in hist] == [1, 2]
    assert hist[-1]["perda_treino"] < hist[0]["perda_treino"]


def test_onnx_fp32_bate_com_o_torch(treinado):
    modelo, tok, _, pasta, val = treinado
    amostra = val[:40]
    sess = ort.InferenceSession(str(pasta / "bert_fp32.onnx"), providers=["CPUExecutionProvider"])
    enc = tok(amostra, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="np")
    lg = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                         "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
    p_onnx = np.exp(lg - lg.max(1, keepdims=True))
    p_onnx /= p_onnx.sum(1, keepdims=True)
    assert np.abs(p_onnx - bert.prever_proba(modelo, tok, amostra, MAX_LEN)).max() < 1e-4


def test_onnx_aceita_lotes_e_comprimentos_diferentes(treinado):
    _, tok, _, pasta, val = treinado
    sess = ort.InferenceSession(str(pasta / "bert_int8.onnx"), providers=["CPUExecutionProvider"])
    for lote in (val[:1], val[:7], ["ok"], [val[0] * 3]):
        enc = tok(lote, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="np")
        lg = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                             "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        assert lg.shape == (len(lote), 4) and np.isfinite(lg).all()


def test_servico_le_o_que_o_treino_exportou(treinado):
    _, tok, _, pasta, val = treinado
    clf = ClassificadorOnnx.de_pasta(pasta, GRAVIDADES, MAX_LEN)
    amostra = val[:20]
    p = clf.predict_proba(amostra)
    assert p.shape == (20, 4) and p.sum(axis=1) == pytest.approx(np.ones(20), abs=1e-5)
    sess = ort.InferenceSession(str(pasta / "bert_int8.onnx"), providers=["CPUExecutionProvider"])
    enc = tok(amostra, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="np")
    lg = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                         "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
    # tokenizers (serviço) e transformers (treino) têm de produzir os mesmos ids
    assert (p.argmax(1) == lg.argmax(1)).all()


def test_int8_e_menor_que_fp32(treinado):
    _, _, _, pasta, _ = treinado
    assert (pasta / "bert_int8.onnx").stat().st_size < 0.8 * (pasta / "bert_fp32.onnx").stat().st_size
