"""DistilBERT multilíngue: fine-tuning da gravidade da OS, exportação ONNX e quantização INT8.

Importar este módulo exige o extra `bert` (transformers + torch). O serviço NÃO importa isto:
ele só usa o ONNX INT8 e o `tokenizer.json` (ver `mp.api.servico.ClassificadorOnnx`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mp.textgen.lexicon import GRAVIDADES

MODELO = "distilbert-base-multilingual-cased"


def _lotes(n: int, tamanho: int, rng: np.random.Generator | None = None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, tamanho):
        yield idx[i : i + tamanho]


def carregar_modelo(pasta_ou_nome: str = MODELO):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(pasta_ou_nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        pasta_ou_nome, num_labels=len(GRAVIDADES),
        id2label=dict(enumerate(GRAVIDADES)), label2id={g: i for i, g in enumerate(GRAVIDADES)},
    )
    return modelo, tok


def prever_proba(modelo, tok, textos: list[str], max_len: int = 64, batch: int = 64) -> np.ndarray:
    import torch

    modelo.eval()
    out = []
    with torch.no_grad():
        for ix in _lotes(len(textos), batch):
            enc = tok([textos[i] for i in ix], truncation=True, padding=True, max_length=max_len,
                      return_tensors="pt")
            out.append(torch.softmax(modelo(**enc).logits, dim=-1).numpy())
    return np.vstack(out)


def f1_macro(y_true: list[int], y_pred: list[int]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=list(range(len(GRAVIDADES))), average="macro",
                          zero_division=0))


def treinar(textos_tr: list[str], y_tr: list[int], textos_va: list[str], y_va: list[int], *,
            seed: int = 42, epocas: int = 3, batch: int = 32, lr: float = 5e-5, max_len: int = 64,
            checkpoint: Path | None = None, log=print):
    """Fine-tuning com AdamW + decaimento linear. Guarda o melhor estado pela F1-macro de validação."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    modelo, tok = carregar_modelo()
    otim = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    passos = epocas * ((len(textos_tr) + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(otim, int(0.1 * passos), passos)
    y_tr_t = torch.tensor(y_tr)
    melhor, melhor_estado, hist = -1.0, None, []
    for ep in range(1, epocas + 1):
        modelo.train()
        perda_total = 0.0
        for k, ix in enumerate(_lotes(len(textos_tr), batch, rng), 1):
            enc = tok([textos_tr[i] for i in ix], truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt")
            perda = modelo(**enc, labels=y_tr_t[ix]).loss
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otim.step()
            sched.step()
            otim.zero_grad()
            perda_total += float(perda)
            if k % 50 == 0:
                log(f"época {ep} passo {k} perda média {perda_total / k:.4f}")
        pred = prever_proba(modelo, tok, textos_va, max_len).argmax(axis=1).tolist()
        f1 = f1_macro(y_va, pred)
        hist.append({"epoca": ep, "perda_treino": perda_total / k, "f1_macro_val": f1})
        log(f"== época {ep}: perda {perda_total / k:.4f} | F1-macro validação {f1:.4f}")
        if f1 > melhor:
            melhor = f1
            melhor_estado = {k_: v.detach().clone() for k_, v in modelo.state_dict().items()}
            if checkpoint:  # sessão que cai não perde a melhor época
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(melhor_estado, checkpoint)
    modelo.load_state_dict(melhor_estado)
    return modelo, tok, hist


def exportar_onnx(modelo, tok, destino: Path, max_len: int = 64) -> None:
    """ONNX fp32 com batch e comprimento dinâmicos; entradas `input_ids` e `attention_mask`."""
    import torch

    modelo.eval()
    enc = tok(["falha total em compressor", "leve ruído"], padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")

    class Envoltorio(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    destino.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Envoltorio(modelo), (enc["input_ids"], enc["attention_mask"]), str(destino),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def quantizar_int8(onnx_fp32: Path, onnx_int8: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8)
