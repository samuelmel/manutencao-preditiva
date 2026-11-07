"""Fase 3: fine-tuning do DistilBERT multilíngue, ONNX fp32 e INT8, comparação com o baseline.

Uso: python -m mp.train_bert [--epocas 3] [--max-len 64] [--batch 32]
Precisa do extra `bert` e de `python -m mp.train_triagem` já executado (silver de texto).
Roda em CPU. A divisão treino/teste é a MESMA do baseline (por motor, mesma seed).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from mp.config import ROOT, load_config
from mp.models.triagem import avaliar, dividir_por_motor
from mp.textgen.lexicon import GRAVIDADES


class Adaptador:
    """Dá a um classificador por probabilidades a interface `.predict` que `avaliar` espera."""

    def __init__(self, proba_fn):
        self.proba_fn = proba_fn

    def predict(self, textos):
        return np.array(GRAVIDADES)[self.proba_fn(list(textos)).argmax(axis=1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()

    from mp.models import bert

    cfg = load_config("data")
    seed = cfg["seed"]
    df = pd.read_parquet(ROOT / "data" / "silver_text" / "os.parquet")
    treino, teste = dividir_por_motor(df, seed)
    treino, val = dividir_por_motor(treino, seed, test_size=0.1)  # validação por motor, dentro do treino
    idx = {g: i for i, g in enumerate(GRAVIDADES)}
    out = ROOT / "artifacts" / "bert"
    out.mkdir(parents=True, exist_ok=True)
    print(f"treino {len(treino)} | validação {len(val)} | teste {len(teste)} OS")

    t0 = time.time()
    modelo, tok, hist = bert.treinar(
        treino["texto"].tolist(), treino["gravidade"].map(idx).tolist(),
        val["texto"].tolist(), val["gravidade"].map(idx).tolist(),
        seed=seed, epocas=a.epocas, batch=a.batch, lr=a.lr, max_len=a.max_len,
        checkpoint=out / "melhor.pt",
    )
    res: dict = {"modelo": bert.MODELO, "epocas": a.epocas, "max_len": a.max_len, "batch": a.batch,
                 "lr": a.lr, "treino_s": round(time.time() - t0), "historico": hist,
                 "n_treino": len(treino), "n_val": len(val), "n_teste": len(teste)}

    proba = lambda ts: bert.prever_proba(modelo, tok, ts, a.max_len)
    res["distilbert_fp32"] = avaliar(Adaptador(proba), teste, "texto")
    modelo.save_pretrained(out / "hf")
    tok.save_pretrained(out / "hf")
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))  # o serviço só precisa deste arquivo

    bert.exportar_onnx(modelo, tok, out / "bert_fp32.onnx", a.max_len)
    bert.quantizar_int8(out / "bert_fp32.onnx", out / "bert_int8.onnx")
    res["tamanho_mb"] = {n: round((out / f"bert_{n}.onnx").stat().st_size / 1e6, 1)
                         for n in ("fp32", "int8")}

    import onnxruntime as ort

    def proba_onnx(caminho):
        sess = ort.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])

        def f(textos):
            saidas = []
            for i in range(0, len(textos), 64):
                enc = tok(textos[i : i + 64], truncation=True, padding=True, max_length=a.max_len,
                          return_tensors="np")
                lg = sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                     "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
                e = np.exp(lg - lg.max(axis=1, keepdims=True))
                saidas.append(e / e.sum(axis=1, keepdims=True))
            return np.vstack(saidas)

        return f

    f32, f8 = proba_onnx(out / "bert_fp32.onnx"), proba_onnx(out / "bert_int8.onnx")
    amostra = teste["texto"].tolist()[:200]
    p_torch, p_32, p_8 = proba(amostra), f32(amostra), f8(amostra)
    res["paridade"] = {
        "onnx_fp32_vs_torch_max_abs_prob": float(np.abs(p_torch - p_32).max()),
        "int8_concorda_com_fp32": float((p_8.argmax(1) == p_32.argmax(1)).mean()),
    }
    res["distilbert_int8"] = avaliar(Adaptador(f8), teste, "texto")
    for nome in ("distilbert_fp32", "distilbert_int8"):
        r = res[nome]
        print(nome, "| F1-macro ruidoso", round(r["ruidoso"]["f1_macro"], 3), "| referência",
              round(r["referencia"]["f1_macro"], 3), "| recall crítica (ruidoso)",
              round(r["ruidoso"]["recall"]["crítica"], 3), "| referência",
              round(r["referencia"]["recall"]["crítica"], 3))
    print("paridade:", res["paridade"], "| tamanhos MB:", res["tamanho_mb"])
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
