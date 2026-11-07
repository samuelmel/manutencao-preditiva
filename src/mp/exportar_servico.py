"""Exporta os artefatos que a API carrega em artifacts/serving.

Uso: python -m mp.exportar_servico [--fd FD001]
Precisa de: mp.pipeline, mp.textgen.build, mp.train_triagem e mp.train_rul já executados.
"""
from __future__ import annotations

import argparse
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from mp.config import ROOT, load_config
from mp.models.triagem import baseline, dividir_por_motor
from mp.textgen.lexicon import GRAVIDADES

OP_COLS = ["op1", "op2", "op3"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    a = ap.parse_args()
    cfg = load_config("data")
    seed = cfg["seed"]
    out = ROOT / "artifacts" / "serving"
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy(ROOT / "artifacts" / a.fd / "lstm.onnx", out / "lstm.onnx")
    silver = ROOT / cfg["paths"]["silver"] / f"fd={a.fd}"
    shutil.copy(silver / "normalizer.json", out / "normalizer.json")
    norm = json.loads((silver / "normalizer.json").read_text(encoding="utf-8"))

    # referência de drift: amostra dos sensores já normalizados do treino
    train = pd.read_parquet(silver / "train.parquet")
    rng = np.random.default_rng(seed)
    amostra = train.iloc[rng.choice(len(train), size=min(3000, len(train)), replace=False)]
    np.savez_compressed(out / "drift_ref.npz", **{s: amostra[s].to_numpy() for s in norm["sensors"]})
    centros = np.asarray(norm["centers"], dtype=float)
    dist = np.linalg.norm(train[OP_COLS].to_numpy()[:, None, :] - centros[None], axis=2).min(axis=1)

    # triagem: baseline TF-IDF + LR treinado só nos motores de treino
    os_df = pd.read_parquet(ROOT / "data" / "silver_text" / "os.parquet")
    tr, _ = dividir_por_motor(os_df, seed)
    modelo = baseline(seed).fit(tr["texto_limpo"], tr["gravidade"])
    joblib.dump(modelo, out / "triagem_tfidf.joblib", compress=3)
    prop = tr["gravidade"].value_counts(normalize=True)
    meta = {
        "fd": a.fd,
        "janela": cfg["window"],
        "limite_distancia": float(dist.max() * 1.5 + 1e-9),
        "classes_ref": {g: float(prop.get(g, 0.0)) for g in GRAVIDADES},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tamanho = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"artefatos em {out} ({tamanho:.2f} MB):", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
