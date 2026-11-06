"""Treina baselines + LSTM em um subconjunto, avalia no teste e exporta ONNX.

Uso: python -m mp.train_rul --fd FD001 [--epochs 100]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")  # antes de importar keras

from mp.config import ROOT, load_config
from mp.models.baselines import MeanBaseline, RFBaseline
from mp.models.metrics import nasa_score, rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    cfg = load_config("data")
    seed = cfg["seed"]
    d = np.load(ROOT / cfg["paths"]["gold"] / f"fd={a.fd}" / "windows.npz")
    xtr, ytr, xva, yva, xte, yte = (d[k] for k in
                                    ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"])
    out = ROOT / a.out / a.fd
    out.mkdir(parents=True, exist_ok=True)
    res = {}

    def report(name, pred):
        res[name] = {"rmse": rmse(yte, pred), "score": nasa_score(yte, pred)}
        print(name, res[name])

    report("media", MeanBaseline().fit(xtr, ytr).predict(xte))
    report("random_forest", RFBaseline(seed).fit(xtr, ytr).predict(xte))

    import keras
    import torch

    from mp.models.lstm import RUL_CAP, build_lstm

    keras.utils.set_random_seed(seed)
    model = build_lstm(xtr.shape[1], xtr.shape[2])
    model.fit(
        xtr, ytr / RUL_CAP, validation_data=(xva, yva / RUL_CAP), epochs=a.epochs, batch_size=256, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )
    pred = np.clip(model.predict(xte, verbose=0).ravel(), 0, None)  # RUL negativo -> 0
    report("lstm", pred)

    # ONNX via PyTorch: embrulha o modelo Keras (backend torch) como nn.Module.
    model.save(out / "lstm.keras")
    torch.onnx.export(
        model, (torch.zeros(1, xtr.shape[1], xtr.shape[2]),), str(out / "lstm.onnx"),
        input_names=["janela"], output_names=["rul"],
        dynamic_axes={"janela": {0: "batch"}, "rul": {0: "batch"}}, dynamo=False,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "lstm.onnx"))
    onnx_pred = np.clip(sess.run(None, {"janela": xte})[0].ravel(), 0, None)
    res["paridade_onnx_max_abs"] = float(np.abs(onnx_pred - pred).max())
    print("paridade ONNX:", res["paridade_onnx_max_abs"])
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(Path(out / "lstm.onnx").stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
