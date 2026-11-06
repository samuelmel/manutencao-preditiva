"""Orquestra bronze -> silver -> gold. Uso: python -m mp.pipeline"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mp.config import ROOT, load_config
from mp.ingest.cmapss import ingest_bronze, read_raw, read_rul_file
from mp.transform.silver import build_silver
from mp.transform.windows import make_windows, split_units


def run(cfg: dict | None = None, base: Path = ROOT) -> None:
    cfg = cfg or load_config("data")
    p = {k: base / v for k, v in cfg["paths"].items()}
    ingest_bronze(p["raw_src"], p["bronze"], cfg["subsets"])
    for fd in cfg["subsets"]:
        n_cl = cfg["n_clusters"] if fd in cfg["multi_condition"] else 1
        train, test, norm = build_silver(
            read_raw(p["bronze"] / f"train_{fd}.txt"),
            read_raw(p["bronze"] / f"test_{fd}.txt"),
            read_rul_file(p["bronze"] / f"RUL_{fd}.txt"),
            cap=cfg["rul_cap"],
            n_clusters=n_cl,
            threshold=cfg["variance_threshold"],
            seed=cfg["seed"],
        )
        out = p["silver"] / f"fd={fd}"
        out.mkdir(parents=True, exist_ok=True)
        train.to_parquet(out / "train.parquet", index=False)
        test.to_parquet(out / "test.parquet", index=False)
        (out / "normalizer.json").write_text(
            json.dumps(
                {
                    "sensors": norm.sensors,
                    "centers": norm.centers_.tolist(),
                    "min": norm.min_.tolist(),
                    "max": norm.max_.tolist(),
                }
            )
        )
        tr_ids, va_ids = split_units(train["unit_id"], cfg["val_fraction"], cfg["seed"])
        w = cfg["window"]
        xtr, ytr, *_ = make_windows(train[train.unit_id.isin(tr_ids)], norm.sensors, w)
        xva, yva, *_ = make_windows(train[train.unit_id.isin(va_ids)], norm.sensors, w)
        xte, yte, ute, lc = make_windows(test, norm.sensors, w, last_only=True)
        g = p["gold"] / f"fd={fd}"
        g.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            g / "windows.npz",
            x_train=xtr, y_train=ytr, x_val=xva, y_val=yva,
            x_test=xte, y_test=yte, unit_test=ute, low_conf_test=lc,
            train_units=tr_ids, val_units=va_ids,
        )
        print(f"{fd}: train={xtr.shape} val={xva.shape} test={xte.shape}")


if __name__ == "__main__":
    run()
