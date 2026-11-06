"""Camada bronze: copia os arquivos C-MAPSS sem alterá-los e registra checksums."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle", *OP_COLS, *SENSOR_COLS]


def raw_files(fd: str) -> list[str]:
    return [f"train_{fd}.txt", f"test_{fd}.txt", f"RUL_{fd}.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_bronze(src: Path, bronze: Path, subsets: list[str]) -> dict[str, str]:
    """Copia os .txt para bronze/ byte a byte. Bronze é imutável: se o arquivo já
    existe com checksum diferente, falha em vez de sobrescrever."""
    bronze.mkdir(parents=True, exist_ok=True)
    manifest_path = bronze / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for fd in subsets:
        for name in raw_files(fd):
            origem = src / name
            if not origem.exists():
                raise FileNotFoundError(f"arquivo C-MAPSS ausente: {origem}")
            destino = bronze / name
            digest = sha256(origem)
            if destino.exists():
                if sha256(destino) != digest:
                    raise ValueError(f"bronze é imutável; {name} difere do original")
            else:
                shutil.copyfile(origem, destino)
            manifest[name] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def read_raw(path: Path) -> pd.DataFrame:
    """Lê um train_/test_FDxxx.txt (separado por espaço, 26 colunas)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df["unit_id"] = df["unit_id"].astype("int32")
    df["cycle"] = df["cycle"].astype("int32")
    return df


def read_rul_file(path: Path) -> pd.Series:
    """RUL real do último ciclo de cada motor de teste (índice = unit_id)."""
    s = pd.read_csv(path, header=None)[0].astype("int32")
    s.index = pd.RangeIndex(1, len(s) + 1, name="unit_id")
    return s
