import numpy as np
import pytest

from mp.config import load_config
from mp.ingest.cmapss import ingest_bronze
from mp.pipeline import run


def _cfg(raw_dir):
    cfg = load_config("data")
    cfg["subsets"] = ["FD001", "FD002"]
    cfg["paths"]["raw_src"] = str(raw_dir)
    return cfg


def test_end_to_end(raw_dir, tmp_path):
    run(_cfg(raw_dir), base=tmp_path)
    for fd in ["FD001", "FD002"]:
        d = np.load(tmp_path / f"data/gold/fd={fd}/windows.npz")
        assert not set(d["train_units"]) & set(d["val_units"])
        for k in ["x_train", "x_val", "x_test"]:
            assert not np.isnan(d[k]).any()
        assert d["x_train"].shape[1:] == (30, d["x_train"].shape[2])
        assert (d["y_train"] <= 125).all()


def test_bronze_is_byte_identical_and_immutable(raw_dir, tmp_path):
    bronze = tmp_path / "bronze"
    ingest_bronze(raw_dir, bronze, ["FD001"])
    assert (bronze / "train_FD001.txt").read_bytes() == (raw_dir / "train_FD001.txt").read_bytes()
    (raw_dir / "train_FD001.txt").write_text("alterado")
    with pytest.raises(ValueError):
        ingest_bronze(raw_dir, bronze, ["FD001"])
