import numpy as np
import pandas as pd

from mp.ingest.cmapss import SENSOR_COLS, read_raw
from mp.transform.silver import Normalizer, add_rul, add_test_rul, constant_sensors


def test_rul_and_cap():
    df = pd.DataFrame({"unit_id": [1] * 200, "cycle": range(1, 201)})
    out = add_rul(df, cap=125)
    assert out["rul"].iloc[-1] == 0
    assert out["rul"].max() == 125
    assert add_rul(df, cap=None)["rul"].iloc[0] == 199


def test_test_rul_uses_final_value():
    df = pd.DataFrame({"unit_id": [1] * 10, "cycle": range(1, 11)})
    out = add_test_rul(df, pd.Series([20], index=[1]), cap=125)
    assert out["rul"].iloc[-1] == 20 and out["rul"].iloc[0] == 29


def test_constant_sensor_detected_from_data(raw_dir):
    df = read_raw(raw_dir / "train_FD001.txt")
    assert constant_sensors(df) == ["s1"]


def test_normalizer_fit_on_train_only_and_no_nan(raw_dir):
    tr = read_raw(raw_dir / "train_FD002.txt")
    te = read_raw(raw_dir / "test_FD002.txt")
    feats = [c for c in SENSOR_COLS if c != "s1"]
    n = Normalizer(feats, n_clusters=6, seed=1).fit(tr)
    a = n.transform(tr)
    assert not a[feats].isna().any().any()
    assert a[feats].min().min() >= 0 and a[feats].max().max() <= 1 + 1e-9
    assert not n.transform(te)[feats].isna().any().any()
    # estatísticas vêm do treino: alterar o teste não muda o ajuste
    lo = n.min_.copy()
    n.transform(te * 1000)
    assert np.array_equal(lo, n.min_)


def test_cluster_distance_flags_unseen_condition(raw_dir):
    tr = read_raw(raw_dir / "train_FD002.txt")
    n = Normalizer(["s2"], n_clusters=6, seed=1).fit(tr)
    far = tr.head(3).copy()
    far["op3"] = 10_000
    assert n.cluster_distance(far).min() > 10 * n.cluster_distance(tr).max() / 10
