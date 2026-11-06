import numpy as np
import pandas as pd

from mp.transform.windows import make_windows, split_units


def _df(n, uid=1):
    return pd.DataFrame(
        {"unit_id": uid, "cycle": range(1, n + 1), "s2": np.arange(n, dtype=float),
         "rul": np.arange(n)[::-1]}
    )


def test_window_shape_and_target():
    x, y, _, lc = make_windows(_df(50), ["s2"], 30)
    assert x.shape == (21, 30, 1) and y[-1] == 0 and not lc.any()


def test_short_engine_padded_and_flagged():
    x, _, _, lc = make_windows(_df(10), ["s2"], 30)
    assert x.shape == (1, 30, 1) and lc.all()
    assert (x[0, :20, 0] == 0).all()  # preenchido com o primeiro valor


def test_last_only_one_window_per_unit():
    df = pd.concat([_df(40, 1), _df(35, 2)])
    x, *_ = make_windows(df, ["s2"], 30, last_only=True)
    assert len(x) == 2


def test_split_by_unit_has_no_overlap():
    tr, va = split_units(np.repeat(np.arange(1, 101), 5), 0.2, seed=0)
    assert not set(tr) & set(va) and len(va) == 20
