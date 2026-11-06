import numpy as np

from mp.models.baselines import MeanBaseline, RFBaseline, window_features
from mp.models.metrics import nasa_score, rmse


def test_rmse_zero_and_value():
    assert rmse([1, 2], [1, 2]) == 0
    assert np.isclose(rmse([0, 0], [3, 4]), np.sqrt(12.5))


def test_score_penalises_late_more_than_early():
    assert nasa_score([50], [60]) > nasa_score([50], [40])  # +10 tarde > -10 cedo
    assert nasa_score([50], [50]) == 0


def test_window_features_slope():
    x = np.tile(np.arange(30, dtype=float)[None, :, None], (2, 1, 1))
    f = window_features(x)
    assert f.shape == (2, 4) and np.allclose(f[:, 3], 1.0)


def test_rf_beats_mean_on_learnable_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 30, 2)).astype(np.float32)
    y = (x[:, -1, 0] * 20 + 60).astype(np.float32)
    mean = MeanBaseline().fit(x, y).predict(x)
    rf = RFBaseline(n_estimators=30).fit(x[:200], y[:200]).predict(x[200:])
    assert rmse(y[200:], rf) < rmse(y[200:], mean[200:])
