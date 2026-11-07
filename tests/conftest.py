import numpy as np
import pytest


def _fake(path, n_units, rng, multi, truncate=False):
    rows = []
    for u in range(1, n_units + 1):
        n = int(rng.integers(40, 80))
        if truncate:
            n = int(rng.integers(20, n))
        for c in range(1, n + 1):
            op = rng.choice(6, size=3) * [10, 0.1, 50] if multi else [0, 0, 100]
            s = rng.normal(size=21) + c * 0.05
            s[0] = 518.67  # sensor constante
            rows.append([u, c, *np.asarray(op, float), *s])
    with open(path, "w") as f:
        f.writelines(" ".join(str(v) for v in r) + "\n" for r in rows)


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    d = tmp_path / "raw"
    d.mkdir()
    for fd, multi in [("FD001", False), ("FD002", True)]:
        _fake(d / f"train_{fd}.txt", 12, rng, multi)
        _fake(d / f"test_{fd}.txt", 6, rng, multi, truncate=True)
        (d / f"RUL_{fd}.txt").write_text("\n".join(str(int(x)) for x in rng.integers(5, 100, 6)))
    return d
