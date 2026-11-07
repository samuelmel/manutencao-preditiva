import os
from pathlib import Path

import yaml

# MP_ROOT permite apontar para a raiz (com configs/) quando o pacote roda instalado, como no Docker.
ROOT = Path(os.environ.get("MP_ROOT") or Path(__file__).resolve().parents[2])


def load_config(name: str = "data") -> dict:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
