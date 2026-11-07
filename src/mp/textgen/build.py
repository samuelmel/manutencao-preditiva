"""Gera a camada bronze de texto. Uso: python -m mp.textgen.build"""
from __future__ import annotations

from collections import Counter

from mp.config import ROOT, load_config
from mp.textgen.generator import gerar_dataset, salvar_jsonl


def main() -> None:
    cfg = load_config("data")
    regs = gerar_dataset(ROOT / cfg["paths"]["silver"], cfg["subsets"], seed=cfg["seed"])
    destino = ROOT / "data" / "bronze_text" / "os.jsonl"
    salvar_jsonl(regs, destino)
    print(len(regs), "OS ->", destino)
    print(Counter(r["gravidade"] for r in regs))
    for r in regs[:5]:
        print(r["gravidade_ref"], "|", r["texto"])


if __name__ == "__main__":
    main()
