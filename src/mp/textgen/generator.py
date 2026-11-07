"""Gerador de OS sintéticas por templates (sem LLM), ligado ao RUL real do C-MAPSS.

A gravidade de referência vem do RUL (<15 crítica, <40 alta, <80 média, resto baixa).
O texto mistura sintomas de gravidades vizinhas e recebe ruído de chão de fábrica
(abreviações, erros de digitação, caixa); cerca de 10% dos rótulos finais são trocados
de propósito (`gravidade`), mas a referência limpa fica em `gravidade_ref`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mp.textgen.lexicon import ABREVIACOES, COMPONENTES, GRAVIDADES, SINTOMAS, TEMPLATES

LIMITES_RUL = (15, 40, 80)
CODIGO_RE = re.compile(r"\b[A-Za-z]{3}-?\d{4}\b")


def gravidade_por_rul(rul: float) -> int:
    for i, lim in enumerate(LIMITES_RUL):
        if rul < lim:
            return 3 - i
    return 0


def _typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 5 or any(ch.isdigit() for ch in word):
        return word
    i = int(rng.integers(1, len(word) - 2))
    if rng.random() < 0.5:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]  # troca vizinhos
    return word[:i] + word[i + 1:]  # remove letra


def _ruido(texto: str, rng: np.random.Generator, p_typo: float = 0.04, p_abrev: float = 0.3) -> str:
    for cheio, abrev in ABREVIACOES.items():
        if cheio in texto and rng.random() < p_abrev:
            texto = texto.replace(cheio, abrev)
    palavras = [_typo(w, rng) if rng.random() < p_typo else w for w in texto.split(" ")]
    texto = " ".join(palavras)
    r = rng.random()
    if r < 0.08:
        texto = texto.upper()
    elif r < 0.16:
        texto = texto.lower()
    return texto


def gerar_os(unit_id: int, ciclo: int, rul: float, rng: np.random.Generator,
             p_vizinha: float = 0.3, p_troca: float = 0.10) -> dict:
    """Gera uma OS e suas entidades de referência (para avaliar o NER)."""
    g_ref = gravidade_por_rul(rul)
    g_texto = g_ref
    if rng.random() < p_vizinha:
        g_texto = int(np.clip(g_ref + rng.choice([-1, 1]), 0, 3))
    sintoma = str(rng.choice(SINTOMAS[g_texto]))
    comp_canon = str(rng.choice(list(COMPONENTES)))
    prefixo, formas = COMPONENTES[comp_canon]
    comp = str(rng.choice(formas))
    codigo = f"{prefixo}-{int(rng.integers(1, 1000)):04d}"
    if rng.random() < 0.1:
        codigo = codigo.replace("-", "")
    tpl = str(rng.choice(TEMPLATES))
    texto = tpl.format(u=unit_id, y=ciclo, s=sintoma, c=comp, k=codigo)
    texto = _ruido(texto, rng)
    g_final = g_ref
    if rng.random() < p_troca:
        g_final = int(rng.choice([g for g in range(4) if g != g_ref]))
    return {
        "texto": texto,
        "gravidade_ref": GRAVIDADES[g_ref],
        "gravidade": GRAVIDADES[g_final],
        "entidades": {"codigo": codigo, "componente": comp, "sintoma": sintoma},
    }


def amostrar_ciclos(df: pd.DataFrame, por_faixa: int, rng: np.random.Generator) -> pd.DataFrame:
    """Por motor, `por_faixa` ciclos de cada faixa de gravidade (classes balanceadas)."""
    faixa = df["rul"].map(gravidade_por_rul)
    partes = []
    for _, g in df.groupby(["unit_id", faixa]):
        n = min(por_faixa, len(g))
        partes.append(g.iloc[rng.choice(len(g), size=n, replace=False)])
    return pd.concat(partes)


def gerar_dataset(silver_dir: Path, subsets: list[str], por_faixa: int = 4,
                  seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for fd in subsets:
        train = pd.read_parquet(silver_dir / f"fd={fd}" / "train.parquet",
                                columns=["unit_id", "cycle", "rul"])
        for r in amostrar_ciclos(train, por_faixa, rng).itertuples():
            os_ = gerar_os(int(r.unit_id), int(r.cycle), float(r.rul), rng)
            out.append({"os_id": f"{fd}-{r.unit_id}-{r.cycle}", "fd": fd,
                        "unit_id": int(r.unit_id), "cycle": int(r.cycle), **os_})
    return out


def salvar_jsonl(registros: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in registros)


def ler_jsonl(caminho: Path) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
