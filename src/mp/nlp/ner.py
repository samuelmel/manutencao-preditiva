"""NER das OS: EntityRuler com regex para código de peça e léxico para componente e sintoma.

O NER genérico do spaCy não reconhece `HPC-0412`, então o código vem de regra. Componente e
sintoma são casados por frase (minúsculas); erros de digitação não casam, e isso é uma
limitação conhecida e medida nos testes.
"""
from __future__ import annotations

import re

from mp.textgen.lexicon import COMPONENTES, SINTOMAS

CODIGO_REGEX = r"^[A-Za-z]{3}-?\d{4}$"
_CODIGO_RE = re.compile(CODIGO_REGEX)
ROTULOS = {"PECA": "codigo", "COMPONENTE": "componente", "SINTOMA": "sintoma"}


def padroes() -> list[dict]:
    pats: list[dict] = [{"label": "PECA", "pattern": [{"TEXT": {"REGEX": CODIGO_REGEX}}]}]
    for _, formas in COMPONENTES.values():
        pats += [{"label": "COMPONENTE", "pattern": f.lower()} for f in formas]
    for grupo in SINTOMAS:
        pats += [{"label": "SINTOMA", "pattern": s.lower()} for s in grupo]
    return pats


def adicionar_ruler(nlp):
    """Adiciona o EntityRuler (antes do `ner`, se existir, para as regras terem prioridade)."""
    cfg = {"phrase_matcher_attr": "LOWER"}
    if "ner" in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner", config=cfg)
    else:
        ruler = nlp.add_pipe("entity_ruler", config=cfg)
    ruler.add_patterns(padroes())
    return nlp


def normalizar_codigo(texto: str) -> str:
    """HPC0412 / hpc-0412 -> HPC-0412."""
    t = texto.strip().upper()
    return f"{t[:3]}-{t[-4:]}" if _CODIGO_RE.match(t) else t


def extrair_entidades(doc) -> dict:
    out: dict = {"codigo": None, "componente": None, "sintoma": None}
    for ent in doc.ents:
        campo = ROTULOS.get(ent.label_)
        if campo and out[campo] is None:
            out[campo] = normalizar_codigo(ent.text) if campo == "codigo" else ent.text.lower()
    return out
