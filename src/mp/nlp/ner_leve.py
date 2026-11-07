"""NER leve (sem spaCy) para o serviço: mesmas regras do EntityRuler, imagem Docker menor."""
from __future__ import annotations

import re

from mp.nlp.ner import normalizar_codigo
from mp.textgen.lexicon import COMPONENTES, SINTOMAS

CODIGO_BUSCA = re.compile(r"\b[A-Za-z]{3}-?\d{4}\b")
_COMPONENTES = sorted({f.lower() for _, fs in COMPONENTES.values() for f in fs}, key=len, reverse=True)
_SINTOMAS = sorted({s.lower() for g in SINTOMAS for s in g}, key=len, reverse=True)


def extrair(texto_limpo: str) -> dict:
    cod = CODIGO_BUSCA.search(texto_limpo)
    return {
        "codigo": normalizar_codigo(cod.group()) if cod else None,
        "componente": next((c for c in _COMPONENTES if c in texto_limpo), None),
        "sintoma": next((s for s in _SINTOMAS if s in texto_limpo), None),
    }
