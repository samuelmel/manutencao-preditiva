"""Silver de texto: limpeza, lematização e NER (spaCy) das OS.

Usa `pt_core_news_sm` quando instalado. Sem ele, cai para `spacy.blank("pt")`: o NER por regras
(códigos de peça, componentes, sintomas) continua valendo, mas não há lematização real e
`lemas` vira o token em minúsculas. O modelo usado fica registrado em `modelo_spacy`.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from mp.nlp.ner import adicionar_ruler, extrair_entidades

MODELO = "pt_core_news_sm"

# inverso de textgen.lexicon.ABREVIACOES (gíria de chão de fábrica -> forma cheia)
EXPANDE = {
    "temp": "temperatura", "rolam.": "rolamento", "vibr.": "vibração", "comp.": "compressor",
    "vaz.": "vazamento", "pres.": "pressão", "turb.": "turbina", "insp.": "inspeção",
}
_EXPANDE_RE = [(re.compile(rf"(?<!\w){re.escape(ab)}(?!\w)"), full) for ab, full in EXPANDE.items()]


def limpar(texto: str) -> str:
    """Minúsculas, abreviações expandidas e espaços normalizados."""
    t = unicodedata.normalize("NFC", texto).lower()
    for rx, full in _EXPANDE_RE:
        t = rx.sub(full, t)
    return re.sub(r"\s+", " ", t).strip()


def carregar_nlp():
    """Devolve (nlp, nome_do_modelo). Cai para `blank("pt")` se o modelo não estiver instalado."""
    import spacy

    try:
        nlp = spacy.load(MODELO)
        nome = MODELO
    except OSError:
        nlp = spacy.blank("pt")
        nome = "blank-pt"
    adicionar_ruler(nlp)
    return nlp, nome


def _lemas(doc) -> list[str]:
    out = []
    for t in doc:
        if t.is_stop or t.is_punct or t.like_num or t.is_space or t.ent_type_ == "PECA":
            continue
        out.append((t.lemma_ or t.lower_).lower())
    return out


def processar(textos: list[str], nlp) -> pd.DataFrame:
    limpos = [limpar(t) for t in textos]
    linhas = []
    for doc in nlp.pipe(limpos, batch_size=256):
        linhas.append({"lemas": " ".join(_lemas(doc)),
                       "entidades_pred": json.dumps(extrair_entidades(doc), ensure_ascii=False)})
    out = pd.DataFrame(linhas)
    out.insert(0, "texto_limpo", limpos)
    return out


def construir_silver_texto(bronze: Path, destino: Path) -> pd.DataFrame:
    from mp.textgen.generator import ler_jsonl

    regs = ler_jsonl(bronze)
    df = pd.DataFrame(regs)
    nlp, nome = carregar_nlp()
    proc = processar(df["texto"].tolist(), nlp)
    df = pd.concat([df.drop(columns=["entidades"]), proc], axis=1)
    df["entidades_ref"] = [json.dumps(r["entidades"], ensure_ascii=False) for r in regs]
    df["modelo_spacy"] = nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    return df
