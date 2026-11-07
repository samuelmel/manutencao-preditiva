"""Fase 2: silver de texto + baseline TF-IDF/LR + Word2Vec + LDA.

Uso: python -m mp.train_triagem   (precisa de `python -m mp.textgen.build` antes)
"""
from __future__ import annotations

import json

import pandas as pd

from mp.config import ROOT, load_config
from mp.models.triagem import avaliar, baseline, dividir_por_motor
from mp.nlp.ner import normalizar_codigo
from mp.nlp.preprocess import construir_silver_texto
from mp.nlp.topicos import (
    escolher_k,
    matriz_vetores,
    os_semelhantes,
    tokens,
    treinar_lda,
    treinar_word2vec,
)


def main() -> None:
    cfg = load_config("data")
    seed = cfg["seed"]
    bronze = ROOT / "data" / "bronze_text" / "os.jsonl"
    silver = ROOT / "data" / "silver_text" / "os.parquet"
    df = construir_silver_texto(bronze, silver)
    print("silver:", len(df), "OS | spaCy:", df["modelo_spacy"].iloc[0])

    out = ROOT / "artifacts" / "triagem"
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"modelo_spacy": df["modelo_spacy"].iloc[0], "n_os": len(df)}

    # NER: acerto do código de peça (exato, após normalização)
    ref = df["entidades_ref"].map(json.loads)
    pred = df["entidades_pred"].map(json.loads)
    ok = [normalizar_codigo(r["codigo"]) == p["codigo"] for r, p in zip(ref, pred, strict=True)]
    res["ner_codigo_acuracia"] = float(pd.Series(ok).mean())
    for campo in ("componente", "sintoma"):
        res[f"ner_{campo}_cobertura"] = float(pred.map(lambda p, c=campo: p[c] is not None).mean())
    print("NER:", {k: round(v, 3) for k, v in res.items() if k.startswith("ner_")})

    treino, teste = dividir_por_motor(df, seed)
    res["n_treino"], res["n_teste"] = len(treino), len(teste)
    for coluna in ("texto_limpo", "lemas"):
        m = baseline(seed).fit(treino[coluna], treino["gravidade"])
        res[f"tfidf_lr[{coluna}]"] = avaliar(m, teste, coluna)
        r = res[f"tfidf_lr[{coluna}]"]
        print(coluna, "| F1-macro ruidoso", round(r["ruidoso"]["f1_macro"], 3),
              "| ref", round(r["referencia"]["f1_macro"], 3),
              "| recall crítica (ruidoso)", round(r["ruidoso"]["recall"]["crítica"], 3))

    corpus = tokens(treino["lemas"].tolist())
    w2v = treinar_word2vec(corpus, seed)
    vet = matriz_vetores(w2v, corpus)
    q = tokens(["vibração forte compressor alta"])[0]
    idx, sim = os_semelhantes(w2v, vet, q, k=3)
    res["w2v_exemplo"] = {"consulta": " ".join(q),
                          "vizinhas": [treino["texto_limpo"].iloc[i] for i in idx],
                          "similaridade": [float(s) for s in sim]}
    w2v.save(str(out / "w2v.model"))

    melhor_k, coer = escolher_k(corpus, seed=seed)
    res["lda"] = {"melhor_k": melhor_k, "coerencia_cv": {str(k): v for k, v in coer.items()}}
    lda, _ = treinar_lda(corpus, melhor_k, seed)
    res["lda"]["topicos"] = [[p for p, _ in lda.show_topic(i, 6)] for i in range(melhor_k)]
    lda.save(str(out / "lda.model"))
    print("LDA k =", melhor_k, {k: round(v, 3) for k, v in coer.items()})

    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
