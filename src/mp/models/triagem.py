"""Triagem de OS: baseline TF-IDF (1 a 3-gramas) + Regressão Logística.

A divisão treino/teste é por motor (`fd` + `unit_id`), nunca por linha, como no módulo de RUL.
Treina-se no rótulo ruidoso (`gravidade`); a avaliação sai nos dois rótulos: o ruidoso (o que
um sistema real veria) e o de referência (`gravidade_ref`, o que mede o modelo de fato).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from mp.textgen.lexicon import GRAVIDADES


def grupo_motor(df: pd.DataFrame) -> pd.Series:
    return df["fd"] + "-" + df["unit_id"].astype(str)


def dividir_por_motor(df: pd.DataFrame, seed: int, test_size: float = 0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=grupo_motor(df)))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def baseline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, random_state=seed)),
    ])


def avaliar(modelo, df: pd.DataFrame, coluna: str) -> dict:
    pred = modelo.predict(df[coluna])
    res = {}
    for nome, alvo in [("ruidoso", "gravidade"), ("referencia", "gravidade_ref")]:
        y = df[alvo]
        rec = recall_score(y, pred, labels=GRAVIDADES, average=None, zero_division=0)
        res[nome] = {
            "f1_macro": float(f1_score(y, pred, labels=GRAVIDADES, average="macro", zero_division=0)),
            "recall": {g: float(r) for g, r in zip(GRAVIDADES, rec, strict=True)},
            "matriz_confusao": confusion_matrix(y, pred, labels=GRAVIDADES).tolist(),
        }
    return res


def prob_alta_ou_critica(modelo, textos) -> np.ndarray:
    """P(gravidade in {alta, crítica}); entrada da fusão (RF07)."""
    proba = modelo.predict_proba(textos)
    cls = list(modelo.classes_)
    return proba[:, [cls.index("alta"), cls.index("crítica")]].sum(axis=1)
