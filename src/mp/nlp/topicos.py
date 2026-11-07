"""Word2Vec (similaridade entre OS, RF05) e LDA (tópicos, RF06) com gensim, treinados no corpus."""
from __future__ import annotations

import numpy as np


def tokens(lemas: list[str]) -> list[list[str]]:
    return [s.split() for s in lemas]


def treinar_word2vec(corpus: list[list[str]], seed: int = 42):
    from gensim.models import Word2Vec

    # workers=1 para ser determinístico com a seed
    return Word2Vec(corpus, vector_size=50, window=5, min_count=2, epochs=30, workers=1, seed=seed)


def vetor_os(w2v, toks: list[str]) -> np.ndarray:
    v = [w2v.wv[t] for t in toks if t in w2v.wv]
    return np.mean(v, axis=0) if v else np.zeros(w2v.vector_size, dtype=np.float32)


def matriz_vetores(w2v, corpus: list[list[str]]) -> np.ndarray:
    return np.vstack([vetor_os(w2v, t) for t in corpus])


def os_semelhantes(w2v, vetores: np.ndarray, consulta: list[str], k: int = 5):
    """Índices e similaridade-cosseno das k OS mais parecidas com a consulta."""
    q = vetor_os(w2v, consulta)
    num = vetores @ q
    den = np.linalg.norm(vetores, axis=1) * (np.linalg.norm(q) + 1e-12) + 1e-12
    sim = num / den
    idx = np.argsort(-sim)[:k]
    return idx, sim[idx]


def treinar_lda(corpus: list[list[str]], k: int, seed: int = 42):
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    dic = Dictionary(corpus)
    dic.filter_extremes(no_below=5, no_above=0.5)
    bow = [dic.doc2bow(d) for d in corpus]
    lda = LdaModel(bow, num_topics=k, id2word=dic, random_state=seed, passes=10, iterations=100)
    return lda, dic


def coerencia_cv(lda, corpus: list[list[str]], dic) -> float:
    from gensim.models import CoherenceModel

    cm = CoherenceModel(model=lda, texts=corpus, dictionary=dic, coherence="c_v", processes=1)
    return float(cm.get_coherence())


def escolher_k(corpus: list[list[str]], ks=(3, 4, 5, 6, 8), seed: int = 42):
    """Treina um LDA por k e devolve (melhor_k, {k: coerência c_v})."""
    res = {}
    for k in ks:
        lda, dic = treinar_lda(corpus, k, seed)
        res[k] = coerencia_cv(lda, corpus, dic)
    return max(res, key=res.get), res
