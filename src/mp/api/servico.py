"""Lógica de inferência do serviço, sem FastAPI: testável com modelos injetados."""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Protocol

import numpy as np

from mp import drift
from mp.fusion import prioridade
from mp.nlp.ner_leve import extrair
from mp.nlp.preprocess import limpar
from mp.textgen.lexicon import GRAVIDADES

OP_COLS = ["op1", "op2", "op3"]
LIMIAR_OOV = 0.6  # acima disso o texto é de outro idioma ou ruído: vai para triagem humana


class ModeloRul(Protocol):
    def predict(self, janelas: np.ndarray) -> np.ndarray: ...


class OnnxRul:
    def __init__(self, caminho: Path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(caminho), providers=["CPUExecutionProvider"])
        self.entrada = self.sess.get_inputs()[0].name

    def predict(self, janelas: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.entrada: janelas.astype(np.float32)})[0].ravel()


class ServicoRul:
    def __init__(self, modelo: ModeloRul, norm: dict, ref: dict[str, np.ndarray],
                 limite_distancia: float, janela: int = 30, cfg: dict | None = None):
        self.modelo, self.janela = modelo, janela
        self.sensores: list[str] = norm["sensors"]
        self.centros = np.asarray(norm["centers"], dtype=float)
        self.minimo = np.asarray(norm["min"], dtype=float)
        self.maximo = np.asarray(norm["max"], dtype=float)
        self.ref, self.limite_distancia = ref, limite_distancia
        self.cfg = cfg or {"psi_limiar": 0.2, "psi_min_sensores": 3}
        self.buffer: deque[np.ndarray] = deque(maxlen=1000)
        self._n = 0
        self._cache: dict | None = None

    def faltando(self, ciclos: list[dict[str, float]]) -> list[str]:
        exigidos = set(self.sensores) | set(OP_COLS)
        return sorted({k for c in ciclos for k in exigidos if k not in c})

    def _normalizar(self, ciclos: list[dict[str, float]]) -> tuple[np.ndarray, float]:
        ops = np.array([[c[k] for k in OP_COLS] for c in ciclos], dtype=float)
        x = np.array([[c[s] for s in self.sensores] for c in ciclos], dtype=float)
        d = np.linalg.norm(ops[:, None, :] - self.centros[None], axis=2)
        cl = d.argmin(axis=1)
        lo, hi = self.minimo[cl], self.maximo[cl]
        faixa = np.where(hi - lo == 0, 1.0, hi - lo)
        return (x - lo) / faixa, float(d.min(axis=1).max())

    def prever(self, ciclos: list[dict[str, float]]) -> dict:
        x, dist = self._normalizar(ciclos)
        self.buffer.extend(x[-self.janela:])
        curto = len(x) < self.janela
        if curto:  # mesmo tratamento do treino: repete o primeiro valor
            x = np.vstack([np.repeat(x[:1], self.janela - len(x), axis=0), x])
        rul = float(self.modelo.predict(x[-self.janela:][None].astype(np.float32))[0])
        self._n += 1
        return {"rul": max(rul, 0.0), "low_confidence": curto,
                "condicao_desconhecida": dist > self.limite_distancia,
                "distancia_condicao": dist}

    def drift_cacheado(self, a_cada: int = 20) -> dict:
        """PSI recalculado só a cada `a_cada` previsões, para não pesar na latência do /rul."""
        if self._cache is None or self._n % a_cada == 0:
            self._cache = self.avaliar_drift(com_ks=False)
        return self._cache

    def avaliar_drift(self, com_ks: bool = True) -> dict:
        """PSI (e KS, sob demanda) por sensor do buffer recente contra a referência de treino."""
        if len(self.buffer) < 100:
            return {"psi": {}, "n_amostras": len(self.buffer), "drift": False,
                    "sensores_acima": 0, "ks": {}}
        atual = np.vstack(self.buffer)
        a = {s: atual[:, i] for i, s in enumerate(self.sensores)}
        psis = drift.psi_por_sensor(self.ref, a)
        psis = {s: min(v, 99.0) for s, v in psis.items()}
        ks = {s: drift.ks(self.ref[s], a[s])[0] for s in psis} if com_ks else {}
        return {"psi": psis, "ks": ks, "n_amostras": len(self.buffer),
                "sensores_acima": drift.sensores_em_drift(psis, self.cfg["psi_limiar"]),
                "drift": drift.drift_de_sensores(psis, self.cfg["psi_limiar"],
                                                 self.cfg["psi_min_sensores"])}


class ServicoTriagem:
    def __init__(self, modelo, ref_classes: dict[str, float]):
        self.modelo = modelo
        self.classes = list(modelo.classes_)
        self.vocab = {t for t in modelo.named_steps["tfidf"].vocabulary_ if " " not in t}
        self.ref_classes = ref_classes
        self.recentes: deque[str] = deque(maxlen=200)
        self.oov: deque[float] = deque(maxlen=200)

    def triar(self, texto: str) -> dict:
        limpo = limpar(texto or "")
        tokens = re.findall(r"[^\W\d_]+", re.sub(r"\b[a-z]{3}-?\d{4}\b", " ", limpo))
        entidades = extrair(limpo)
        base = {"entidades": entidades, "texto_limpo": limpo}
        if not tokens:
            return {**base, "gravidade": "indefinida", "motivo": "sem texto descritivo",
                    "p_grave": None, "probabilidades": {}}
        oov = drift.taxa_oov([limpo], self.vocab)
        self.oov.append(oov)
        if oov > LIMIAR_OOV:
            return {**base, "gravidade": "indefinida", "motivo": "vocabulário desconhecido",
                    "p_grave": None, "probabilidades": {}}
        proba = self.modelo.predict_proba([limpo])[0]
        probs = {c: float(p) for c, p in zip(self.classes, proba, strict=True)}
        grav = max(probs, key=probs.get)
        self.recentes.append(grav)
        return {**base, "gravidade": grav, "motivo": None, "probabilidades": probs,
                "p_grave": probs.get("alta", 0.0) + probs.get("crítica", 0.0)}

    def drift_texto(self) -> dict:
        n = len(self.recentes)
        cont = Counter(self.recentes)
        atual = {g: cont.get(g, 0) / n for g in GRAVIDADES} if n else {}
        return {"oov_taxa": float(np.mean(self.oov)) if self.oov else 0.0,
                "psi_classes": drift.psi_categorico(self.ref_classes, atual) if n >= 50 else 0.0}


class Fila:
    """Estado em memória da fila de manutenção (um registro por motor)."""

    def __init__(self, cfg: dict):
        self.cfg, self._d, self._lock = cfg, {}, threading.Lock()

    def atualizar(self, unit_id: int, *, rul: float | None = None,
                  p_grave: float | None = None, gravidade: str | None = None) -> dict:
        with self._lock:
            r = self._d.setdefault(unit_id, {"unit_id": unit_id, "rul": None, "p_grave": None,
                                             "gravidade_os": None})
            if rul is not None:
                r["rul"] = rul
            if p_grave is not None:
                r["p_grave"] = p_grave
                r["gravidade_os"] = gravidade
            if r["rul"] is not None:
                r["prioridade"] = prioridade(r["rul"], r["p_grave"], self.cfg["w_rul"],
                                             self.cfg["w_os"], self.cfg["rul_cap"])
                r["critico"] = r["rul"] < self.cfg["rul_critico"]
            r["atualizado_em"] = time.time()
            return dict(r)

    def ordenada(self, top: int | None = None) -> list[dict]:
        with self._lock:
            itens = [dict(v) for v in self._d.values() if "prioridade" in v]
        itens.sort(key=lambda r: r["prioridade"], reverse=True)
        return itens[:top] if top else itens


def carregar(pasta: Path, cfg: dict):
    import joblib

    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    norm = json.loads((pasta / "normalizer.json").read_text(encoding="utf-8"))
    ref_npz = np.load(pasta / "drift_ref.npz")
    ref = {s: ref_npz[s] for s in norm["sensors"]}
    rul = ServicoRul(OnnxRul(pasta / "lstm.onnx"), norm, ref, meta["limite_distancia"],
                     meta["janela"], cfg)
    tri = ServicoTriagem(joblib.load(pasta / "triagem_tfidf.joblib"), meta["classes_ref"])
    return rul, tri
