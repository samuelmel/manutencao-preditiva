"""API FastAPI: /rul, /triagem, /prioridade, /drift e /metrics. Uso: uvicorn mp.api.app:criar_app --factory"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from mp.api.metricas import Metricas
from mp.api.servico import Fila, ServicoRul, ServicoTriagem, carregar
from mp.config import ROOT, load_config


class RulIn(BaseModel):
    unit_id: int = Field(ge=0)
    ciclos: list[dict[str, float]] = Field(min_length=1, max_length=2000)


class TriagemIn(BaseModel):
    texto: str = Field(max_length=2000)
    unit_id: int | None = Field(default=None, ge=0)


class OsRotulada(BaseModel):
    texto: str = Field(max_length=2000)
    gravidade: Literal["baixa", "média", "alta", "crítica"]


class AmostraIn(BaseModel):
    itens: list[OsRotulada] = Field(min_length=1, max_length=5000)


def criar_app(rul: ServicoRul | None = None, triagem: ServicoTriagem | None = None,
              metricas: Metricas | None = None, cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config("fusion")
    if rul is None or triagem is None:
        pasta = Path(os.environ.get("MP_ARTIFACTS", ROOT / "artifacts" / "serving"))
        rul, triagem = carregar(pasta, cfg)
    metricas = metricas or Metricas()
    fila = Fila(cfg)
    app = FastAPI(title="Manutenção preditiva 4.0", version="0.1.0")

    @app.middleware("http")
    async def medir(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        rota = request.scope.get("route")
        handler = getattr(rota, "path", "desconhecida")
        metricas.latencia.labels(handler, request.method, str(resp.status_code)).observe(
            time.perf_counter() - t0)
        return resp

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/rul")
    def prever_rul(corpo: RulIn):
        faltando = rul.faltando(corpo.ciclos)
        if faltando:
            raise HTTPException(422, f"campos ausentes nos ciclos: {faltando}")
        r = rul.prever(corpo.ciclos)
        uid = str(corpo.unit_id)
        metricas.rul.labels(uid).set(r["rul"])
        if r["condicao_desconhecida"]:
            metricas.condicao_desconhecida.inc()
        item = fila.atualizar(corpo.unit_id, rul=r["rul"])
        metricas.prioridade.labels(uid).set(item["prioridade"])
        d = rul.drift_cacheado()
        for s, v in d["psi"].items():
            metricas.psi.labels(s).set(v)
        metricas.sensores_drift.set(d["sensores_acima"])
        return {"unit_id": corpo.unit_id, **r, "prioridade": item["prioridade"],
                "alerta_drift": d["drift"]}

    @app.post("/triagem")
    def triar(corpo: TriagemIn):
        t = triagem.triar(corpo.texto)
        metricas.os_total.labels(t["gravidade"]).inc()
        dt = triagem.drift_texto()
        metricas.texto_oov.set(dt["oov_taxa"])
        metricas.texto_psi.set(dt["psi_classes"])
        t["revisao_humana"] = t["gravidade"] == "indefinida"
        if corpo.unit_id is not None and t["p_grave"] is not None:
            item = fila.atualizar(corpo.unit_id, p_grave=t["p_grave"], gravidade=t["gravidade"])
            if "prioridade" in item:
                metricas.prioridade.labels(str(corpo.unit_id)).set(item["prioridade"])
        return t

    @app.post("/triagem/avaliar")
    def avaliar_amostra(corpo: AmostraIn):
        """Recebe OS com a gravidade real e devolve o F1-macro contra a referencia do modelo."""
        try:
            r = triagem.avaliar_f1([(i.texto, i.gravidade) for i in corpo.itens],
                                   cfg.get("f1_queda_pontos", 10.0), cfg.get("f1_amostra_min", 30))
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        metricas.f1_amostra.set(r["f1_macro"])
        metricas.f1_queda.set(r["queda_pontos"])
        metricas.amostra_n.set(r["n_avaliadas"])
        return r

    @app.get("/prioridade")
    def prioridade_(top: int | None = None):
        return {"itens": fila.ordenada(top)}

    @app.get("/drift")
    def drift_():
        return {"sensores": rul.avaliar_drift(), "texto": triagem.drift_texto()}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(metricas.registry), media_type=CONTENT_TYPE_LATEST)

    return app
