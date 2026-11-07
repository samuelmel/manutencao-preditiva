"""Métricas Prometheus. Rótulos só `unit_id`, `sensor`, `gravidade` e rota: nunca texto de OS."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metricas:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.rul = Gauge("mp_rul_ciclos", "RUL estimado por motor (ciclos)", ["unit_id"], registry=r)
        self.prioridade = Gauge("mp_prioridade", "Prioridade combinada por motor (0 a 1)",
                                ["unit_id"], registry=r)
        self.psi = Gauge("mp_psi", "PSI por sensor contra o treino", ["sensor"], registry=r)
        self.sensores_drift = Gauge("mp_sensores_drift", "Sensores com PSI acima do limiar",
                                    registry=r)
        self.condicao_desconhecida = Counter("mp_condicao_desconhecida_total",
                                             "Previsões em condição operacional nunca vista",
                                             registry=r)
        self.os_total = Counter("mp_os_total", "OS triadas por gravidade", ["gravidade"],
                                registry=r)
        self.texto_oov = Gauge("mp_texto_oov_taxa", "Taxa de tokens fora do vocabulário de treino",
                               registry=r)
        self.texto_psi = Gauge("mp_texto_psi_classes", "PSI da distribuição de gravidade das OS",
                               registry=r)
        self.latencia = Histogram("http_request_duration_seconds", "Latência HTTP",
                                  ["handler", "method", "status"], registry=r,
                                  buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
