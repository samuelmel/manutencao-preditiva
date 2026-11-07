"""Gera grafana/dashboards/manutencao.json (versionado; nada é configurado pela interface).

Uso: python scripts/gerar_dashboard.py
"""
import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
INF = {"type": "yesoreyeram-infinity-datasource", "uid": "infinity"}


def painel(id_, tipo, titulo, x, y, w, h, **kw):
    return {"id": id_, "type": tipo, "title": titulo, "gridPos": {"x": x, "y": y, "w": w, "h": h}, **kw}


def prom(expr, legenda="", instant=False, ref="A"):
    return {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legenda, "instant": instant,
            "range": not instant}


def cores(*passos):
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for c, v in passos]}


COR_GRAVIDADE = {"crítica": "red", "alta": "orange", "média": "yellow", "baixa": "green"}

fila = painel(
    1, "table", "Fila de manutenção (RUL + gravidade da OS)", 0, 0, 24, 10, datasource=INF,
    targets=[{
        "refId": "A", "datasource": INF, "type": "json", "source": "url", "format": "table",
        "url": "http://api:8000/prioridade", "url_options": {"method": "GET", "data": ""},
        "root_selector": "itens",
        "columns": [
            {"selector": "unit_id", "text": "Motor", "type": "number"},
            {"selector": "rul", "text": "RUL (ciclos)", "type": "number"},
            {"selector": "prioridade", "text": "Prioridade", "type": "number"},
            {"selector": "gravidade_os", "text": "Gravidade da OS", "type": "string"},
            {"selector": "critico", "text": "Crítico", "type": "boolean"},
        ],
    }],
    fieldConfig={"defaults": {}, "overrides": [
        {"matcher": {"id": "byName", "options": "Prioridade"}, "properties": [
            {"id": "custom.cellOptions", "value": {"type": "color-background"}},
            {"id": "thresholds", "value": cores(("green", None), ("yellow", 0.4), ("red", 0.7))},
            {"id": "decimals", "value": 2}]},
        {"matcher": {"id": "byName", "options": "RUL (ciclos)"}, "properties": [
            {"id": "decimals", "value": 0}]},
        {"matcher": {"id": "byName", "options": "Gravidade da OS"}, "properties": [
            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
            {"id": "mappings", "value": [{"type": "value", "options": {
                g: {"color": c, "index": i} for i, (g, c) in enumerate(COR_GRAVIDADE.items())}}]}]},
    ]},
    options={"sortBy": [{"displayName": "Prioridade", "desc": True}], "showHeader": True},
)

rul = painel(
    2, "bargauge", "RUL por motor (15 menores)", 0, 10, 12, 10, datasource=PROM,
    targets=[prom("bottomk(15, mp_rul_ciclos)", "Motor {{unit_id}}", instant=True)],
    fieldConfig={"defaults": {"min": 0, "max": 125, "unit": "none", "decimals": 0,
                              "thresholds": cores(("red", None), ("orange", 15), ("yellow", 40),
                                                  ("green", 80))}, "overrides": []},
    options={"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
             "reduceOptions": {"calcs": ["lastNotNull"], "values": False}},
)

psi_linha = painel(
    3, "timeseries", "Drift por sensor (PSI, limiar 0,2)", 12, 10, 12, 10, datasource=PROM,
    targets=[prom("mp_psi", "{{sensor}}")],
    fieldConfig={"defaults": {"min": 0, "custom": {"thresholdsStyle": {"mode": "line"}, "lineWidth": 2},
                              "thresholds": cores(("green", None), ("red", 0.2))}, "overrides": []},
    options={"legend": {"displayMode": "table", "placement": "right", "calcs": ["lastNotNull"]}},
)

psi_mapa = painel(
    4, "state-timeline", "Mapa de calor do PSI por sensor", 0, 20, 12, 8, datasource=PROM,
    targets=[prom("mp_psi", "{{sensor}}")],
    fieldConfig={"defaults": {"min": 0, "max": 0.4, "color": {"mode": "thresholds"},
                              "thresholds": cores(("green", None), ("yellow", 0.1), ("red", 0.2)),
                              "custom": {"fillOpacity": 85}}, "overrides": []},
    options={"showValue": "never", "rowHeight": 0.9, "mergeValues": False},
)

pizza = painel(
    5, "piechart", "Gravidade das OS triadas", 12, 20, 12, 8, datasource=PROM,
    targets=[prom("sum by (gravidade) (mp_os_total)", "{{gravidade}}", instant=True)],
    fieldConfig={"defaults": {}, "overrides": [
        {"matcher": {"id": "byName", "options": g},
         "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]}
        for g, c in COR_GRAVIDADE.items()]},
    options={"pieType": "donut", "legend": {"displayMode": "table", "placement": "right",
                                            "values": ["value"]},
             "reduceOptions": {"calcs": ["lastNotNull"], "values": False}},
)

latencia = painel(
    6, "timeseries", "Latência p95 por rota", 0, 28, 8, 8, datasource=PROM,
    targets=[prom("histogram_quantile(0.95, sum by (le, handler) "
                  "(rate(http_request_duration_seconds_bucket[5m])))", "{{handler}}")],
    fieldConfig={"defaults": {"unit": "s", "min": 0}, "overrides": []},
)

erros = painel(
    7, "timeseries", "Taxa de erro 5xx", 8, 28, 8, 8, datasource=PROM,
    targets=[prom('sum(rate(http_request_duration_seconds_count{status=~"5.."}[5m]))', "5xx/s")],
    fieldConfig={"defaults": {"unit": "reqps", "min": 0, "color": {"mode": "fixed", "fixedColor": "red"}},
                 "overrides": []},
)

saude = painel(
    8, "stat", "Drift: sensores e texto", 16, 28, 8, 8, datasource=PROM,
    targets=[prom("mp_sensores_drift", "Sensores com PSI > 0,2", instant=True, ref="A"),
             prom("mp_texto_oov_taxa", "Tokens fora do vocabulário", instant=True, ref="B"),
             prom("mp_texto_psi_classes", "PSI das classes de OS", instant=True, ref="C")],
    fieldConfig={"defaults": {"decimals": 2, "thresholds": cores(("green", None), ("red", 0.2))},
                 "overrides": [{"matcher": {"id": "byName", "options": "Sensores com PSI > 0,2"},
                                "properties": [{"id": "decimals", "value": 0},
                                               {"id": "thresholds",
                                                "value": cores(("green", None), ("red", 4))}]}]},
    options={"colorMode": "value", "textMode": "value_and_name",
             "reduceOptions": {"calcs": ["lastNotNull"], "values": False}},
)

dashboard = {
    "uid": "manutencao-preditiva", "title": "Manutenção preditiva 4.0", "schemaVersion": 39,
    "version": 1, "editable": False, "refresh": "15s", "time": {"from": "now-30m", "to": "now"},
    "tags": ["manutencao", "mlops"], "timezone": "browser",
    "panels": [fila, rul, psi_linha, psi_mapa, pizza, latencia, erros, saude],
}

if __name__ == "__main__":
    destino = Path(__file__).resolve().parents[1] / "grafana" / "dashboards" / "manutencao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("gerado:", destino)
