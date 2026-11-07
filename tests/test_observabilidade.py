"""Valida os arquivos de observabilidade sem subir Docker: JSON/YAML válidos e coerentes com a API."""
import json
import re

import pytest
import yaml

from mp.config import ROOT

DASH = ROOT / "grafana" / "dashboards" / "manutencao.json"
PROV = ROOT / "grafana" / "provisioning"

METRICAS_API = {
    "mp_rul_ciclos", "mp_prioridade", "mp_psi", "mp_sensores_drift", "mp_os_total",
    "mp_texto_oov_taxa", "mp_texto_psi_classes", "mp_condicao_desconhecida_total",
    "http_request_duration_seconds",
}


def _metricas_em(expr: str) -> set[str]:
    achadas = set(re.findall(r"\b(?:mp|http)_[a-z_]+", expr))
    return {re.sub(r"_(bucket|count|sum)$", "", m) for m in achadas}


@pytest.fixture(scope="module")
def dashboard():
    return json.loads(DASH.read_text(encoding="utf-8"))


def test_dashboard_e_json_valido_com_uid_fixo(dashboard):
    assert dashboard["uid"] == "manutencao-preditiva"
    assert len(dashboard["panels"]) >= 5
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids))


def test_dashboard_usa_so_metricas_que_a_api_expoe(dashboard):
    usadas = set()
    for p in dashboard["panels"]:
        for t in p.get("targets", []):
            usadas |= _metricas_em(t.get("expr", ""))
    assert usadas, "nenhuma métrica Prometheus encontrada"
    assert usadas <= METRICAS_API, usadas - METRICAS_API


def test_dashboard_aponta_para_datasources_provisionados(dashboard):
    ds = yaml.safe_load((PROV / "datasources" / "datasources.yml").read_text(encoding="utf-8"))
    uids = {d["uid"] for d in ds["datasources"]}
    for p in dashboard["panels"]:
        assert p["datasource"]["uid"] in uids


def test_fila_vem_do_endpoint_prioridade_via_infinity(dashboard):
    fila = next(p for p in dashboard["panels"] if p["title"].startswith("Fila"))
    t = fila["targets"][0]
    assert t["url"].endswith("/prioridade") and t["root_selector"] == "itens"
    assert t["parser"] == "backend"  # sem isso a tabela só é montada no navegador


def test_regras_de_alerta_cobrem_drift_e_rul_critico():
    regras = yaml.safe_load((PROV / "alerting" / "regras.yml").read_text(encoding="utf-8"))
    rules = {r["uid"]: r for g in regras["groups"] for r in g["rules"]}
    assert {"mp-drift-sensores", "mp-rul-critico"} <= set(rules)
    for r in rules.values():
        assert r["annotations"]["description"]
        for d in r["data"]:
            assert d["datasourceUid"] in {"prometheus", "__expr__"}
            if d["datasourceUid"] == "prometheus":
                assert _metricas_em(d["model"]["expr"]) <= METRICAS_API
    limiar_drift = rules["mp-drift-sensores"]["data"][2]["model"]["conditions"][0]["evaluator"]
    assert limiar_drift == {"type": "gt", "params": [3]}  # "mais de 3 sensores"
    limiar_rul = rules["mp-rul-critico"]["data"][2]["model"]["conditions"][0]["evaluator"]
    assert limiar_rul == {"type": "lt", "params": [15]}


def test_compose_limita_retencao_do_prometheus_e_nao_expoe_na_rede():
    c = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "--storage.tsdb.retention.size=200MB" in c["services"]["prometheus"]["command"]
    for nome, s in c["services"].items():
        for porta in s.get("ports", []):
            assert str(porta).startswith("127.0.0.1:"), (nome, porta)


def test_prometheus_coleta_a_api_a_cada_15s():
    p = yaml.safe_load((ROOT / "prometheus" / "prometheus.yml").read_text(encoding="utf-8"))
    assert p["global"]["scrape_interval"] == "15s"
    assert p["scrape_configs"][0]["static_configs"][0]["targets"] == ["api:8000"]


def test_dashboard_no_repo_esta_atualizado_com_o_gerador(dashboard):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gd", ROOT / "scripts" / "gerar_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.dashboard == dashboard
