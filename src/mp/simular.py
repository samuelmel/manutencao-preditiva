"""Simula tráfego na API para encher os painéis do Grafana.

Uso: python -m mp.simular [--url http://127.0.0.1:8000] [--drift]
Envia o histórico de cada motor de teste do FD001 e uma OS sintética coerente com o RUL real.
Com --drift, desloca 6 sensores fora da faixa de treino e dispara o alerta de drift.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

from mp.config import ROOT
from mp.ingest.cmapss import read_raw, read_rul_file
from mp.textgen.generator import gerar_os


def _post(url: str, corpo: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(corpo).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")  # 127.0.0.1: "localhost" tenta ::1 antes no Windows
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--f1", action="store_true",
                    help="envia uma amostra rotulada de OS e mostra o gatilho de F1-macro")
    ap.add_argument("--f1-degradado", action="store_true",
                    help="como --f1, mas com os rotulos embaralhados (simula o modelo envelhecido)")
    a = ap.parse_args()
    if a.f1 or a.f1_degradado:
        rng = np.random.default_rng(123)
        oss = [gerar_os(i % 100, i, float(rng.integers(0, 125)), rng) for i in range(300)]
        rotulos = [o["gravidade"] for o in oss]
        if a.f1_degradado:
            rotulos = list(rng.permutation(rotulos))
        itens = [{"texto": o["texto"], "gravidade": g} for o, g in zip(oss, rotulos, strict=True)]
        r = _post(f"{a.url}/triagem/avaliar", {"itens": itens})
        print(f"F1-macro {r['f1_macro']:.3f} (referencia {r['f1_macro_ref']:.3f}) | queda "
              f"{r['queda_pontos']:.1f} pontos | gatilho de retreino: {r['gatilho_retreino']}")
        return
    df = read_raw(ROOT / "data" / "bronze" / f"test_{a.fd}.txt")
    rul_final = read_rul_file(ROOT / "data" / "bronze" / f"RUL_{a.fd}.txt")
    rng = np.random.default_rng(0)
    norm = json.loads((ROOT / "data" / "silver" / f"fd={a.fd}" / "normalizer.json").read_text())
    sens = norm["sensors"][:6]  # sensores que o modelo usa (os constantes foram descartados)
    amplitude = (df[sens].max() - df[sens].min()).to_dict()
    for uid, g in df.groupby("unit_id"):
        g = g.sort_values("cycle")
        if a.drift:  # sai bem da faixa de treino em 6 sensores
            g = g.copy()
            for s in sens:
                g[s] = g[s] + 3 * amplitude[s]
        ciclos = g[[c for c in g.columns if c not in ("unit_id", "cycle")]].to_dict("records")
        r = _post(f"{a.url}/rul", {"unit_id": int(uid), "ciclos": ciclos})
        os_ = gerar_os(int(uid), int(g["cycle"].max()), float(rul_final.loc[uid]), rng)
        _post(f"{a.url}/triagem", {"texto": os_["texto"], "unit_id": int(uid)})
        print(f"motor {uid:>3}: RUL {r['rul']:6.1f} prioridade {r['prioridade']:.2f}"
              f"{' ALERTA DRIFT' if r['alerta_drift'] else ''}")


if __name__ == "__main__":
    main()
