"""Gera os gráficos de resultados em img/ a partir de docs/resultados/.

Uso: python scripts/gerar_graficos.py   (precisa do extra `docs`: matplotlib)
Os números vêm dos arquivos de métricas e do CSV de previsões. Os poucos valores que não estão
em arquivo (janelas maiores e gatilho de F1) estão nas constantes abaixo, com a origem anotada.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "docs" / "resultados"
IMG = RAIZ / "img"
IMG.mkdir(exist_ok=True)

AZUL, LARANJA, VERDE, ROXO, CINZA, VERMELHO = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#898781", "#d03b3b"
CLASSES = ["baixa", "média", "alta", "crítica"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 100,
})

# Experimentos de janela maior (docs/RESULTADOS.md, scripts avulsos do scratchpad): RMSE no teste do FD001
JANELAS = {"Random Forest": [11.96, 13.12, 14.95], "LSTM": [13.23, 14.33, 15.87]}
# Gatilho de F1 medido no contêiner com `python -m mp.simular --f1` e `--f1-degradado`
F1_REF, F1_BOA, F1_DEGRADADA = 0.690, 0.656, 0.215


def virgula(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def salvar(fig, nome):
    fig.savefig(IMG / nome, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("gerado:", IMG / nome)


def rul_comparacao():
    modelos = [("Mistura 30% LSTM + 70% RF", 11.76, ROXO), ("Random Forest", 11.96, AZUL),
               ("LSTM + estatísticas (híbrido)", 12.62, VERDE), ("LSTM, média de 5 seeds", 13.13, LARANJA),
               ("LSTM, receita final", 13.23, LARANJA), ("LSTM, receita inicial", 13.42, LARANJA)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(modelos))[::-1]
    ax.barh(y, [m[1] for m in modelos], color=[m[2] for m in modelos], height=0.62)
    for yi, (_, v, _) in zip(y, modelos, strict=True):
        ax.text(v + 0.15, yi, virgula(v), va="center", fontweight="bold")
    ax.set_yticks(y, [m[0] for m in modelos])
    ax.set_xlim(0, 15.5)
    ax.set_xlabel("RMSE em ciclos de vida restante (menor é melhor). Média constante, fora do gráfico: 40,7")
    ax.set_title("Previsão de vida útil no FD001: o Random Forest vence o LSTM")
    ax.grid(axis="y", visible=False)
    salvar(fig, "rul_comparacao.png")


def rul_previsto_vs_real():
    df = pd.read_csv(RES / "rul_fd001_predicoes.csv")
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, col, titulo, cor in [(eixos[0], "random_forest", "Random Forest", AZUL),
                                 (eixos[1], "lstm_media_5_seeds", "LSTM (5 seeds)", LARANJA)]:
        rmse = float(np.sqrt(((df[col] - df.rul_real) ** 2).mean()))
        ax.scatter(df.rul_real, df[col], s=28, color=cor, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.plot([0, 135], [0, 135], color=CINZA, linestyle="--", linewidth=1.2)
        ax.set_title(f"{titulo}: RMSE {virgula(rmse)}")
        ax.set_xlabel("RUL real (ciclos)")
        ax.set_xlim(-3, 135)
        ax.set_ylim(-3, 135)
        ax.text(0.97, 0.05, "abaixo da linha:\nprevisão atrasada", transform=ax.transAxes, ha="right",
                color=CINZA, fontsize=9)
    eixos[0].set_ylabel("RUL previsto (ciclos)")
    fig.suptitle("100 motores de teste do FD001: previsto contra real", x=0.07, ha="left", fontweight="bold",
                 fontsize=13, y=1.02)
    salvar(fig, "rul_previsto_vs_real.png")


def janela_vs_rmse():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for nome, ys, cor in [("Random Forest", JANELAS["Random Forest"], AZUL), ("LSTM", JANELAS["LSTM"], LARANJA)]:
        ax.plot([30, 50, 70], ys, marker="o", color=cor, linewidth=2.4, markersize=8, label=nome)
        for x, v in zip([30, 50, 70], ys, strict=True):
            ax.annotate(virgula(v), (x, v), textcoords="offset points", xytext=(0, 9), ha="center", color=cor,
                        fontweight="bold")
    ax.set_xticks([30, 50, 70])
    ax.set_xlabel("Tamanho da janela (ciclos)")
    ax.set_ylabel("RMSE no teste (menor é melhor)")
    ax.set_ylim(10, 18)
    ax.set_title("Janela maior piora o teste nos dois modelos")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.03, "Motores curtos no teste são preenchidos com o primeiro valor", transform=ax.transAxes,
            ha="right", color=CINZA, fontsize=9)
    salvar(fig, "janela_vs_rmse.png")


def _matriz(ax, m, titulo):
    m = np.array(m, dtype=float)
    pct = m / m.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            cor = "white" if pct[i, j] > 0.5 else "#1a1a19"
            ax.text(j, i, f"{pct[i, j] * 100:.0f}%\n({int(m[i, j])})", ha="center", va="center", color=cor,
                    fontsize=10)
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Gravidade prevista")
    ax.grid(False)
    ax.set_title(titulo)


def triagem_matriz_confusao():
    base = json.loads((RES / "baseline_triagem_metrics.json").read_text(encoding="utf-8"))
    bert = json.loads((RES / "distilbert_metrics.json").read_text(encoding="utf-8"))
    fig, eixos = plt.subplots(1, 2, figsize=(11, 5.0))
    _matriz(eixos[0], base["tfidf_lr[texto_limpo]"]["referencia"]["matriz_confusao"], "TF-IDF + Regressão Logística")
    _matriz(eixos[1], bert["distilbert_int8"]["referencia"]["matriz_confusao"], "DistilBERT INT8")
    eixos[0].set_ylabel("Gravidade real")
    fig.suptitle("Matriz de confusão da triagem de OS (rótulo de referência, % por linha)", x=0.06, ha="left",
                 fontweight="bold", fontsize=13, y=1.0)
    fig.tight_layout()
    salvar(fig, "triagem_matriz_confusao.png")


def triagem_recall_critica():
    base = json.loads((RES / "baseline_triagem_metrics.json").read_text(encoding="utf-8"))
    bert = json.loads((RES / "distilbert_metrics.json").read_text(encoding="utf-8"))
    modelos = [("TF-IDF + Reg. Logística", base["tfidf_lr[texto_limpo]"]),
               ("DistilBERT fp32", bert["distilbert_fp32"]), ("DistilBERT INT8", bert["distilbert_int8"])]
    x = np.arange(len(modelos))
    larg = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, (rotulo, chave, cor) in enumerate([("Rótulo ruidoso (10% trocados)", "ruidoso", CINZA),
                                              ("Rótulo de referência", "referencia", AZUL)]):
        vals = [m[1][chave]["recall"]["crítica"] for m in modelos]
        barras = ax.bar(x + (k - 0.5) * larg, vals, larg, color=cor, label=rotulo)
        for b, v in zip(barras, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, virgula(v, 3), ha="center", fontweight="bold",
                    fontsize=10)
    ax.axhline(0.85, color=VERMELHO, linestyle="--", linewidth=1.4)
    ax.text(-0.47, 0.857, "meta 0,85", color=VERMELHO, ha="left", va="bottom", fontsize=10)
    ax.set_xticks(x, [m[0] for m in modelos])
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("Recall da classe crítica")
    ax.set_title("Recall da classe crítica: só o rótulo de referência chega à meta")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    salvar(fig, "triagem_recall_critica.png")


def distilbert_treino():
    bert = json.loads((RES / "distilbert_metrics.json").read_text(encoding="utf-8"))
    h = bert["historico"]
    ep = [x["epoca"] for x in h]
    fig, eixos = plt.subplots(1, 2, figsize=(10, 3.9))
    eixos[0].plot(ep, [x["perda_treino"] for x in h], marker="o", color=LARANJA, linewidth=2.4)
    eixos[0].set_title("Perda de treino")
    eixos[0].set_ylabel("Entropia cruzada")
    eixos[1].plot(ep, [x["f1_macro_val"] for x in h], marker="o", color=AZUL, linewidth=2.4)
    for e, v in zip(ep, [x["f1_macro_val"] for x in h], strict=True):
        eixos[1].annotate(virgula(v, 3), (e, v), textcoords="offset points", xytext=(0, 9), ha="center")
    eixos[1].set_title("F1-macro na validação")
    eixos[1].set_ylim(0.65, 0.72)
    for ax in eixos:
        ax.set_xticks(ep)
        ax.set_xlim(0.6, 4.4)
        ax.set_xlabel("Época")
    fig.suptitle(f"DistilBERT no Colab (GPU T4): {bert['treino_s']} s de treino, platô na época 3", x=0.06,
                 ha="left", fontweight="bold", fontsize=13, y=1.04)
    salvar(fig, "distilbert_treino.png")


def gatilho_f1():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    rotulos = ["Referência do modelo", "Amostra rotulada boa", "Amostra com rótulos embaralhados"]
    vals = [F1_REF, F1_BOA, F1_DEGRADADA]
    barras = ax.bar(rotulos, vals, color=[CINZA, VERDE, VERMELHO], width=0.55)
    for b, v in zip(barras, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, virgula(v, 3), ha="center", fontweight="bold")
    limite = F1_REF - 0.10
    ax.axhline(limite, color=VERMELHO, linestyle="--", linewidth=1.4)
    ax.text(2.45, limite + 0.012, f"limite: {virgula(limite, 2)} (queda de 10 pontos)", color=VERMELHO,
            ha="right", fontsize=10)
    ax.text(1, 0.30, "queda de\n3,4 pontos\nsem disparo", ha="center", va="center", color="white", fontsize=11,
            fontweight="bold")
    ax.text(2, 0.105, "queda de 47,5\npontos: dispara", ha="center", va="center", color="white", fontsize=10,
            fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("F1-macro")
    ax.set_title("Gatilho de retreino por queda de F1-macro (POST /triagem/avaliar)")
    ax.grid(axis="x", visible=False)
    salvar(fig, "gatilho_f1.png")


if __name__ == "__main__":
    rul_comparacao()
    rul_previsto_vs_real()
    janela_vs_rmse()
    triagem_matriz_confusao()
    triagem_recall_critica()
    distilbert_treino()
    gatilho_f1()
