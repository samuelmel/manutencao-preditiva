"""Empacota o código e o corpus de OS num zip para subir no Colab (sem precisar de push no GitHub).

Uso: python scripts/empacotar_colab.py   ->  artifacts/colab/mp-colab.zip
Precisa de data/silver_text/os.parquet (gerado por `make text`). Gera também o notebook.
"""
import json
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "artifacts" / "colab"
CORPUS = RAIZ / "data" / "silver_text" / "os.parquet"


def md(texto):
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(True)}


def codigo(texto):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": texto.strip("\n").splitlines(True)}


CELULAS = [
    md("""
# DistilBERT para triagem de OS (Fase 3)

Este notebook só **chama o pacote `mp`** (a lógica está em `src/mp`, testada). Ele treina o
`distilbert-base-multilingual-cased` na GPU, exporta ONNX INT8 e devolve 3 arquivos para o repositório.

**Antes de rodar:** `Ambiente de execução > Alterar o tipo de ambiente de execução > GPU T4`.
Você precisa do arquivo `mp-colab.zip` (gerado por `python scripts/empacotar_colab.py`).
"""),
    codigo("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
assert torch.cuda.is_available(), "Sem GPU: troque o tipo de ambiente de execução para GPU T4"
"""),
    md("## 1. Enviar o pacote e (opcional) guardar o checkpoint no Drive"),
    codigo("""
from google.colab import files
import zipfile, os
enviado = files.upload()                     # selecione mp-colab.zip
zipfile.ZipFile(next(iter(enviado))).extractall("/content/mp")
os.chdir("/content/mp")
print(sorted(os.listdir(".")))
"""),
    codigo("""
# Opcional, mas recomendado: a sessão do Colab pode cair. Com isto o melhor checkpoint
# (artifacts/bert/melhor.pt) fica no seu Drive a cada época.
from google.colab import drive
drive.mount("/content/drive")
os.makedirs("/content/drive/MyDrive/mp_bert", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
if not os.path.exists("artifacts/bert"):
    os.symlink("/content/drive/MyDrive/mp_bert", "artifacts/bert")
"""),
    md("## 2. Instalar o pacote"),
    codigo("""
!pip -q install -e ".[bert]"
"""),
    md("""
## 3. Treinar

Mesma divisão por motor e mesma seed do baseline TF-IDF, para a comparação ser justa.
Na T4, 4 épocas levam alguns minutos (fp16). O modelo (~540 MB) é baixado do Hugging Face aqui dentro.
"""),
    codigo("""
!python -m mp.train_bert --epocas 4 --batch 32 --max-len 64
"""),
    md("## 4. Conferir o resultado"),
    codigo("""
import json
m = json.load(open("artifacts/bert/metrics.json", encoding="utf-8"))
for nome in ("distilbert_fp32", "distilbert_int8"):
    r = m[nome]
    print(nome, "| F1-macro ruidoso", round(r["ruidoso"]["f1_macro"], 3),
          "| referência", round(r["referencia"]["f1_macro"], 3),
          "| recall crítica ruidoso", round(r["ruidoso"]["recall"]["crítica"], 3),
          "| referência", round(r["referencia"]["recall"]["crítica"], 3))
print("paridade:", m["paridade"])
print("tamanhos (MB):", m["tamanho_mb"], "| treino (s):", m["treino_s"])
"""),
    md("""
## 5. Levar os arquivos de volta

Baixe os 3 arquivos abaixo e coloque-os em `artifacts/bert/` no repositório. Depois, no seu computador:
`python -m mp.exportar_servico` e reinicie a API (`podman compose restart api`).
"""),
    codigo("""
import shutil
os.makedirs("/content/saida", exist_ok=True)
for f in ("bert_int8.onnx", "tokenizer.json", "metrics.json"):
    shutil.copy(f"artifacts/bert/{f}", "/content/saida/")
shutil.make_archive("/content/bert-saida", "zip", "/content/saida")
files.download("/content/bert-saida.zip")
"""),
]


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"falta {CORPUS}: rode `make text` antes")
    SAIDA.mkdir(parents=True, exist_ok=True)
    destino = SAIDA / "mp-colab.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for pasta in ("src", "configs"):
            for f in sorted((RAIZ / pasta).rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
                    z.write(f, f.relative_to(RAIZ).as_posix())
        z.write(RAIZ / "pyproject.toml", "pyproject.toml")
        z.write(CORPUS, "data/silver_text/os.parquet")
    nb = {"cells": CELULAS, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"}}}
    caminho_nb = RAIZ / "notebooks" / "treinar_distilbert.ipynb"
    caminho_nb.parent.mkdir(exist_ok=True)
    caminho_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{destino} ({destino.stat().st_size / 1e6:.2f} MB)\n{caminho_nb}")


if __name__ == "__main__":
    main()
