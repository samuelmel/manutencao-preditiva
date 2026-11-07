.PHONY: install data train-rul text serving colab graficos reproduce up simular test lint
COMPOSE ?= docker compose

install:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev,train,nlp,serve,bert,docs]"

# Pipeline de dados: bronze -> silver -> gold do C-MAPSS (precisa dos .txt em data/raw_download)
data:
	uv run python -m mp.pipeline

train-rul:
	uv run python -m mp.train_rul --fd FD001

# OS sintéticas (bronze de texto), silver (spaCy), baseline TF-IDF, Word2Vec e LDA
text:
	uv run python -m mp.textgen.build
	uv run python -m mp.train_triagem

# Exporta os artefatos que a API carrega (artifacts/serving); inclui o DistilBERT se já treinado
serving:
	uv run python -m mp.exportar_servico

# Zip + notebook para treinar o DistilBERT no Colab
colab:
	uv run python scripts/empacotar_colab.py

# Gráficos de resultados em img/ (a partir de docs/resultados)
graficos:
	uv run python scripts/gerar_graficos.py

# RNF06: refaz tudo a partir do bronze (seeds fixas). O DistilBERT é treinado à parte, no Colab.
reproduce: data train-rul text serving

# Sobe api + prometheus + grafana. Com Podman: make up COMPOSE="podman compose"
up:
	$(COMPOSE) up --build -d

simular:
	uv run python -m mp.simular

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
