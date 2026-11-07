.PHONY: install data text train-rul serving colab up simular test lint
install:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev,train]"
data:
	uv run python -m mp.pipeline
test:
	uv run pytest -q
lint:
	uv run ruff check src tests
