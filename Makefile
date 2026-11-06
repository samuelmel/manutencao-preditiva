.PHONY: install data test lint
install:
	pip install --no-cache-dir -e ".[dev]"
data:
	python -m mp.pipeline
test:
	pytest -q
lint:
	ruff check src tests
