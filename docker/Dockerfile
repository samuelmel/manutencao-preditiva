# Imagem de inferência: só ONNX Runtime, sem PyTorch (RNF05, meta < 1 GB). Contexto: raiz do repo.
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[serve]"

FROM python:3.12-slim
ENV MP_ROOT=/app \
    MP_ARTIFACTS=/app/artifacts/serving \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=build /install /usr/local
COPY configs /app/configs
WORKDIR /app
RUN useradd --system --no-create-home mp
USER mp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "mp.api.app:criar_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
