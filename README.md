# Manutenção Preditiva 4.0

RUL por LSTM (NASA C-MAPSS) + triagem de ordens de serviço por PLN, fundidos em uma fila de prioridade.
Plano completo de engenharia: arquitetura medalhão (bronze/silver/gold), API FastAPI + ONNX, Prometheus + Grafana.

## Status

| Fase | Estado |
|---|---|
| 0: repo, CI, bronze → silver → gold do C-MAPSS | feito (testado com dados sintéticos) |
| 1: baselines + LSTM (FD001) + ONNX | pendente |
| 2: OS sintéticas + spaCy + TF-IDF/Word2Vec/LDA | pendente |
| 3: DistilBERT, fusão, API, Grafana, drift | pendente |

## Uso

1. Baixe o C-MAPSS (NASA Prognostics Data Repository) e coloque `train_FD00x.txt`, `test_FD00x.txt`
   e `RUL_FD00x.txt` em `data/raw_download/`.
2. `make install && make data && make test`

`data/` não é versionado. Bronze é imutável: o pipeline copia os arquivos byte a byte, grava
checksums em `MANIFEST.json` e falha se um original mudar.

## Decisões

- **RUL** = ciclo máximo do motor − ciclo atual, com teto em 125 (degradação inicial não é observável).
- **Teste**: RUL do último ciclo vem de `RUL_FDxxx.txt`; os anteriores somam a distância ao fim.
- **Sensores constantes** são detectados pela variância no treino, não por lista fixa.
- **Normalização**: Min-Max ajustado só no treino; em FD002/FD004, por cluster K-Means (k=6) das 3 condições operacionais.
  `Normalizer.cluster_distance` expõe a distância ao centroide para o alerta de condição nunca vista.
- **Divisão treino/validação por `unit_id`**, nunca por linha (teste automático garante).
- **Motor com < 30 ciclos**: início repetido e `low_conf_test=True`.
- Limitação: o scaler é ajustado no arquivo de treino inteiro, antes da separação treino/validação por motor.
