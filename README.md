# Manutenção Preditiva 4.0

RUL por LSTM (NASA C-MAPSS) + triagem de ordens de serviço por PLN, fundidos em uma fila de prioridade.
Plano completo de engenharia: arquitetura medalhão (bronze/silver/gold), API FastAPI + ONNX, Prometheus + Grafana.

## Status

| Fase | Estado |
|---|---|
| 0: repo, CI, bronze → silver → gold do C-MAPSS | feito (testado com dados sintéticos) |
| 1: baselines + LSTM (FD001) + ONNX | feito; LSTM não superou o Random Forest (ver Resultados da Fase 1) |
| 2: OS sintéticas + spaCy + TF-IDF/Word2Vec/LDA | pendente |
| 3: DistilBERT, fusão, API, Grafana, drift | pendente |

## Uso

1. Baixe o C-MAPSS (NASA Prognostics Data Repository) e coloque `train_FD00x.txt`, `test_FD00x.txt`
   e `RUL_FD00x.txt` em `data/raw_download/`.
2. `make install && make data && make test` (usa [uv](https://docs.astral.sh/uv/); sem pip)

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

## Resultados da Fase 1 (FD001)

RMSE em ciclos no conjunto de teste (100 motores, último ciclo de cada um). Menor é melhor.
Janela de 30 ciclos, Adam, MSE, EarlyStopping; alvo escalado por RUL/125.

| Modelo | RMSE | Score NASA |
|---|---|---|
| Média constante | 40,69 | 19887 |
| **Random Forest** (média, desvio, último valor e inclinação da janela) | **11,96** | 218 |
| LSTM + features da janela (híbrido, média de 5 seeds) | 12,62 | 236 |
| LSTM puro, receita final (dropout 0,1 + ReduceLROnPlateau, média de 3 seeds) | 13,23 | n/d |
| LSTM puro, receita inicial (3 seeds) | 13,42 | n/d |
| Mistura 30% LSTM + 70% RF | 11,76 | 209 |

Exportação ONNX do LSTM: 0,32 MB, diferença máxima para o Keras de 8,4e-5 (tolerância 1e-4).

**Conclusão: o LSTM não bate o Random Forest no FD001.** Cinco abordagens deram o mesmo padrão
(LSTM entre 12,6 e 13,5 contra 11,96 do RF):

- Hiperparâmetros: batch 64/128/256, lr 1e-3 e 3e-3, dropout 0,1 e 0,2, scheduler. Variação de ~0,2 ciclo, ruído.
- Média de 5 seeds: 13,13. Os modelos erram quase igual, então a média ajuda pouco.
- Mistura LSTM + RF: 11,76 contra 11,96. Bootstrap (2000 reamostragens dos motores): IC 95% de −0,65 a +0,77,
  indistinguível de ruído; a proporção foi escolhida olhando o teste, então não é um resultado limpo.
- Híbrido (ramo LSTM + ramo denso com as features do RF): 12,62; bootstrap dá 16% de chance de ser melhor que o RF.
- Janela maior: piora o teste (50 → LSTM 14,3 / RF 13,1; 70 → LSTM 15,9 / RF 14,9), mesmo com a validação melhorando.
  O teste avalia só o fim de cada motor e há motores com poucos ciclos (mínimo 31): o início é preenchido com o
  primeiro valor, o que injeta informação falsa. Vale para os dois modelos, por isso W=30 fica.

Explicação provável: FD001 tem poucos dados, sinais limpos e degradação quase linear, o que favorece modelos simples
com boas estatísticas feitas à mão. O teste tem só 100 motores, então diferenças abaixo de ~1 ciclo não são conclusivas.
A comparação deve ser refeita em FD002/FD004 (várias condições de operação), onde sequências costumam ajudar mais.
Os experimentos foram feitos com scripts avulsos; a receita final ainda não está em `train_rul.py`.
