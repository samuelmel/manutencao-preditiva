# Manutenção Preditiva 4.0

RUL por LSTM (NASA C-MAPSS) + triagem de ordens de serviço por PLN, fundidos em uma fila de prioridade.
Plano completo de engenharia: arquitetura medalhão (bronze/silver/gold), API FastAPI + ONNX, Prometheus + Grafana.

## Status

| Fase | Estado |
|---|---|
| 0: repo, CI, bronze → silver → gold do C-MAPSS | feito (testado com dados sintéticos) |
| 1: baselines + LSTM (FD001) + ONNX | feito; LSTM não superou o Random Forest (ver Resultados da Fase 1) |
| 2: OS sintéticas + spaCy + TF-IDF/Word2Vec/LDA | feito (ver Resultados da Fase 2) |
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

## Resultados da Fase 2 (triagem de OS)

**Os textos são sintéticos.** Não existe dataset público de OS em português, então `mp.textgen` gera as OS por
templates (sem LLM), a partir do RUL real do C-MAPSS: 4 ciclos por faixa de gravidade por motor, nos 4 subconjuntos
(11.344 OS, classes balanceadas). A gravidade de referência vem do RUL (<15 crítica, <40 alta, <80 média, resto baixa).
Para não ficar fácil demais: 30% das OS usam sintomas de uma gravidade vizinha, há abreviações ("rolam.", "temp"),
erros de digitação, caixa alta/baixa, códigos com e sem hífen, e **10% dos rótulos são trocados de propósito**
(`gravidade`; a referência limpa fica em `gravidade_ref`). Por isso o teto realista de acerto é bem abaixo de 100%.

Pipeline: `python -m mp.textgen.build` (bronze JSONL) → `python -m mp.train_triagem` (silver: limpeza, lemas, NER;
depois baseline, Word2Vec e LDA). Divisão treino/teste por motor, nunca por OS (`GroupShuffleSplit`).

Baseline TF-IDF (1 a 3-gramas) + Regressão Logística, 20% dos motores no teste:

| Entrada | F1-macro (rótulo ruidoso) | F1-macro (referência) | Recall da classe crítica (ruidoso) |
|---|---|---|---|
| `texto_limpo` | 0,693 | 0,771 | 0,763 |
| `lemas` (spaCy `pt_core_news_sm`) | 0,692 | 0,769 | 0,761 |

- Lemas e texto limpo empatam: o vocabulário sintético é pequeno e dirigido por templates, então lematizar não
  acrescenta informação. O modelo usado fica registrado em `modelo_spacy` no parquet e em `artifacts/triagem/metrics.json`.
- Com 10% de rótulos trocados, o recall da crítica de ≥ 0,85 (meta do plano) ainda não foi atingido; é a meta do
  classificador final da Fase 3 (DistilBERT), não do baseline. Os erros se concentram entre classes vizinhas.
- NER (EntityRuler): código de peça 100% (inclusive `HPC0412` e `hpc-0412`, que o NER genérico não pega);
  cobertura de componente 93% e de sintoma 91%. O que falta são erros de digitação, que não casam com o léxico.
- Word2Vec (50 dimensões, treinado no corpus) acha OS parecidas: para "vibração forte compressor alta" volta OS com o
  mesmo sintoma, mesmo com palavras erradas ("verfiicar", "mootr").
- LDA: k escolhido por coerência `c_v` entre 3, 4, 5, 6 e 8 → k = 8 (0,317). A coerência é baixa e os tópicos seguem
  as palavras dos templates ("motor", "peça", "durante ronda"), efeito esperado de texto sintético; k = 8 é a borda do
  intervalo testado.

## Ambiente

`make install` usa Python 3.12. O `gensim` ainda não tem pacote pronto para Python 3.14, então crie o ambiente com 3.12:
`uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[dev,train,nlp]"`. O extra `nlp` já inclui o
modelo `pt_core_news_sm` 3.8.0 (wheel da explosion/spacy-models). Sem ele o código cai para `blank("pt")`.
