# Manutenção Preditiva 4.0

RUL por LSTM (NASA C-MAPSS) + triagem de ordens de serviço por PLN, fundidos em uma fila de prioridade.
Plano completo de engenharia: arquitetura medalhão (bronze/silver/gold), API FastAPI + ONNX, Prometheus + Grafana.

## Status

| Fase | Estado |
|---|---|
| 0: repo, CI, bronze → silver → gold do C-MAPSS | feito (testado com dados sintéticos) |
| 1: baselines + LSTM (FD001) + ONNX | feito; LSTM não superou o Random Forest (ver Resultados da Fase 1) |
| 2: OS sintéticas + spaCy + TF-IDF/Word2Vec/LDA | feito (ver Resultados da Fase 2) |
| 3: fusão, drift, API, Docker/Prometheus/Grafana | feito e verificado no Podman; DistilBERT: código testado, treino no Colab pendente (ver Fase 3) |

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

## Fase 3 (serviço, drift e observabilidade)

Fluxo: `make data` → `make train-rul` → `make text` → `make serving` (gera `artifacts/serving`, 0,9 MB) → `make up`.

- **Fusão** (`mp.fusion`, pesos em `configs/fusion.yaml`): `0,6·(1 − min(RUL,125)/125) + 0,4·P(gravidade ∈ {alta, crítica})`.
  É regra de negócio, não modelo treinado. Motor sem OS conhecida entra com a parcela da OS igual a 0.
- **API** (`mp.api`, FastAPI + ONNX Runtime, sem PyTorch): `/rul`, `/triagem`, `/prioridade` (fila ordenada),
  `/drift`, `/metrics` (Prometheus) e `/health`. Entrada inválida devolve 422. O classificador de gravidade em serviço
  é o baseline TF-IDF + Regressão Logística (o DistilBERT ainda não foi treinado).
- **Casos de borda:** motor com menos de 30 ciclos → `low_confidence=true`; RUL negativo → 0; OS vazia, só com código de
  peça ou em outro idioma (>60% dos tokens fora do vocabulário) → `gravidade=indefinida` e `revisao_humana=true`;
  condição operacional longe dos centroides de treino → `condicao_desconhecida=true` e contador no Prometheus.
- **Drift:** PSI por sensor contra uma amostra do treino (gatilho: PSI > 0,2 em **mais de 3** sensores; KS sob demanda em
  `/drift`), taxa de tokens fora do vocabulário e PSI da distribuição de gravidade das OS. O PSI dos sensores é recalculado
  a cada 20 previsões: com KS a cada requisição o `/rul` levava 170 ms; agora a previsão custa ~1 ms.
  O PSI das classes de OS usa uma **linha de base de produção** (as 100 primeiras OS), não o treino: o treino é balanceado
  por construção e alarmaria sempre contra uma população real. Só passa a valer depois de 150 OS.
- **Métricas** só com rótulos `unit_id`, `sensor` e `gravidade` (nunca texto de OS).
- **Grafana versionado** (`grafana/`): fontes Prometheus e Infinity, dashboard gerado por `scripts/gerar_dashboard.py` e três
  alertas (drift de sensores, RUL < 15, drift de classes de OS) carregados por *provisioning*. O mapa de calor do PSI é um
  `state-timeline` por sensor. Prometheus com `--storage.tsdb.retention.size=200MB`; portas presas em `127.0.0.1`;
  senha do Grafana em `GRAFANA_ADMIN_PASSWORD` (padrão local `admin`).
- **Simulador:** `make simular` envia os 100 motores de teste e uma OS coerente com o RUL real de cada um;
  `python -m mp.simular --drift` desloca 6 sensores e dispara o alerta.

### Verificado nesta etapa
- 81 testes passam (fusão, PSI/KS, contrato da API e 422, casos de borda, métricas, coerência dos JSON/YAML do Grafana com a API).
- Teste com artefatos reais: RMSE pela API de motores de teste do FD001 < 16 (LSTM offline: ~13), `/rul` p95 < 50 ms e
  `/triagem` p95 < 300 ms em CPU (RNF03 e RNF04).
- Tráfego normal: 0 sensores em drift e PSI de texto 0. Com `--drift`: 6 sensores em drift e `alerta_drift=true` na API.

### Verificado com `podman compose up` (Podman 5.8, Windows/WSL 2)
- Imagem da API: **741 MB** (meta < 1 GB). Prometheus coleta a API (alvo `up`, 100 motores).
- Grafana 11.3: fontes Prometheus e Infinity (health OK), dashboard e 3 alertas carregados por *provisioning*,
  sem configuração pela interface. As consultas dos 8 painéis rodam sem erro e devolvem dados; a tabela da fila
  usa `parser: backend` no Infinity (sem isso o plugin só monta a tabela no navegador).
- Drift simulado (`python -m mp.simular --drift`) → alerta **Drift em mais de 3 sensores** em `firing`; motores com
  RUL < 15 → **Motor com RUL crítico** em `firing`; o alerta de distribuição de OS segue `inactive` com tráfego normal.
- Subir: `podman compose up --build -d` (ou `docker compose`); Grafana em `http://127.0.0.1:3000`
  (admin / `GRAFANA_ADMIN_PASSWORD`, padrão `admin`). Na primeira vez o Grafana leva ~3 min migrando o banco e baixando o plugin.
  Nomes de imagem com `docker.io/` porque o Podman não resolve nomes curtos. Para liberar memória: `podman compose down`
  e `podman machine stop`.

### Visto no navegador (Grafana 11.3 + Infinity 3.7.1)
- A fila de manutenção, o RUL por motor, o PSI por sensor (linhas e mapa de calor), a pizza de gravidade, a latência e os
  indicadores de drift renderizam com dados. O plugin Infinity está **fixado em 3.7.1**: a 4.x exige Grafana ≥ 11.6 e
  quebrava a tabela no navegador (`react/jsx-runtime` 404), sem erro algum nas consultas pela API. A ordem das colunas
  da tabela é fixada por transformação (o parser de backend as ordena por nome).
- A fila fica em memória na API: reiniciar a API esvazia a tabela até chegarem novos `/rul` (limitação conhecida).

### Ainda não feito
- Gatilho de queda de 10 pontos no F1-macro numa amostra rotulada: a API não recebe rótulos; hoje só existe o proxy de
  PSI das classes de OS.
- Treino real do DistilBERT (ver abaixo) e, com ele, a meta de recall da crítica ≥ 0,85.

### DistilBERT multilíngue (treino no Colab)
Código pronto e testado; o **treino real ainda não rodou**. Como o PyTorch local é só CPU, o treino é no Colab (GPU T4, fp16):
1. `make colab` gera `artifacts/colab/mp-colab.zip` (código + corpus de OS, 1,2 MB, sem precisar de push) e o
   notebook `notebooks/treinar_distilbert.ipynb`, que só chama `python -m mp.train_bert`.
2. No Colab: envia o zip, treina (mesma divisão por motor e mesma seed do baseline), exporta ONNX INT8 e devolve
   `bert_int8.onnx`, `tokenizer.json` e `metrics.json`. O melhor checkpoint vai para o seu Drive a cada época.
3. Coloque os 3 arquivos em `artifacts/bert/`, rode `make serving` e reinicie a API: com `bert_int8.onnx` e
   `tokenizer.json` em `artifacts/serving`, o serviço usa o DistilBERT (ONNX Runtime + `tokenizers`, sem PyTorch na imagem).
   O TF-IDF continua como baseline e como detector de texto em outro idioma.

Verificado sem baixar o modelo (`tests/test_bert_pipeline.py`, DistilBERT minúsculo de pesos aleatórios): o laço de treino
memoriza um lote pequeno (acusaria rótulos desalinhados), o ONNX fp32 bate com o PyTorch (< 1e-4), o ONNX aceita lotes e
comprimentos variados, o INT8 é menor, e o serviço lê o que o treino exporta com os mesmos ids de token que o `transformers`.
**Não verificado:** a qualidade do modelo real (F1-macro e recall da crítica) e a latência `/triagem` com o DistilBERT.
Com 10% de rótulos trocados de propósito, o teto de recall medido no rótulo ruidoso fica perto de 0,9; os números saem
nos dois rótulos (ruidoso e de referência) em `metrics.json`.
