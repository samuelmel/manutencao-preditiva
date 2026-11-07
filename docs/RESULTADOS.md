# Resultados e decisões técnicas

Este arquivo reúne os números do projeto, as decisões de engenharia e o que foi verificado em cada etapa. Para uma
visão geral, leia o [README](../README.md). Para a história de como o projeto foi feito, leia o
[README-PROCESSO](../README-PROCESSO.md). Os arquivos de métricas brutas estão em
[`docs/resultados/`](resultados/).

## Decisões de dados

- RUL é o ciclo máximo do motor menos o ciclo atual, com teto em 125. No começo da vida do motor a degradação não é
  observável, e o teto evita que o modelo tente prever isso.
- No conjunto de teste, o RUL do último ciclo vem de `RUL_FDxxx.txt`, e os anteriores somam a distância até o fim.
- Sensores constantes são detectados pela variância no treino, não por uma lista fixa. No FD001 sobram 14 sensores.
- A normalização Min-Max é ajustada só no treino. No FD002 e no FD004 ela é feita por grupo (K-Means com k = 6 sobre as
  três condições de operação). `Normalizer.cluster_distance` expõe a distância ao centroide, que serve para o alerta de
  condição nunca vista.
- A divisão entre treino e validação é por `unit_id`, nunca por linha. Um teste automático garante isso.
- Motor com menos de 30 ciclos tem o início repetido e a marca `low_conf_test=True`.
- O bronze é imutável: o pipeline copia os arquivos byte a byte, grava checksums em `MANIFEST.json` e falha se um
  original mudar.
- Limitação conhecida: o normalizador é ajustado no arquivo de treino inteiro, antes da separação entre treino e
  validação por motor.

## Fase 1: previsão de vida útil (FD001)

RMSE em ciclos no conjunto de teste (100 motores, último ciclo de cada um). Menor é melhor. Janela de 30 ciclos, Adam,
MSE, EarlyStopping, alvo escalado por RUL/125.

| Modelo | RMSE | Score NASA |
|---|---|---|
| Média constante | 40,69 | 19.887 |
| Random Forest (média, desvio, último valor e inclinação da janela) | 11,96 | 218 |
| LSTM com as estatísticas da janela (híbrido, média de 5 seeds) | 12,62 | 236 |
| LSTM puro, receita final (dropout 0,1 e ReduceLROnPlateau, média de 3 seeds) | 13,23 | não medido |
| LSTM puro, receita inicial (3 seeds) | 13,42 | não medido |
| Mistura de 30% LSTM e 70% Random Forest | 11,76 | 209 |

O ONNX do LSTM pesa 0,32 MB e difere do Keras em no máximo 8,4e-5 (tolerância de 1e-4). O que o Score e as outras
métricas medem está em [METRICAS.md](METRICAS.md), e a teoria do modelo em [TEORIA_LSTM.md](TEORIA_LSTM.md).

O LSTM não bate o Random Forest no FD001. Cinco abordagens deram o mesmo padrão, com o LSTM entre 12,6 e 13,5 contra
11,96 do Random Forest:

- Hiperparâmetros: batch de 64, 128 e 256, learning rate de 1e-3 e 3e-3, dropout de 0,1 e 0,2, com e sem scheduler. A
  variação ficou em torno de 0,2 ciclo, o que é ruído.
- Média de 5 seeds: 13,13. Os modelos erram quase igual, então a média ajuda pouco.
- Mistura com o Random Forest: 11,76 contra 11,96. O bootstrap (2.000 reamostragens dos motores) deu intervalo de 95%
  entre −0,65 e +0,77, indistinguível de ruído. A proporção foi escolhida olhando o teste, então o resultado não é limpo.
- Híbrido (ramo LSTM mais ramo denso com as estatísticas do Random Forest): 12,62, com 16% de chance de ser melhor que o
  Random Forest.
- Janela maior piora o teste: com 50 ciclos o LSTM fica em 14,3 e o Random Forest em 13,1, com 70 ciclos ficam em 15,9 e
  14,9, mesmo com a validação melhorando. O teste avalia só o fim de cada motor, e alguns motores têm poucos ciclos (o
  menor tem 31). O início da janela é preenchido com o primeiro valor, o que injeta informação falsa. Isso vale para os
  dois modelos, e por isso a janela fica em 30.

O FD001 tem poucos motores, sinais limpos e degradação quase linear, o que favorece modelos simples com boas
estatísticas feitas à mão. Com 100 motores de teste, diferenças abaixo de um ciclo não são conclusivas. A comparação
ainda não foi refeita no FD002 e no FD004, onde as condições de operação variam e sequências costumam ajudar mais.

## Fase 2: triagem de ordens de serviço

Os textos são sintéticos. Não existe base pública de OS em português, então `mp.textgen` gera as ordens por templates,
sem modelo de linguagem, a partir do RUL real do C-MAPSS. São 4 ciclos por faixa de gravidade por motor, nos quatro
subconjuntos, num total de 11.344 OS com classes balanceadas. A gravidade de referência vem do RUL: abaixo de 15 é
crítica, abaixo de 40 é alta, abaixo de 80 é média, e o resto é baixa.

Para o problema não ficar fácil demais, 30% das OS usam sintomas de uma gravidade vizinha, há abreviações ("rolam.",
"temp"), erros de digitação, caixa alta e baixa, códigos de peça com e sem hífen, e 10% dos rótulos são trocados de
propósito (`gravidade`; a referência limpa fica em `gravidade_ref`). O teto realista de acerto fica bem abaixo de 100%.

O pipeline é `python -m mp.textgen.build` (bronze em JSONL) e `python -m mp.train_triagem` (silver com limpeza, lemas e
NER, depois baseline, Word2Vec e LDA). A divisão entre treino e teste é por motor (`GroupShuffleSplit`).

Baseline TF-IDF de 1 a 3 gramas com Regressão Logística, 20% dos motores no teste:

| Entrada | F1-macro (ruidoso) | F1-macro (referência) | Recall da crítica (ruidoso) |
|---|---|---|---|
| `texto_limpo` | 0,693 | 0,771 | 0,763 |
| `lemas` (spaCy `pt_core_news_sm`) | 0,692 | 0,769 | 0,761 |

- Lemas e texto limpo empatam. O vocabulário sintético é pequeno e guiado por templates, então lematizar não acrescenta
  informação.
- NER com EntityRuler: o código de peça é achado em 100% das OS, inclusive `HPC0412` e `hpc-0412`, que o NER genérico não
  reconhece. A cobertura de componente é 93% e a de sintoma é 91%. O que falta são erros de digitação, que não casam com
  o léxico.
- Word2Vec (50 dimensões, treinado no próprio corpus) acha OS parecidas. Para "vibração forte compressor alta" volta OS
  com o mesmo sintoma, mesmo com palavras escritas errado ("verfiicar", "mootr").
- LDA: o k foi escolhido por coerência `c_v` entre 3, 4, 5, 6 e 8, e deu k = 8 (coerência 0,317). A coerência é baixa e
  os tópicos seguem as palavras dos templates ("motor", "peça", "durante ronda"), o que é esperado em texto sintético. O
  k = 8 é a borda do intervalo testado.

### DistilBERT multilíngue

Modelo `distilbert-base-multilingual-cased`, treinado no Colab numa GPU T4 com fp16: 4 épocas, batch de 32, learning rate
de 5e-5, janela de 64 tokens, texto bruto da OS. O treino levou 143 segundos. A divisão por motor e a seed são as mesmas
do baseline, e o teste tem 2.272 OS de motores que o modelo não viu.

| Modelo | F1-macro (ruidoso) | F1-macro (referência) | Recall da crítica (ruidoso) | Recall da crítica (referência) |
|---|---|---|---|---|
| TF-IDF e Regressão Logística (`texto_limpo`) | 0,693 | 0,771 | 0,763 | 0,857 |
| DistilBERT fp32 | 0,696 | 0,773 | 0,761 | 0,852 |
| DistilBERT INT8 (o que o serviço usa) | 0,690 | 0,768 | 0,746 | 0,836 |

- O DistilBERT não supera o TF-IDF neste corpus. A diferença de 0,003 no F1-macro é ruído. O texto é sintético e guiado
  por templates, e o limite de acerto vem do desenho dos dados, com sintomas de gravidade vizinha e 10% de rótulos
  trocados, e não da capacidade do modelo. Com OS reais, com paráfrases e vocabulário livre, a vantagem de um modelo
  pré-treinado deveria aparecer, mas este corpus não permite afirmar isso.
- A meta de recall da crítica de 0,85 foi atingida no rótulo de referência pelo TF-IDF (0,857) e pelo DistilBERT fp32
  (0,852). O INT8 (0,836) fica abaixo, e no rótulo ruidoso, que é o que um sistema real veria, nenhum chega lá (0,75 a
  0,76).
- A quantização INT8 custou cerca de 0,6 ponto de F1-macro e 1,6 ponto de recall da crítica, em troca de um modelo quatro
  vezes menor (135,7 MB contra 541,4 MB). O INT8 concorda com o fp32 em 97,5% das OS.
- O ONNX fp32 difere do PyTorch em no máximo 3,6e-7 na probabilidade (tolerância de 1e-4).
- Para produção, neste corpus o TF-IDF (menos de 5 MB, sem GPU) rende o mesmo. O DistilBERT fica integrado como opção, e
  só OS reais poderiam mostrar vantagem dele.

Para reproduzir: `make colab` gera o zip e o notebook `notebooks/treinar_distilbert.ipynb`. Os três arquivos que o
notebook devolve vão para `artifacts/bert/`, e depois rodam `make serving` e um reinício da API. O pipeline inteiro
também é coberto por `tests/test_bert_pipeline.py`, que usa um modelo minúsculo e não baixa nada.

## Fase 3: serviço, drift e observabilidade

O fluxo completo é `make data`, `make train-rul`, `make text`, `make serving` e `make up`. O `make reproduce` encadeia
os quatro primeiros, com seeds fixas, a partir do bronze. O DistilBERT é treinado à parte, no Colab. O `make` precisa
estar instalado (no Windows, `winget install GnuWin32.Make`, ou rode à mão os comandos `uv run` do Makefile). Com
Podman: `make up COMPOSE="podman compose"`.

- A fusão (`mp.fusion`, pesos em `configs/fusion.yaml`) calcula `0,6 · (1 − min(RUL, 125) / 125) + 0,4 · P(gravidade em
  {alta, crítica})`. É uma regra de negócio, não um modelo treinado. Um motor sem OS conhecida entra com a parcela da OS
  igual a zero.
- A API (`mp.api`, FastAPI com ONNX Runtime, sem PyTorch) expõe `/rul`, `/triagem`, `/triagem/avaliar`, `/prioridade`
  (a fila ordenada), `/drift`, `/metrics` e `/health`. Entrada inválida devolve 422.
- Casos de borda: motor com menos de 30 ciclos devolve `low_confidence=true`, RUL negativo vira 0, e OS vazia, só com
  código de peça ou em outro idioma (mais de 60% dos tokens fora do vocabulário) devolve `gravidade=indefinida` e
  `revisao_humana=true`. Condição de operação longe dos centroides de treino devolve `condicao_desconhecida=true` e
  incrementa um contador no Prometheus.
- O drift dos sensores usa PSI contra uma amostra do treino. O gatilho é PSI acima de 0,2 em mais de 3 sensores. O KS
  fica sob demanda em `/drift`. O PSI é recalculado a cada 20 previsões: com o KS a cada requisição o `/rul` levava 170
  ms, e agora a previsão custa cerca de 1 ms.
- O drift do texto mede a taxa de tokens fora do vocabulário e o PSI da distribuição de gravidade. O PSI das classes
  usa uma linha de base de produção (as 100 primeiras OS), não o treino, porque o treino é balanceado por construção e
  alarmaria sempre contra uma população real. Ele só passa a valer depois de 150 OS.
- O gatilho de retreino por F1-macro funciona assim: `POST /triagem/avaliar` recebe OS com a gravidade real, calcula o
  F1-macro e compara com o F1 de referência do classificador servido (guardado em `meta.json`). Se a queda chega a 10
  pontos, devolve `gatilho_retreino=true`. OS marcadas como indefinidas não entram na conta, e amostras com menos de 30
  OS são só informativas. A avaliação não mexe nas janelas de drift. No contêiner, uma amostra boa deu queda de 3,4
  pontos (sem disparo) e uma amostra com rótulos embaralhados deu queda de 47,5 pontos (dispara).
- As métricas só usam rótulos `unit_id`, `sensor` e `gravidade`, nunca texto de OS, para não explodir a cardinalidade.
- O Grafana é versionado em `grafana/`: fontes Prometheus e Infinity, um dashboard gerado por
  `scripts/gerar_dashboard.py` e quatro alertas (drift de sensores, RUL abaixo de 15, drift de classes de OS e queda de
  F1), todos carregados por provisioning. O mapa de calor do PSI é um `state-timeline` por sensor.
- O Prometheus roda com `--storage.tsdb.retention.size=200MB`, as portas ficam presas em `127.0.0.1`, e a senha do
  Grafana vem de `GRAFANA_ADMIN_PASSWORD` (o padrão local é `admin`).
- O simulador (`make simular`) envia os 100 motores de teste e uma OS coerente com o RUL real de cada um.
  `python -m mp.simular --drift` desloca 6 sensores e dispara o alerta de drift. `--f1` e `--f1-degradado` enviam uma
  amostra rotulada.

### O que foi verificado

Em testes automáticos (112 testes): fusão, PSI e KS, contrato da API e respostas 422, casos de borda, métricas expostas,
gatilho de F1, coerência dos arquivos do Grafana com a API, o pipeline do DistilBERT com um modelo minúsculo, e a
existência de todo comando `make` citado na documentação.

Com os artefatos reais: o RMSE medido pela API sobre os motores de teste do FD001 ficou abaixo de 16 (o LSTM offline dá
cerca de 13), `/rul` ficou com p95 abaixo de 50 ms e `/triagem` com p95 abaixo de 300 ms, em CPU.

Com `podman compose up` (Podman 5.8, Windows com WSL 2):

- A imagem da API tem 777 MB, abaixo da meta de 1 GB. Dentro do contêiner, `/triagem` com o DistilBERT leva 13,6 ms de
  mediana e 33 ms de p95 via HTTP. A API usa 365 MB de RAM.
- O Prometheus coleta a API (alvo `up`, 100 motores). No Grafana 11.3 as duas fontes passam no health check, o dashboard e
  os alertas carregam por provisioning, e as consultas dos 8 painéis rodam sem erro e devolvem dados.
- Com tráfego normal, nenhum sensor entra em drift. Com `--drift`, o alerta de drift de sensores fica em `firing`. Motores
  com RUL abaixo de 15 deixam o alerta de RUL crítico em `firing`, e a amostra de F1 degradada deixa o alerta de F1 em
  `firing`. O alerta de distribuição de OS segue `inactive` com tráfego normal.
- Num navegador, a fila de manutenção, o RUL por motor, o PSI por sensor, a pizza de gravidade, a latência e os
  indicadores de drift renderizam com dados.

Dois cuidados saíram dessa verificação. O plugin Infinity está fixado na versão 3.7.1, porque a 4.x exige Grafana 11.6
ou mais novo e quebrava a tabela no navegador (erro 404 em `react/jsx-runtime`) sem nenhum erro nas consultas pela API.
Os nomes de imagem levam o prefixo `docker.io/`, porque o Podman não resolve nomes curtos.

### Limitações conhecidas

- A fila de manutenção fica em memória na API. Reiniciar a API esvazia a tabela até chegarem novas leituras.
- O LSTM foi treinado e servido só para o FD001.
- Word2Vec e LDA existem como código e métricas, mas não estão expostos na API.
- O texto das OS é sintético, e nenhum resultado de triagem vale como evidência sobre OS reais.
- A imagem foi construída só para amd64.
- Os modelos finais não foram publicados em GitHub Releases.
- Os experimentos não estão registrados em MLflow nem em CSV. Os números estão neste arquivo e em `docs/resultados/`.

## Ambiente

O `make install` usa Python 3.12. O `gensim` ainda não tem pacote pronto para Python 3.14, então o ambiente precisa ser
criado com 3.12: `uv venv --python 3.12 .venv` e `uv pip install --python .venv -e ".[dev,train,nlp,serve,bert]"`. O
extra `nlp` já inclui o modelo `pt_core_news_sm` 3.8.0. Sem ele, o código cai para `spacy.blank("pt")`.
